"""Score saved pretrained retrieval rankings against the frozen development reference.

Run from the repository root. This command has no model or embedding client.
"""
import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path

from evaluation.compare import write_bundle
from evaluation.run_audit import fingerprint
from evaluation.score import (
    _interval, _match_predicted_track, _mean_track_iou, _temporal_iou,
    _track_boxes, read_jsonl, score_results,
)
from trackrag.emt import load_annotated_tracks, open_recording


RANKERS = {
    "lexical": ["published-task4"],
    "tfidf": ["expanded-tfidf-temporal", "expanded-tfidf-static"],
    "qwen": ["expanded-qwen8b-temporal", "expanded-qwen8b-static"],
}
SOURCES = {"annotated": "annotated", "botsort-gmc": "predicted", "botsort-no-gmc": "predicted"}
FIELDS = ("recording_id", "source", "track_id", "start_frame", "end_frame")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(rows, key):
    indexed = {}
    for row in rows:
        identifier = key(row)
        require(identifier not in indexed, f"Duplicate record {identifier}")
        indexed[identifier] = row
    return indexed


def checked(path, digest, inputs):
    path = Path(path)
    actual = fingerprint(path)
    require(actual == digest, f"Hash mismatch for {path}")
    inputs[str(path)] = actual


def interval_id(row):
    token = ":".join(str(row[field]) for field in FIELDS)
    return sha256(token.encode()).hexdigest()[:24]


def inventory(raw, question, source, boxes, scene_counts, available_frames):
    require(raw["question"] == question, "Inventory differs from the frozen public question")
    candidates = unique(raw["candidates"], lambda c: c["candidate_id"])
    recording, cutoff = question["recording_id"], question["query_frame"]
    for cid, c in candidates.items():
        start, end = _interval(c["start_frame"], c["end_frame"])
        require(c["recording_id"] == recording and c["source"] == source
                and end <= cutoff and start in available_frames and end in available_frames,
                "Invalid candidate recording, source or cutoff")
        require(cid == interval_id(c), "Candidate ID differs from its evidence interval")
        track = c["track_id"]
        if track is None:
            require(start == end, "A scene must identify one frame")
            expected = [start] * scene_counts[start]
        else:
            require(type(track) is int and (recording, track) in boxes, "Unknown track")
            expected = sorted(f for f in boxes[recording, track] if start <= f <= end)
            require(expected and expected[0] == start and expected[-1] == end,
                    "Track interval endpoints must be observed")
        require(c["observed_frames"] == expected, "Inventory frames differ from saved track observations")
    published = raw["published_pool_candidate_ids"]
    require(len(set(published)) == len(published) and set(published) <= candidates.keys(),
            "Invalid published candidate subset")
    require(set(published) == {cid for cid,c in candidates.items() if c["kind"] == "short observed window"},
            "Published pool differs from short-window inventory")
    return candidates, {cid: candidates[cid] for cid in published}


def ranked_rows(ranking, pool):
    require(len(ranking) <= 3, "Expected at most three ranked items")
    seen = set()
    rows = []
    for rank, item in enumerate(ranking, 1):
        cid = interval_id(item)
        require(type(item["rank"]) is int and item["rank"] == rank and math.isfinite(item["score"]),
                "Invalid rank or score")
        require(cid in pool and cid not in seen, "Ranking uses an unknown or duplicate candidate")
        require(all(item[field] == pool[cid][field] for field in FIELDS), "Ranking binding mismatch")
        seen.add(cid)
        rows.append(pool[cid])
    return rows


def identity_matcher(predicted, annotated, source):
    @lru_cache(maxsize=None)
    def associated(recording, track, start, end):
        candidate = dict(recording_id=recording, track_id=track, start_frame=start, end_frame=end)
        return _match_predicted_track(candidate, predicted, annotated, 0.5)[0]

    @lru_cache(maxsize=None)
    def matches(recording, track, start, end, target):
        if source == "annotated":
            return track == target
        # A pair below the shared scorer's edge threshold cannot be assigned.
        overlap = _mean_track_iou(predicted[recording, track], annotated[recording, target], start, end)
        return overlap >= 0.5 and associated(recording, track, start, end) == target
    return matches


def coverage(rows, key, disposition, matches):
    """Component, primary and full-bundle availability in a ranked prefix or full pool."""
    identity_hit = False
    scene_hit = False
    found_scenes = set()
    observed = set()
    endpoints = set(disposition.get("full_evidence_requires_observed_track_endpoints", []))
    targets = [ref for ref in key["acceptable_evidence"] if ref["track_id"] is not None]
    valid_scenes = {ref["start_frame"] for ref in key["acceptable_evidence"] if ref["track_id"] is None}
    for row in rows:
        if row["track_id"] is None:
            found_scenes.add(row["start_frame"])
            scene_hit |= row["start_frame"] in valid_scenes
            continue
        for ref in targets:
            relevant_span = _temporal_iou(row, ref) >= 0.5
            possible_endpoints = endpoints.intersection(row["observed_frames"])
            if (relevant_span or possible_endpoints) and matches(
                row["recording_id"], row["track_id"], row["start_frame"], row["end_frame"], ref["track_id"]
            ):
                identity_hit |= relevant_span
                # A track can switch identities within a correctly associated
                # interval. Credit an endpoint only if its own box is assigned
                # to the target at that exact frame as well.
                observed.update(f for f in possible_endpoints if matches(
                    row["recording_id"], row["track_id"], f, f, ref["track_id"]
                ))
    primary = identity_hit or scene_hit
    extra_scene = disposition.get("full_evidence_requires_additional_exact_scene")
    bundle = primary and endpoints <= observed and (extra_scene is None or extra_scene in found_scenes)
    return dict(primary=primary, identity=identity_hit, scene=scene_hit, bundle=bundle)


def score_run(run_dir, vision_dir, data_root, key_path, review_path):
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_bytes())
    vision_path = vision_dir / "summary.json"
    vision = json.loads(vision_path.read_bytes())
    review = json.loads(review_path.read_bytes())
    inputs = {str(p): fingerprint(p) for p in [summary_path, review_path, Path(__file__)]}
    checked(key_path, review["key"]["sha256"], inputs)
    checked(run_dir / "rankings.jsonl", summary["ranking_sha256"], inputs)
    checked(run_dir / "candidates.jsonl", summary["candidate_sha256"], inputs)
    for record in [*review["evidence_inputs"], *review["logic_inputs"], *summary["inputs"]]:
        checked(record["path"], record["sha256"], inputs)
    require(any(Path(r["path"]).resolve() == vision_path.resolve() for r in summary["inputs"]),
            "Vision summary was not bound to the retrieval run")
    public = [r for p in review["evidence_inputs"] if p["path"].endswith("-questions.jsonl")
              for r in read_jsonl(Path(p["path"]))]
    questions = unique(public, lambda q: q["question_id"])
    dispositions = unique(review["dispositions"], lambda d: d["question_id"])
    gold = unique(read_jsonl(key_path), lambda g: g["question_id"])
    score_results([], gold.values())
    require(set(questions) == set(dispositions) and len(questions) == summary["question_count"]
            == review["counts"]["all_questions"], "Incomplete public questions")
    require(set(gold) == {qid for qid,d in dispositions.items() if d["disposition"] == "scored"}
            and len(gold) == review["counts"]["scored_primary"], "Incomplete scored reference")
    for qid, key in gold.items():
        require(key["answerable"] and all(key[f] == questions[qid][f] for f in ("recording_id","query_frame")),
                "Gold question metadata mismatch")

    conditions = summary["conditions"]
    variants = [v for ranker in summary["rankers"] for v in RANKERS[ranker]]
    require(len(set(conditions)) == len(conditions) and len(set(variants)) == len(variants), "Duplicate conditions")
    inventories = unique(read_jsonl(run_dir / "candidates.jsonl"), lambda r: (r["condition"], r["question"]["question_id"]))
    rankings = unique(read_jsonl(run_dir / "rankings.jsonl"), lambda r: (r["condition"], r["question"]["question_id"],r["variant"]))
    counts = unique(summary["candidate_counts"], lambda r: (r["condition"],r["question_id"]))
    expected = {(c,q) for c in conditions for q in questions}
    require(set(inventories) == set(counts) == expected
            and set(rankings) == {(c,q,v) for c,q in expected for v in variants}, "Incomplete condition/variant matrix")
    track_inputs = unique(summary["track_inputs"], lambda r: (r["condition"], r["recording_id"]))
    recordings = sorted({q["recording_id"] for q in public})
    vision_records = unique(vision["recordings"], lambda r:r["recording_id"])
    record_data = {}
    for name in recordings:
        rec = open_recording(data_root, name)
        require(rec.annotation_path.resolve() in {Path(p["path"]).resolve() for p in review["evidence_inputs"]},
                "Annotation source was not bound to the freeze")
        labels = load_annotated_tracks(rec)
        record_data[name] = (rec, labels, _track_boxes(labels, "annotated"))

    per_case = []
    ceilings = {}
    scores = {}
    for condition in conditions:
        source = SOURCES[condition]
        by_recording = {}
        for name,(rec,labels,annotated_boxes) in record_data.items():
            tracks = labels
            if source == "predicted":
                method = {"botsort-gmc":"sparseOptFlow","botsort-no-gmc":"none"}[condition]
                tracker = unique(vision_records[name]["trackers"],lambda t:t["gmc_method"])[method]
                checked(tracker["tracks_path"], tracker["tracks_sha256"], inputs)
                tracks = read_jsonl(Path(tracker["tracks_path"]))
            digest = sha256(json.dumps(tracks,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            binding = track_inputs[condition,name]
            require(binding["canonical_rows_sha256"] == digest and binding["source"] == source
                    and binding["observation_count"] == len(tracks), "Saved tracks differ from inference input")
            boxes = _track_boxes(tracks,source) if source == "predicted" else annotated_boxes
            by_recording[name] = (boxes, Counter(r["frame_id"] for r in tracks), set(rec.frame_ids),
                                  identity_matcher(boxes,annotated_boxes,source))
        for qid,q in questions.items():
            boxes,scene_counts,frames,matches = by_recording[q["recording_id"]]
            expanded,published = inventory(inventories[condition,qid],q,source,boxes,scene_counts,frames)
            count = counts[condition,qid]
            require((len(published),len(expanded)) == (count["published_candidates"],count["expanded_candidates"]),
                    "Inventory count differs from inference summary")
            result = dict(condition=condition,question_id=qid,disposition=dispositions[qid]["disposition"])
            if qid in gold:
                def evaluate(rows):
                    return coverage(rows, gold[qid], dispositions[qid], matches)
                result["oracle"] = {}
                for label,pool in (("published",published),("expanded",expanded)):
                    flags = evaluate(pool.values())
                    result["oracle"][label] = flags
                    ceilings.setdefault((condition,label),Counter()).update(flags)
                result["rankings"] = {}
            for variant in variants:
                r = rankings[condition,qid,variant]
                require(r["question"] == q and r["source"] == source, "Ranking question/source mismatch")
                pool = published if variant == "published-task4" else expanded
                policies = {p:ranked_rows(r[p],pool) for p in ("raw","reserved_current_scene")}
                scenes = [c for c in pool.values() if c["track_id"] is None]
                raw = policies["raw"]
                raw_scores = [item["score"] for item in r["raw"]]
                require(raw_scores == sorted(raw_scores, reverse=True), "Raw scores are not descending")
                expected_reserved = raw
                if scenes:
                    current = max(scenes,key=lambda c:c["start_frame"])
                    expected_reserved = [c for c in raw if c != current][:2] + [current]
                require(policies["reserved_current_scene"] == expected_reserved, "Reserved-scene policy mismatch")
                if qid not in gold:
                    continue
                result["rankings"][variant] = {}
                for policy,rows in policies.items():
                    prefixes = {f"top{k}":evaluate(rows[:k]) for k in (1,3)}
                    result["rankings"][variant][policy] = prefixes
                    for prefix,flags in prefixes.items():
                        scores.setdefault((condition,variant,policy,prefix),Counter()).update(flags)
            per_case.append(result)
    aggregates = []
    for (condition, variant, policy, prefix), flags in scores.items():
        pool = "published" if variant == "published-task4" else "expanded"
        possible = ceilings[condition,pool]["primary"]
        aggregates.append(dict(condition=condition, variant=variant, policy=policy, prefix=prefix,
                               **flags, full_denominator=len(gold), pool_covered=possible,
                               primary_rate=flags["primary"] / len(gold),
                               primary_rate_when_pool_covers=flags["primary"] / possible if possible else None))
    return {
        "scope":"Development retrieval only. Bundle measures availability in selected intervals, before image sampling.",
        "temporal_iou":0.5,"predicted_box_iou":0.5,"primary_denominator":len(gold),
        "candidate_endpoint_policy":"Require each endpoint to be observed on a target-associated interval and associated to the target at that exact frame. Spanning it does not establish availability.",
        "inputs":inputs,"dispositions":list(dispositions.values()),
        "target_kind_denominators":review["target_kind_counts"],
        "oracle":[dict(condition=c,pool=p,**flags) for (c,p),flags in ceilings.items()],
        "scores":aggregates,
        "per_case":per_case,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--vision-run",type=Path,required=True)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--key",type=Path,default=Path("experiments/pretrained-reference.jsonl"))
    parser.add_argument("--review",type=Path,default=Path("experiments/pretrained-reference-review.json"))
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args(argv)
    report = score_run(args.run,args.vision_run,args.data_root,args.key,args.review)
    cases = report.pop("per_case")
    write_bundle(args.out,{"summary.json":json.dumps(report,indent=2)+"\n",
                           "cases.jsonl":"".join(json.dumps(r)+"\n" for r in cases)})
    print(json.dumps({"questions":len(report["dispositions"]),"primary_denominator":report["primary_denominator"],
                      "oracle":report["oracle"]}))


if __name__ == "__main__":
    main()
