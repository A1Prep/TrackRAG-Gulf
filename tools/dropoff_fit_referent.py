"""Fit and score a predicted-object ranker on the two frozen, disjoint public partitions."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import minimize

from tools.dropoff_anchor_fit import TRAIN_SHA, DEV_SHA
from tools.dropoff_bootstrap_regseg import SCENE_METADATA_SHA256, SCENE_METADATA_URL, METADATA_URL, METADATA_SHA256
from tools.dropoff_public_candidates import CANDIDATES_SHA256, read_candidates
from trackrag.dropoff.data import load_cases, validate_disjoint_groups
from trackrag.dropoff.referent import (
    KIND, REGION_NAMES, ALIGN_NAMES, DEFAULT_REG, ReferentRanker, category, words, region_features,
    alignment, encode, overlap, ordered_alternatives,
)


def checked(path, digest):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Pinned file differs: " + str(path))
    return data


def corners(xywh, size):
    x, y, w, h = map(float, xywh)
    box = [max(0., x), max(0., y), min(size[0], x+w), min(size[1], y+h)]
    return box if np.isfinite(box).all() and box[0] < box[2] and box[1] < box[3] else None


def cases_for_training(rows, detections, commands):
    result = []
    for i, case in enumerate(rows, 1):
        req, src = case["request"], case["source"]
        key, image_path = src["command_token"], Path(req["scene"]["frame"]["image_path"])
        official, proposals = commands[key], detections[str(src["source_index"])]
        if (key != proposals["command_token"] or req["text"].strip() != official["command"].strip()
                or src["scene_token"] != official["scene_token"] or src["sample_token"] != official["sample_token"]
                or image_path.name != official["t2c_img"]):
            raise ValueError("Unverified source join for " + case["case_id"])
        with Image.open(image_path) as image:
            size = image.size
            if size != tuple(req["scene"]["frame"][axis] for axis in ("width", "height")):
                raise ValueError("Unexpected original image dimensions")
            anchor = corners(official["2d_box"], size)
            if not anchor:
                raise ValueError("Empty visible reference annotation")
            boxes, categories = [], []
            for det in proposals["predictions"]:
                visible = corners(det["bbox"], size)
                if visible:
                    boxes.append(visible)
                    categories.append(category(det["class"]))
            regions, hist, middle = region_features(image, boxes, categories)
        aligned = alignment(req["text"], categories, regions, hist, middle)
        ious = overlap(boxes, anchor)
        weights = np.append(np.where(ious >= .5, ious, 0.), float(not np.any(ious >= .5)))
        weights /= weights.sum()
        result.append({"id": case["case_id"], "group": case["scene_group"], "text": req["text"],
                       "image_sha": src["image_sha256"], "targets": weights, "ious": ious,
                       "categories": categories, "boxes": boxes, "regions": regions, "aligned": aligned,
                       "raw_proposals": len(proposals["predictions"])})
        if i % 100 == 0:
            print(f"Loaded {i}/{len(rows)} source images and predicted proposals", flush=True)
    return result


def train(rows, use_language):
    terms = Counter()
    for row in rows:
        terms.update(words(row["text"]))
    vocab = [term for term, count in sorted(terms.items(), key=lambda x: (-x[1], x[0]))
             if count >= 5][:130] if use_language else []
    text = np.stack([encode(row["text"], vocab) for row in rows])
    sizes = [len(row["regions"]) for row in rows]
    starts = np.cumsum([0, *sizes[:-1]])
    owner = np.repeat(np.arange(len(rows)), sizes)
    z = np.concatenate([row["regions"] for row in rows])
    a = np.concatenate([row["aligned"] for row in rows])
    truth = np.concatenate([row["targets"] for row in rows])
    d, v, f = z.shape[1], len(vocab), a.shape[1]
    n = len(rows)
    regularization = np.r_[np.full(d, DEFAULT_REG["base"]), np.full(v*d, DEFAULT_REG["words"]),
                           np.full(f, DEFAULT_REG["align"])]
    if not use_language:
        a.fill(0)

    def loss(params):
        base, interactions, aligned = params[:d], params[d:(v+1)*d].reshape(v, d), params[(v+1)*d:]
        scores = np.einsum("ij,ij->i", z, (base + text @ interactions)[owner]) + a @ aligned
        shifted = scores - np.maximum.reduceat(scores, starts)[owner]
        mass = np.exp(shifted)
        denom = np.add.reduceat(mass, starts)
        logloss = np.log(denom).sum() - truth @ shifted
        err = mass/denom[owner] - truth
        by_case = np.add.reduceat(err[:, None]*z, starts, axis=0)
        grad = np.r_[by_case.sum(axis=0), (text.T @ by_case).ravel(), a.T @ err]
        return (logloss + .5*np.dot(regularization, params*params))/n, (grad + regularization*params)/n

    opt = minimize(loss, np.zeros(d*(v+1)+f), jac=True, method="L-BFGS-B",
                   options={"maxiter": 140, "ftol": 1e-9, "gtol": 8e-6, "maxcor": 10})
    values = opt.x
    result = {"kind": KIND, "region_names": list(REGION_NAMES), "align_names": list(ALIGN_NAMES),
              "vocabulary": vocab, "base": values[:d].tolist(), "interactions": values[d:(v+1)*d].reshape(v,d).tolist(),
              "aligned": values[(v+1)*d:].tolist(), "training_split_sha256": TRAIN_SHA,
              "landmark_labels_sha256": SCENE_METADATA_SHA256,
              "input": "current predicted boxes, pooled RGB from current original frame and instruction",
              "uses_language": use_language}
    return result, {"iterations": opt.nit, "termination": str(opt.message), "regularized_ce": float(opt.fun),
                    "gradient_max": float(np.abs(opt.jac).max()), "parameters": len(values),
                    "regularization": DEFAULT_REG, "maxiter_predeclared": 140, "vocab_terms": len(vocab)}


def evaluate(model, rows):
    predictions = []
    for r in rows:
        probs = model.scores(r["text"], r["regions"], r["aligned"])
        order = ordered_alternatives(r["boxes"], r["categories"], probs[:-1])
        null = not order or probs[-1] >= probs[order[0]]
        ious = r["ious"]
        best = float(ious[order[0]]) if order else 0.
        top3 = max((float(ious[j]) for j in order[:3]), default=0.)
        predictions.append({"id": r["id"], "scene_group": r["group"], "image_sha256": r["image_sha"],
                            "match_available": bool(np.any(ious>=.5)), "top1_object_iou": best,
                            "top1_object_hit": best>=.5, "top3_distinct_object_hit": top3>=.5,
                            "null_wins": bool(null), "null_score": float(probs[-1]),
                            "selection_correct": bool((null and not (ious>=.5).any()) or (not null and best>=.5)),
                            "visible_proposals": len(ious), "raw_proposals": r["raw_proposals"],
                            "positive_proposals": int((ious>=.5).sum()), "distinct_proposals": len(order)})
    count = lambda key: sum(bool(r[key]) for r in predictions)
    avail = count("match_available")
    return {"rows":len(predictions), "groups":len({r["group"] for r in rows}),
            "matching_detector_candidate_available":avail,
            "top1_object_hit_all":count("top1_object_hit"), "top3_distinct_object_hit_all":count("top3_distinct_object_hit"),
            "selection_correct_including_null":count("selection_correct"),
            "top1_object_given_available":count("top1_object_hit")/avail if avail else None,
            "absent_correctly_selected_null":sum(r["null_wins"] and not r["match_available"] for r in predictions),
            "false_null_on_visible_reference":sum(r["null_wins"] and r["match_available"] for r in predictions),
            "null_selected":count("null_wins"),
            "mean_top1_object_iou":float(np.mean([r["top1_object_iou"] for r in predictions])),
            "total_distinct_detector_candidates":sum(r["distinct_proposals"] for r in predictions),
            "total_discarded_empty_or_invisible_proposals":sum(r["raw_proposals"]-r["visible_proposals"] for r in predictions)
            }, predictions


def write(path, data):
    raw = (json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":"))+"\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def fit(splits, metadata, proposals, out):
    if out.exists():
        raise ValueError("Choose a fresh directory for this fixed training run")
    out.mkdir(parents=True)
    commands = {r["command_token"]:r for r in json.loads(checked(metadata, SCENE_METADATA_SHA256))["commands"]}
    detections = read_candidates(proposals)
    proposal_file_sha = hashlib.sha256(Path(proposals).read_bytes()).hexdigest()
    rows = {}
    for partition, digest in (("train", TRAIN_SHA), ("development", DEV_SHA)):
        path = splits / (partition+".jsonl")
        rows[partition] = load_cases(path, allowed_partitions={partition}, payload=checked(path, digest))
    validate_disjoint_groups(rows["train"], rows["development"])
    ids = {str(r["source"]["source_index"]) for values in rows.values() for r in values}
    if set(detections) != ids or len(ids) != 625:
        raise ValueError("Proposals are not exactly the pre-frozen 500 training and 125 development inputs")
    train_rows = cases_for_training(rows["train"], detections, commands)
    # Fit and persist both heads before accessing any development images or reference labels.
    receipt = {"model":KIND, "train_sha256":TRAIN_SHA, "development_sha256":DEV_SHA,
               "public_candidates_sha256":proposal_file_sha, "canonical_candidates_sha256":CANDIDATES_SHA256,
               "public_metadata_url": METADATA_URL,
               "public_metadata_sha256":METADATA_SHA256, "reference_labels_url":SCENE_METADATA_URL,
               "reference_labels_sha256":SCENE_METADATA_SHA256, "training_rows":len(train_rows),
               "training_groups":len({r["group"] for r in train_rows}),
               "train_match_available":sum((r["ious"] >= .5).any().item() for r in train_rows),
               "source_dataset":"Talk2Car-RegSeg training partition; pinned official CenterNet predictions; original Talk2Car reference boxes",
               "positive_set":"All valid visible predicted boxes with IoU >= 0.5 to the single original referent, weighted by IoU; otherwise train NULL.",
               "development_used_for_model_selection":False, "official_final_or_blind_examined":False,
               "limitations":["Single annotated referent cannot establish true passenger ambiguity or safe stopping.",
                              "Scores are uncalibrated. Transfer to UAE tracks and arbitrary spoken requests is unmeasured."],
               "conditions":{}}
    for name, use_language in (("conditioned",True), ("language_blind",False)):
        print("Fitting "+name, flush=True)
        coeffs, optimization = train(train_rows, use_language)
        coeffs["official_proposal_subset_sha256"] = proposal_file_sha
        receipt["conditions"][name] = {"checkpoint_sha256": write(out/(name+".json"), coeffs),
                                      "optimization": optimization}
    dev_rows = cases_for_training(rows["development"], detections, commands)
    all_predictions = {}
    for name, condition in receipt["conditions"].items():
        checkpoint = ReferentRanker.load(out/(name+".json"), expected_sha256=condition["checkpoint_sha256"])
        condition["development"], predictions = evaluate(checkpoint, dev_rows)
        condition["prediction_sha256"] = write(out/(name+"-public-development.json"), predictions)
        all_predictions[name] = predictions
    lhs, rhs = all_predictions["conditioned"], all_predictions["language_blind"]
    receipt["paired_top1_object"] = {
        "conditioned_only":sum(a["top1_object_hit"] and not b["top1_object_hit"] for a,b in zip(lhs,rhs)),
        "blind_only":sum(b["top1_object_hit"] and not a["top1_object_hit"] for a,b in zip(lhs,rhs)),
        "both":sum(a["top1_object_hit"] and b["top1_object_hit"] for a,b in zip(lhs,rhs))}
    write(out/"fit.json",receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("splits", "metadata", "proposals", "out"):
        parser.add_argument("--"+field, type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(fit(a.splits,a.metadata,a.proposals,a.out), indent=2))
