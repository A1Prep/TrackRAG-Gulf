"""Prepare and verify one bounded public side-ground trial using the normal vision dependencies."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import cv2

from tools.passenger_release import PUBLIC_PHOTO_LISTS, ROOT
from trackrag.dropoff import manual_surfaces, vehicle_curb_ground, vehicle_ground_reference as identity
from trackrag.dropoff import vehicle_side_ground as side, vehicle_side_view as view
from trackrag.dropoff.bay_packets import read_photo_packet


TRIAL = ROOT / "runs/passenger-placement-field-20261124/vehicle-side"
GALLERY = ROOT / "data/uae-parking-study"
MACHINE_REPORT = "passenger-vehicle-side-ground-observations-20261124.json"
INDEPENDENT_REPORT = "passenger-vehicle-lateral-independent-photo-labels-20261124.json"


def prepare():
    allowlist = {p["sha256"] for name in PUBLIC_PHOTO_LISTS
                 for p in json.loads((ROOT/"experiments/dropoff"/name).read_bytes())["photos"]}
    seen, jobs, blinded = set(), [], {}
    for line in (GALLERY/"passenger-requests.jsonl").read_bytes().splitlines():
        scene = json.loads(line)["scene"]
        key = scene["provenance"]["image_sha256"]
        if (key in seen or key not in allowlist or scene["source"] != "predicted"
                or scene["provenance"].get("network_policy") == "local_only"):
            continue
        seen.add(key)
        if not read_photo_packet(identity.BANK, key, 8192):
            continue
        surface = manual_surfaces.get_packet(key)
        frame = scene["frame"]
        classes = manual_surfaces.decode(surface, frame, key)
        specs = vehicle_curb_ground.describe(scene, classes)
        vehicle = side.identity_packet(key, specs)
        rows = side.geometry_rows(scene, classes, specs, vehicle["body"]["answers"])
        if not rows:
            continue
        source = (GALLERY / (key + ".jpg")).resolve()
        if sha256(source.read_bytes()).hexdigest() != key:
            raise ValueError("Registered public original changed")
        jobs.append({"source": key, "context": side.context(scene, surface, vehicle), "rows": rows})
        for row in rows:
            ident = view.digest([key, row["anchor_box"], row["point"]])
            blinded[ident] = {"id": ident, "image_sha256": key, "original_path": str(source),
                              "recording_id": frame["recording_id"], "frame_id": frame["frame_id"],
                              "anchor_box": row["anchor_box"], "point": row["point"]}
    return jobs, [blinded[key] for key in sorted(blinded)]


def grade():
    """Join the sealed independent pixel judgements to candidate identities after the model has finished."""
    from trackrag.dropoff.vehicle_side_ground import release_passes
    model_report = json.loads((TRIAL/MACHINE_REPORT).read_bytes())
    labels_path = ROOT/"experiments/dropoff"/INDEPENDENT_REPORT
    independent = json.loads(labels_path.read_bytes())
    if independent["independent_inputs_only_sha256"] != sha256((TRIAL/"independent-inputs-only.json").read_bytes()).hexdigest():
        raise ValueError("Independent labels were collected against other photo inputs")
    labels = {row["id"]: row for row in independent["records"]}
    if set(labels) != {case["point_id"] for case in model_report["cases"]}:
        raise ValueError("Incomplete independent image review")
    counts = {"independently_supported_offers": 0, "accepted_negative_controls": 0, "accepted_wrong_places": 0}
    by_role = defaultdict(Counter)
    for case in model_report["cases"]:
        answer = labels[case["point_id"]]["independent_grade"]
        if answer not in {"LEFT","RIGHT","NOT","UNCERTAIN"}:
            raise ValueError("Invalid independent side grade")
        grade = ("supported" if answer == case["asked_side"].upper() else
                 "uncertain" if answer == "UNCERTAIN" else "incorrect")
        role = case["role"]
        by_role[role]["all_"+grade] += 1
        if case["wide"] == case["close"] == "OPEN":
            by_role[role]["dual_open_"+grade] += 1
        if case["passed_source_checks"]:
            by_role[role]["source_checked_"+grade] += 1
            if role != "candidate":
                counts["accepted_negative_controls"] += 1
            elif grade == "supported":
                counts["independently_supported_offers"] += 1
            elif grade == "incorrect":
                counts["accepted_wrong_places"] += 1
    declared = {"recipe":view.RECIPE, "bank_sha256":model_report["archive_sha256"],
                "review_artifact":INDEPENDENT_REPORT,"review_sha256":sha256(labels_path.read_bytes()).hexdigest(),
                "model_observations":MACHINE_REPORT, "model_observations_sha256":sha256((TRIAL/MACHINE_REPORT).read_bytes()).hexdigest(),
                "reviewer":"Independent AI inspection of the original public photographs. No human annotation.",
                "release_counts":counts,"by_input_role":dict(by_role),"enabled":release_passes(counts),
                "scope":"One global decision. Reviewer judgements never select individual photos or vehicles in the running app."}
    (TRIAL/side.RELEASE.name).write_text(json.dumps(declared, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(declared, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "publish", "grade"))
    action = parser.parse_args().action
    if action == "grade":
        grade()
        return
    cv2.setNumThreads(1)
    jobs, blinded = prepare()
    TRIAL.mkdir(parents=True, exist_ok=True)
    if action == "prepare":
        (TRIAL/"jobs.json").write_text(json.dumps(jobs, allow_nan=False), encoding="utf-8")
        (TRIAL/"independent-inputs-only.json").write_text(json.dumps(blinded, indent=2, allow_nan=False), encoding="utf-8")
        print(f"{len(jobs)} original photos, {sum(len(j['rows']) for j in jobs)} inputs, {len(blinded)} distinct points", flush=True)
        return
    if json.loads((TRIAL/"jobs.json").read_bytes()) != jobs:
        raise ValueError("Current source-bound inputs differ from the prepared batch")
    packets = json.loads((TRIAL/"responses.json").read_bytes())
    if set(packets) != {job["source"] for job in jobs}:
        raise ValueError("Incomplete lateral-ground batch")
    # Keep the signed trial out of the live app until the independent photo review is complete.
    target_bank = TRIAL/side.BANK.name
    temporary = target_bank.with_suffix(".pending")
    counts = defaultdict(Counter)
    cases = []
    failures = []
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        for job in sorted(jobs, key=lambda row: row["source"]):
            key = job["source"]
            photo = (GALLERY/(key+".jpg")).read_bytes()
            verdicts = side.verify_rows(photo, key, job["rows"], packets[key], job["context"])
            qualified = side.qualified_indices(job["rows"], verdicts)
            for index, (row, pair) in enumerate(zip(job["rows"], verdicts, strict=True)):
                role = row["role"]
                stats = counts[role]
                stats["total"] += 1
                stats["wide_open"] += pair[0] == "OPEN"
                stats["close_open"] += pair[1] == "OPEN"
                stats["wide_and_close_open"] += pair == ("OPEN", "OPEN")
                stats["returned_after_source_controls"] += index in qualified
                if role != "candidate" and pair == ("OPEN", "OPEN"):
                    failures.append({"photo": key, "role": role})
                cases.append({"point_id": view.digest([key, row["anchor_box"], row["point"]]),
                              "photo": key, "track_id": row["track_id"], "asked_side": row["side"], "role": role,
                              "wide": pair[0], "close": pair[1], "passed_source_checks": index in qualified})
            print(key[:12], len(verdicts), "verified", flush=True)
            member = ZipInfo(key+".json", date_time=(1980,1,1,0,0,0))
            member.compress_type = ZIP_DEFLATED
            archive.writestr(member, json.dumps(packets[key], sort_keys=True, separators=(",", ":")))
    temporary.replace(target_bank)
    report = {"study": "Public lateral vehicle-ground checks", "scope": "Existing public KartaView originals.",
              "model": identity.MODEL, "revision": identity.REVISION, "recipe": view.RECIPE,
              "archive": target_bank.name, "archive_sha256": sha256(target_bank.read_bytes()).hexdigest(),
              "images": len(packets), "individual_model_views": 2*len(cases),
              "method": "Separate wide and close readings by the same model. A dual OPEN on a wrong-side, occupied or road control rejects all candidates on that original.",
              "counts": dict(counts), "failed_source_controls": failures, "cases": cases}
    target = TRIAL/MACHINE_REPORT
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"Verified {len(packets)} original photographs in the trial folder: {target_bank.name}", flush=True)


if __name__ == "__main__":
    main()
