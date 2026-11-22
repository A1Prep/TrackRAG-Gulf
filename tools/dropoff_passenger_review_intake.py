"""Validate individual human exports from the two nonfinal passenger review forms.

An intake is an immutable snapshot. Individual exports, conflicting answers and
human adjudication are retained separately from the four-field training files.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unicodedata

from tools.dropoff_passenger_references_pack import PARTITIONS, review_inputs
from trackrag.dropoff.contracts import FULL_FIELDS, STATUSES, validate_target
from trackrag.dropoff.data import _unique_object, supervision_sha256, validate_case, validate_disjoint_groups


FIELDS = sorted(FULL_FIELDS)
ROW_KEYS = {"request_id", "reviewer", "reviewed_at", "recording_id", "frame_id", "image_sha256",
            "input_sha256", "target", "notes"}
EXPORT_KEYS = {"kind", "version", "reviewer", "input_bank_sha256", "expected_cases", "completed"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return sha256(value).hexdigest()


def read_json(path):
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError(f"return file is too large: {path.name}")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object)
    canonical(value)  # Reject nonfinite JSON anywhere, including optional notes/metadata.
    require(isinstance(value, dict), "expected a JSON object")
    return raw, value


def write_json(path, value):
    path.write_bytes((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def write_rows(path, rows):
    path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))


def timestamp(value):
    require(isinstance(value, str), "a signed timestamp is required")
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(date.tzinfo is not None, "review timestamp needs a timezone")
    return date.astimezone(timezone.utc)


def reviewer_key(name):
    require(isinstance(name, str) and 0 < len(name.strip()) <= 100, "a named reviewer or team ID is required")
    require(not any(unicodedata.category(c).startswith("C") for c in name), "invalid characters in reviewer name")
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def attested(value, name, fixture):
    kind = "software_fixture" if fixture else "human"
    require(value.get("source") == kind and value.get("human_attested") is (not fixture),
            f"expected an explicit {kind} source and a matching attestation")
    identity = reviewer_key(name)
    if not fixture:
        flags = identity.replace("_", " ").replace("-", " ")
        require(not any(tag in flags for tag in ("software fixture", "test fixture", "ai reviewer", "ai visual annotator"))
                and flags not in {"codex", "chatgpt"}, "software or AI reviews cannot enter human intake")


def legacy_confirmations(path, archive, fixture):
    if path is None:
        return {}
    raw, sheet = read_json(path)
    require(set(sheet) == {"kind", "version", "confirmations"} and sheet["kind"] == "legacy_passenger_review_confirmations"
            and type(sheet["version"]) is int and sheet["version"] == 1 and isinstance(sheet["confirmations"], list),
            "invalid legacy confirmation sheet")
    (archive / "legacy-confirmations.json").write_bytes(raw)
    result = {}
    for entry in sheet["confirmations"]:
        require(isinstance(entry, dict) and set(entry) == {"export_sha256", "reviewer", "reviewed_at", "source", "human_attested"},
                "invalid named legacy confirmation")
        attested(entry, entry["reviewer"], fixture)
        timestamp(entry["reviewed_at"])
        key = entry["export_sha256"]
        require(isinstance(key, str) and len(key) == 64 and key not in result, "duplicate or invalid legacy export SHA256")
        result[key] = entry
    return result


def collect(returns, banks, archive, confirmations, fixture):
    by_hash = {bank_hash: (partition, rows, requests) for partition, (bank_hash, rows, requests) in banks.items()}
    ballots, files, seen, used, pending = defaultdict(dict), [], set(), set(), []
    for path in sorted(returns.iterdir()):
        require(path.is_file() and path.suffix.lower() == ".json", f"place only individual review JSONs in the returns folder: {path.name}")
        raw, export = read_json(path)
        sha = digest(raw)
        if sha in seen:
            continue
        seen.add(sha)
        version = export.get("version")
        require(type(version) is int and version in (1, 2) and export.get("kind") == "independent_passenger_review",
                "unsupported reviewer export")
        extra = {"partition", "source", "human_attested"} if version == 2 else set()
        require(set(export) == EXPORT_KEYS | extra, "missing or unexpected reviewer export fields")
        name = export["reviewer"]
        reviewer_key(name)
        bank_hash = export["input_bank_sha256"]
        require(isinstance(bank_hash, str) and bank_hash in by_hash, "export belongs to neither frozen nonfinal instruction bank")
        partition, rows, requests = by_hash[bank_hash]
        require(type(export["expected_cases"]) is int and export["expected_cases"] == len(rows),
                "wrong frozen bank size on review")
        confirmation = confirmations.get(sha)
        signed = version == 2 or confirmation is not None
        if version == 2:
            require(export["partition"] == partition, "review partition does not match the frozen bank")
            attested(export, name, fixture)
        elif confirmation:
            require(confirmation["reviewer"] == name, "legacy attestation must come from the named original reviewer")
            used.add(sha)
        else:
            pending.append({"export_sha256": sha, "reviewer": name, "source": "human", "human_attested": False, "reviewed_at": None})
        completed = export["completed"]
        require(isinstance(completed, list) and 0 < len(completed) <= len(rows), "invalid completed review count")
        completed_ids = set()
        for row in completed:
            require(isinstance(row, dict) and set(row) == ROW_KEYS, "missing or unexpected individual answer fields")
            case_id = row["request_id"]
            require(isinstance(case_id, str) and case_id in rows and case_id not in completed_ids,
                    "foreign or duplicate case in individual export")
            completed_ids.add(case_id)
            source = rows[case_id]
            require(row["reviewer"] == name and type(row["frame_id"]) is int and all(row[k] == source[k]
                    for k in ("recording_id", "frame_id", "image_sha256", "input_sha256")), "individual answer has stale input or reviewer binding")
            when = timestamp(row["reviewed_at"])
            if confirmation:
                require(when <= timestamp(confirmation["reviewed_at"]), "legacy signature predates the reviewed answer")
            require(isinstance(row["notes"], str) and len(row["notes"]) <= 2000, "invalid reviewer notes")
            frame = requests[case_id]["scene"]["frame"]
            target = validate_target(row["target"], FIELDS, frame)
            if target["status"] == "proposed":
                x, y = target["destination_point_xy"]
                box = target["destination_box_xyxy"]
                require(x < box[2] and y < box[3], "point must be inside the review form's half-open box")
            if signed:
                row_sha = digest(canonical(row))
                if row_sha not in ballots[case_id]:
                    ballots[case_id][row_sha] = {**row, "target": target, "row_sha256": row_sha,
                        "supervision_sha256": supervision_sha256(target, FIELDS, frame), "export_sha256": []}
                ballots[case_id][row_sha]["export_sha256"].append(sha)
        (archive / (sha + ".json")).write_bytes(raw)
        files.append({"sha256": sha, "reviewer": name, "partition": partition, "named_attestation": signed,
                      "completed_rows": len(completed)})
    require(used == confirmations.keys(), "legacy confirmation must match an actual legacy export in this intake")
    return ballots, files, pending


def conflict_sheet(case_id, answers):
    choices = []
    for answer in sorted(answers, key=lambda a: a["row_sha256"]):
        choices.append({k: answer[k] for k in ("reviewer", "reviewed_at", "row_sha256", "supervision_sha256", "target", "notes")})
    return {"request_id": case_id, "recording_id": answers[0]["recording_id"], "frame_id": answers[0]["frame_id"],
            "image_sha256": answers[0]["image_sha256"], "input_sha256": answers[0]["input_sha256"],
            "review_set_sha256": digest(canonical(choices)), "choices": choices, "chosen_supervision_sha256": None,
            "reason": ""}


def signed_decisions(path, conflicts, archive, fixture):
    if path is None:
        return {}
    raw, sheet = read_json(path)
    require(set(sheet) == {"kind", "version", "source", "human_attested", "adjudicator", "reviewed_at", "decisions"}
            and sheet["kind"] == "passenger_review_adjudication" and type(sheet["version"]) is int and sheet["version"] == 1
            and isinstance(sheet["decisions"], list), "invalid adjudication sheet")
    signer = sheet["adjudicator"]
    attested(sheet, signer, fixture)
    when = timestamp(sheet["reviewed_at"])
    decisions, seen = {}, set()
    for row in sheet["decisions"]:
        require(isinstance(row, dict) and isinstance(row.get("request_id"), str), "missing adjudicated case")
        identity = row["request_id"]
        require(identity in conflicts and identity not in seen, "duplicate or inapplicable adjudicated case")
        seen.add(identity)
        expected = conflicts[identity]
        require(set(row) == set(expected) and all(row[k] == expected[k] for k in expected
                if k not in ("chosen_supervision_sha256", "reason")), "adjudication is stale or altered")
        choice, reason = row["chosen_supervision_sha256"], row["reason"]
        if choice is None:
            require(reason == "", "an unresolved adjudication must remain blank")
            continue
        require(isinstance(reason, str) and reason.strip() and len(reason) <= 2000,
                "the adjudicator must explain the selected target")
        require(choice in {c["supervision_sha256"] for c in expected["choices"]},
                "adjudication must choose an exact submitted target")
        require(all(timestamp(c["reviewed_at"]) <= when for c in expected["choices"]), "adjudication predates individual responses")
        decisions[identity] = {"adjudicator": signer, "reviewed_at": sheet["reviewed_at"], "reason": reason,
                               "review_set_sha256": expected["review_set_sha256"], "chosen_supervision_sha256": choice,
                               "adjudication_file_sha256": digest(raw)}
    (archive / "adjudication.json").write_bytes(raw)
    return decisions


def resolve(answers, decision):
    """Return the exact selected individual answer and the people actually supporting it."""
    if not answers:
        return "awaiting_review", None, []
    choices = {a["supervision_sha256"] for a in answers}
    if len(choices) > 1 and decision is None:
        return "awaiting_adjudication", None, []
    selected = decision["chosen_supervision_sha256"] if decision else next(iter(choices))
    answer = next(a for a in answers if a["supervision_sha256"] == selected)
    votes = defaultdict(set)
    for row in answers:
        votes[reviewer_key(row["reviewer"])].add(row["supervision_sha256"])
    names = {reviewer_key(a["reviewer"]): a["reviewer"] for a in answers
             if votes[reviewer_key(a["reviewer"])] == {selected}}
    if decision:
        names[reviewer_key(decision["adjudicator"])] = decision["adjudicator"]
    supporters = [names[key] for key in sorted(names)]
    if answer["target"]["status"] == "needs_clarification" and (len(votes) < 2 or len(supporters) < 2):
        return "awaiting_second_clarification_review", None, supporters
    method = "human_adjudicated" if decision else ("exact_agreement" if len(supporters) >= 2 else "single_review")
    return method, answer, supporters


def reviewed_case(request, row, target, supporters, answers, method, decision, fixture):
    dates = [timestamp(a["reviewed_at"]) for a in answers]
    if decision:
        dates.append(timestamp(decision["reviewed_at"]))
    initial_votes = defaultdict(set)
    for answer in answers:
        initial_votes[reviewer_key(answer["reviewer"])].add(answer["supervision_sha256"])
    initial_names = {reviewer_key(a["reviewer"]): a["reviewer"] for a in answers}
    review = {"status": "human_reviewed", "reviewers": supporters, "resolution": method,
              "independent_reviewers": [initial_names[k] for k in sorted(initial_names)],
              "independent_supporters_before_adjudication": [initial_names[k] for k in sorted(initial_names)
                                                           if initial_votes[k] == {target["supervision_sha256"]}],
              "reviewed_at": max(dates).isoformat(), "image_sha256": row["image_sha256"],
              "input_sha256": row["input_sha256"], "supervision_sha256": target["supervision_sha256"],
              "individual_row_sha256": sorted(a["row_sha256"] for a in answers),
              "export_sha256": sorted({sha for a in answers for sha in a["export_sha256"]}),
              "adjudication": decision, "software_fixture": fixture}
    return {"schema_version": 1, "case_id": row["request_id"], "scene_group": "emt:" + row["recording_id"],
            "partition": row["partition"], "source": {"dataset": "EMT", "source_split": "train",
            "recording_id": row["recording_id"], "image_sha256": row["image_sha256"],
            "instruction_origin": "AI-authored daytime diagnostic", "target_origin": "software_fixture" if fixture else "named_human_review"},
            "request": request, "target": target["target"], "supervised_fields": FIELDS, "review": review}


def ingest(bank, requests, returns, out, *, decisions=None, legacy=None, _software_fixture=False):
    if out.exists():
        raise FileExistsError("choose a new output directory for each intake snapshot")
    require(returns.is_dir(), "create a dedicated returns directory before recording intake")
    require(set(requests) == set(PARTITIONS), "provide both nonfinal prepared partitions")
    banks = {partition: review_inputs(bank, requests[partition], partition=partition) for partition in PARTITIONS}
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".passenger-intake-", dir=out.parent) as temporary:
        # The staging directory and destination share a volume. Publish only after all validations.
        stage = Path(temporary)
        archive = stage / "original-exports"
        archive.mkdir(parents=True)
        confirmations = legacy_confirmations(legacy, archive, _software_fixture)
        ballots, files, pending = collect(returns, banks, archive, confirmations, _software_fixture)
        conflicts = {case_id: conflict_sheet(case_id, list(rows.values())) for case_id, rows in ballots.items()
                     if len({r["supervision_sha256"] for r in rows.values()}) > 1}
        signed = signed_decisions(decisions, conflicts, archive, _software_fixture)
        choices = {"kind": "passenger_review_adjudication", "version": 1, "source": "human",
                   "human_attested": False, "adjudicator": "", "reviewed_at": None,
                   "decisions": [conflicts[key] for key in sorted(conflicts)]}
        write_json(stage / "adjudication-template.json", choices)
        if pending:
            write_json(stage / "legacy-confirmations-template.json", {"kind": "legacy_passenger_review_confirmations",
                       "version": 1, "confirmations": pending})
        traces, ready, summary = [], {}, {}
        for partition, (bank_sha, rows, prepared) in banks.items():
            accepted = []
            status_counts = dict.fromkeys(sorted(STATUSES), 0)
            resolutions = {}
            for identity, row in rows.items():
                answers = sorted(ballots[identity].values(), key=lambda vote: vote["row_sha256"])
                decision = signed.get(identity)
                method, selected, supporters = resolve(answers, decision)
                resolutions[method] = resolutions.get(method, 0) + 1
                trace = {"partition": partition, "request_id": identity, "recording_id": row["recording_id"],
                         "frame_id": row["frame_id"], "input_sha256": row["input_sha256"], "image_sha256": row["image_sha256"],
                         "resolution": method, "supporters_of_selected_target": supporters,
                         "selected_supervision_sha256": selected["supervision_sha256"] if selected else None,
                         "adjudication": decision, "individual_answers": answers}
                traces.append(trace)
                if selected is None:
                    continue
                case = reviewed_case(prepared[identity], {**row, "partition": partition}, selected, supporters,
                                     answers, method, decision, _software_fixture)
                validated = validate_case(case, allowed_partitions=(partition,))
                accepted.append(validated)
                status_counts[validated["target"]["status"]] += 1
            ready[partition] = accepted
            summary[partition] = {"recording_id": PARTITIONS[partition][0], "expected_cases": len(rows),
                                  "input_bank_sha256": bank_sha, "eligible_cases": len(accepted),
                                  "resolutions": resolutions, "eligible_status_counts": status_counts}
        validate_disjoint_groups(ready["train"], ready["development"])
        written = {}
        for partition, cases in ready.items():
            if not cases:
                continue
            target = stage / (partition + ".jsonl")
            for case in cases:
                # Relative to this immutable snapshot. A teammate can rebuild it from their own prepared originals.
                frame = case["request"]["scene"]["frame"]
                try:
                    frame["image_path"] = os.path.relpath(frame["image_path"], out)
                except ValueError:  # Windows drive letters can differ.
                    pass
            write_rows(target, cases)
            written[partition] = {"file": target.name, "sha256": digest(target.read_bytes()), "rows": len(cases)}
        write_rows(stage / "individual-responses.jsonl", traces)
        data_ready = bool(ready["development"]) and all(summary["train"]["eligible_status_counts"].values())
        if data_ready:
            from tools.dropoff_train import TrainConfig, load_training_cases
            load_training_cases(stage / "train.jsonl", stage / "development.jsonl",
                                TrainConfig(run_kind="adaptation", max_train_cases=None, max_dev_cases=None))
        human_files = sum(f["named_attestation"] for f in files) if not _software_fixture else 0
        receipt = {"schema_version": 1, "kind": "nonfinal_passenger_human_review_intake",
                   "created_utc": datetime.now(timezone.utc).isoformat(), "software_fixture": _software_fixture,
                   "identity_basis": "Named reviewer self-attestation", "externally_authenticated": False,
                   "partitions": summary, "returned_distinct_exports": len(files), "attested_human_exports": human_files,
                   "unattested_legacy_exports": len(pending), "cases_with_attested_human_returns":
                   sum(bool(t["individual_answers"]) for t in traces) if not _software_fixture else 0,
                   "conflicts_seen": len(conflicts), "conflicts_adjudicated": len(signed),
                   "eligible_human_cases": sum(map(len, ready.values())) if not _software_fixture else 0,
                   "has_training_status_coverage_and_development": data_ready,
                   "training_loader_validated": data_ready,
                   "model_continuation_preflight_run": False, "training_performed": False,
                   "reviewed_files": written, "individual_response_ledger_sha256": digest((stage / "individual-responses.jsonl").read_bytes()),
                   "archived_exports": files}
        write_json(stage / "receipt.json", receipt)
        if out.exists():
            raise FileExistsError("another intake snapshot already exists at the destination")
        stage.rename(out)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("annotations/dropoff/passenger-development"))
    parser.add_argument("--train-requests", type=Path, required=True)
    parser.add_argument("--development-requests", type=Path, required=True)
    parser.add_argument("--returns", type=Path, required=True, help="Directory containing only individual reviewer JSON exports")
    parser.add_argument("--out", type=Path, required=True, help="New directory for the immutable receipt, choices and reviewed cases")
    parser.add_argument("--decisions", type=Path, help="Optional adjudicator-completed conflict sheet")
    parser.add_argument("--legacy-confirmations", type=Path, help="Optional named confirmations for original version 1 returns")
    args = parser.parse_args(argv)
    result = ingest(args.bank, {"train": args.train_requests, "development": args.development_requests},
                    args.returns, args.out, decisions=args.decisions, legacy=args.legacy_confirmations)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
