"""Export an ungraded, artifact-bound review packet from a saved run."""

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

from .compare import _md, write_bundle
from .run_audit import audit_run, fingerprint, validate_question
from .score import read_jsonl


def export_review(directory, output_dir, reviewer=""):
    directory, output_dir = Path(directory).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Review output already exists; choose a new directory")
    if not isinstance(reviewer, str):
        raise ValueError("Reviewer label must be a string")
    rows = read_jsonl(directory / "results.jsonl")
    if not rows:
        raise ValueError("No saved results to review")
    expected = {}
    for row in rows:
        validate_question(row["question"])
        qid = row["question"]["question_id"]
        if qid in expected:
            raise ValueError("Duplicate question in saved run")
        expected[qid] = row["question"]
    audited = audit_run(directory, expected)
    snapshot = {"version": 1, "files": [
        {"path": Path(item["path"]).relative_to(directory).as_posix(), "sha256": item["sha256"]}
        for item in audited["files"]
    ]}
    binding = {"directory": str(directory), "review": {
        "path": str(output_dir / "review.jsonl"),
        "results_sha256": fingerprint(directory / "results.jsonl"),
        "artifacts": str(output_dir / "artifacts.json"),
    }}
    reviews = [{"question_id": row["question"]["question_id"], "grade": None, "note": "", "reviewer": reviewer,
                "run_id": directory.name} for row in rows]
    worksheet = ["# Saved-answer review worksheet", "",
        "All grades start blank. Exporting this packet does not validate any answers or certify a human review.", "",
        "Inspect the answer and original/model-input images, then fill grade, note and reviewer in review.jsonl. "
        "Use pass, qualified or fail. Use manual only for diagnostic questions declared in the comparison manifest. "
        "Keep pending grades null; they remain unreviewed. A delivered error must not receive a successful grade.", "",
        "The binding includes the result file and a snapshot of all audited run files. If they change, export and review a new packet. "
        "binding.json contains a run entry for a comparison manifest. Its paths are absolute on this machine; update locations if moved.", ""]
    for row in rows:
        question, answer = row["question"], row["answer"]
        worksheet += [f"## {_md(question['question_id'])}", "", _md(question["question"]), "",
            f"Recording: {_md(question['recording_id'])}; query frame: {question['query_frame']}.",
            f"Delivered status: {answer['answer_status']}.", "", _md(answer["answer"] or "No answer text."), "",
            "Citations: " + ", ".join(_md(x) for x in answer["cited_evidence_ids"]), ""]
        for evidence in row["evidence"]:
            interval = evidence["interval"]
            subject = "full scene" if interval["track_id"] is None else f"track {interval['track_id']}"
            worksheet.append(f"{_md(evidence['evidence_id'])}: {subject}, frames {interval['start_frame']}–{interval['end_frame']}.")
            worksheet.extend(f"- [Open frame {image['frame_id']}](<{(directory / image['image_path']).as_posix()}>)" for image in evidence["images"])
        # Provider inputs are useful for judging privacy/visibility limitations.
        hosted = directory / "hosted" / sha256(question["question_id"].encode()).hexdigest()
        inputs = [Path(item["path"]) for item in audited["files"] if item["role"] == "provider_input" and Path(item["path"]).parent == hosted]
        if inputs:
            worksheet += ["", "Processed inputs supplied to the model:"]
            worksheet.extend(f"- [Open {_md(path.name)}](<{path.as_posix()}>)" for path in inputs)
        worksheet.append("")
    write_bundle(output_dir, {
        "review.jsonl": "".join(json.dumps(row) + "\n" for row in reviews),
        "binding.json": json.dumps(binding, indent=2) + "\n",
        "artifacts.json": json.dumps(snapshot, indent=2) + "\n",
        "worksheet.md": "\n".join(worksheet) + "\n",
    })
    return output_dir / "worksheet.md"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reviewer", default="", help="Reviewer label; this does not certify a human review")
    args = parser.parse_args(argv)
    try:
        path = export_review(args.run, args.out, args.reviewer)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Review export failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved ungraded review packet to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
