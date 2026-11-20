"""Prepare real Task 4 candidate packets and a pending review worksheet.

Questions, source and recording splits are explicit inputs. This command assigns
no relevance labels and calls no ranker, answerer or training model. It does not
modify Task 2/3 data or Task 5 evaluation. Its short-window default is a new fixed
experiment policy, not a tuned replacement for the production baseline.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import sys
import tempfile

from .questions import load_questions
from .retrieval import RetrievalConfig, build_index
from .retrieval_candidates import build_candidate_batch
from .retrieval_dataset import _load_pack, _loads, _read, _splits, prepare_review
from .retrieval_training_data import canonical, write_json


def prepare_realdata(data_root, questions_path, split_manifest, output_dir, *,
                     source="annotated", predicted_tracks=None, config=None):
    """Use one entry point for all supplied train/dev recordings, preserving IDs."""
    from .emt import open_recording, load_annotated_tracks

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Choose a new real-data preparation directory")
    if source not in ("annotated", "predicted") or bool(predicted_tracks) != (source == "predicted"):
        raise ValueError("Predicted source requires saved predictions; annotated source must not use them")
    questions = load_questions(questions_path)
    assigned_manifest = _read(split_manifest)
    assigned = _splits(assigned_manifest)
    if not questions or any(assigned.get(question["recording_id"]) not in ("train", "dev") for question in questions):
        raise ValueError("Supply public questions from assigned train/dev recordings only")
    config = config if config is not None else RetrievalConfig(window_frames=10, stride_frames=10)
    if not isinstance(config, RetrievalConfig):
        raise TypeError("config must be RetrievalConfig")
    recording_ids = {question["recording_id"] for question in questions}
    predictions = None
    if predicted_tracks:
        predictions = [_loads(line) for line in Path(predicted_tracks).read_text(encoding="utf-8-sig").splitlines()
                       if line.strip()]
        if any(not isinstance(row, dict) or row.get("recording_id") not in recording_ids
               or row.get("source") != "predicted" for row in predictions):
            raise ValueError("Prediction identities/source must match the selected recordings")
    indices = {}
    for identity in sorted(recording_ids):
        recording = open_recording(data_root, identity)
        tracks = load_annotated_tracks(recording) if predictions is None else [row for row in predictions if row["recording_id"] == identity]
        indices[identity] = build_index(recording, tracks, config=config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".task4-realdata-", dir=output.parent) as temporary:
        staging = Path(temporary) / "preparation"
        (staging / "candidates").mkdir(parents=True)
        (staging / "questions.jsonl").write_text("".join(canonical(question) + "\n" for question in questions), encoding="utf-8")
        write_json(staging / "splits.json", assigned_manifest)
        write_json(staging / "retrieval-config.json", asdict(config))
        question_files = {}
        for number, identity in enumerate(sorted(recording_ids), 1):
            relative = f"questions-{number:04d}.jsonl"
            (staging / relative).write_text("".join(canonical(question) + "\n" for question in questions
                                                   if question["recording_id"] == identity), encoding="utf-8")
            question_files[identity] = relative
        packets, cases = [], []
        for number, question in enumerate(questions, 1):
            print(f"Preparing question {number}/{len(questions)}: {question['question_id']}", file=sys.stderr, flush=True)
            indices[question["recording_id"]].recording.get_frame(question["query_frame"])
            batch = build_candidate_batch(question, indices[question["recording_id"]], source=source)
            path = staging / "candidates" / f"candidate-{number:04d}.json"
            write_json(path, batch)
            packets.append(path)
            cases.append({"question": question, "split": assigned[question["recording_id"]],
                          "candidate_count": len(batch["candidates"]),
                          "candidate_kinds": dict(Counter(candidate["kind"] for candidate in batch["candidates"])),
                          "metadata_sha256": batch["metadata_sha256"]})
        pack = prepare_review(packets, staging / "splits.json", staging / "review")
        verified, lookup = _load_pack(staging / "review")
        if verified["pack_sha256"] != pack["pack_sha256"]:
            raise RuntimeError("Prepared review pack did not pass readback verification")
        summary = {
            "schema_version": "trackrag.realdata-preparation.v1", "status": "prepared_unreviewed",
            "source": source, "recordings": sorted(recording_ids), "questions": len(questions),
            "question_files_by_recording": question_files,
            "retrieval_config": asdict(config), "cases": cases, "candidate_count": len(lookup),
            "pending_labels": len(lookup), "human_labels": 0, "assets": len(pack["assets"]),
            "pack_sha256": pack["pack_sha256"], "worksheet": "review/review.html",
            "questions_file_sha256": sha256(Path(questions_path).read_bytes()).hexdigest(),
            "source_files_sha256": {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                    for name in ("retrieval.py", "retrieval_candidates.py", "retrieval_dataset.py", "retrieval_realdata.py")},
            "training_completed": False, "accuracy_metrics": None,
            "limitations": ["Questions and relevance remain subject to human review.",
                            "All labels are pending; preparation creates no training labels.",
                            "No model scoring, real-data training, final test evaluation or answer generation was performed.",
                            "Use the saved pool policy for subsequent comparisons; do not equate this with the original 30-frame baseline pool."],
        }
        write_json(staging / "preparation.json", summary)
        if output.exists():
            raise FileExistsError("Preparation output was created concurrently")
        staging.rename(output)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source", choices=("annotated", "predicted"), default="annotated")
    parser.add_argument("--predicted-tracks", type=Path)
    parser.add_argument("--retrieval-config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = prepare_realdata(args.data_root, args.questions, args.split_manifest, args.out,
                                   source=args.source, predicted_tracks=args.predicted_tracks,
                                   config=RetrievalConfig(**_read(args.retrieval_config)) if args.retrieval_config else None)
        print(f"Prepared {summary['questions']} questions and {summary['pending_labels']} pending labels at {args.out / 'review/review.html'}")
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
