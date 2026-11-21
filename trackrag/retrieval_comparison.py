"""Task 4's fixed-pool lexical/Qwen ranking experiment.

This is an explicitly selected experiment, not a replacement for the production
lexical ranker or Task 5's answer/evidence evaluator. Both rankers score every
member of one validated, query-bounded CandidateBatch. They share selection and
tie policies, but deliberately retain their existing feature representations.
No relevance labels, accuracy estimates, parameter updates or answerer are used.
"""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter

from .contracts import Recording
from .retrieval import (
    RetrievalConfig, TrackWindow, _describe_window, _frame_dimensions,
    _question_features, _text_score, build_index,
)
from .retrieval_candidates import CandidateBatch, build_candidate_batch, candidate_to_interval
from .retrieval_dataset import _validate_batch
from .retrieval_qwen import (
    INPUT_VERSION, INSTRUCTION, PREFIX, SUFFIX, QwenConfig, QwenTextScorer,
    _query_text, rank_candidate_batch, text_documents,
)


SCHEMA_VERSION = "trackrag.ranking-comparison.v1"
TIE_BREAK = "start_frame, end_frame, tracked before scene, track_id, candidate_id"


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return sha256(_json(value).encode("utf-8")).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _features_record(features):
    return {name: sorted(getattr(features, name))
            for name in ("class_terms", "location_terms", "movement_terms")}


def _ordered(candidates, scores):
    """Match Qwen's existing deterministic sort, including zero and negative scores."""
    return sorted(zip(scores, candidates), key=lambda item: (
        -item[0], item[1]["start_frame"], item[1]["end_frame"], item[1]["track_id"] is None,
        item[1]["track_id"] if item[1]["track_id"] is not None else -1, item[1]["candidate_id"],
    ))


def _ranking_result(candidates, scores, k):
    ordered = _ordered(candidates, scores)
    return {
        "scores": [float(score) for score in scores],
        "ranking": [{"candidate_id": candidate["candidate_id"], "position": position, "score": float(score)}
                    for position, (score, candidate) in enumerate(ordered, 1)],
        "intervals": [candidate_to_interval(candidate, rank=rank, score=score)
                      for rank, (score, candidate) in enumerate(ordered[:k], 1)],
    }


def _lexical_inputs(batch, recording):
    """Reuse the baseline's exact descriptors on each candidate's OWN observations.

    Snapshot original dimensions for every permitted observed frame, including
    interior frames not in the representative-image sample. Neither a constant
    image size nor interpolation across gaps is assumed. Pixel data is not a
    ranking feature; the recording adapter may decode images for metadata.
    """
    if recording.recording_id != batch["question"]["recording_id"]:
        raise ValueError("Recording does not match the candidate batch")
    histories = {history["history_id"]: history["observations"] for history in batch["histories"]}
    dimensions = {}
    for frame_id in sorted({row["frame_id"] for rows in histories.values() for row in rows}):
        dimensions[frame_id] = _frame_dimensions(recording, frame_id)
    for candidate in batch["candidates"]:
        if candidate["kind"] == "scene":
            continue
        for image in candidate["images"]:
            if dimensions[image["frame_id"]] != (image["width"], image["height"]):
                raise ValueError("Recording dimensions changed after candidate construction")
    documents = []
    descriptions = []
    for candidate in batch["candidates"]:
        if candidate["kind"] == "scene":
            descriptions.append(None)
            documents.append({"candidate_id": candidate["candidate_id"], "features": None,
                              "unsupported_reason": "The lexical baseline has no scene descriptor"})
            continue
        rows = tuple(row for row in histories[candidate["history_id"]]
                     if candidate["start_frame"] <= row["frame_id"] <= candidate["end_frame"])
        for row in rows:
            width, height = dimensions[row["frame_id"]]
            if row["bbox_xyxy"][2] > width or row["bbox_xyxy"][3] > height:
                raise ValueError("Observation box is outside current original frame dimensions")
        window = TrackWindow(candidate["recording_id"], candidate["source"], candidate["track_id"],
                             candidate["start_frame"], candidate["end_frame"], rows)
        description = _describe_window(window, recording, dimensions)
        descriptions.append(description)
        documents.append({"candidate_id": candidate["candidate_id"], "features": _features_record(description),
                          "unsupported_reason": None})
    frame_sizes = [{"frame_id": frame_id, "width": size[0], "height": size[1]}
                   for frame_id, size in sorted(dimensions.items())]
    return descriptions, documents, frame_sizes


class ComparisonFailure(RuntimeError):
    """A failed model comparison carries its partial receipt, never fallback results."""

    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(receipt["qwen"]["error"])


def compare_candidate_batch(batch: CandidateBatch, recording: Recording, *,
                            lexical_config: RetrievalConfig | None = None,
                            qwen_config: QwenConfig | None = None, scorer=None, k=3):
    """Return both complete score vectors and top-k rankings over one frozen pool.

    k is 1..3: this experiment scores the full pool, even when selecting only one
    interval. Empty pools skip the scorer. An injected scorer is explicitly named
    in provenance; contract tests must not be described as pretrained runs.
    Caller inputs are copied. No gold labels or reviewer judgments are accepted.
    """
    if type(k) is not int or not 1 <= k <= 3:
        raise ValueError("Comparison k must be an integer from 1 through 3")
    snapshot = deepcopy(batch)
    _validate_batch(snapshot)
    policy = snapshot["policy"]
    lexical_config = lexical_config if lexical_config is not None else RetrievalConfig(
        window_frames=policy["window_frames"], stride_frames=policy["stride_frames"],
        max_gap_frames=policy["max_gap_frames"])
    qwen_config = qwen_config if qwen_config is not None else QwenConfig()
    if not isinstance(lexical_config, RetrievalConfig) or not isinstance(qwen_config, QwenConfig):
        raise TypeError("Supply RetrievalConfig and QwenConfig instances")
    if any(getattr(lexical_config, name) != policy[name]
           for name in ("window_frames", "stride_frames", "max_gap_frames")):
        raise ValueError("Lexical configuration must match the fixed pool window policy")
    if isinstance(scorer, QwenTextScorer) and scorer.config != qwen_config:
        raise ValueError("Scorer configuration does not match the declared Qwen configuration")
    candidates = snapshot["candidates"]
    question = snapshot["question"]
    receipt = {
        "schema_version": SCHEMA_VERSION, "status": "running", "question": question.copy(),
        "source": snapshot["source"], "k": k, "metadata_sha256": snapshot["metadata_sha256"],
        "candidate_policy": deepcopy(policy), "candidate_count": len(candidates),
        "candidate_kinds": dict(sorted(Counter(candidate["kind"] for candidate in candidates).items())),
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "selection_policy": {"complete_pool": True, "zero_scores_retained": True,
                             "relevance_threshold": None, "tie_break": TIE_BREAK},
        "scope": "Same candidates and cutoff; rankers retain different text representations",
        "evaluation_status": "requires_reviewed_relevance_labels", "accuracy_metrics": None,
        "scores_directly_comparable": False,
    }
    started = perf_counter()
    descriptions, lexical_documents, dimensions = _lexical_inputs(snapshot, recording)
    query_features = _question_features(question["question"])
    lexical_inputs = {"question_features": _features_record(query_features), "documents": lexical_documents,
                      "frame_dimensions": dimensions, "config": asdict(lexical_config)}
    scores = [_text_score(query_features, description, lexical_config) if description is not None else 0.0
              for description in descriptions]
    receipt["lexical"] = {
        "variant": "lexical_shared_pool", "input_policy": "Original class/location/movement descriptors on each candidate's observations; scenes score zero",
        "training_state": "untrained_heuristics", **lexical_inputs,
        "inputs_sha256": _hash(lexical_inputs), **_ranking_result(candidates, scores, k),
        "elapsed_seconds": perf_counter() - started,
    }
    documents = text_documents(snapshot)
    model_inputs = {"query": _query_text(question), "instruction": INSTRUCTION,
                    "prefix": PREFIX, "suffix": SUFFIX, "input_version": INPUT_VERSION, "documents": documents}
    receipt["qwen"] = {"variant": "qwen_text_shared_pool", "input_version": INPUT_VERSION,
                       "config": asdict(qwen_config), "input_modality": "text_metadata_only",
                       "visual_pixels_used": False, **model_inputs,
                       "model_inputs_sha256": _hash(model_inputs), "scorer_called": False}
    started = perf_counter()
    try:
        scorer = scorer if scorer is not None else QwenTextScorer(qwen_config)
        receipt["qwen"]["scorer_called"] = bool(documents)
        intervals, details = rank_candidate_batch(snapshot, scorer, k=k)
        result = _ranking_result(candidates, details["scores"], k)
        if result["intervals"] != intervals or details["documents"] != documents:
            raise RuntimeError("Comparison selection disagrees with the Qwen adapter")
        provenance = getattr(scorer, "provenance", None)
        result["scorer"] = provenance() if callable(provenance) and documents else {"backend": type(scorer).__name__}
        receipt["qwen"].update(result, status="completed")
    except Exception as error:
        receipt["status"] = "error"
        receipt["qwen"].update(status="error", error_type=type(error).__name__, error=str(error)[:2000])
        raise ComparisonFailure(receipt) from error
    finally:
        receipt["qwen"]["elapsed_seconds"] = perf_counter() - started
    left = [row["candidate_id"] for row in receipt["lexical"]["ranking"][:k]]
    right = [row["candidate_id"] for row in receipt["qwen"]["ranking"][:k]]
    receipt["agreement"] = {"top_1_same_candidate": left[0] == right[0] if left else None,
                            "shared_top_k_candidates": len(set(left) & set(right)),
                            "top_k_order_equal": left == right,
                            "interpretation": "Agreement is not evidence relevance or accuracy"}
    receipt["status"] = "completed"
    return receipt


def main(argv=None):
    """Save a fixed-pool comparison on real inputs, without invoking an answerer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--source", choices=("annotated", "predicted"), required=True)
    parser.add_argument("--predicted-tracks", type=Path)
    parser.add_argument("--qwen-config", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Use a saved trained adapter instead of frozen Qwen")
    parser.add_argument("--allow-synthetic", action="store_true", help="Explicitly allow a synthetic smoke checkpoint")
    parser.add_argument("--retrieval-config", type=Path)
    parser.add_argument("--k", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from .emt import open_recording, load_annotated_tracks
        from .questions import load_questions

        if args.out.exists():
            raise FileExistsError("Choose a new output directory")
        if bool(args.predicted_tracks) != (args.source == "predicted"):
            raise ValueError("Predicted mode requires --predicted-tracks; annotated mode must not use it")
        if args.allow_synthetic and not args.checkpoint:
            raise ValueError("--allow-synthetic requires --checkpoint")
        questions = load_questions(args.questions)
        if not questions or any(question["recording_id"] != args.recording for question in questions):
            raise ValueError("Question file must contain this recording's public questions only")
        qwen_config = QwenConfig(**json.loads(args.qwen_config.read_text(encoding="utf-8-sig"))) if args.qwen_config else None
        config = RetrievalConfig(**json.loads(args.retrieval_config.read_text(encoding="utf-8-sig"))) if args.retrieval_config else RetrievalConfig()
        recording = open_recording(args.data_root, args.recording)
        tracks = ([json.loads(line) for line in args.predicted_tracks.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
                  if args.predicted_tracks else load_annotated_tracks(recording))
        index = build_index(recording, tracks, config=config)
        if index.source is not None and index.source != args.source:
            raise ValueError("Track source does not match the selected run source")
        if args.checkpoint:
            from .retrieval_qwen_trained import TrainedQwenTextScorer

            scorer = TrainedQwenTextScorer(args.checkpoint, config=qwen_config, allow_synthetic=args.allow_synthetic)
            qwen_config = scorer.config
        else:
            qwen_config = qwen_config if qwen_config is not None else QwenConfig()
            scorer = QwenTextScorer(qwen_config)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".ranking-comparison-", dir=args.out.parent) as temporary:
            staging = Path(temporary) / "run"
            staging.mkdir()
            manifest = {"schema_version": SCHEMA_VERSION, "status": "running", "source": args.source,
                        "recording_id": args.recording, "question_count": len(questions), "completed_questions": 0,
                        "questions_file_sha256": sha256(args.questions.read_bytes()).hexdigest(),
                        "source_files_sha256": {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                                for name in ("retrieval.py", "retrieval_candidates.py", "retrieval_dataset.py",
                                                             "retrieval_qwen.py", "retrieval_comparison.py")},
                        "retrieval_config": asdict(config), "qwen_config": asdict(qwen_config),
                        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                        "evaluation_status": "requires_reviewed_relevance_labels", "accuracy_metrics": None}
            rows = []
            try:
                for number, question in enumerate(questions, 1):
                    print(f"Comparing question {number}/{len(questions)}: {question['question_id']}", file=sys.stderr, flush=True)
                    batch = build_candidate_batch(question, index, source=args.source)
                    _write(staging / f"candidates-{number:04d}.json", batch)
                    receipt = compare_candidate_batch(batch, recording, lexical_config=config,
                                                      qwen_config=qwen_config, scorer=scorer, k=args.k)
                    _write(staging / f"comparison-{number:04d}.json", receipt)
                    rows.append({"question": question, "metadata_sha256": receipt["metadata_sha256"],
                                 "candidate_count": receipt["candidate_count"],
                                 "lexical_intervals": receipt["lexical"]["intervals"],
                                 "qwen_intervals": receipt["qwen"]["intervals"], "agreement": receipt["agreement"]})
                    manifest["completed_questions"] += 1
                manifest["status"] = "completed"
            except Exception as error:
                manifest.update(status="error", error_type=type(error).__name__, error=str(error)[:2000])
                if isinstance(error, ComparisonFailure):
                    _write(staging / "failure.json", error.receipt)
                raise
            finally:
                _write(staging / "run.json", manifest)
                (staging / "comparisons.jsonl").write_text("".join(_json(row) + "\n" for row in rows), encoding="utf-8")
                if args.out.exists():
                    raise FileExistsError("Output directory was created concurrently")
                staging.rename(args.out)
        print(f"Saved fixed-pool ranking comparison to {args.out}")
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
