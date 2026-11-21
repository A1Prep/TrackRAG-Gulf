"""Fixed-pool contracts use injected scorers, never claim model accuracy."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from trackrag.contracts import RankedInterval
from trackrag.retrieval import RetrievalConfig, build_index, rank_evidence
from trackrag.retrieval_candidates import build_candidate_batch
from trackrag.retrieval_comparison import ComparisonFailure, compare_candidate_batch, main
from trackrag.retrieval_qwen import QwenConfig, QwenTextScorer


class Recording:
    def __init__(self, identity="recording-a"):
        self.recording_id = identity
        self.requested = []
        self.frames = {frame: {"recording_id": identity, "frame_id": frame, "width": 100,
                              "height": 80, "timestamp_s": frame / 10,
                              "image_path": f"not-read/{frame}.jpg"} for frame in range(121)}

    def iter_frames(self):
        raise AssertionError("Comparison must not scan the full recording")

    def get_frame(self, frame):
        self.requested.append(frame)
        return self.frames[frame]


def observation(frame, track=7, label="Bus", source="annotated", identity="recording-a"):
    return {"recording_id": identity, "source": source, "frame_id": frame,
            "track_id": track, "class_name": label, "bbox_xyxy": (10, 20, 30, 40)}


def question(text="Which bus is visible?", cutoff=100, identity="recording-a"):
    return {"question_id": "q1", "recording_id": identity, "query_frame": cutoff, "question": text}


class ClassScorer:
    def __init__(self):
        self.calls = []

    def score(self, query, documents):
        self.calls.append((deepcopy(query), documents))
        target = "Bus" if "bus" in query["question"].lower() else "Car"
        return [0.9 if (json.loads(document)["observed_track_facts"] or {}).get("class_counts", {}).get(target)
                else 0.1 for document in documents]


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.recording = Recording()
        self.rows = [observation(frame) for frame in range(90, 111)]
        self.rows += [observation(frame, 42, "Car") for frame in range(90, 111)]
        self.config = RetrievalConfig(window_frames=5, stride_frames=5)

    def batch(self, query=None, rows=None, config=None):
        index = build_index(self.recording, self.rows if rows is None else rows, config=config or self.config)
        return build_candidate_batch(query or question(), index, source=index.source or "annotated")

    def compare(self, batch=None, **kwargs):
        return compare_candidate_batch(batch if batch is not None else self.batch(), self.recording,
                                       scorer=kwargs.pop("scorer", ClassScorer()), **kwargs)

    def test_bus_car_changes_both_rankings_on_identical_complete_pools(self):
        for text, track in (("bus", 7), ("car", 42)):
            with self.subTest(text=text):
                batch = self.batch(question(text))
                scorer = ClassScorer()
                result = self.compare(batch, scorer=scorer, k=1)
                self.assertEqual(result["lexical"]["intervals"][0]["track_id"], track)
                self.assertEqual(result["qwen"]["intervals"][0]["track_id"], track)
                self.assertGreater(len(batch["candidates"]), 3)
                self.assertEqual(len(scorer.calls[0][1]), len(batch["candidates"]))
                expected = set(result["candidate_ids"])
                for name in ("lexical", "qwen"):
                    self.assertEqual(set(row["candidate_id"] for row in result[name]["ranking"]), expected)
                    self.assertEqual(len(result[name]["scores"]), len(expected))
                    self.assertEqual(set(result[name]["intervals"][0]), RankedInterval.__required_keys__)
                self.assertIsNone(result["accuracy_metrics"])
                self.assertFalse(result["scores_directly_comparable"])
                self.assertEqual(result["qwen"]["scorer"]["backend"], "ClassScorer")

    def test_original_baseline_scores_reused_for_matching_interval_bounds(self):
        index = build_index(self.recording, self.rows, config=self.config)
        original = rank_evidence(question("bus on the left"), index)
        batch = build_candidate_batch(question("bus on the left"), index, source="annotated")
        before = deepcopy(index.observations)
        result = self.compare(batch)
        scores = {(candidate["track_id"], candidate["start_frame"], candidate["end_frame"]): score
                  for candidate, score in zip(batch["candidates"], result["lexical"]["scores"])}
        for interval in original:
            key = (interval["track_id"], interval["start_frame"], interval["end_frame"])
            self.assertEqual(scores[key], interval["score"])
        self.assertEqual(rank_evidence(question("bus on the left"), index), original)
        self.assertEqual(index.observations, before)

    def test_no_match_keeps_zero_scores_scenes_and_common_tie_policy(self):
        batch = self.batch(question("helicopter"))
        scorer = SimpleNamespace(score=lambda query, documents: [0.0] * len(documents))
        result = self.compare(batch, scorer=scorer)
        self.assertEqual(result["lexical"]["scores"], [0.0] * len(batch["candidates"]))
        self.assertEqual(result["lexical"]["ranking"], result["qwen"]["ranking"])
        self.assertEqual(len(result["lexical"]["intervals"]), 3)
        scenes = [document for document in result["lexical"]["documents"] if document["features"] is None]
        self.assertTrue(scenes)
        self.assertTrue(all(document["unsupported_reason"] for document in scenes))

    def test_exact_ties_prefer_tracks_before_scenes_at_same_frame(self):
        batch = self.batch(question("helicopter", 90), rows=[observation(90, 42, "Car"), observation(90)])
        result = self.compare(batch, scorer=SimpleNamespace(score=lambda q, docs: [0.0] * len(docs)))
        for name in ("lexical", "qwen"):
            self.assertEqual([row["track_id"] for row in result[name]["intervals"]], [7, 42, None])
            self.assertEqual([row["rank"] for row in result[name]["intervals"]], [1, 2, 3])

    def test_future_changes_cannot_change_inputs_scores_or_rankings(self):
        prefix = [row for row in self.rows if row["frame_id"] <= 100]
        expected = self.compare(self.batch(rows=prefix))
        for future in ([], [observation(101, 7, "Car")], [observation(110, 0, "Truck")]):
            self.recording = Recording()
            batch = self.batch(rows=prefix + future)
            for frame in range(101, 121):
                del self.recording.frames[frame]
            self.recording.requested.clear()
            result = self.compare(batch)
            for key in ("metadata_sha256", "candidate_ids", "candidate_count", "agreement"):
                self.assertEqual(result[key], expected[key])
            for name in ("lexical", "qwen"):
                for key in ("documents", "scores", "ranking", "intervals"):
                    self.assertEqual(result[name][key], expected[name][key])
            self.assertEqual(result["lexical"]["inputs_sha256"], expected["lexical"]["inputs_sha256"])
            self.assertEqual(result["qwen"]["model_inputs_sha256"], expected["qwen"]["model_inputs_sha256"])
            self.assertTrue(all(frame <= 100 for frame in self.recording.requested))

    def test_interior_geometry_and_varying_frame_sizes_reuse_baseline_features(self):
        rows = [observation(frame) for frame in range(90, 95)]
        rows[1]["bbox_xyxy"] = (70, 20, 90, 40)
        self.recording.frames[91]["width"] = 200
        batch = self.batch(question("moving bus", 94), rows=rows)
        result = self.compare(batch)
        tracked = next(document for document in result["lexical"]["documents"] if document["features"])
        self.assertIn("moving", tracked["features"]["movement_terms"])
        self.assertIn({"frame_id": 91, "width": 200, "height": 80}, result["lexical"]["frame_dimensions"])
        self.assertEqual(result["lexical"]["scores"][0], 0.75)

    def test_paths_are_not_scoring_inputs_and_callers_are_not_mutated(self):
        batch = self.batch()
        original = deepcopy(batch)
        class MutatingScorer(ClassScorer):
            def score(self, query, documents):
                query["question"] = "car"
                return super().score(query, documents)
        self.compare(batch, scorer=MutatingScorer())
        self.assertEqual(batch, original)
        first = self.compare(batch)
        for candidate in batch["candidates"]:
            for image in candidate["images"]:
                image["image_path"] = "different-machine.jpg"
        second = self.compare(batch)
        self.assertEqual(first["lexical"]["inputs_sha256"], second["lexical"]["inputs_sha256"])
        self.assertEqual(first["qwen"]["model_inputs_sha256"], second["qwen"]["model_inputs_sha256"])

    def test_missing_pool_skips_scorer_and_has_no_accuracy_claim(self):
        index = build_index(self.recording, [])
        del self.recording.frames[100]
        batch = build_candidate_batch(question(), index, source="annotated")
        scorer = ClassScorer()
        result = self.compare(batch, scorer=scorer)
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(scorer.calls, [])
        self.assertFalse(result["qwen"]["scorer_called"])
        self.assertIsNone(result["agreement"]["top_1_same_candidate"])
        self.assertIsNone(result["accuracy_metrics"])

    def test_empty_tracks_can_still_produce_an_exact_scene(self):
        result = self.compare(self.batch(rows=[]))
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["lexical"]["scores"], [0.0])
        self.assertIsNone(result["qwen"]["intervals"][0]["track_id"])

    def test_foreign_recording_tampered_packet_gold_and_bad_k_rejected_before_scoring(self):
        scorer = ClassScorer()
        batch = self.batch()
        for k in (0, 4, True, "3"):
            with self.subTest(k=k), self.assertRaises(ValueError):
                self.compare(batch, scorer=scorer, k=k)
        for key, value in (("end_frame", 101), ("source", "predicted")):
            changed = deepcopy(batch)
            changed["candidates"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.compare(changed, scorer=scorer)
        changed = deepcopy(batch)
        changed["question"]["gold_answer"] = "bus"
        with self.assertRaises(ValueError):
            self.compare(changed, scorer=scorer)
        with self.assertRaises(ValueError):
            compare_candidate_batch(batch, Recording("other"), scorer=scorer)
        self.assertEqual(scorer.calls, [])

    def test_source_and_recording_identities_are_retained(self):
        rows = [dict(row, source="predicted") for row in self.rows]
        batch = self.batch(rows=rows)
        result = self.compare(batch)
        for name in ("lexical", "qwen"):
            self.assertTrue(all(row["source"] == "predicted" for row in result[name]["intervals"]))
        self.recording = Recording("recording-b")
        other_rows = [dict(row, recording_id="recording-b") for row in rows]
        other = self.batch(question(identity="recording-b"), rows=other_rows)
        self.assertFalse(set(batch["candidates"][i]["candidate_id"] for i in range(len(batch["candidates"]))) &
                         set(candidate["candidate_id"] for candidate in other["candidates"]))

    def test_config_policy_and_changed_dimensions_fail_without_model_call(self):
        batch = self.batch()
        scorer = ClassScorer()
        with self.assertRaises(ValueError):
            self.compare(batch, scorer=scorer, lexical_config=RetrievalConfig())
        with self.assertRaises(TypeError):
            self.compare(batch, scorer=scorer, qwen_config={})
        self.recording.frames[90]["width"] = 200
        with self.assertRaises(ValueError):
            self.compare(batch, scorer=scorer)
        self.assertEqual(scorer.calls, [])

    def test_bad_model_scores_and_exceptions_preserve_failure_not_fallback(self):
        for value in ([], [float("nan")], [float("inf")], [True], ["0.5"], RuntimeError("failed model")):
            def score(query, documents):
                if isinstance(value, Exception):
                    raise value
                return value if len(value) != 1 else value * len(documents)
            with self.subTest(value=value):
                with self.assertRaises(ComparisonFailure) as caught:
                    self.compare(scorer=SimpleNamespace(score=score))
                receipt = caught.exception.receipt
                self.assertEqual(receipt["status"], "error")
                self.assertIn("intervals", receipt["lexical"])
                self.assertNotIn("intervals", receipt["qwen"])
                self.assertNotIn("agreement", receipt)

    def test_negative_scores_and_scene_selection_preserve_interval_identity(self):
        batch = self.batch()
        scorer = SimpleNamespace(score=lambda q, docs: [1.0 if json.loads(doc)["evidence_kind"] == "scene"
                                                        else -2.0 for doc in docs])
        result = self.compare(batch, scorer=scorer)
        self.assertIsNone(result["qwen"]["intervals"][0]["track_id"])
        self.assertTrue(any(row["score"] < 0 for row in result["qwen"]["ranking"]))
        self.assertTrue(all(row["candidate_id"] in result["candidate_ids"] for row in result["qwen"]["ranking"]))

    def test_declared_and_actual_qwen_config_must_match(self):
        with self.assertRaises(ValueError):
            self.compare(scorer=QwenTextScorer(QwenConfig(batch_size=1)))


class ComparisonCommandTests(unittest.TestCase):
    def command(self, directory, questions, source="annotated", extra=()):
        path = Path(directory) / "questions.jsonl"
        path.write_text("".join(json.dumps(query) + "\n" for query in questions), encoding="utf-8")
        return ["--data-root", directory, "--recording", "recording-a", "--questions", str(path),
                "--source", source, "--out", str(Path(directory) / "comparison"), *extra]

    def test_cli_saves_same_pool_complete_rankings_and_public_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            scorer = ClassScorer()
            args = self.command(directory, [question(cutoff=90)])
            with patch("trackrag.emt.open_recording", return_value=Recording()), \
                 patch("trackrag.emt.load_annotated_tracks", return_value=[observation(90)]), \
                 patch.object(QwenTextScorer, "score", side_effect=scorer.score), \
                 patch.object(QwenTextScorer, "provenance", return_value={"backend": "injected_class_scorer"}), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 0)
            out = Path(directory) / "comparison"
            run = json.loads((out / "run.json").read_text())
            batch = json.loads((out / "candidates-0001.json").read_text())
            receipt = json.loads((out / "comparison-0001.json").read_text())
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["completed_questions"], 1)
            self.assertIn("retrieval_comparison.py", run["source_files_sha256"])
            self.assertTrue(all(len(value) == 64 for value in run["source_files_sha256"].values()))
            self.assertEqual(batch["metadata_sha256"], receipt["metadata_sha256"])
            self.assertIsNone(run["accuracy_metrics"])
            self.assertEqual(len(receipt["qwen"]["scores"]), len(batch["candidates"]))
            row = json.loads((out / "comparisons.jsonl").read_text())
            self.assertEqual(row["candidate_count"], len(batch["candidates"]))

    def test_cli_persists_model_failure_and_completed_previous_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = 0
            def score(query, documents):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("model unavailable")
                return [0.1] * len(documents)
            queries = [question(cutoff=90), dict(question(cutoff=90), question_id="q2")]
            args = self.command(directory, queries)
            with patch("trackrag.emt.open_recording", return_value=Recording()), \
                 patch("trackrag.emt.load_annotated_tracks", return_value=[observation(90)]), \
                 patch.object(QwenTextScorer, "score", side_effect=score), \
                 patch.object(QwenTextScorer, "provenance", return_value={"backend": "injected_failure_scorer"}), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(args)
            self.assertEqual(caught.exception.code, 2)
            out = Path(directory) / "comparison"
            run = json.loads((out / "run.json").read_text())
            self.assertEqual(run["status"], "error")
            self.assertEqual(run["completed_questions"], 1)
            failure = json.loads((out / "failure.json").read_text())
            self.assertEqual(failure["question"]["question_id"], "q2")
            self.assertNotIn("intervals", failure["qwen"])
            self.assertEqual(len((out / "comparisons.jsonl").read_text().splitlines()), 1)

    def test_cli_rejects_existing_output_foreign_questions_and_missing_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.command(directory, [question(identity="other")])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            args = self.command(directory, [question()], source="predicted")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            args = self.command(directory, [question()])
            out = Path(directory) / "comparison"
            out.mkdir()
            marker = out / "keep.txt"
            marker.write_text("original")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertEqual(marker.read_text(), "original")

    def test_cli_reads_saved_predictions_without_running_an_annotated_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(json.dumps(observation(90, source="predicted")) + "\n", encoding="utf-8")
            args = self.command(directory, [question(cutoff=90)], source="predicted",
                                extra=("--predicted-tracks", str(path)))
            with patch("trackrag.emt.open_recording", return_value=Recording()), \
                 patch("trackrag.emt.load_annotated_tracks", side_effect=AssertionError("Do not load human tracks")), \
                 patch.object(QwenTextScorer, "score", side_effect=ClassScorer().score), \
                 patch.object(QwenTextScorer, "provenance", return_value={"backend": "injected_class_scorer"}), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 0)
            result = json.loads((Path(directory) / "comparison/comparison-0001.json").read_text())
            self.assertEqual(result["source"], "predicted")
            self.assertTrue(all(row["source"] == "predicted" for row in result["qwen"]["intervals"]))

    def test_cli_can_select_a_checkpoint_without_changing_the_candidate_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            scorer = ClassScorer()
            scorer.config = QwenConfig(local_files_only=True)
            args = self.command(directory, [question(cutoff=90)], extra=("--checkpoint", "saved-checkpoint"))
            with patch("trackrag.emt.open_recording", return_value=Recording()), \
                 patch("trackrag.emt.load_annotated_tracks", return_value=[observation(90)]), \
                 patch("trackrag.retrieval_qwen_trained.TrainedQwenTextScorer", return_value=scorer) as loader, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 0)
            loader.assert_called_once_with(Path("saved-checkpoint"), config=None, allow_synthetic=False)
            run = json.loads((Path(directory) / "comparison/run.json").read_text())
            self.assertEqual(run["checkpoint"], "saved-checkpoint")
            self.assertTrue(run["qwen_config"]["local_files_only"])

    def test_synthetic_opt_in_requires_an_explicit_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.command(directory, [question()], extra=("--allow-synthetic",))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)


if __name__ == "__main__":
    unittest.main()
