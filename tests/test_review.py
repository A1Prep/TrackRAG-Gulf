"""Task 5 integration checks use temporary images and injected adapters."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from evaluation.score import read_jsonl, score_results
from trackrag.__main__ import check_adapters, main
from trackrag.review import review_recording


class RecordingFixture:
    recording_id = "recording-a"

    def __init__(self, directory):
        self.frames = {}
        self.requested = []
        for frame_id in range(1, 7):
            path = directory / f"{frame_id}.png"
            Image.new("RGB", (32, 24), (frame_id, 0, 0)).save(path)
            self.frames[frame_id] = {
                "recording_id": self.recording_id, "frame_id": frame_id,
                "timestamp_s": (frame_id - 1) / 10, "width": 32, "height": 24,
                "image_path": str(path),
            }

    def iter_frames(self):
        return iter(self.frames.values())

    def get_frame(self, frame_id):
        self.requested.append(frame_id)
        return self.frames[frame_id]


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.recording = RecordingFixture(self.root)
        self.question = {
            "question_id": "q1", "recording_id": "recording-a",
            "query_frame": 5, "question": "Which bus entered?",
        }
        self.observations = [{
            "recording_id": "recording-a", "frame_id": frame_id, "track_id": 7,
            "class_name": "Bus", "bbox_xyxy": (4, 5, 20, 18), "source": "annotated",
        } for frame_id in (1, 3, 5, 6)]
        self.interval = {
            "recording_id": "recording-a", "source": "annotated", "track_id": 7,
            "start_frame": 1, "end_frame": 5, "rank": 1, "score": 0.8,
        }
        self.load_tracks = Mock(return_value=self.observations)
        self.build_index = Mock(return_value="index")
        self.rank_evidence = Mock(side_effect=lambda question, index, k: [self.interval])

    def run_review(self, **overrides):
        arguments = {
            "source": "annotated", "load_tracks": self.load_tracks,
            "build_index": self.build_index, "rank_evidence": self.rank_evidence,
            "output_dir": self.root / "run",
        }
        arguments.update(overrides)
        return review_recording(self.recording, [self.question], **arguments)

    def read_result(self, path):
        return json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    def test_saved_images_show_original_frame_and_box(self):
        result_path = self.run_review()
        result = self.read_result(result_path)
        self.assertEqual(result["answer"]["answer_status"], "not_requested")
        evidence = result["evidence"][0]
        self.assertEqual([image["frame_id"] for image in evidence["images"]], [1, 3, 5])
        for image in evidence["images"]:
            relative = Path(image["image_path"])
            self.assertFalse(relative.is_absolute())
            with Image.open(result_path.parent / relative) as marked:
                self.assertEqual(marked.getpixel((4, 5)), (255, 0, 0))
                self.assertEqual(marked.getpixel((0, 0)), (image["frame_id"], 0, 0))
        self.assertNotIn(6, self.recording.requested)

    def test_answerer_can_open_exact_cited_images(self):
        def answer(question, evidence, run_dir):
            for item in evidence:
                for image in item["images"]:
                    self.assertLessEqual(image["frame_id"], question["query_frame"])
                    self.assertTrue((run_dir / image["image_path"]).is_file())
            return {"answer_status": "answered", "answer": "The bus in E1.", "cited_evidence_ids": ["E1"]}

        result = self.read_result(self.run_review(answer_question=answer))
        self.assertEqual(result["answer"]["cited_evidence_ids"], [result["evidence"][0]["evidence_id"]])

    def test_unknown_citation_is_an_error(self):
        answerer = Mock(return_value={"answer_status": "answered", "answer": "Bus.", "cited_evidence_ids": ["E9"]})
        result = self.read_result(self.run_review(answer_question=answerer))
        self.assertEqual(result["answer"]["answer_status"], "error")
        self.assertEqual(result["answer"]["cited_evidence_ids"], [])

    def test_ranker_cannot_change_the_original_query_time(self):
        def mutate(question, index, k):
            question["query_frame"] = 6
            return [dict(self.interval, end_frame=6)]

        self.rank_evidence.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "future"):
            self.run_review()
        self.assertEqual(self.question["query_frame"], 5)
        self.assertFalse((self.root / "run").exists())

    def test_answerer_cannot_invent_evidence_or_change_saved_frames(self):
        def mutate(question, evidence, run_dir):
            question["query_frame"] = 6
            evidence[0]["images"][0]["frame_id"] = 6
            evidence.append(dict(evidence[0], evidence_id="E9"))
            return {"answer_status": "answered", "answer": "Bus.", "cited_evidence_ids": ["E9"]}

        result = self.read_result(self.run_review(answer_question=mutate))
        self.assertEqual(result["answer"]["answer_status"], "error")
        self.assertEqual(result["question"]["query_frame"], 5)
        self.assertEqual([item["evidence_id"] for item in result["evidence"]], ["E1"])
        self.assertEqual(result["evidence"][0]["images"][0]["frame_id"], 1)

    def test_index_builder_cannot_rewrite_exported_observations(self):
        def mutate(recording, tracks):
            tracks[0]["bbox_xyxy"] = (-1, -1, -1, -1)
            return "index"

        self.build_index.side_effect = mutate
        result = self.read_result(self.run_review())
        self.assertEqual(result["evidence"][0]["images"][0]["bbox_xyxy"], [4, 5, 20, 18])

    def test_provider_timeout_does_not_hide_remaining_questions(self):
        answerer = Mock(side_effect=[
            TimeoutError("provider unavailable"),
            {"answer_status": "insufficient_evidence", "answer": None, "cited_evidence_ids": []},
        ])
        second = dict(self.question, question_id="q2")
        path = review_recording(
            self.recording, [self.question, second], source="annotated",
            load_tracks=self.load_tracks, build_index=self.build_index,
            rank_evidence=self.rank_evidence, output_dir=self.root / "run", answer_question=answerer,
        )
        results = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([row["answer"]["answer_status"] for row in results], ["error", "insufficient_evidence"])
        self.load_tracks.assert_called_once()
        self.build_index.assert_called_once()

    def test_empty_predictions_do_not_call_answerer(self):
        self.load_tracks.return_value = []
        self.rank_evidence.side_effect = lambda *args, **kwargs: []
        answerer = Mock()
        result = self.read_result(self.run_review(answer_question=answerer))
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["answer"]["answer_status"], "insufficient_evidence")
        answerer.assert_not_called()

    def test_failed_detector_produces_no_completed_run(self):
        self.load_tracks.side_effect = RuntimeError("detector failed")
        with self.assertRaisesRegex(RuntimeError, "detector failed"):
            self.run_review()
        self.assertFalse((self.root / "run").exists())

    def test_bad_retrieval_cannot_publish_partial_output(self):
        for changes in (
            {"end_frame": 6}, {"recording_id": "recording-b"},
            {"source": "predicted"}, {"track_id": 99}, {"rank": 2},
        ):
            with self.subTest(changes=changes):
                self.rank_evidence.side_effect = lambda *args, **kwargs: [dict(self.interval, **changes)]
                with self.assertRaises(ValueError):
                    self.run_review()
                self.assertFalse((self.root / "run").exists())
                self.assertEqual(list(self.root.glob(".trackrag-*")), [])

    def test_invalid_question_fails_before_track_or_index_calls(self):
        self.question["recording_id"] = "wrong-recording"
        with self.assertRaisesRegex(ValueError, "different recording"):
            self.run_review()
        self.load_tracks.assert_not_called()
        self.build_index.assert_not_called()

    def test_answer_key_fields_cannot_enter_inference(self):
        self.question["expected_answer"] = "bus 7"
        with self.assertRaisesRegex(ValueError, "gold labels"):
            self.run_review()
        self.load_tracks.assert_not_called()

    def test_actual_run_output_is_accepted_by_offline_scorer(self):
        path = self.run_review()
        gold = [{
            "question_id": "q1", "recording_id": "recording-a", "query_frame": 5,
            "answerable": True,
            "acceptable_evidence": [{"track_id": 7, "start_frame": 1, "end_frame": 5}],
        }]
        scores = score_results(read_jsonl(path), gold)
        self.assertEqual(scores["counts"]["top1_correct"], 1)
        self.assertEqual(read_jsonl(path.parent / "tracks.jsonl"), [
            dict(item, bbox_xyxy=list(item["bbox_xyxy"])) for item in self.observations
        ])

    def test_existing_results_are_preserved(self):
        output = self.root / "run"
        output.mkdir()
        (output / "results.jsonl").write_text("previous run")
        with self.assertRaises(FileExistsError):
            self.run_review()
        self.assertEqual((output / "results.jsonl").read_text(), "previous run")

    def test_image_dimensions_must_match_metadata(self):
        self.recording.frames[1]["width"] = 31
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.run_review()
        self.assertFalse((self.root / "run").exists())

    def test_cli_uses_only_selected_adapters_with_same_application(self):
        for source in ("annotated", "predicted"):
            with self.subTest(source=source):
                tracks = [dict(item, source=source) for item in self.observations]
                ranked = dict(self.interval, source=source)
                emt = SimpleNamespace(open_recording=Mock(return_value=self.recording))
                modules = {
                    "trackrag.emt": emt,
                    "trackrag.questions": SimpleNamespace(load_questions=Mock(return_value=[self.question])),
                    "trackrag.retrieval": SimpleNamespace(
                        build_index=Mock(return_value="index"), rank_evidence=Mock(return_value=[ranked]),
                    ),
                }
                if source == "annotated":
                    emt.load_annotated_tracks = Mock(return_value=tracks)
                else:
                    modules["trackrag.vision"] = SimpleNamespace(predict_tracks=Mock(return_value=tracks))
                with patch("trackrag.__main__.import_module", side_effect=modules.__getitem__) as imports:
                    with redirect_stdout(io.StringIO()):
                        code = main([
                            "review", "--data-root", str(self.root), "--recording", "recording-a",
                            "--questions", "questions.jsonl", "--source", source, "--answerer", "none",
                            "--out", str(self.root / source),
                        ])
                self.assertEqual(code, 0)
                self.assertTrue((self.root / source / "results.jsonl").is_file())
                self.assertNotIn("trackrag.answering", [call.args[0] for call in imports.call_args_list])
                if source == "predicted":
                    self.assertFalse(hasattr(emt, "load_annotated_tracks"))

    def test_missing_module_is_reported_clearly(self):
        with patch("trackrag.__main__.import_module", side_effect=ModuleNotFoundError("trackrag.emt")):
            with redirect_stderr(io.StringIO()) as stderr:
                code = main([
                    "review", "--data-root", ".", "--recording", "recording-a", "--questions", "q.jsonl",
                    "--source", "annotated", "--out", str(self.root / "run"),
                ])
        self.assertEqual(code, 1)
        self.assertIn("Task 2", stderr.getvalue())

    def test_adapter_check_reports_all_missing_interfaces(self):
        with patch("trackrag.__main__.import_module", side_effect=ModuleNotFoundError("not merged")):
            report = check_adapters("predicted", "gemini")
        self.assertFalse(report["ready"])
        self.assertEqual(len(report["adapters"]), 6)
        self.assertTrue(all(item["status"] == "unavailable" for item in report["adapters"]))

    def test_adapter_check_rejects_incompatible_signature_without_running_it(self):
        function = Mock()
        wrong_signature = lambda: []
        module = SimpleNamespace(
            open_recording=function, load_questions=function, build_index=function,
            rank_evidence=wrong_signature, load_annotated_tracks=function,
        )
        with patch("trackrag.__main__.import_module", return_value=module):
            report = check_adapters("annotated", "none")
        self.assertFalse(report["ready"])
        failed = [item for item in report["adapters"] if item["status"] == "unavailable"]
        self.assertEqual([item["function"] for item in failed], ["trackrag.retrieval.rank_evidence"])
        function.assert_not_called()

    def test_cli_reports_runtime_adapter_failure_without_traceback(self):
        module = SimpleNamespace(
            open_recording=Mock(return_value=self.recording),
            load_questions=Mock(return_value=[self.question]),
            predict_tracks=Mock(side_effect=RuntimeError("detector failed")),
            build_index=Mock(), rank_evidence=Mock(),
        )
        with patch("trackrag.__main__.import_module", return_value=module):
            with redirect_stderr(io.StringIO()) as stderr:
                code = main([
                    "review", "--data-root", str(self.root), "--recording", "recording-a",
                    "--questions", "q.jsonl", "--source", "predicted", "--out", str(self.root / "run"),
                ])
        self.assertEqual(code, 1)
        self.assertIn("detector failed", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertFalse((self.root / "run").exists())

    def test_cli_rejects_a_loader_returning_the_wrong_selected_recording(self):
        module = SimpleNamespace(
            open_recording=Mock(return_value=self.recording),
            load_questions=Mock(return_value=[self.question]),
            predict_tracks=Mock(), build_index=Mock(), rank_evidence=Mock(),
        )
        with patch("trackrag.__main__.import_module", return_value=module):
            with redirect_stderr(io.StringIO()) as stderr:
                code = main([
                    "review", "--data-root", str(self.root), "--recording", "recording-b",
                    "--questions", "q.jsonl", "--source", "predicted", "--out", str(self.root / "run"),
                ])
        self.assertEqual(code, 1)
        self.assertIn("different recording", stderr.getvalue())
        module.predict_tracks.assert_not_called()
        module.load_questions.assert_not_called()


if __name__ == "__main__":
    unittest.main()
