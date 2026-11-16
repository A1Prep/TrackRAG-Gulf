"""Falsification and handoff checks using synthetic images, without model calls."""

import json
import io
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from evaluation.run_audit import audit_run, fingerprint
from evaluation.score import read_jsonl
from evaluation.temporal_checks import checked_ranking, same_ranking
from evaluation.verify_retrieval import check_recording, verify_suite
from tools.emt_trial_adapters import build_index, rank_evidence
from trackrag.emt import load_annotated_tracks, open_recording


def bounded_rank(question, index, k=3):
    rows = [row for row in index if row["frame_id"] <= question["query_frame"]]
    if not rows:
        return []
    track = min(row["track_id"] for row in rows)
    rows = [row for row in rows if row["track_id"] == track]
    return [{"recording_id": question["recording_id"], "source": rows[0]["source"],
             "track_id": track, "start_frame": rows[0]["frame_id"],
             "end_frame": rows[-1]["frame_id"], "rank": 1, "score": 1.0}]


class RetrievalVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "emt"
        self.questions = []
        for recording_id in ("rec_a", "rec_b"):
            directory = self.data / "images" / "train" / "images" / recording_id
            directory.mkdir(parents=True)
            for frame in range(1, 5):
                Image.new("RGB", (40, 30), (frame, 40, 60)).save(directory / f"{frame:05d}.png")
            annotations = self.data / "annotations" / "train"
            annotations.mkdir(parents=True, exist_ok=True)
            (annotations / f"{recording_id}.txt").write_text("".join(
                f"{frame:05d} 7 Bus 0 0 0 3 4 20 22\n" for frame in range(1, 5)
            ), encoding="utf-8")
            self.questions.append({"question_id": recording_id, "recording_id": recording_id,
                                   "query_frame": 2, "question": "Which bus is on the left?"})
        self.question_path = self.root / "questions.jsonl"
        self.write_jsonl(self.question_path, self.questions, encoding="utf-8-sig")
        self.recording = open_recording(self.data, "rec_a")
        self.rows = load_annotated_tracks(self.recording)

    @staticmethod
    def write_jsonl(path, rows, **options):
        Path(path).write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                              encoding=options.get("encoding", "utf-8"))

    def check(self, rank=bounded_rank, build=None, questions=None):
        report, _ = check_recording(self.recording, questions or self.questions[:1], self.rows,
                                    "annotated", build or (lambda recording, rows: rows), rank,
                                    self.root / "controls")
        return report

    def test_positive_bounded_adapter_and_real_protocol_wrappers(self):
        seen_sizes = []

        def build(recording, rows):
            seen_sizes.append((len(list(recording.iter_frames())), len(rows)))
            return rows

        report = self.check(build=build)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["positive_temporal_comparisons"], 3)
        self.assertEqual(seen_sizes[0], (4, 4))  # Full baseline, without hidden truncation.
        self.assertIn((2, 2), seen_sizes)

    def test_future_observation_leak_is_detected_despite_past_only_output(self):
        def leaky(question, index, k=3):
            result = bounded_rank(question, index)
            result[0]["score"] = float(len(index))
            return result

        report = self.check(leaky)
        checks = report["questions"][0]["checks"]
        self.assertEqual(report["status"], "failed")
        self.assertEqual(checks["future_observations"]["status"], "failed")
        self.assertEqual(checks["prefix_only"]["status"], "failed")
        self.assertEqual(checks["raw_contract"]["status"], "passed")

    def test_future_pixel_leak_is_detected(self):
        def build(recording, rows):
            last = list(recording.iter_frames())[-1]
            with Image.open(last["image_path"]) as image:
                pixel = image.convert("RGB").getpixel((0, 0))[0]
            return rows, pixel

        def leaky(question, index, k=3):
            result = bounded_rank(question, index[0])
            result[0]["score"] = float(index[1])
            return result

        checks = self.check(leaky, build)["questions"][0]["checks"]
        self.assertEqual(checks["future_pixels"]["status"], "failed")
        self.assertEqual(checks["wrapper_control"]["status"], "passed")

    def test_unstable_control_is_inconclusive(self):
        calls = 0

        def unstable(question, index, k=3):
            nonlocal calls
            calls += 1
            result = bounded_rank(question, index)
            result[0]["score"] = float(calls)
            return result

        report = self.check(unstable)
        self.assertEqual(report["status"], "inconclusive")
        self.assertEqual(report["positive_temporal_comparisons"], 0)

    def test_private_recording_attribute_dependency_is_inconclusive(self):
        def build(recording, rows):
            _ = recording.frame_ids
            return rows

        report = self.check(build=build)
        self.assertEqual(report["status"], "inconclusive")
        self.assertEqual(report["questions"][0]["checks"]["wrapper_control"]["status"], "inconclusive")

    def test_empty_outputs_do_not_count_as_positive_checks(self):
        report = self.check(lambda *args, **kwargs: [])
        self.assertEqual(report["status"], "inconclusive")
        self.assertEqual(report["positive_temporal_comparisons"], 0)
        self.assertEqual(report["empty_temporal_comparisons"], 3)

    def test_zero_score_scene_context_is_separate_from_retrieval_matches(self):
        report = self.check(rank_evidence, build_index,
                            questions=[dict(self.questions[0], question="Is there an overpass?")])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["ranking_kind_counts"], {"zero_score_scene_context": 1})
        self.assertEqual(report["context_temporal_comparisons"], 3)
        self.assertEqual(report["positive_temporal_comparisons"], 3)

    def test_end_of_recording_query_has_no_positive_temporal_check(self):
        report = self.check(questions=[dict(self.questions[0], query_frame=4)])
        self.assertEqual(report["status"], "inconclusive")
        self.assertEqual(report["questions"][0]["checks"]["future_tests"]["status"], "not_applicable")

    def test_mixed_cutoff_cohort_keeps_untested_query_gap_visible(self):
        questions = [self.questions[0], dict(self.questions[0], question_id="last", query_frame=4)]
        report = self.check(questions=questions)
        self.assertEqual(report["status"], "passed_with_gaps")
        self.assertEqual(report["positive_temporal_comparisons"], 3)

    def test_raw_rank_contract_rejects_invalid_adapters(self):
        changes = [{"end_frame": 4}, {"source": "predicted"}, {"track_id": 88},
                   {"score": float("nan")}, {"score": True}, {"score": "1"},
                   {"answer": "gold"}, {"rank": 2}, {"track_id": None}]
        for change in changes:
            with self.subTest(change=change):
                def invalid(question, index, k=3):
                    return [dict(bounded_rank(question, index)[0], **change)]
                report = self.check(invalid)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["questions"][0]["checks"]["raw_contract"]["status"], "failed")
                self.assertIn("returned_intervals_repr", report["questions"][0]["baseline"])

    def test_scene_evidence_is_valid_without_tracks(self):
        scene = dict(bounded_rank(self.questions[0], self.rows)[0], track_id=None, start_frame=2, end_frame=2)
        self.assertEqual(checked_ranking([scene], self.questions[0], "annotated", [],
                                        list(self.recording.iter_frames())), [scene])

    def test_query_mutation_is_not_accepted(self):
        def mutating(question, index, k=3):
            result = bounded_rank(question, index)
            question["question"] = "changed"
            return result
        report = self.check(mutating)
        self.assertEqual(report["status"], "failed")
        self.assertIn("modified", report["questions"][0]["baseline"]["detail"])
        self.assertIn("Which bus", self.questions[0]["question"])

    def test_tolerance_applies_only_to_scores(self):
        original = bounded_rank(self.questions[0], self.rows)
        close = [dict(original[0], score=1.0000001)]
        self.assertTrue(same_ranking(original, close, 1e-6))
        self.assertFalse(same_ranking(original, [dict(original[0], score=1.01)], 1e-6))
        self.assertFalse(same_ranking(original, [dict(original[0], track_id=8)], 1e-6))

    def test_trial_index_preserves_prediction_source_and_identity(self):
        predicted = [dict(row, source="predicted", track_id=99) for row in self.rows]
        result = rank_evidence(self.questions[0], build_index(self.recording, predicted))
        self.assertTrue(result)
        self.assertTrue(all(row["source"] == "predicted" and row["track_id"] == 99 for row in result))
        with self.assertRaises(ValueError):
            build_index(self.recording, predicted + self.rows)

    def test_two_recording_cli_exports_auditable_evidence_without_answerer(self):
        output = self.root / "cli"
        process = subprocess.run([
            sys.executable, "-m", "evaluation", "verify-retrieval", "--data-root", str(self.data),
            "--questions", str(self.question_path), "--source", "annotated",
            "--retrieval-module", "tools.emt_trial_adapters", "--out", str(output),
        ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=40)
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["two_recordings_exercised"])
        for item in report["recordings"]:
            audit_run(output / item["run_directory"], {q["question_id"]: q for q in self.questions})
            result = read_jsonl(output / item["run_directory"] / "results.jsonl")[0]
            self.assertEqual(result["answer"]["answer_status"], "not_requested")
            self.assertTrue(all(e["interval"]["end_frame"] <= 2 for e in result["evidence"]))

    def test_saved_prediction_handoff_never_calls_detector_and_keeps_original_boxes(self):
        saved = [dict(row, source="predicted", track_id=99) for row in self.rows]
        path = self.root / "predictions.jsonl"
        self.write_jsonl(path, saved)
        questions = self.root / "one.jsonl"
        self.write_jsonl(questions, self.questions[:1])
        output = self.root / "predicted"
        with patch("evaluation.verify_retrieval._adapter", side_effect=lambda module, name, task: {
            "build_index": build_index, "rank_evidence": rank_evidence,
        }[name]) as adapters:
            report = verify_suite(self.data, [questions], "predicted", output,
                                  retrieval_module="tools.emt_trial_adapters", predicted_paths=[path])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(adapters.call_count, 2)
        self.assertFalse(report["detector_adapter_called"])
        self.assertIn("not verified", report["observation_origin"])
        self.assertEqual(report["prediction_inputs"][0]["sha256"], fingerprint(path))
        item = report["recordings"][0]
        self.assertEqual(read_jsonl(output / item["run_directory"] / "tracks.jsonl"), read_jsonl(path))

    def test_invalid_prediction_source_or_foreign_recording_is_rejected(self):
        for row in (self.rows[0], dict(self.rows[0], source="predicted", recording_id="foreign")):
            path = self.root / "bad.jsonl"
            self.write_jsonl(path, [row])
            with self.subTest(row=row), self.assertRaises(ValueError):
                verify_suite(self.data, [self.question_path], "predicted", self.root / "bad-output",
                             retrieval_module="tools.emt_trial_adapters", predicted_paths=[path])

    def test_duplicate_or_out_of_bounds_predictions_are_recorded_as_failures(self):
        good = dict(self.rows[0], source="predicted", track_id=99)
        for rows in ([good, good], [dict(good, bbox_xyxy=[1, 1, 50, 20])]):
            with self.subTest(rows=rows):
                path = self.root / "bad.jsonl"
                self.write_jsonl(path, rows)
                output = self.root / f"invalid-{len(rows)}"
                report = verify_suite(self.data, [self.question_path], "predicted", output,
                                      retrieval_module="tools.emt_trial_adapters", predicted_paths=[path])
                self.assertEqual(report["status"], "failed")
                self.assertIn("error", report["recordings"][0])
                self.assertTrue((output / "report.json").is_file())

    def test_no_overwrite_or_gold_question_fields_or_invalid_tolerance(self):
        with self.assertRaises(FileExistsError):
            verify_suite(self.data, [self.question_path], "annotated", self.root)
        for tolerance in (True, -1, float("nan")):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                verify_suite(self.data, [self.question_path], "annotated", self.root / "unused", tolerance=tolerance)
        path = self.root / "gold.jsonl"
        self.write_jsonl(path, [dict(self.questions[0], answer="Bus")])
        with self.assertRaises(ValueError):
            verify_suite(self.data, [path], "annotated", self.root / "unused")

    def test_one_recording_failure_does_not_hide_other_recordings(self):
        (self.data / "annotations" / "train" / "rec_a.txt").unlink()
        report = verify_suite(self.data, [self.question_path], "annotated", self.root / "partial",
                              retrieval_module="tools.emt_trial_adapters")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["recordings"][0]["status"], "failed")
        self.assertEqual(report["recordings"][1]["status"], "passed")

    def test_live_batch_preserves_two_real_handoffs_with_http_replaced(self):
        from tools.live_trial import main as live_main
        def wire(request,**kwargs):
            body=json.loads(request.data)
            required=body['generationConfig']['responseSchema']['required']
            response={'selected':['T7'],'requested_frame_ids':[]} if 'selected' in required else {
                'answer_status':'answered','answer':'The bus.','cited_evidence_ids':['E1']}
            return io.BytesIO(json.dumps({'modelVersion':'test-model','candidates':[{'content':{'parts':[{'text':json.dumps(response)}]}}]}).encode())
        output=self.root/'live-mocked'
        with patch.dict(os.environ,{'GEMINI_API_KEY':'unused-test-key','TRACKRAG_TRIAL_CACHE':str(self.root/'model-cache')}):
            with patch('urllib.request.urlopen',side_effect=wire):
                status=live_main(['--data-root',str(self.data),'--questions',str(self.question_path),
                                  '--adapters','tools.grounded_trial_adapters','--model','test-model',
                                  '--min-interval','0','--out',str(output)])
        self.assertEqual(status,0)
        summary=json.loads((output/'summary.json').read_text())
        self.assertTrue(summary['all_planned_cases_have_fresh_responses'])
        self.assertEqual(len(summary['cases']),2)
        self.assertEqual(sum(event['role']=='answer' for case in summary['cases'] for event in case['attempts'][0]['provider']['events']),2)

    def test_live_batch_retains_quota_error_body_and_stops_before_second_case(self):
        from tools.live_trial import main as live_main
        provider={'error':{'status':'RESOURCE_EXHAUSTED','message':'Daily limit',
                           'details':[{'violations':[{'quotaId':'RequestsPerDay'}]}]}}
        def limited(request,**kwargs):
            raise urllib.error.HTTPError('https://example.invalid',429,'limit',{},io.BytesIO(json.dumps(provider).encode()))
        output=self.root/'live-limited'
        with patch.dict(os.environ,{'GEMINI_API_KEY':'unused-test-key','TRACKRAG_TRIAL_CACHE':str(self.root/'model-cache')}):
            with patch('urllib.request.urlopen',side_effect=limited) as network:
                status=live_main(['--data-root',str(self.data),'--questions',str(self.question_path),
                                  '--adapters','tools.grounded_trial_adapters','--model','test-model',
                                  '--min-interval','0','--out',str(output)])
        self.assertEqual(status,1)
        self.assertEqual(network.call_count,1)
        summary=json.loads((output/'summary.json').read_text())
        self.assertEqual(summary['unattempted_question_ids'],['rec_b'])
        self.assertIn('Daily provider limit',summary['stop_reason'])

    def test_saved_file_missing_recording_is_inconclusive_even_with_scene_results(self):
        path = self.root / "predictions.jsonl"
        self.write_jsonl(path, [dict(row, source="predicted", track_id=99) for row in self.rows])

        def scene(question, index, k=3):
            return [{"recording_id": question["recording_id"], "source": "predicted", "track_id": None,
                     "start_frame": 2, "end_frame": 2, "rank": 1, "score": 1.0}]

        with patch("evaluation.verify_retrieval._adapter", side_effect=lambda module, name, task: {
            "build_index": lambda recording, rows: rows, "rank_evidence": scene,
        }[name]):
            report = verify_suite(self.data, [self.question_path], "predicted", self.root / "coverage",
                                  predicted_paths=[path])
        self.assertEqual(report["status"], "inconclusive")
        self.assertEqual(report["recordings"][1]["saved_prediction_coverage"], "unverified_empty")
        self.assertEqual(report["recordings"][1]["positive_temporal_comparisons"], 3)

    def test_export_failure_preserves_original_checks(self):
        with patch("evaluation.verify_retrieval.audit_run", side_effect=ValueError("invalid artifact")):
            report = verify_suite(self.data, [self.question_path], "annotated", self.root / "audit-failure",
                                  retrieval_module="tools.emt_trial_adapters")
        self.assertEqual(report["status"], "failed")
        for item in report["recordings"]:
            self.assertIn("invalid artifact", item["error"]["detail"])
            self.assertEqual(item["positive_temporal_comparisons"], 3)
            self.assertTrue(item["questions"][0]["baseline"]["intervals"])

    def test_detector_adapter_called_flag_and_preflight_query_bounds(self):
        calls = []

        def detector(recording):
            calls.append(recording.recording_id)
            return [dict(row, source="predicted", track_id=99) for row in load_annotated_tracks(recording)]

        bad = self.root / "unavailable-frame.jsonl"
        self.write_jsonl(bad, [dict(self.questions[0], query_frame=999), self.questions[1]])
        with patch("evaluation.verify_retrieval._adapter", side_effect=lambda module, name, task: {
            "build_index": build_index, "rank_evidence": rank_evidence, "predict_tracks": detector,
        }[name]):
            report = verify_suite(self.data, [bad], "predicted", self.root / "detector")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(calls, ["rec_b"])
        self.assertTrue(report["detector_adapter_called"])
        self.assertEqual(report["recordings"][1]["source"], "predicted")


if __name__ == "__main__":
    unittest.main()
