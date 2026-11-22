"""Offline comparison integrity cases using small saved runs, not model mocks."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from evaluation.__main__ import main
from evaluation.compare import compare_manifest, write_comparison
from evaluation.run_audit import fingerprint
from evaluation.review_template import export_review


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.questions = [self.question("a1", "a"), self.question("a2", "a"), self.question("b1", "b")]
        self.write_lines(self.root / "questions.jsonl", self.questions)
        self.runs = {}
        for variant in ("base", "tuned"):
            for recording in ("a", "b"):
                rows = [self.result(q) for q in self.questions if q["recording_id"] == recording]
                self.runs[(variant, recording)] = self.saved_run(variant, recording, rows)
        self.manifest = {
            "version": 1, "name": "Synthetic hand-calculated comparison", "purpose": "development",
            "questions": ["questions.jsonl"],
            "variants": [{"id": variant, "runs": [{"directory": f"{variant}-{recording}"} for recording in ("a", "b")]} for variant in ("base", "tuned")],
        }
        self.manifest_path = self.root / "comparison.json"
        self.save_manifest()

    @staticmethod
    def question(qid, recording):
        return {"question_id": qid, "recording_id": recording, "query_frame": 5, "question": "Which bus is visible?"}

    @staticmethod
    def result(question, source="annotated", track=7):
        return {"question": dict(question), "source": source, "evidence": [{
            "evidence_id": "E1", "interval": {"recording_id": question["recording_id"], "source": source,
                "track_id": track, "start_frame": 5, "end_frame": 5, "rank": 1, "score": 1.0},
            "images": [{"frame_id": 5, "bbox_xyxy": [2, 2, 10, 10], "image_path": "evidence/5.png"}],
        }], "answer": {"answer_status": "answered", "answer": "The bus in E1.", "cited_evidence_ids": ["E1"]}}

    @staticmethod
    def write_lines(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def saved_run(self, variant, recording, rows, source="annotated", track=7):
        directory = self.root / f"{variant}-{recording}"
        (directory / "evidence").mkdir(parents=True)
        Image.new("RGB", (20, 16), "grey").save(directory / "evidence/5.png")
        self.write_lines(directory / "results.jsonl", rows)
        self.write_lines(directory / "tracks.jsonl", [{"recording_id": recording, "source": source,
            "track_id": track, "frame_id": 5, "class_name": "Bus", "bbox_xyxy": [2, 2, 10, 10]}])
        (directory / "run.json").write_text(json.dumps({"recording_id": recording, "source": source,
            "question_count": len(rows), "observation_count": 1, "configuration": {"adapter_module": "synthetic_fixture"}}))
        return directory

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def rewrite_results(self, variant, recording, transform):
        directory = self.runs[(variant, recording)]
        rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        transform(rows)
        self.write_lines(directory / "results.jsonl", rows)
        metadata = json.loads((directory / "run.json").read_text())
        metadata["question_count"] = len(rows)
        (directory / "run.json").write_text(json.dumps(metadata))

    def add_reviews(self, variant, grades=None):
        for entry, recording in zip(next(v for v in self.manifest["variants"] if v["id"] == variant)["runs"], ("a", "b")):
            directory = self.runs[(variant, recording)]
            reviews = [{"question_id": q["question_id"], "grade": (grades or {}).get(q["question_id"], "pass"),
                        "note": "Checked the evidence.", "reviewer": "Synthetic test reviewer"}
                       for q in self.questions if q["recording_id"] == recording]
            path = self.root / f"{variant}-{recording}-review.jsonl"
            self.write_lines(path, reviews)
            entry["review"] = {"path": path.name, "results_sha256": fingerprint(directory / "results.jsonl")}
        self.save_manifest()

    def add_gold(self):
        rows = [{"question_id": q["question_id"], "recording_id": q["recording_id"], "query_frame": 5,
                 "answerable": True, "acceptable_evidence": [{"track_id": 7, "start_frame": 5, "end_frame": 5}]}
                for q in self.questions]
        self.write_lines(self.root / "gold.jsonl", rows)
        self.manifest["gold"] = ["gold.jsonl"]
        self.save_manifest()

    def test_two_recordings_have_hand_calculated_grades_and_paired_changes_without_network(self):
        self.add_reviews("base", {"a2": "fail"})
        self.add_reviews("tuned")
        self.add_gold()
        with patch("socket.create_connection", side_effect=AssertionError("No network allowed")):
            report = compare_manifest(self.manifest_path)
        self.assertEqual([v["review_counts"]["pass"] for v in report["variants"]], [2, 3])
        self.assertEqual(report["paired_comparisons"][0]["clean_pass_gains"], ["a2"])
        self.assertEqual(report["paired_comparisons"][0]["clean_pass_losses"], [])
        self.assertEqual(report["variants"][0]["retrieval"]["counts"]["top1_correct"], 3)
        self.assertTrue(any(x["role"] == "evidence_image" for x in report["input_files"]))
        self.assertEqual({b["scope"] for b in report["variants"][0]["review_bindings"]}, {"results_only"})

    def test_missing_results_and_provider_errors_are_retained_in_the_same_cohort(self):
        self.rewrite_results("base", "a", lambda rows: rows.pop())
        self.rewrite_results("base", "b", lambda rows: rows[0].update(answer={"answer_status": "error", "answer": "Provider unavailable", "cited_evidence_ids": []}))
        self.add_gold()
        report = compare_manifest(self.manifest_path)
        baseline = report["variants"][0]
        self.assertEqual(baseline["status_counts"]["missing"], 1)
        self.assertEqual(baseline["status_counts"]["error"], 1)
        self.assertEqual(baseline["ordinary_total"], 3)
        self.assertEqual(baseline["review_counts"]["unreviewed"], 3)
        self.assertIsNone(baseline["clean_pass_fraction"])
        self.assertEqual(baseline["retrieval"]["counts"]["missing_results"], 1)

    def test_sparse_reviews_do_not_produce_a_factual_pass_rate(self):
        self.add_reviews("base")
        entry = self.manifest["variants"][0]["runs"][0]
        path = self.root / entry["review"]["path"]
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        self.write_lines(path, rows[:1])
        variant = compare_manifest(self.manifest_path)["variants"][0]
        self.assertEqual(variant["ordinary_reviewed"], 2)
        self.assertEqual(variant["review_counts"]["unreviewed"], 1)
        self.assertIsNone(variant["clean_pass_fraction"])

    def test_stale_review_binding_is_rejected(self):
        self.add_reviews("base")
        self.rewrite_results("base", "a", lambda rows: rows[0]["answer"].update(answer="A changed answer."))
        with self.assertRaisesRegex(ValueError, "different results file"):
            compare_manifest(self.manifest_path)

    def test_error_or_unrequested_answer_cannot_be_marked_pass(self):
        for status in ("error", "not_requested"):
            with self.subTest(status=status):
                self.rewrite_results("base", "a", lambda rows: rows[0].update(answer={"answer_status": status, "answer": None, "cited_evidence_ids": []}))
                self.add_reviews("base")
                with self.assertRaisesRegex(ValueError, "cannot receive"):
                    compare_manifest(self.manifest_path)

    def test_manual_diagnostic_exclusions_are_shared_and_cannot_be_invented_by_reviewers(self):
        self.add_reviews("base", {"a2": "manual"})
        with self.assertRaisesRegex(ValueError, "Manual exclusions"):
            compare_manifest(self.manifest_path)
        self.manifest["manual_question_ids"] = ["a2"]
        self.save_manifest()
        report = compare_manifest(self.manifest_path)
        self.assertEqual(report["ordinary_questions"], 2)
        self.assertEqual([v["ordinary_total"] for v in report["variants"]], [2, 2])

    def test_changed_question_text_recording_or_cutoff_is_rejected(self):
        original = (self.runs[("base", "a")] / "results.jsonl").read_bytes()
        for change in ({"question": "A different question"}, {"query_frame": 4}, {"recording_id": "b"}):
            with self.subTest(change=change):
                (self.runs[("base", "a")] / "results.jsonl").write_bytes(original)
                self.rewrite_results("base", "a", lambda rows: rows[0]["question"].update(change))
                with self.assertRaisesRegex(ValueError, "differs"):
                    compare_manifest(self.manifest_path)

    def test_duplicate_unknown_and_answer_key_question_fields_are_rejected(self):
        original = (self.runs[("base", "a")] / "results.jsonl").read_bytes()
        for mutate in (
            lambda rows: rows.append(deepcopy(rows[0])),
            lambda rows: rows[0]["question"].update(question_id="unknown"),
            lambda rows: rows[0]["question"].update(expected_answer="The bus"),
        ):
            (self.runs[("base", "a")] / "results.jsonl").write_bytes(original)
            self.rewrite_results("base", "a", mutate)
            with self.assertRaises(ValueError):
                compare_manifest(self.manifest_path)

    def test_future_frames_wrong_boxes_and_invalid_citations_are_rejected(self):
        original = (self.runs[("base", "a")] / "results.jsonl").read_bytes()
        for mutate in (
            lambda rows: rows[0]["evidence"][0]["interval"].update(end_frame=6),
            lambda rows: rows[0]["evidence"][0]["images"][0].update(frame_id=6),
            lambda rows: rows[0]["evidence"][0]["images"][0].update(bbox_xyxy=[3, 2, 10, 10]),
            lambda rows: rows[0]["answer"].update(cited_evidence_ids=["E9"]),
        ):
            (self.runs[("base", "a")] / "results.jsonl").write_bytes(original)
            self.rewrite_results("base", "a", mutate)
            with self.assertRaises(ValueError):
                compare_manifest(self.manifest_path)

    def test_missing_and_escaping_assets_fail_before_publishing(self):
        for relative in ("missing.png", "../outside.png"):
            self.rewrite_results("base", "a", lambda rows: rows[0]["evidence"][0]["images"][0].update(image_path=relative))
            with self.assertRaisesRegex(ValueError, "missing or outside"):
                write_comparison(self.manifest_path, self.root / "out")
            self.assertFalse((self.root / "out").exists())

    def test_provider_receipt_cannot_claim_an_unsupplied_frame_even_before_query(self):
        directory = self.runs[("base", "a")]
        hosted = directory / "hosted" / sha256(b"a1").hexdigest()
        hosted.mkdir(parents=True)
        (hosted / "receipt.json").write_text(json.dumps({"question_id": "a1", "model": "test",
            "sent": [{"evidence_id": "E1", "frame_id": 4, "image": "unused.jpg"}]}))
        with self.assertRaisesRegex(ValueError, "unsupplied"):
            compare_manifest(self.manifest_path)

    def test_failed_or_contradictory_final_receipts_cannot_support_successful_answers(self):
        hosted = self.runs[("base", "a")] / "hosted" / sha256(b"a1").hexdigest()
        hosted.mkdir(parents=True)
        for field, receipt in ((field, receipt) for field in ("response", "answer") for receipt in (
            {"request_status": "error"},
            {"request_status": "completed", "response": {"answer_status": "insufficient_evidence", "cited_evidence_ids": []}},
            {"request_status": "completed", "response": {"answer_status": "answered", "cited_evidence_ids": ["E9"]}},
        )):
            if "response" in receipt and field == "answer":
                receipt = {"request_status": receipt["request_status"], "answer": receipt["response"]}
            with self.subTest(field=field, receipt=receipt):
                (hosted / "receipt.json").write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, "contradict"):
                    compare_manifest(self.manifest_path)
        (hosted / "receipt.json").write_text(json.dumps({"request_status": "completed", "response": {
            "answer_status": "answered", "cited_evidence_ids": ["E1"]}}))
        (hosted / "draft-receipt.json").write_text(json.dumps({"request_status": "completed", "response": {
            "answer_status": "insufficient_evidence", "cited_evidence_ids": []}}))
        self.assertEqual(compare_manifest(self.manifest_path)["variants"][0]["status_counts"]["answered"], 3)

    def test_rejected_completed_provider_response_can_remain_an_application_error(self):
        self.rewrite_results("base", "a", lambda rows: rows[0].update(answer={
            "answer_status": "error", "answer": "Invalid provider citations", "cited_evidence_ids": []}))
        hosted = self.runs[("base", "a")] / "hosted" / sha256(b"a1").hexdigest()
        hosted.mkdir(parents=True)
        (hosted / "receipt.json").write_text(json.dumps({"request_status": "completed", "response": {
            "answer_status": "answered", "cited_evidence_ids": ["E9"]}}))
        self.add_reviews("base", {"a1": "fail"})
        report = compare_manifest(self.manifest_path)
        self.assertEqual(report["variants"][0]["status_counts"]["error"], 1)
        self.assertEqual(report["variants"][0]["review_counts"]["pass"], 2)

    def test_predicted_identity_is_scored_with_annotations_not_numeric_id_equality(self):
        for recording in ("a", "b"):
            directory = self.runs[("tuned", recording)]
            self.rewrite_results("tuned", recording, lambda rows: [r.update(source="predicted") or r["evidence"][0]["interval"].update(source="predicted", track_id=99) for r in rows])
            tracks = [{"recording_id": recording, "source": "predicted", "frame_id": 5, "track_id": 99,
                       "class_name": "Bus", "bbox_xyxy": [2, 2, 10, 10]}]
            self.write_lines(directory / "tracks.jsonl", tracks)
            meta = json.loads((directory / "run.json").read_text()); meta["source"] = "predicted"
            (directory / "run.json").write_text(json.dumps(meta))
            annotations = self.root / f"{recording}-annotations.jsonl"
            self.write_lines(annotations, [dict(tracks[0], source="annotated", track_id=7)])
            entry = next(e for e in self.manifest["variants"][1]["runs"] if e["directory"] == directory.name)
            entry["annotated_tracks"] = annotations.name
        self.add_gold()
        report = compare_manifest(self.manifest_path)
        self.assertEqual(report["variants"][1]["retrieval"]["counts"]["top1_correct"], 3)
        self.assertEqual(report["variants"][1]["retrieval"]["predicted_track_matching"]["matched"], 3)
        self.manifest["variants"][1]["runs"][1].pop("annotated_tracks")
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "own annotated reference"):
            compare_manifest(self.manifest_path)

    def test_gold_cannot_drop_duplicate_or_retime_questions(self):
        self.add_gold()
        original = (self.root / "gold.jsonl").read_bytes()
        for mutation in (lambda rows: rows.pop(), lambda rows: rows.append(deepcopy(rows[0])),
                         lambda rows: rows[0].update(query_frame=4)):
            (self.root / "gold.jsonl").write_bytes(original)
            rows = [json.loads(x) for x in (self.root / "gold.jsonl").read_text().splitlines()]
            mutation(rows)
            self.write_lines(self.root / "gold.jsonl", rows)
            with self.assertRaises(ValueError):
                compare_manifest(self.manifest_path)

    def test_unreadable_images_are_rejected_without_partial_output(self):
        (self.runs[("base", "a")] / "evidence/5.png").write_bytes(b"not an image")
        with self.assertRaises(OSError):
            write_comparison(self.manifest_path, self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_scene_evidence_does_not_require_a_fake_track_identity(self):
        self.rewrite_results("base", "a", lambda rows: [(r["evidence"][0]["interval"].update(track_id=None),
            r["evidence"][0]["images"][0].update(bbox_xyxy=None)) for r in rows])
        report = compare_manifest(self.manifest_path)
        self.assertEqual(report["variants"][0]["status_counts"]["answered"], 3)
        self.assertIsNone(report["variants"][0]["retrieval"])

    def test_command_process_writes_atomic_json_and_markdown_without_overwriting(self):
        result = subprocess.run([sys.executable, "-m", "evaluation", "compare", "--manifest", str(self.manifest_path),
                                 "--out", str(self.root / "output")], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "output/summary.json").is_file())
        self.assertIn("No content review supplied", (self.root / "output/report.md").read_text(encoding="utf-8"))
        with self.assertRaises(FileExistsError):
            write_comparison(self.manifest_path, self.root / "output")
        self.assertEqual(list(self.root.glob(".comparison-*")), [])

    def test_cli_reports_a_manifest_typo_clearly_and_keeps_existing_scoring_interface(self):
        self.manifest["variantz"] = self.manifest.pop("variants")
        self.save_manifest()
        with redirect_stderr(io.StringIO()) as stderr:
            code = main(["compare", "--manifest", str(self.manifest_path), "--out", str(self.root / "out")])
        self.assertEqual(code, 1)
        self.assertIn("Unknown fields", stderr.getvalue())

    def use_review_packet(self):
        path = self.root / "review-packet"
        export_review(self.runs[("base", "a")], path)
        binding = json.loads((path / "binding.json").read_text())
        self.manifest["variants"][0]["runs"][0]["review"] = binding["review"]
        self.save_manifest()
        return path

    def test_exported_review_packet_starts_ungraded_and_only_completed_rows_count(self):
        path = self.root / "review-packet"
        with redirect_stdout(io.StringIO()):
            code = main(["review-template", "--run", str(self.runs[("base", "a")]), "--out", str(path)])
        self.assertEqual(code, 0)
        rows = [json.loads(x) for x in (path / "review.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["grade"] is None for row in rows))
        self.assertIn("does not validate", (path / "worksheet.md").read_text(encoding="utf-8"))
        binding = json.loads((path / "binding.json").read_text())
        self.manifest["variants"][0]["runs"][0]["review"] = binding["review"]
        self.save_manifest()
        baseline = compare_manifest(self.manifest_path)["variants"][0]
        self.assertEqual(baseline["ordinary_reviewed"], 0)
        self.assertEqual(baseline["review_counts"]["unreviewed"], 3)
        rows[0].update(grade="pass", note="Reviewed the answer and image.", reviewer="Test reviewer")
        self.write_lines(path / "review.jsonl", rows)
        baseline = compare_manifest(self.manifest_path)["variants"][0]
        self.assertEqual(baseline["ordinary_reviewed"], 1)
        self.assertEqual(baseline["review_bindings"][0]["scope"], "results_and_audited_artifacts")
        self.assertIsNone(baseline["clean_pass_fraction"])

    def test_review_packet_rejects_changed_image_bytes_even_when_results_are_unchanged(self):
        self.use_review_packet()
        Image.new("RGB", (20, 16), "blue").save(self.runs[("base", "a")] / "evidence/5.png")
        with self.assertRaisesRegex(ValueError, "snapshot differs"):
            compare_manifest(self.manifest_path)

    def test_review_snapshot_cannot_omit_an_audited_file(self):
        path = self.use_review_packet()
        snapshot = json.loads((path / "artifacts.json").read_text())
        snapshot["files"].pop()
        (path / "artifacts.json").write_text(json.dumps(snapshot))
        with self.assertRaisesRegex(ValueError, "snapshot differs"):
            compare_manifest(self.manifest_path)

    def test_duplicate_pending_review_rows_are_not_silently_ignored(self):
        path = self.use_review_packet()
        rows = [json.loads(x) for x in (path / "review.jsonl").read_text().splitlines()]
        self.write_lines(path / "review.jsonl", rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicated"):
            compare_manifest(self.manifest_path)


if __name__ == "__main__":
    unittest.main()
