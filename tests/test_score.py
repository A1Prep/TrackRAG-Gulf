"""Hand-calculated retrieval cases for the offline evaluation command."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from evaluation.__main__ import main
from evaluation.score import _resolve_assignment, score_results


def gold_row(question_id, answerable=True):
    return {
        "question_id": question_id, "recording_id": "a", "query_frame": 20,
        "answerable": answerable,
        "acceptable_evidence": [{"track_id": 7, "start_frame": 1, "end_frame": 5}] if answerable else [],
    }


def interval(track=7, start=1, end=5, rank=1, source="annotated"):
    return {
        "recording_id": "a", "source": source, "track_id": track,
        "start_frame": start, "end_frame": end, "rank": rank, "score": 1 / rank,
    }


def result_row(question_id, intervals, source="annotated", status="not_requested"):
    return {
        "question": {"question_id": question_id, "recording_id": "a", "query_frame": 20, "question": "Which bus?"},
        "source": source,
        "evidence": [{"evidence_id": f"E{i}", "interval": item, "images": []} for i, item in enumerate(intervals, 1)],
        "answer": {"answer_status": status, "answer": None, "cited_evidence_ids": []},
    }


def observation(track_id, frame_id, box, source):
    return {"recording_id": "a", "track_id": track_id, "frame_id": frame_id, "bbox_xyxy": box, "source": source}


class ScoreTests(unittest.TestCase):
    def test_hand_calculated_first_third_wrong_vehicle_wrong_time_missing(self):
        gold = [gold_row(name) for name in ("first", "third", "vehicle", "time", "missing")]
        gold.append(gold_row("unanswerable", False))
        results = [
            result_row("first", [interval()]),
            result_row("third", [interval(track=1), interval(track=2, rank=2), interval(rank=3)]),
            result_row("vehicle", [interval(track=9)]),
            result_row("time", [interval(start=10, end=15)]),
            result_row("unanswerable", [], status="insufficient_evidence"),
        ]
        scores = score_results(results, gold)
        self.assertEqual(scores["counts"], {
            "answerable": 5, "unanswerable": 1, "top1_correct": 1, "top3_correct": 2, "missing_results": 1,
        })
        self.assertEqual(scores["top1_accuracy"], 0.2)
        self.assertEqual(scores["top3_recall"], 0.4)
        self.assertIsNone(scores["questions"][-1]["top1_correct"])

    def test_inclusive_frame_iou_and_alternative_support(self):
        gold = gold_row("q")
        gold["acceptable_evidence"] = [
            {"track_id": 1, "start_frame": 1, "end_frame": 3},
            {"track_id": 7, "start_frame": 3, "end_frame": 3},
        ]
        scores = score_results([result_row("q", [interval(start=3, end=3)])], [gold])
        self.assertEqual(scores["counts"]["top1_correct"], 1)
        gold["acceptable_evidence"] = [{"track_id": 7, "start_frame": 1, "end_frame": 3}]
        scores = score_results([result_row("q", [interval(start=2, end=3)])], [gold], temporal_iou=2/3)
        self.assertEqual(scores["counts"]["top1_correct"], 1)

    def test_predicted_id_matches_by_boxes_instead_of_numeric_identity(self):
        annotated = [observation(7, frame, (0, 0, 10, 10), "annotated") for frame in range(1, 6)]
        predicted = [observation(99, frame, (0, 0, 10, 10), "predicted") for frame in range(1, 6)]
        results = [result_row("q", [interval(track=99, source="predicted")], source="predicted")]
        scores = score_results(results, [gold_row("q")], annotated_tracks=annotated, predicted_tracks=predicted)
        self.assertEqual(scores["counts"]["top1_correct"], 1)
        wrong = [observation(7, frame, (20, 20, 30, 30), "predicted") for frame in range(1, 6)]
        results[0]["evidence"][0]["interval"]["track_id"] = 7
        scores = score_results(results, [gold_row("q")], annotated_tracks=annotated, predicted_tracks=wrong)
        self.assertEqual(scores["counts"]["top1_correct"], 0)

    def test_missing_track_frames_reduce_box_overlap(self):
        annotated = [observation(7, frame, (0, 0, 10, 10), "annotated") for frame in range(1, 6)]
        predicted = [observation(99, 1, (0, 0, 10, 10), "predicted")]
        result = result_row("q", [interval(track=99, source="predicted")], source="predicted")
        scores = score_results([result], [gold_row("q")], annotated_tracks=annotated, predicted_tracks=predicted)
        self.assertEqual(scores["counts"]["top1_correct"], 0)

    def test_nearby_wrong_vehicle_does_not_receive_gold_identity(self):
        annotated = [
            observation(7, frame, (0, 0, 10, 10), "annotated") for frame in range(1, 6)
        ] + [
            observation(8, frame, (1, 0, 11, 10), "annotated") for frame in range(1, 6)
        ]
        predicted = [observation(99, frame, (1, 0, 11, 10), "predicted") for frame in range(1, 6)]
        result = result_row("q", [interval(track=99, source="predicted")], source="predicted")
        scores = score_results([result], [gold_row("q")], annotated_tracks=annotated, predicted_tracks=predicted)
        self.assertEqual(scores["counts"]["top1_correct"], 0)
        self.assertEqual(scores["questions"][0]["track_matches"][0]["annotated_track_id"], 8)

    def test_duplicate_predictions_are_reported_as_ambiguous(self):
        annotated = [observation(7, frame, (0, 0, 10, 10), "annotated") for frame in range(1, 6)]
        predicted = [
            observation(track, frame, (0, 0, 10, 10), "predicted")
            for track in (99, 100) for frame in range(1, 6)
        ]
        for track_id in (99, 100):
            with self.subTest(track_id=track_id):
                result = result_row("q", [interval(track=track_id, source="predicted")], source="predicted")
                scores = score_results([result], [gold_row("q")], annotated_tracks=annotated, predicted_tracks=predicted)
                self.assertEqual(scores["counts"]["top1_correct"], 0)
                self.assertEqual(scores["predicted_track_matching"]["ambiguous"], 1)

    def test_global_assignment_uses_competitors_and_detects_global_ties(self):
        cases = [
            ([[0.8, 0.1], [0.9, 1.0]], 0, 0, "matched"),
            ([[0.9, 0.9], [0.9, 0.1]], 0, 1, "matched"),
            ([[0.9, 0.8], [0.8, 0.7]], 0, None, "ambiguous"),
            ([[1.0], [1.0]], 0, None, "ambiguous"),
            ([[1.0], [1.0]], 1, None, "ambiguous"),
            ([[1.0], [0.6]], 1, None, "unmatched"),
            ([[0.0]], 0, None, "unmatched"),
            ([], 0, None, "unmatched"),
            ([[]], 0, None, "unmatched"),
        ]
        for weights, row, expected_column, expected_status in cases:
            with self.subTest(weights=weights):
                column, score, status = _resolve_assignment(weights, row)
                self.assertEqual((column, status), (expected_column, expected_status))

    def test_scene_gold_rejects_a_time_range(self):
        reference = gold_row("q")
        reference["acceptable_evidence"][0]["track_id"] = None
        with self.assertRaisesRegex(ValueError, "Scene gold"):
            score_results([result_row("q", [interval()])], [reference])

    def test_scene_evidence_matches_exact_frame_for_either_source_without_tracks(self):
        reference = gold_row("q")
        reference["acceptable_evidence"] = [{"track_id": None, "start_frame": 5, "end_frame": 5}]
        for source in ("annotated", "predicted"):
            with self.subTest(source=source):
                result = result_row("q", [interval(track=None, start=5, end=5, source=source)], source=source)
                self.assertEqual(score_results([result], [reference])["counts"]["top1_correct"], 1)
                result["evidence"][0]["interval"].update(start_frame=4, end_frame=4)
                self.assertEqual(score_results([result], [reference])["counts"]["top1_correct"], 0)

    def test_unmatched_prediction_is_not_scene_evidence(self):
        reference = gold_row("q")
        reference["acceptable_evidence"] = [{"track_id": None, "start_frame": 5, "end_frame": 5}]
        result = result_row("q", [interval(track=99, start=5, end=5, source="predicted")], source="predicted")
        scores = score_results(
            [result], [reference],
            annotated_tracks=[observation(7, 5, (0, 0, 10, 10), "annotated")],
            predicted_tracks=[observation(99, 5, (20, 20, 30, 30), "predicted")],
        )
        self.assertEqual(scores["counts"]["top1_correct"], 0)
        self.assertEqual(scores["predicted_track_matching"]["unmatched"], 1)

    def test_scene_result_rejects_a_time_range(self):
        result = result_row("q", [interval(track=None)])
        with self.assertRaisesRegex(ValueError, "Scene evidence"):
            score_results([result], [gold_row("q")])

    def test_predicted_future_boxes_cannot_help_match_an_earlier_interval(self):
        annotated = [observation(7, frame, (0, 0, 10, 10), "annotated") for frame in (1, 2, 30)]
        predicted = [observation(99, frame, (20, 20, 30, 30), "predicted") for frame in (1, 2)]
        predicted.append(observation(99, 30, (0, 0, 10, 10), "predicted"))
        result = result_row("q", [interval(track=99, source="predicted")], source="predicted")
        scores = score_results([result], [gold_row("q")], annotated_tracks=annotated, predicted_tracks=predicted)
        self.assertEqual(scores["counts"]["top1_correct"], 0)

    def test_predicted_ids_require_observations(self):
        result = result_row("q", [interval(source="predicted")], source="predicted")
        with self.assertRaisesRegex(ValueError, "requires predicted and annotated"):
            score_results([result], [gold_row("q")])

    def test_result_recording_and_query_time_must_match_gold(self):
        for change in ({"recording_id": "other"}, {"query_frame": 19}):
            with self.subTest(change=change):
                result = result_row("q", [interval()])
                result["question"].update(change)
                with self.assertRaisesRegex(ValueError, "disagree"):
                    score_results([result], [gold_row("q")])

    def test_duplicate_and_unknown_questions_are_rejected(self):
        result = result_row("q", [])
        for results, gold in (
            ([result, result], [gold_row("q")]),
            ([result], [gold_row("q"), gold_row("q")]),
            ([result], [gold_row("other")]),
        ):
            with self.subTest(results=results, gold=gold):
                with self.assertRaises(ValueError):
                    score_results(results, gold)

    def test_all_unanswerable_has_no_retrieval_denominator(self):
        scores = score_results([result_row("q", [])], [gold_row("q", False)])
        self.assertIsNone(scores["top1_accuracy"])
        self.assertIsNone(scores["top3_recall"])

    def test_score_command_reads_separate_files(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            results = directory / "results.jsonl"
            gold = directory / "gold.jsonl"
            results.write_text(json.dumps(result_row("q", [interval()])) + "\n", encoding="utf-8")
            gold.write_text(json.dumps(gold_row("q")) + "\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()) as stdout:
                code = main(["--results", str(results), "--gold", str(gold)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["counts"]["top1_correct"], 1)


if __name__ == "__main__":
    unittest.main()
