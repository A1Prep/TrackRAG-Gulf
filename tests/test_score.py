"""Hand-calculated retrieval cases for the offline evaluation command."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from evaluation.__main__ import main
from evaluation.score import score_results


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
