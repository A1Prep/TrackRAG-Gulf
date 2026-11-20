"""Contract tests use injected scorers; real pretrained runs are recorded separately."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from trackrag.contracts import RankedInterval
from trackrag.retrieval import RetrievalConfig, build_index as baseline_index, rank_evidence as baseline_rank
from trackrag.retrieval_candidates import build_candidate_batch
from trackrag.retrieval_qwen import (
    QwenConfig, QwenTextScorer, build_index, config_from_environment,
    rank_candidate_batch, rank_evidence, text_documents, _query_text,
)


class Recording:
    recording_id = "test-recording"

    def __init__(self):
        self.requested = []
        self.frames = {frame: {"recording_id": self.recording_id, "frame_id": frame,
                              "width": 100, "height": 80, "timestamp_s": frame / 10,
                              "image_path": f"not-opened/{frame}.jpg"} for frame in range(121)}

    def iter_frames(self):
        raise AssertionError("Do not scan future frames")

    def get_frame(self, frame):
        self.requested.append(frame)
        return self.frames[frame]


def observation(frame, track=7, label="Bus", source="annotated"):
    return {"recording_id": "test-recording", "source": source, "frame_id": frame,
            "track_id": track, "class_name": label, "bbox_xyxy": (10, 20, 30, 40)}


def question(text="Which bus is visible?", cutoff=100):
    return {"question_id": "test-q", "recording_id": "test-recording", "query_frame": cutoff, "question": text}


class ObservedClassScorer:
    """Test double, not a model or a claimed learned result."""

    def __init__(self):
        self.calls = []

    def score(self, query, documents):
        self.calls.append((query.copy(), documents))
        target = "Bus" if "bus" in query["question"].lower() else "Car"
        return [0.8 if (json.loads(doc)["observed_track_facts"] or {}).get("class_counts", {}).get(target)
                else 0.1 for doc in documents]


class QwenConfigTests(unittest.TestCase):
    def test_configuration_is_immutable_pinned_and_explicit(self):
        config = QwenConfig()
        self.assertEqual(len(config.revision), 40)
        self.assertEqual(config.device, "cpu")
        with self.assertRaises(FrozenInstanceError):
            config.device = "cuda"
        for key, value in (("revision", "main"), ("model_id", "unknown"), ("device", "auto"),
                           ("batch_size", 0), ("batch_size", True), ("max_tokens", 100),
                           ("max_tokens", 32769), ("cpu_threads", 0), ("local_files_only", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                QwenConfig(**{key: value})

    def test_json_config_and_unknown_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"batch_size": 1, "local_files_only": True}), encoding="utf-8")
            with patch.dict(os.environ, {"TRACKRAG_QWEN_CONFIG": str(path)}):
                self.assertEqual(config_from_environment().batch_size, 1)
                path.write_text('{"unexpected": 1}', encoding="utf-8")
                with self.assertRaises(ValueError):
                    config_from_environment()


class QwenRankingTests(unittest.TestCase):
    def setUp(self):
        self.recording = Recording()
        self.rows = [observation(frame) for frame in range(90, 111)]
        self.rows += [observation(frame, 42, "Car") for frame in range(90, 111)]

    def index(self, rows=None, scorer=None, **kwargs):
        return build_index(self.recording, self.rows if rows is None else rows,
                           source="annotated", scorer=scorer or ObservedClassScorer(), **kwargs)

    def test_same_contract_and_question_changes_ranking_for_injected_scorer(self):
        scorer = ObservedClassScorer()
        index = self.index(scorer=scorer)
        for text, target in (("bus", 7), ("car", 42)):
            intervals = rank_evidence(question(text), index)
            self.assertEqual(intervals[0]["track_id"], target)
            self.assertTrue(all(set(row) == RankedInterval.__required_keys__ for row in intervals))
            self.assertEqual([row["rank"] for row in intervals], list(range(1, len(intervals) + 1)))
            self.assertEqual(index.last_receipt["status"], "completed")
            self.assertEqual(len(index.last_receipt["scores"]), len(index.last_receipt["candidate_ids"]))
        self.assertEqual(len(scorer.calls), 2)

    def test_score_all_candidates_without_top_three_or_lexical_prefilter(self):
        scorer = ObservedClassScorer()
        index = self.index(scorer=scorer, retrieval_config=RetrievalConfig(window_frames=5, stride_frames=5))
        result = rank_evidence(question("helicopter"), index, k=1)
        self.assertEqual(len(result), 1)
        batch = build_candidate_batch(question("helicopter"), index.tracks, source="annotated")
        self.assertGreater(len(scorer.calls[0][1]), 3)
        self.assertEqual(len(scorer.calls[0][1]), len(batch["candidates"]))

    def test_scene_scoring_is_possible_without_claiming_visual_information(self):
        class SceneScorer:
            def score(self, query, documents):
                return [0.9 if json.loads(doc)["evidence_kind"] == "scene" else 0.1 for doc in documents]

        index = self.index(scorer=SceneScorer())
        result = rank_evidence(question(), index)
        self.assertIsNone(result[0]["track_id"])
        self.assertEqual(result[0]["start_frame"], result[0]["end_frame"])
        self.assertFalse(index.last_receipt["visual_pixels_used"])

    def test_future_changes_preserve_all_documents_scores_and_intervals(self):
        prefix = [row for row in self.rows if row["frame_id"] <= 100]
        expected_index = self.index(rows=prefix)
        expected = rank_evidence(question(), expected_index)
        for future in ([observation(110, 0, "Truck")], [observation(101, 7, "Car")], []):
            self.recording = Recording()
            index = self.index(rows=prefix + future)
            for frame in range(101, 121):
                del self.recording.frames[frame]
            self.recording.requested.clear()
            self.assertEqual(rank_evidence(question(), index), expected)
            for name in ("metadata_sha256", "model_inputs_sha256", "candidate_ids", "documents", "scores"):
                self.assertEqual(index.last_receipt[name], expected_index.last_receipt[name])
            self.assertTrue(all(frame <= 100 for frame in self.recording.requested))

    def test_annotated_predicted_use_identical_code_with_own_identities(self):
        rows = [dict(row, source="predicted") for row in self.rows]
        index = build_index(self.recording, rows, source="predicted", scorer=ObservedClassScorer())
        result = rank_evidence(question(), index)
        self.assertTrue(all(row["source"] == "predicted" for row in result))
        with self.assertRaises(ValueError):
            build_index(self.recording, rows, source="annotated")

    def test_source_must_be_explicit_even_for_empty_indexes(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "SOURCE"):
                build_index(self.recording, [])
        with patch.dict(os.environ, {"TRACKRAG_QWEN_SOURCE": "predicted"}):
            index = build_index(self.recording, [], scorer=ObservedClassScorer())
            self.assertEqual(index.source, "predicted")

    def test_empty_and_future_only_rows_generate_identical_scene_inputs(self):
        empty = self.index(rows=[])
        future = self.index(rows=[observation(110)])
        self.assertEqual(rank_evidence(question(), empty), rank_evidence(question(), future))
        self.assertEqual(empty.last_receipt["model_inputs_sha256"], future.last_receipt["model_inputs_sha256"])

    def test_missing_scene_and_no_tracks_do_not_load_or_call_a_model(self):
        index = self.index(rows=[])
        del self.recording.frames[100]
        self.assertEqual(rank_evidence(question(), index), [])
        self.assertEqual(index.scorer.calls, [])

    def test_invalid_k_question_or_packet_cannot_reach_the_scorer(self):
        index = self.index()
        for k in (-1, 4, True, "3"):
            with self.subTest(k=k), self.assertRaises(ValueError):
                rank_evidence(question(), index, k=k)
        with self.assertRaises(ValueError):
            rank_evidence(dict(question(), gold_answer="bus"), index)
        batch = build_candidate_batch(question(), index.tracks, source="annotated")
        batch["candidates"][0]["end_frame"] = 101
        with self.assertRaises(ValueError):
            rank_candidate_batch(batch, index.scorer)
        self.assertEqual(index.scorer.calls, [])

    def test_zero_k_and_index_build_are_lazy(self):
        with patch("trackrag.retrieval_qwen.QwenTextScorer._load", side_effect=AssertionError("No loading")):
            index = build_index(self.recording, self.rows, source="annotated")
            self.assertEqual(rank_evidence(question(), index, k=0), [])

    def test_failed_question_does_not_leave_a_previous_success_receipt(self):
        index = self.index()
        rank_evidence(question(), index)
        self.assertIsNotNone(index.last_receipt)
        with self.assertRaises(ValueError):
            rank_evidence(dict(question(), gold_answer="bus"), index)
        self.assertIsNone(index.last_receipt)

    def test_bad_score_count_values_and_failure_never_fall_back(self):
        for kind in ("short", "long", "nan", "bool", "string", "failure"):
            class BadScorer:
                def score(self, query, documents):
                    if kind == "failure":
                        raise RuntimeError("model failed")
                    if kind == "short":
                        return []
                    if kind == "long":
                        return [0.5] * (len(documents) + 1)
                    return [{"nan": float("nan"), "bool": True, "string": "0.5"}[kind]] * len(documents)

            index = self.index(scorer=BadScorer())
            with self.subTest(kind=kind), self.assertRaises((ValueError, RuntimeError)):
                rank_evidence(question(), index)
            self.assertEqual(index.last_receipt["status"], "error")
            self.assertNotIn("intervals", index.last_receipt)

    def test_ties_are_deterministic_and_intervals_unique(self):
        scorer = SimpleNamespace(score=lambda q, docs: [0.5] * len(docs))
        index = self.index(scorer=scorer)
        result = rank_evidence(question(), index)
        reverse = self.index(rows=list(reversed(self.rows)), scorer=scorer)
        self.assertEqual(rank_evidence(question(), reverse), result)
        self.assertEqual(len({(row["track_id"], row["start_frame"], row["end_frame"]) for row in result}), len(result))

    def test_baseline_and_public_inputs_remain_unchanged(self):
        query = question()
        rows = deepcopy(self.rows)
        baseline = baseline_index(self.recording, rows)
        before = baseline_rank(query, baseline)
        index = self.index()
        original = deepcopy(index.tracks.observations)
        rank_evidence(query, index)
        self.assertEqual(query, question())
        self.assertEqual(index.tracks.observations, original)
        self.assertEqual(baseline_rank(query, baseline), before)

    def test_model_text_excludes_paths_ids_reviewer_notes_and_unseen_pixels(self):
        index = self.index()
        batch = build_candidate_batch(question(), index.tracks, source="annotated")
        docs = text_documents(batch)
        for doc in docs:
            self.assertNotIn("not-opened", doc)
            self.assertNotIn("candidate_id", doc)
            self.assertNotIn("history_id", doc)
            self.assertNotIn("test-recording", doc)
            self.assertIn("No image pixels", doc)
        relocated = deepcopy(batch)
        for candidate in relocated["candidates"]:
            for image in candidate["images"]:
                image["image_path"] = "another-location.jpg"
        self.assertEqual(text_documents(relocated), docs)
        self.assertNotIn("test-q", _query_text(question()))

    def test_compact_model_document_preserves_missing_observation_ranges(self):
        index = self.index(rows=[observation(frame) for frame in (90, 91, 93, 100)])
        batch = build_candidate_batch(question(), index.tracks, source="annotated")
        doc = json.loads(text_documents(batch)[0])
        facts = doc["observed_track_facts"]
        self.assertEqual(facts["observed_frame_ranges"], [[90, 91], [93, 93], [100, 100]])
        self.assertEqual(facts["missing_observation_ranges"], [[92, 92], [94, 99]])
        self.assertEqual(facts["observation_count"], 4)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Optional torch dependency is not installed")
class QwenScoringMathTests(unittest.TestCase):
    def scorer(self):
        import torch

        class Tokenizer:
            def __call__(self, prompts, **kwargs):
                return {"input_ids": [[3] * (10 + index) for index in range(len(prompts))]}

            def pad(self, batch, **kwargs):
                width = max(len(row) for row in batch["input_ids"])
                return {"input_ids": torch.tensor([[0] * (width-len(row)) + row for row in batch["input_ids"]]),
                        "attention_mask": torch.tensor([[0] * (width-len(row)) + [1]*len(row) for row in batch["input_ids"]])}

        class Model:
            def __call__(self, **inputs):
                assert inputs["logits_to_keep"] == 1 and inputs["use_cache"] is False
                count = inputs["input_ids"].shape[0]
                return SimpleNamespace(logits=torch.tensor([[[9.0, 1.0, 3.0]]] * count))

        scorer = QwenTextScorer(QwenConfig())
        scorer.torch, scorer.tokenizer, scorer.model = torch, Tokenizer(), Model()
        scorer.prefix, scorer.suffix, scorer.no_id, scorer.yes_id = [4], [5], 1, 2
        return scorer

    def test_yes_no_score_uses_only_final_binary_logits(self):
        scorer = self.scorer()
        scores = scorer.score(question(), ["doc-a", "doc-b", "doc-c"])
        self.assertEqual(len(scores), 3)
        for score in scores:
            self.assertAlmostEqual(score, 1 / (1 + math.exp(-2)), places=6)
        self.assertEqual(scorer.last_token_counts, [12, 13, 14])

    def test_no_silent_truncation_on_context_overflow(self):
        scorer = self.scorer()
        scorer.config = QwenConfig(max_tokens=128)
        scorer.prefix = [4] * 128
        with self.assertRaisesRegex(ValueError, "exceeds"):
            scorer.score(question(), ["doc"])


if __name__ == "__main__":
    unittest.main()
