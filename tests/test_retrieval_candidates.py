"""Candidate-pool checks using synthetic tracks and frame metadata, without models."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import unittest

from trackrag.contracts import Frame, RankedInterval, TrackObservation
from trackrag.retrieval import RetrievalConfig, build_index, rank_evidence
from trackrag.retrieval_candidates import build_candidate_batch, candidate_to_interval


class CandidateRecording:
    """The batch may look up named frames, but must never scan the recording."""

    def __init__(self, frame_ids=range(121), recording_id="recording-a"):
        self.recording_id = recording_id
        self.requested = []
        self.frames = {
            frame_id: {
                "recording_id": recording_id,
                "frame_id": frame_id,
                "timestamp_s": frame_id / 10,
                "width": 100,
                "height": 80,
                "image_path": f"not-on-disk/{recording_id}/{frame_id}.jpg",
            }
            for frame_id in frame_ids
        }

    def get_frame(self, frame_id):
        self.requested.append(frame_id)
        return self.frames[frame_id]

    def iter_frames(self):
        raise AssertionError("Candidate construction must not scan future frames")


def observed(frame_id, track_id=7, **overrides):
    row = {
        "recording_id": "recording-a", "frame_id": frame_id,
        "track_id": track_id, "source": "annotated", "class_name": "Bus",
        "bbox_xyxy": (10, 20, 30, 40),
    }
    row.update(overrides)
    return row


def public_question(query_frame=100, **overrides):
    question = {
        "question_id": "q-candidates", "recording_id": "recording-a",
        "query_frame": query_frame, "question": "Which bus entered the scene?",
    }
    question.update(overrides)
    return question


class CandidateBatchTests(unittest.TestCase):
    def make_index(self, tracks, *, recording=None, config=None):
        recording = recording or CandidateRecording()
        index = build_index(recording, tracks, config=config)
        recording.requested.clear()
        return index

    def test_pool_keeps_wrong_vehicles_and_wrong_times_for_later_ranking(self):
        tracks = [observed(frame) for frame in range(90, 111)]
        tracks += [observed(frame, 42, class_name="Car") for frame in range(90, 111)]
        index = self.make_index(
            tracks, config=RetrievalConfig(window_frames=5, stride_frames=5),
        )
        batch = build_candidate_batch(public_question(), index, source="annotated")
        histories = {row["track_id"]: row for row in batch["histories"]}
        self.assertEqual(set(histories), {7, 42})
        tracked = [row for row in batch["candidates"] if row["track_id"] is not None]
        self.assertEqual({row["track_id"] for row in tracked}, {7, 42})
        for track_id in (7, 42):
            candidates = [row for row in tracked if row["track_id"] == track_id]
            self.assertEqual(
                {(row["kind"], row["start_frame"], row["end_frame"]) for row in candidates},
                {("track_history", 90, 100), ("track_window", 90, 94),
                 ("track_window", 95, 99), ("track_window", 100, 100)},
            )
            self.assertTrue(all(row["history_id"] == histories[track_id]["history_id"]
                                for row in candidates))
        self.assertTrue(all(row["end_frame"] <= 100 for row in batch["candidates"]))

    def test_whole_history_and_identical_local_window_are_deduplicated(self):
        index = self.make_index([observed(90), observed(91)])
        batch = build_candidate_batch(public_question(), index, source="annotated")
        tracked = [row for row in batch["candidates"] if row["track_id"] is not None]
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0]["kind"], "track_history")
        ids = [row["candidate_id"] for row in batch["candidates"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_features_preserve_class_changes_positions_and_explicit_gaps(self):
        index = self.make_index([
            observed(10), observed(12, bbox_xyxy=(20, 20, 40, 40)),
            observed(20, class_name="Truck", bbox_xyxy=(30, 40, 50, 60)),
        ], config=RetrievalConfig(max_gap_frames=2))
        batch = build_candidate_batch(public_question(20), index, source="annotated")
        features = batch["histories"][0]["features"]
        self.assertEqual(features["observed_frame_ids"], [10, 12, 20])
        self.assertEqual(features["observation_count"], 3)
        self.assertEqual(features["class_counts"], {"Bus": 2, "Truck": 1})
        self.assertEqual(tuple(features["first_box_xyxy"]), (10, 20, 30, 40))
        self.assertEqual(tuple(features["last_box_xyxy"]), (30, 40, 50, 60))
        self.assertEqual(tuple(features["first_center_normalized"]), (0.2, 0.375))
        self.assertEqual(tuple(features["last_center_normalized"]), (0.4, 0.625))
        self.assertAlmostEqual(features["center_displacement_normalized"][0], 0.2)
        self.assertAlmostEqual(features["center_displacement_normalized"][1], 0.25)
        self.assertEqual([tuple(pair) for pair in features["missing_observation_ranges"]],
                         [(11, 11), (13, 19)])
        self.assertEqual([tuple(pair) for pair in features["continuity_breaks"]], [(13, 19)])
        self.assertAlmostEqual(features["center_range_normalized"][0], 0.2)
        self.assertAlmostEqual(features["center_range_normalized"][1], 0.25)
        # Every saved observation retains the shared six-field contract.
        self.assertTrue(all(set(row) == TrackObservation.__required_keys__
                            for row in batch["histories"][0]["observations"]))

    def test_single_observation_does_not_claim_measured_displacement(self):
        index = self.make_index([observed(100)])
        batch = build_candidate_batch(public_question(), index, source="annotated")
        features = batch["histories"][0]["features"]
        self.assertIsNone(features["center_displacement_normalized"])
        self.assertEqual(features["missing_observation_ranges"], [])
        self.assertEqual(features["continuity_breaks"], [])

    def test_images_include_reviewer_samples_and_explicit_legal_frames(self):
        index = self.make_index([observed(frame) for frame in range(90, 101)],
                                config=RetrievalConfig(window_frames=30, stride_frames=30))
        question = public_question(question="Compare the bus in frame 93 with frame 100.")
        batch = build_candidate_batch(question, index, source="annotated")
        history = next(row for row in batch["candidates"] if row["kind"] == "track_history")
        self.assertEqual([image["frame_id"] for image in history["images"]], [90, 93, 95, 100])
        scene = [row for row in batch["candidates"] if row["track_id"] is None]
        self.assertEqual({row["start_frame"] for row in scene}, {90, 93, 100})
        for candidate in batch["candidates"]:
            for image in candidate["images"]:
                self.assertEqual(set(image), Frame.__required_keys__ | {"bbox_xyxy"})
                self.assertEqual(image["recording_id"], "recording-a")
                if candidate["track_id"] is None:
                    self.assertEqual(candidate["kind"], "scene")
                    self.assertEqual(len(candidate["images"]), 1)
                    self.assertIsNone(image["bbox_xyxy"])
                    self.assertEqual(candidate["start_frame"], candidate["end_frame"])
                    self.assertIsNone(candidate["history_id"])
                    self.assertIsNone(candidate["features"])
                else:
                    self.assertEqual(tuple(image["bbox_xyxy"]), (10, 20, 30, 40))
        self.assertTrue(all(count == 1 for count in Counter(index.recording.requested).values()))

    def test_even_observation_count_uses_lower_middle_like_reviewer(self):
        index = self.make_index([observed(frame) for frame in (90, 91, 92, 93)])
        batch = build_candidate_batch(public_question(), index, source="annotated")
        history = next(row for row in batch["candidates"] if row["kind"] == "track_history")
        self.assertEqual([image["frame_id"] for image in history["images"]], [90, 91, 93])

    def test_missing_scene_frames_are_reported_without_substituting_nearby_frames(self):
        recording = CandidateRecording((90, 95, 100))
        index = self.make_index([observed(frame) for frame in (90, 95, 100)], recording=recording)
        question = public_question(105, question="Compare frame 97 with frame 105.")
        batch = build_candidate_batch(question, index, source="annotated")
        self.assertEqual(batch["unavailable_scene_frames"], [97, 105])
        scenes = [row for row in batch["candidates"] if row["track_id"] is None]
        self.assertEqual([row["start_frame"] for row in scenes], [90])
        self.assertEqual(batch["excluded_future_frame_requests"], [])

    def test_explicit_future_frames_are_reported_and_never_requested(self):
        index = self.make_index([observed(90), observed(100), observed(110)])
        question = public_question(question="Compare frame 90, frame 110 and frame 150.")
        batch = build_candidate_batch(question, index, source="annotated")
        self.assertEqual(batch["excluded_future_frame_requests"], [110, 150])
        self.assertTrue(all(frame <= 100 for frame in index.recording.requested))
        self.assertTrue(all(row["end_frame"] <= 100 for row in batch["candidates"]))

    def test_future_changes_leave_entire_batch_including_metadata_hash_unchanged(self):
        prefix = [observed(frame, bbox_xyxy=(frame - 90, 20, frame - 70, 40))
                  for frame in range(90, 101)]
        prefix += [observed(95, 42, class_name="Car")]
        question = public_question(question="Which bus appears in frame 95 and frame 110?")
        baseline = build_candidate_batch(question, self.make_index(prefix), source="annotated")
        for future in (
            [observed(frame) for frame in range(101, 121)],
            [observed(105, class_name="Truck", bbox_xyxy=(70, 40, 95, 70))],
            [observed(110, 0, class_name="Bicycle")],
        ):
            with self.subTest(future=future):
                index = self.make_index(prefix + future)
                # Future metadata can disappear after the index is constructed.
                for frame in range(101, 121):
                    del index.recording.frames[frame]
                actual = build_candidate_batch(question, index, source="annotated")
                self.assertEqual(actual, baseline)
                self.assertTrue(all(frame <= 100 for frame in index.recording.requested))

    def test_input_order_and_repeated_calls_are_deterministic(self):
        tracks = [observed(90), observed(95), observed(95, 42, class_name="Car"), observed(100)]
        index = self.make_index(tracks)
        baseline = build_candidate_batch(public_question(), index, source="annotated")
        self.assertEqual(build_candidate_batch(public_question(), index, source="annotated"), baseline)
        reversed_index = self.make_index(list(reversed(tracks)))
        self.assertEqual(build_candidate_batch(public_question(), reversed_index, source="annotated"),
                         baseline)

    def test_caller_mutation_cannot_change_index_question_or_recording(self):
        question = public_question()
        index = self.make_index([observed(90), observed(100)])
        observations_before = deepcopy(index.observations)
        frames_before = deepcopy(index.recording.frames)
        question_before = deepcopy(question)
        batch = build_candidate_batch(question, index, source="annotated")
        baseline = deepcopy(batch)
        batch["question"]["question"] = "Changed question"
        batch["histories"][0]["observations"][0]["class_name"] = "Changed class"
        batch["histories"][0]["features"]["class_counts"]["Bus"] = 999
        batch["candidates"][0]["images"][0]["image_path"] = "changed.jpg"
        self.assertEqual(question, question_before)
        self.assertEqual(index.observations, observations_before)
        self.assertEqual(index.recording.frames, frames_before)
        self.assertEqual(build_candidate_batch(question, index, source="annotated"), baseline)

    def test_metadata_hash_is_portable_across_image_path_relocation(self):
        index = self.make_index([observed(90), observed(100)])
        original = build_candidate_batch(public_question(), index, source="annotated")
        for frame in index.recording.frames.values():
            frame["image_path"] = "relocated/" + frame["image_path"]
        relocated = build_candidate_batch(public_question(), index, source="annotated")
        self.assertNotEqual(original["candidates"], relocated["candidates"])
        self.assertEqual(original["metadata_sha256"], relocated["metadata_sha256"])
        self.assertEqual([row["candidate_id"] for row in original["candidates"]],
                         [row["candidate_id"] for row in relocated["candidates"]])

    def test_finite_json_batch_schema_and_canonical_metadata_digest(self):
        index = self.make_index([observed(90), observed(100)])
        batch = build_candidate_batch(public_question(), index, source="annotated")
        self.assertEqual(set(batch), {
            "schema_version", "question", "source", "policy", "histories", "candidates",
            "unavailable_scene_frames", "excluded_future_frame_requests", "metadata_sha256",
        })
        self.assertEqual(batch["schema_version"], "trackrag.candidates.v1")
        self.assertIsInstance(batch["policy"], dict)
        self.assertIsInstance(json.loads(json.dumps(batch, allow_nan=False)), dict)

        portable = deepcopy(batch)
        del portable["metadata_sha256"]
        for candidate in portable["candidates"]:
            for image in candidate["images"]:
                del image["image_path"]
        canonical = json.dumps(portable, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False)
        self.assertEqual(batch["metadata_sha256"], hashlib.sha256(canonical.encode("utf-8")).hexdigest())

    def test_identity_is_scoped_to_recording_and_source(self):
        batches = []
        for recording_id, source in (("recording-a", "annotated"),
                                     ("recording-b", "annotated"),
                                     ("recording-a", "predicted")):
            index = self.make_index(
                [observed(90, recording_id=recording_id, source=source)],
                recording=CandidateRecording(recording_id=recording_id),
            )
            batch = build_candidate_batch(public_question(recording_id=recording_id), index,
                                          source=source)
            self.assertTrue(all(row["recording_id"] == recording_id and row["source"] == source
                                for row in batch["candidates"]))
            batches.append(batch)
        history_ids = [batch["histories"][0]["history_id"] for batch in batches]
        self.assertEqual(len(set(history_ids)), 3)
        candidate_ids = [next(row["candidate_id"] for row in batch["candidates"]
                              if row["kind"] == "track_history") for batch in batches]
        self.assertEqual(len(set(candidate_ids)), 3)

    def test_source_is_explicit_and_must_match_eligible_observations(self):
        index = self.make_index([observed(90)])
        with self.assertRaises(TypeError):
            build_candidate_batch(public_question(), index)
        for source in (None, "emt_ground_truth", True, "mixed", "predicted"):
            with self.subTest(source=source), self.assertRaises((TypeError, ValueError)):
                build_candidate_batch(public_question(), index, source=source)

    def test_empty_and_future_only_tracks_use_explicit_source_for_scene_evidence(self):
        for rows in ([], [observed(110, source="annotated")]):
            with self.subTest(rows=rows):
                index = self.make_index(rows)
                batch = build_candidate_batch(public_question(), index, source="predicted")
                self.assertEqual(batch["source"], "predicted")
                self.assertEqual(batch["histories"], [])
                self.assertEqual(len(batch["candidates"]), 1)
                scene = batch["candidates"][0]
                self.assertIsNone(scene["track_id"])
                self.assertEqual(scene["source"], "predicted")
                self.assertEqual((scene["start_frame"], scene["end_frame"]), (100, 100))
                self.assertEqual(index.recording.requested, [100])

    def test_no_observations_and_missing_query_frame_produce_empty_pool(self):
        index = self.make_index([], recording=CandidateRecording(()))
        batch = build_candidate_batch(public_question(), index, source="annotated")
        self.assertEqual(batch["histories"], [])
        self.assertEqual(batch["candidates"], [])
        self.assertEqual(batch["unavailable_scene_frames"], [100])

    def test_nonfinite_frame_timestamp_cannot_enter_json_candidate_payload(self):
        for timestamp in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(timestamp=timestamp):
                index = self.make_index([])
                index.recording.frames[100]["timestamp_s"] = timestamp
                with self.assertRaises(ValueError):
                    build_candidate_batch(public_question(), index, source="annotated")

    def test_missing_observed_frame_is_an_error_not_an_omitted_scene(self):
        index = self.make_index([observed(90), observed(100)])
        del index.recording.frames[90]
        with self.assertRaisesRegex(ValueError, "Observed frame 90"):
            build_candidate_batch(public_question(), index, source="annotated")

    def test_invalid_scene_metadata_is_rejected_before_export(self):
        for field, value in (
            ("recording_id", "another-recording"), ("frame_id", 101),
            ("width", 0), ("height", True), ("timestamp_s", -1),
            ("timestamp_s", True), ("image_path", ""), ("image_path", None),
        ):
            with self.subTest(field=field, value=value):
                index = self.make_index([])
                index.recording.frames[100][field] = value
                with self.assertRaises(ValueError):
                    build_candidate_batch(public_question(), index, source="annotated")

    def test_question_validation_rejects_gold_fields_and_wrong_recording(self):
        index = self.make_index([])
        for question in (public_question(gold_answer="Bus"),
                         public_question(recording_id="recording-b"),
                         public_question(query_frame=True),
                         public_question(question="")):
            with self.subTest(question=question), self.assertRaises(ValueError):
                build_candidate_batch(question, index, source="annotated")
        self.assertEqual(index.recording.requested, [])

    def test_candidate_conversion_preserves_exact_shared_output_fields(self):
        index = self.make_index([observed(90), observed(100)])
        batch = build_candidate_batch(public_question(), index, source="annotated")
        tracked = next(row for row in batch["candidates"] if row["track_id"] is not None)
        scene = next(row for row in batch["candidates"] if row["track_id"] is None)
        for rank, candidate in enumerate((tracked, scene), start=1):
            with self.subTest(kind=candidate["kind"]):
                interval = candidate_to_interval(candidate, rank=rank, score=0.75)
                self.assertEqual(set(interval), RankedInterval.__required_keys__)
                self.assertEqual(interval, {
                    "recording_id": candidate["recording_id"], "source": candidate["source"],
                    "track_id": candidate["track_id"], "start_frame": candidate["start_frame"],
                    "end_frame": candidate["end_frame"], "rank": rank, "score": 0.75,
                })

    def test_constructing_candidates_leaves_baseline_rankings_unchanged(self):
        index = self.make_index([observed(90), observed(95), observed(100),
                                 observed(90, 42, class_name="Car")])
        question = public_question()
        before = rank_evidence(question, index)
        build_candidate_batch(question, index, source="annotated")
        self.assertEqual(rank_evidence(question, index), before)

    def test_projection_requires_valid_rank_and_finite_score(self):
        batch = build_candidate_batch(public_question(), self.make_index([]), source="annotated")
        candidate = batch["candidates"][0]
        for rank in (0, 4, True, 1.0):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                candidate_to_interval(candidate, rank=rank, score=0.5)
        for score in (True, "0.5", float("nan"), float("inf"), 10**1000):
            with self.subTest(score=score), self.assertRaises(ValueError):
                candidate_to_interval(candidate, rank=1, score=score)
        self.assertEqual(candidate_to_interval(candidate, rank=1, score=-2)["score"], -2.0)


if __name__ == "__main__":
    unittest.main()
