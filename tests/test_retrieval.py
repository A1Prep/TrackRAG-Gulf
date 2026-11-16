"""Checks for Task 4's configuration, causal windows and ranked text retrieval."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from inspect import signature
import json
import unittest

from trackrag.retrieval import RetrievalConfig, build_index, candidate_windows, rank_evidence


class RetrievalConfigTests(unittest.TestCase):
    def test_valid_overrides_allow_short_windows_and_disabled_features(self):
        config = RetrievalConfig(
            window_frames=1, stride_frames=1, max_gap_frames=0,
            class_weight=1, location_weight=0, movement_weight=0,
        )
        self.assertEqual(config.window_frames, 1)
        self.assertEqual(config.max_gap_frames, 0)
        with self.assertRaises(FrozenInstanceError):
            setattr(config, "window_frames", 20)

    def test_invalid_temporal_settings_are_rejected(self):
        for name, values in (
            ("window_frames", (0, -1, 1.5, True, "30", None)),
            ("stride_frames", (0, -1, 1.5, True, "10", None)),
            ("max_gap_frames", (-1, 1.5, True, "5", None)),
        ):
            for value in values:
                with self.subTest(field=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        RetrievalConfig(**{name: value})

    def test_stride_cannot_leave_unsearched_frame_ranges(self):
        with self.assertRaisesRegex(ValueError, "stride_frames"):
            RetrievalConfig(window_frames=10, stride_frames=11)

    def test_invalid_feature_weights_are_rejected(self):
        for name in ("class_weight", "location_weight", "movement_weight"):
            for value in (-1, float("nan"), float("inf"), -float("inf"),
                          True, "1", None, 10**1000):
                with self.subTest(field=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        RetrievalConfig(**{name: value})

    def test_weight_sum_must_allow_a_finite_nonzero_normalizer(self):
        for weights in ((0, 0, 0), (1e308, 1e308, 0), (10**308, 10**308, 0)):
            with self.subTest(weights=weights):
                with self.assertRaisesRegex(ValueError, "positive sum"):
                    RetrievalConfig(
                        class_weight=weights[0], location_weight=weights[1],
                        movement_weight=weights[2],
                    )


class RetrievalInterfaceTests(unittest.TestCase):
    def test_task5_calls_bind_without_additional_arguments(self):
        # Match the downstream adapter checker without needing a real recording.
        build_call = signature(build_index).bind(object(), [])
        build_call.apply_defaults()
        self.assertIsNone(build_call.arguments["config"])
        rank_call = signature(rank_evidence).bind({}, object())
        rank_call.apply_defaults()
        self.assertEqual(rank_call.arguments["k"], 3)
        signature(rank_evidence).bind({}, object(), k=3)

    def test_configuration_override_is_keyword_only(self):
        config = RetrievalConfig()
        signature(build_index).bind(object(), iter(()), config=config)
        with self.assertRaises(TypeError):
            signature(build_index).bind(object(), [], config)


class RecordingFixture:
    """Frame metadata only; building the index must not open image files."""

    recording_id = "recording-a"

    def __init__(self, frame_ids=range(13)):
        self.requested = []
        self.frames = {
            frame_id: {
                "recording_id": self.recording_id, "frame_id": frame_id,
                "timestamp_s": frame_id / 10, "width": 100, "height": 80,
                "image_path": f"not-on-disk/{frame_id}.jpg",
            }
            for frame_id in frame_ids
        }

    def iter_frames(self):
        raise AssertionError("The builder should look up only referenced frames")

    def get_frame(self, frame_id):
        self.requested.append(frame_id)
        return self.frames[frame_id]


def observation(frame_id=1, track_id=7, **overrides):
    row = {
        "recording_id": "recording-a", "frame_id": frame_id, "track_id": track_id,
        "class_name": "Bus", "bbox_xyxy": (10, 20, 30, 40), "source": "annotated",
    }
    row.update(overrides)
    return row


def question(query_frame=100, **overrides):
    row = {
        "question_id": "q1", "recording_id": "recording-a",
        "query_frame": query_frame, "question": "Which bus entered the scene?",
    }
    row.update(overrides)
    return row


class BuildIndexTests(unittest.TestCase):
    def setUp(self):
        self.recording = RecordingFixture()

    def test_unsorted_bus_and_car_tracks_have_separate_ordered_histories(self):
        tracks = [
            observation(3), observation(2, 42, class_name="Car"),
            observation(1, 42, class_name="Car"), observation(1),
        ]
        index = build_index(self.recording, tracks)
        self.assertEqual(list(index.histories), [
            ("recording-a", "annotated", 7), ("recording-a", "annotated", 42),
        ])
        bus = index.histories[("recording-a", "annotated", 7)]
        car = index.histories[("recording-a", "annotated", 42)]
        self.assertEqual([row["frame_id"] for row in bus], [1, 3])
        self.assertEqual([row["frame_id"] for row in car], [1, 2])
        self.assertEqual([row["class_name"] for row in car], ["Car", "Car"])
        self.assertEqual(
            [(row["frame_id"], row["track_id"]) for row in index.observations],
            [(1, 7), (1, 42), (2, 42), (3, 7)],
        )
        self.assertEqual(index.source, "annotated")
        self.assertIs(index.recording, self.recording)
        self.assertEqual(index.config, RetrievalConfig())
        self.assertCountEqual(self.recording.requested, [1, 2, 3])
        with self.assertRaises(TypeError):
            index.histories[("recording-a", "annotated", 99)] = ()

    def test_input_order_does_not_change_the_index(self):
        tracks = [observation(3), observation(1), observation(2, 42)]
        forward = build_index(self.recording, tracks)
        backward = build_index(self.recording, reversed(tracks))
        self.assertEqual(forward.observations, backward.observations)
        self.assertEqual(forward.histories, backward.histories)

    def test_generator_reusing_a_mutable_row_is_snapshotted_at_each_yield(self):
        row = observation(bbox_xyxy=[10, 20, 30, 40])

        def tracks():
            yield row
            row["frame_id"] = 3
            row["bbox_xyxy"][0] = 15
            yield row
            row["frame_id"] = 999
            row["bbox_xyxy"][0] = -100

        index = build_index(self.recording, tracks())
        self.assertEqual([item["frame_id"] for item in index.observations], [1, 3])
        self.assertEqual([item["bbox_xyxy"][0] for item in index.observations], [10, 15])

    def test_caller_data_and_index_do_not_share_observation_dicts_or_box_lists(self):
        tracks = [observation(3, bbox_xyxy=[10, 20, 30, 40]), observation(1)]
        original = deepcopy(tracks)
        index = build_index(self.recording, tracks)
        self.assertEqual(tracks, original)
        self.assertIsNot(index.observations[1], tracks[0])
        self.assertIsInstance(index.observations[1]["bbox_xyxy"], tuple)
        tracks[0]["class_name"] = "Truck"
        tracks[0]["bbox_xyxy"][0] = -100
        tracks.clear()
        self.assertEqual(index.observations[1]["class_name"], "Bus")
        self.assertEqual(index.observations[1]["bbox_xyxy"], (10, 20, 30, 40))
        self.assertEqual(index.windows[0].observations[-1]["class_name"], "Bus")
        self.assertEqual(index.windows[0].observations[-1]["bbox_xyxy"], (10, 20, 30, 40))

    def test_annotated_and_predicted_tracks_use_the_same_builder(self):
        config = RetrievalConfig(window_frames=10, stride_frames=5)
        for source in ("annotated", "predicted"):
            with self.subTest(source=source):
                index = build_index(
                    self.recording, [observation(source=source)], config=config,
                )
                self.assertEqual(index.source, source)
                self.assertEqual(list(index.histories), [("recording-a", source, 7)])
                self.assertIs(index.config, config)

    def test_empty_inputs_return_an_empty_index_without_looking_up_frames(self):
        for tracks in ([], iter(())):
            index = build_index(self.recording, tracks)
            self.assertEqual(index.observations, ())
            self.assertEqual(dict(index.histories), {})
            self.assertIsNone(index.source)
            self.assertEqual(index.config, RetrievalConfig())
        self.assertEqual(self.recording.requested, [])

    def test_duplicate_track_frame_is_rejected_even_if_boxes_agree(self):
        for second in (observation(), observation(bbox_xyxy=(20, 20, 40, 40))):
            with self.subTest(second=second):
                with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
                    build_index(self.recording, [observation(), second])

    def test_same_track_id_cannot_introduce_another_recording_or_source(self):
        for change, message in (
            ({"recording_id": "recording-b"}, "recording"),
            ({"source": "predicted"}, "source"),
        ):
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, message):
                    build_index(self.recording, [observation(), observation(2, **change)])

    def test_every_shared_observation_field_is_required(self):
        for field in observation():
            row = observation()
            del row[field]
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    build_index(self.recording, [row])

    def test_invalid_scalar_fields_are_rejected(self):
        cases = {
            "recording_id": (None, "", " ", 7),
            "frame_id": (-1, 1.5, True, "1", None),
            "track_id": (-1, 1.5, True, "7", None),
            "class_name": ("", " ", None, 7),
            "source": ("emt_ground_truth", "unknown", None, True),
        }
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        build_index(self.recording, [observation(**{field: value})])

    def test_invalid_boxes_are_rejected_instead_of_clipped(self):
        boxes = (
            None, "1234", {0: 10, 1: 20, 2: 30, 3: 40},
            (10, 20, 30), (10, 20, 30, 40, 50),
            (True, 20, 30, 40), ("10", 20, 30, 40),
            (float("nan"), 20, 30, 40), (10, 20, float("inf"), 40),
            (10, 20, 10**1000, 40), (-1, 20, 30, 40), (10, -1, 30, 40),
            (30, 20, 10, 40), (10, 40, 30, 20), (10, 20, 10, 40),
            (10, 20, 30, 20), (10, 20, 101, 40), (10, 20, 30, 81),
        )
        for box in boxes:
            with self.subTest(box=box):
                with self.assertRaisesRegex(ValueError, "bbox_xyxy"):
                    build_index(self.recording, [observation(bbox_xyxy=box)])

    def test_zero_ids_and_boxes_on_image_edges_are_valid(self):
        index = build_index(self.recording, [
            observation(frame_id=0, track_id=0, bbox_xyxy=(0, 0, 100, 80)),
        ])
        self.assertEqual(index.observations[0]["bbox_xyxy"], (0, 0, 100, 80))
        self.assertIn(("recording-a", "annotated", 0), index.histories)

    def test_missing_frame_or_incorrect_frame_lookup_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "frame.*99"):
            build_index(self.recording, [observation(99)])
        for changes in (
            {"recording_id": "recording-b"}, {"frame_id": 2}, {"frame_id": True},
        ):
            with self.subTest(changes=changes):
                recording = RecordingFixture()
                recording.frames[1].update(changes)
                with self.assertRaisesRegex(ValueError, "frame|recording"):
                    build_index(recording, [observation()])

    def test_frame_dimensions_must_be_positive_integers(self):
        for dimension in ("width", "height"):
            for value in (0, -1, True, 100.5, "100", None):
                with self.subTest(dimension=dimension, value=value):
                    recording = RecordingFixture()
                    recording.frames[1][dimension] = value
                    with self.assertRaisesRegex(ValueError, dimension):
                        build_index(recording, [observation()])

    def test_per_frame_class_changes_are_preserved_without_global_voting(self):
        index = build_index(self.recording, [
            observation(1, class_name="Bus"), observation(3, class_name="Truck"),
        ])
        self.assertEqual([row["class_name"] for row in index.observations], ["Bus", "Truck"])

    def test_non_record_inputs_and_invalid_configuration_fail_clearly(self):
        for row in (None, [], "not an observation"):
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "observation"):
                    build_index(self.recording, [row])
        with self.assertRaisesRegex(TypeError, "config"):
            build_index(self.recording, [], config={})
        with self.assertRaisesRegex(TypeError, "tracks"):
            build_index(self.recording, None)
        self.recording.recording_id = " "
        with self.assertRaisesRegex(ValueError, "recording_id"):
            build_index(self.recording, [])


class TrackWindowTests(unittest.TestCase):
    def make_index(self, frame_ids, **settings):
        frame_ids = tuple(frame_ids)
        return build_index(
            RecordingFixture(frame_ids),
            (observation(frame_id) for frame_id in reversed(frame_ids)),
            config=RetrievalConfig(**settings),
        )

    def bounds(self, index):
        return [(window.start_frame, window.end_frame) for window in index.windows]

    def test_default_windows_overlap_and_keep_partial_tails(self):
        index = self.make_index(range(1, 46))
        self.assertEqual(self.bounds(index), [
            (1, 30), (11, 40), (21, 45), (31, 45), (41, 45),
        ])
        for window, expected in zip(index.windows, (
            range(1, 31), range(11, 41), range(21, 46), range(31, 46), range(41, 46),
        )):
            self.assertEqual(
                [row["frame_id"] for row in window.observations], list(expected),
            )

    def test_custom_length_and_stride_use_inclusive_bounds(self):
        index = self.make_index(range(1, 13), window_frames=5, stride_frames=3)
        self.assertEqual(self.bounds(index), [(1, 5), (4, 8), (7, 11), (10, 12)])
        self.assertEqual(
            [row["frame_id"] for row in index.windows[1].observations], [4, 5, 6, 7, 8],
        )

    def test_gap_threshold_keeps_small_gaps_and_reanchors_after_large_ones(self):
        index = self.make_index(
            [10, 13, 17, 18, 21], window_frames=6, stride_frames=3, max_gap_frames=2,
        )
        self.assertEqual(self.bounds(index), [(10, 13), (13, 13), (17, 21), (21, 21)])
        self.assertEqual(
            [row["frame_id"] for row in index.windows[0].observations], [10, 13],
        )
        self.assertEqual(
            [row["frame_id"] for row in index.windows[2].observations], [17, 18, 21],
        )
        self.assertFalse(any(window.start_frame <= 13 < 17 <= window.end_frame
                             for window in index.windows))

    def test_zero_gap_allowance_splits_at_one_missing_frame(self):
        index = self.make_index([1, 2, 4], max_gap_frames=0)
        self.assertEqual(self.bounds(index), [(1, 2), (4, 4)])

    def test_bounds_snap_to_observed_frames_without_inventing_boxes(self):
        index = self.make_index(
            [50, 54, 58], window_frames=5, stride_frames=3, max_gap_frames=4,
        )
        self.assertEqual(self.bounds(index), [(50, 54), (54, 54), (58, 58)])
        self.assertEqual(
            [[row["frame_id"] for row in window.observations] for window in index.windows],
            [[50, 54], [54], [58]],
        )

    def test_window_length_counts_original_frame_numbers_not_observations(self):
        index = self.make_index(
            [10, 12, 14], window_frames=3, stride_frames=3, max_gap_frames=2,
        )
        self.assertEqual(self.bounds(index), [(10, 12), (14, 14)])

    def test_sparse_windows_skip_empty_ranges_without_resetting_the_stride(self):
        # A permissive gap setting must not cause a billion empty iterations.
        index = self.make_index(
            [0, 1_000_000_001, 1_000_000_002],
            window_frames=3, stride_frames=2, max_gap_frames=1_000_000_000,
        )
        self.assertEqual(self.bounds(index), [
            (0, 0), (1_000_000_001, 1_000_000_002), (1_000_000_002, 1_000_000_002),
        ])

    def test_exact_duplicate_observed_intervals_are_emitted_once(self):
        index = self.make_index(
            [1, 6], window_frames=4, stride_frames=1, max_gap_frames=4,
        )
        self.assertEqual(self.bounds(index), [(1, 1), (6, 6)])

    def test_short_tracks_single_observations_and_empty_input(self):
        for frames, expected in (([], []), ([7], [(7, 7)]), ([7, 8], [(7, 8)])):
            with self.subTest(frames=frames):
                self.assertEqual(self.bounds(self.make_index(frames)), expected)

    def test_unit_windows_and_nonoverlapping_windows_preserve_all_frames(self):
        for length, stride, expected in (
            (1, 1, [(5, 5), (6, 6), (7, 7), (8, 8)]),
            (2, 2, [(5, 6), (7, 8)]),
        ):
            with self.subTest(length=length, stride=stride):
                index = self.make_index(range(5, 9), window_frames=length, stride_frames=stride)
                self.assertEqual(self.bounds(index), expected)

    def test_vehicle_identity_source_and_input_order_are_preserved(self):
        for source in ("annotated", "predicted"):
            with self.subTest(source=source):
                tracks = [
                    observation(4, 42, class_name="Car", source=source),
                    observation(3, source=source), observation(1, source=source),
                    observation(2, 42, class_name="Car", source=source),
                ]
                config = RetrievalConfig(window_frames=3, stride_frames=2)
                index = build_index(RecordingFixture(), tracks, config=config)
                reversed_index = build_index(RecordingFixture(), reversed(tracks), config=config)
                self.assertEqual(index.windows, reversed_index.windows)
                self.assertEqual([
                    (window.recording_id, window.source, window.track_id,
                     window.start_frame, window.end_frame) for window in index.windows
                ], [
                    ("recording-a", source, 7, 1, 3), ("recording-a", source, 7, 3, 3),
                    ("recording-a", source, 42, 2, 4), ("recording-a", source, 42, 4, 4),
                ])
                for window in index.windows:
                    self.assertTrue(all(
                        (row["recording_id"], row["source"], row["track_id"])
                        == (window.recording_id, window.source, window.track_id)
                        for row in window.observations
                    ))


class QueryCutoffTests(unittest.TestCase):
    def make_index(self, tracks, config=None):
        return build_index(RecordingFixture(range(161)), tracks, config=config)

    def test_crossing_windows_are_rebuilt_from_observations_through_frame_100(self):
        index = self.make_index([observation(frame) for frame in range(90, 121)])
        self.assertTrue(any(window.start_frame <= 100 < window.end_frame
                            for window in index.windows))
        windows = candidate_windows(question(), index)
        self.assertEqual([(window.start_frame, window.end_frame) for window in windows],
                         [(90, 100), (100, 100)])
        self.assertEqual([row["frame_id"] for row in windows[0].observations],
                         list(range(90, 101)))
        self.assertTrue(all(row["frame_id"] <= 100 for window in windows
                            for row in window.observations))
        prefix = self.make_index([observation(frame) for frame in range(90, 101)])
        self.assertEqual(windows, prefix.windows)

    def test_changes_after_100_leave_complete_candidates_identical_for_both_sources(self):
        for source in ("annotated", "predicted"):
            prefix = [observation(frame, source=source) for frame in range(90, 101)]
            prefix.append(observation(95, 42, source=source, class_name="Car"))
            expected = candidate_windows(question(), self.make_index(prefix))
            future_variants = (
                [observation(frame, source=source) for frame in range(101, 121)],
                [observation(frame, source=source, class_name="Truck",
                             bbox_xyxy=(60, 10, 90, 50)) for frame in range(101, 121)],
                [observation(frame, source=source) for frame in (101, 140, 150)],
                # A future-only track with a lower ID changes the full index's order.
                [observation(101, 0, source=source), observation(102, 0, source=source)],
                [],
            )
            for variant_number, future in enumerate(future_variants):
                with self.subTest(source=source, variant=variant_number):
                    actual = candidate_windows(question(), self.make_index(prefix + future))
                    self.assertEqual(actual, expected)
                    self.assertEqual({window.track_id for window in actual}, {7, 42})

    def test_cutoff_in_gap_uses_observed_bounds_and_deduplicates_partial_windows(self):
        config = RetrievalConfig(window_frames=10, stride_frames=2)
        index = build_index(
            RecordingFixture([90, 95, 101]),
            [observation(frame) for frame in (90, 95, 101)], config=config,
        )
        windows = candidate_windows(question(100), index)
        self.assertEqual([(window.start_frame, window.end_frame) for window in windows],
                         [(90, 95), (95, 95)])
        self.assertEqual([[row["frame_id"] for row in window.observations]
                          for window in windows], [[90, 95], [95]])

    def test_query_order_does_not_mutate_the_index_or_questions(self):
        index = self.make_index([observation(frame) for frame in range(90, 121)])
        original_observations = deepcopy(index.observations)
        original_histories = deepcopy(dict(index.histories))
        original_windows = deepcopy(index.windows)
        early, late = question(100), question(120)
        original_questions = deepcopy((early, late))
        late_result = candidate_windows(late, index)
        early_result = candidate_windows(early, index)
        self.assertNotEqual(late_result, early_result)
        self.assertEqual(candidate_windows(late, index), late_result)
        self.assertEqual(candidate_windows(early, index), early_result)
        self.assertEqual((early, late), original_questions)
        self.assertEqual(index.observations, original_observations)
        self.assertEqual(dict(index.histories), original_histories)
        self.assertEqual(index.windows, original_windows)

    def test_modifying_returned_observations_cannot_corrupt_later_queries(self):
        index = self.make_index([observation(frame) for frame in range(90, 121)])
        expected = deepcopy(candidate_windows(question(), index))
        result = candidate_windows(question(), index)
        result[0].observations[0]["class_name"] = "Truck"
        result[0].observations[0]["bbox_xyxy"] = (50, 10, 90, 50)
        result[0].observations[0]["frame_id"] = 999
        self.assertEqual(candidate_windows(question(), index), expected)
        self.assertEqual(index.observations[0], observation(90))

    def test_candidate_generation_does_not_read_future_frame_metadata(self):
        index = self.make_index([observation(frame) for frame in range(90, 121)])
        expected = candidate_windows(question(), index)
        index.recording.requested.clear()
        for frame in range(101, 121):
            del index.recording.frames[frame]
        self.assertEqual(candidate_windows(question(), index), expected)
        self.assertEqual(index.recording.requested, [])

    def test_empty_future_only_and_boundary_cutoffs(self):
        empty = self.make_index([])
        future = self.make_index([observation(101)])
        self.assertEqual(candidate_windows(question(), empty), ())
        self.assertEqual(candidate_windows(question(), future), ())
        self.assertEqual(rank_evidence(question(), empty), [])
        self.assertEqual(rank_evidence(question(), future), [])
        index = self.make_index([observation(0), observation(1)])
        self.assertEqual([(window.start_frame, window.end_frame)
                          for window in candidate_windows(question(0), index)], [(0, 0)])
        self.assertEqual(candidate_windows(question(160), index), index.windows)

    def test_question_fields_are_required_and_gold_fields_are_rejected(self):
        index = self.make_index([])
        for field in question():
            invalid = question()
            del invalid[field]
            with self.subTest(missing=field):
                with self.assertRaisesRegex(ValueError, field):
                    candidate_windows(invalid, index)
        for field in ("expected_answer", "acceptable_evidence", "answerable"):
            with self.subTest(extra=field):
                with self.assertRaisesRegex(ValueError, "four public fields"):
                    candidate_windows(question(**{field: "gold"}), index)
        for invalid in (None, [], "question"):
            with self.subTest(question=invalid):
                with self.assertRaisesRegex(ValueError, "question"):
                    candidate_windows(invalid, index)

    def test_invalid_question_values_are_rejected_even_for_zero_results(self):
        index = self.make_index([])
        fields = {
            "question_id": ("", " ", None, 1),
            "recording_id": ("", " ", None, 1, "another-recording"),
            "question": ("", " ", None, 1),
            "query_frame": (-1, True, False, 100.5, "100", None),
        }
        for field, values in fields.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    invalid = question(**{field: value})
                    with self.assertRaisesRegex(ValueError, field):
                        candidate_windows(invalid, index)
                    with self.assertRaisesRegex(ValueError, field):
                        rank_evidence(invalid, index, k=0)

    def test_recording_mismatch_cannot_retrieve_an_identical_track_id(self):
        index = self.make_index([observation(90, track_id=7)])
        wrong_recording = question(recording_id="recording-b")
        with self.assertRaisesRegex(ValueError, "recording_id"):
            candidate_windows(wrong_recording, index)
        with self.assertRaisesRegex(ValueError, "recording_id"):
            rank_evidence(wrong_recording, index, k=0)

    def test_ranker_validates_result_count_and_accepts_zero(self):
        index = self.make_index([observation(90)])
        self.assertEqual(rank_evidence(question(), index, k=0), [])
        for count in (-1, 4, True, False, 1.5, "3", None):
            with self.subTest(k=count):
                with self.assertRaisesRegex(ValueError, "k"):
                    rank_evidence(question(), index, k=count)


class RankingTests(unittest.TestCase):
    def make_index(self, tracks, *, config=None):
        frame_ids = sorted({row["frame_id"] for row in tracks})
        return build_index(RecordingFixture(frame_ids), tracks, config=config)

    def test_bus_and_car_questions_change_winner_for_both_sources(self):
        for source in ("annotated", "predicted"):
            with self.subTest(source=source):
                index = self.make_index([
                    observation(1, track_id=7, source=source),
                    observation(1, track_id=9, source=source, class_name="Car"),
                ])
                for text, expected_id in (("Which bus is visible?", 7),
                                          ("Which car is visible?", 9)):
                    ranked = rank_evidence(question(1, question=text), index)
                    self.assertEqual(len(ranked), 1)
                    self.assertEqual(ranked[0]["track_id"], expected_id)
                    self.assertEqual(ranked[0]["source"], source)
                    self.assertEqual(ranked[0]["rank"], 1)
                    self.assertEqual(ranked[0]["score"], 0.5)

    def test_entering_question_selects_earlier_interval_of_same_vehicle(self):
        index = self.make_index([
            observation(1, bbox_xyxy=(0, 20, 10, 40)),
            observation(5, bbox_xyxy=(20, 20, 30, 40)),
            observation(11, bbox_xyxy=(40, 20, 50, 40)),
            observation(15, bbox_xyxy=(40, 20, 50, 40)),
        ], config=RetrievalConfig(window_frames=10, stride_frames=10))
        ranked = rank_evidence(question(15, question="Which bus entered the scene?"), index)
        self.assertEqual(len(ranked), 2)
        self.assertEqual([(row["start_frame"], row["end_frame"]) for row in ranked],
                         [(1, 5), (11, 15)])
        self.assertEqual({row["track_id"] for row in ranked}, {7})
        self.assertGreater(ranked[0]["score"], ranked[1]["score"])
        self.assertEqual(ranked[0]["score"], 0.75)

    def test_direction_words_are_distinct_from_image_location(self):
        index = self.make_index([
            observation(1, track_id=1, bbox_xyxy=(6, 20, 14, 40)),
            observation(5, track_id=1, bbox_xyxy=(26, 20, 34, 40)),
            observation(1, track_id=2, bbox_xyxy=(86, 20, 94, 40)),
            observation(5, track_id=2, bbox_xyxy=(66, 20, 74, 40)),
        ])
        for text, expected_id in (
            ("Which bus is moving left?", 2),
            ("Which bus is heading to the left?", 2),
            ("Which left-moving bus is visible?", 2),
            ("Which bus is moving leftward?", 2),
            ("Which bus is moving rightwards?", 1),
            ("Which bus is moving right?", 1),
            ("Which bus is on the left?", 1),
            ("Which bus is on the right?", 2),
        ):
            with self.subTest(question=text):
                ranked = rank_evidence(question(5, question=text), index)
                self.assertEqual(ranked[0]["track_id"], expected_id)
                self.assertGreater(ranked[0]["score"], ranked[1]["score"])
                self.assertEqual([row["score"] for row in ranked], [0.75, 0.5])

    def test_stationary_and_out_and_back_motion_are_distinguished(self):
        index = self.make_index([
            observation(1, track_id=1), observation(3, track_id=1),
            observation(1, track_id=2),
            observation(2, track_id=2, bbox_xyxy=(30, 20, 50, 40)),
            observation(3, track_id=2), observation(1, track_id=3),
        ])
        for text, expected_id in (("Which bus is stationary?", 1),
                                  ("Which bus is parked?", 1),
                                  ("Which bus is moving?", 2)):
            with self.subTest(question=text):
                ranked = rank_evidence(question(3, question=text), index)
                self.assertEqual(ranked[0]["track_id"], expected_id)
                self.assertEqual(ranked[0]["score"], 0.75)
                self.assertTrue(all(row["score"] == 0.5 for row in ranked[1:]))

    def test_single_observation_does_not_claim_motion_or_entry(self):
        index = self.make_index([observation(1, bbox_xyxy=(0, 20, 10, 40))])
        for text in ("moving", "stationary", "entering", "exiting"):
            with self.subTest(question=text):
                self.assertEqual(rank_evidence(question(1, question=text), index), [])

    def test_motion_threshold_includes_exactly_two_percent_as_stationary(self):
        index = self.make_index([
            observation(1, track_id=1, bbox_xyxy=(25, 20, 35, 40)),
            observation(2, track_id=1, bbox_xyxy=(27, 20, 37, 40)),
            observation(1, track_id=2, bbox_xyxy=(25, 20, 35, 40)),
            observation(2, track_id=2, bbox_xyxy=(27.1, 20, 37.1, 40)),
        ])
        stationary = rank_evidence(question(2, question="stationary"), index)
        moving = rank_evidence(question(2, question="rightward"), index)
        self.assertEqual([row["track_id"] for row in stationary], [1])
        self.assertEqual([row["track_id"] for row in moving], [2])

    def test_entry_and_exit_at_five_percent_margin_are_symmetric_for_all_edges(self):
        for edge, interior in (
            ((5, 32, 15, 48), (25, 32, 35, 48)),
            ((85, 32, 95, 48), (65, 32, 75, 48)),
            ((40, 4, 60, 12), (40, 20, 60, 28)),
            ((40, 68, 60, 76), (40, 52, 60, 60)),
        ):
            for first, last, expected, other in (
                (edge, interior, "entering", "exiting"),
                (interior, edge, "exiting", "entering"),
            ):
                with self.subTest(first=first, last=last):
                    index = self.make_index([
                        observation(1, bbox_xyxy=first), observation(2, bbox_xyxy=last),
                    ])
                    ranked = rank_evidence(question(2, question=expected), index)
                    self.assertEqual(len(ranked), 1)
                    self.assertEqual(ranked[0]["score"], 0.25)
                    self.assertEqual(rank_evidence(question(2, question=other), index), [])

    def test_box_size_change_and_reappearance_alone_do_not_imply_entry(self):
        index = self.make_index([
            observation(1, bbox_xyxy=(0, 20, 20, 40)),
            observation(2, bbox_xyxy=(8, 20, 12, 40)),
            observation(20, bbox_xyxy=(40, 20, 60, 40)),
            observation(21, bbox_xyxy=(40, 20, 60, 40)),
        ])
        self.assertEqual(rank_evidence(question(21, question="entering"), index), [])
        self.assertEqual(rank_evidence(question(21, question="exiting"), index), [])

    def test_location_and_motion_use_normalized_frame_dimensions(self):
        recording = RecordingFixture((1, 2))
        recording.frames[2]["width"] = 200
        recording.frames[2]["height"] = 160
        index = build_index(recording, [
            observation(1, bbox_xyxy=(10, 8, 30, 24)),
            observation(2, bbox_xyxy=(20, 16, 60, 48)),
        ])
        ranked = rank_evidence(
            question(2, question="Which stationary bus is at the top left?"), index,
        )
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["score"], 1.0)

    def test_class_case_punctuation_plural_and_unknown_label(self):
        bus_index = self.make_index([
            observation(1, class_name="BUS!"), observation(2, class_name="buses"),
            observation(3, class_name="Car"),
        ])
        ranked = rank_evidence(question(3, question="Which buses are visible?"), bus_index)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["score"], 0.5)
        car_index = self.make_index([observation(1, class_name="Car")])
        self.assertEqual(rank_evidence(
            question(1, question="Which cars are visible?"), car_index,
        )[0]["score"], 0.5)
        self.assertEqual(rank_evidence(question(1, question="scar"), car_index), [])
        robot_index = self.make_index([observation(1, class_name="Delivery robot")])
        self.assertEqual(rank_evidence(
            question(1, question="Find the delivery robot"), robot_index,
        )[0]["score"], 0.5)

    def test_ties_use_start_then_end_then_track_id_and_honor_k(self):
        tracks = [
            observation(1, track_id=9), observation(2, track_id=9),
            observation(1, track_id=3), observation(2, track_id=3),
            observation(0, track_id=4), observation(2, track_id=4),
            observation(0, track_id=2), observation(1, track_id=2),
        ]
        index = self.make_index(list(reversed(tracks)))
        query = question(2, question="bus")
        for k in (1, 2, 3):
            with self.subTest(k=k):
                ranked = rank_evidence(query, index, k=k)
                self.assertEqual([row["track_id"] for row in ranked], [2, 4, 3][:k])
                self.assertEqual([row["rank"] for row in ranked], list(range(1, k + 1)))
                self.assertTrue(all(row["score"] == 0.5 for row in ranked))
        self.assertEqual(rank_evidence(query, index), rank_evidence(query, index, k=3))
        self.assertEqual(rank_evidence(query, index),
                         rank_evidence(query, self.make_index(tracks)))

    def test_feature_weights_can_change_winner(self):
        tracks = [
            observation(1, track_id=1, bbox_xyxy=(40, 20, 60, 40)),
            observation(3, track_id=1, bbox_xyxy=(40, 20, 60, 40)),
            observation(1, track_id=2, class_name="Car", bbox_xyxy=(60, 20, 80, 40)),
            observation(3, track_id=2, class_name="Car", bbox_xyxy=(20, 20, 40, 40)),
        ]
        query = question(3, question="Which bus is moving left?")
        ordinary = rank_evidence(query, self.make_index(tracks))
        motion_weighted = rank_evidence(query, self.make_index(
            tracks, config=RetrievalConfig(class_weight=1, location_weight=0, movement_weight=3),
        ))
        self.assertEqual(ordinary[0]["track_id"], 1)
        self.assertEqual(motion_weighted[0]["track_id"], 2)
        self.assertEqual(motion_weighted[0]["score"], 0.75)

    def test_unsupported_question_returns_no_zero_score_candidates(self):
        index = self.make_index([observation(1)])
        self.assertEqual(rank_evidence(
            question(1, question="Find the purple helicopter"), index,
        ), [])

    def test_future_changes_preserve_complete_rankings_and_scores(self):
        query = question(100, question="Which bus is moving right?")
        for source in ("annotated", "predicted"):
            prefix = [
                observation(frame, source=source,
                            bbox_xyxy=(10 + frame - 90, 20, 30 + frame - 90, 40))
                for frame in range(90, 101)
            ] + [
                observation(frame, track_id=42, source=source, class_name="Car",
                            bbox_xyxy=(60, 20, 80, 40))
                for frame in range(90, 101)
            ]
            future_variants = [
                [observation(frame, source=source, bbox_xyxy=(80, 0, 100, 10))
                 for frame in range(101, 141)],
                [observation(frame, source=source, class_name="Truck",
                             bbox_xyxy=(0, 60, 20, 80)) for frame in range(101, 141, 7)]
                + [observation(105, track_id=0, source=source),
                   observation(106, track_id=0, source=source)],
                [],
            ]
            expected = rank_evidence(query, self.make_index(prefix))
            self.assertEqual((expected[0]["start_frame"], expected[0]["end_frame"]), (90, 100))
            self.assertGreater(expected[0]["score"], expected[1]["score"])
            for variant_number, future in enumerate(future_variants):
                with self.subTest(source=source, variant=variant_number):
                    recording = RecordingFixture(range(90, 141))
                    index = build_index(recording, prefix + future)
                    if variant_number == 0:
                        self.assertTrue(any(window.start_frame <= 100 < window.end_frame
                                            for window in index.windows))
                    # Future footage metadata is unavailable after indexing.
                    for frame in list(recording.frames):
                        if frame > 100:
                            del recording.frames[frame]
                    recording.requested.clear()
                    self.assertEqual(rank_evidence(query, index), expected)
                    self.assertTrue(all(frame <= 100 for frame in recording.requested))
                    self.assertEqual(len(recording.requested), len(set(recording.requested)))

    def test_nonchronological_queries_preserve_results_and_index(self):
        index = self.make_index([
            observation(frame, bbox_xyxy=(10 + frame - 90, 20, 30 + frame - 90, 40))
            for frame in range(90, 121)
        ])
        before = deepcopy((index.observations, index.windows))
        earlier_query, later_query = question(100, question="bus"), question(120, question="bus")
        questions_before = deepcopy((earlier_query, later_query))
        earlier = rank_evidence(earlier_query, index)
        later = rank_evidence(later_query, index)
        self.assertEqual(rank_evidence(earlier_query, index), earlier)
        self.assertEqual(rank_evidence(later_query, index), later)
        self.assertEqual((index.observations, index.windows), before)
        self.assertEqual((earlier_query, later_query), questions_before)

    def test_output_uses_only_serializable_public_interval_fields(self):
        ranked = rank_evidence(question(1, question="bus"), self.make_index([observation(1)]))
        self.assertEqual(set(ranked[0]), {
            "recording_id", "source", "track_id", "start_frame", "end_frame", "rank", "score",
        })
        self.assertEqual(ranked[0]["recording_id"], "recording-a")
        self.assertEqual(ranked[0]["start_frame"], 1)
        self.assertEqual(ranked[0]["end_frame"], 1)
        self.assertIsInstance(ranked[0]["score"], float)
        self.assertEqual(json.loads(json.dumps(ranked, allow_nan=False)), ranked)


if __name__ == "__main__":
    unittest.main()
