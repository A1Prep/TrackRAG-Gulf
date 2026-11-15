"""Checks for Task 4's configuration, observation windows and adapter signatures."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from inspect import signature
import unittest

from trackrag.retrieval import RetrievalConfig, build_index, rank_evidence


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
        # Match the downstream adapter checker without running unfinished search.
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


if __name__ == "__main__":
    unittest.main()
