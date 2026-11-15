"""Checks for Task 4's configuration and public adapter signatures."""

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


if __name__ == "__main__":
    unittest.main()
