"""Exercise the real module command and its public output through subprocesses."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

from trackrag.contracts import RankedInterval


class RetrievalDemoTests(unittest.TestCase):
    def run_demo(self, *args):
        return subprocess.run(
            [sys.executable, "-B", "-m", "trackrag.retrieval_demo", *args],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=15,
        )

    def test_default_command_prints_ranked_table_and_marks_data_as_synthetic(self):
        result = self.run_demo()
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines[0].split(), ["Rank", "Recording", "Source", "Track", "Start", "End", "Score"])
        self.assertEqual(lines[1].split(), ["1", "demo", "annotated", "7", "90", "100", "0.7500"])
        self.assertIn("Synthetic", result.stderr)

    def test_json_is_clean_shared_records_and_question_changes_vehicle(self):
        for source in ("annotated", "predicted"):
            for text, track_id in (("bus", 7), ("car", 42)):
                with self.subTest(source=source, question=text):
                    result = self.run_demo("--format", "json", "--source", source,
                                           "--question", text, "--query-frame", "100", "--k", "1")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    rows = json.loads(result.stdout)
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(set(rows[0]), RankedInterval.__required_keys__)
                    self.assertEqual(rows[0], {
                        "recording_id": "demo", "source": source, "track_id": track_id,
                        "start_frame": 90, "end_frame": 100, "rank": 1, "score": 0.5,
                    })

    def test_configuration_flags_reach_index_and_ranker(self):
        result = self.run_demo(
            "--format", "json", "--question", "bus", "--query-frame", "110",
            "--window-frames", "5", "--stride-frames", "5", "--max-gap-frames", "0",
            "--class-weight", "1", "--location-weight", "0", "--movement-weight", "0",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = json.loads(result.stdout)
        self.assertEqual([(row["start_frame"], row["end_frame"]) for row in rows],
                         [(90, 94), (95, 99), (100, 104)])
        self.assertEqual([row["score"] for row in rows], [1.0, 1.0, 1.0])

    def test_empty_searches_succeed_with_no_invented_results(self):
        for args in (("--query-frame", "89"), ("--question", "helicopter"), ("--k", "0")):
            with self.subTest(args=args):
                result = self.run_demo("--format", "json", *args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), [])
        result = self.run_demo("--query-frame", "89")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No matching intervals.", result.stdout)

    def test_invalid_arguments_are_cli_errors_without_tracebacks(self):
        for args in (
            ("--query-frame", "-1"), ("--question", "   "), ("--k", "4"),
            ("--source", "unknown"), ("--window-frames", "5", "--stride-frames", "10"),
            ("--movement-weight", "nan"),
            ("--class-weight", "0", "--location-weight", "0", "--movement-weight", "0"),
        ):
            with self.subTest(args=args):
                result = self.run_demo(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("error:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_help_explains_synthetic_data_and_available_options(self):
        result = self.run_demo("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("synthetic", result.stdout)
        self.assertIn("--query-frame", result.stdout)
        self.assertIn("--window-frames", result.stdout)
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
