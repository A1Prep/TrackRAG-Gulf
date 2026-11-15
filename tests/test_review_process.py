"""Run the production commands in child processes with temporary adapter modules.

These checks verify the command and on-disk handoffs. They make no model calls and
do not establish accuracy on EMT footage.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


class CommandProcessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = Path(__file__).resolve().parents[1]
        package = self.root / "adapters" / "trackrag"
        package.mkdir(parents=True)
        modules = {
            "__init__.py": f"__path__.append({str(self.project / 'trackrag')!r})\n",
            "emt.py": '''
import json
from pathlib import Path
class LocalRecording:
    def __init__(self, root, recording_id):
        self.root = Path(root) / recording_id
        self.recording_id = recording_id
        rows = json.loads((self.root / "frames.json").read_text())
        self.frames = {row["frame_id"]: row for row in rows}
    def get_frame(self, frame_id):
        return self.frames[frame_id]
    def iter_frames(self):
        return iter(self.frames.values())
def open_recording(data_root, recording_id):
    return LocalRecording(data_root, recording_id)
def load_annotated_tracks(recording):
    return json.loads((recording.root / "annotations.json").read_text())
''',
            "vision.py": '''
import json
def predict_tracks(recording):
    return json.loads((recording.root / "predictions.json").read_text())
''',
            "questions.py": '''
import json
def load_questions(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]
''',
            "retrieval.py": '''
def build_index(recording, tracks):
    return tracks
def rank_evidence(question, index, k=3):
    eligible = [row for row in index if row["frame_id"] <= question["query_frame"]]
    if not eligible:
        return []
    track = eligible[0]
    frames = [row["frame_id"] for row in eligible if row["track_id"] == track["track_id"]]
    return [{"recording_id": track["recording_id"], "source": track["source"],
             "track_id": track["track_id"], "start_frame": min(frames),
             "end_frame": max(frames), "rank": 1, "score": 1.0}]
''',
            "answering.py": '''
def answer_question(question, evidence, run_dir):
    return {"answer_status": "answered", "answer": "The vehicle in E1.",
            "cited_evidence_ids": ["E1"]}
''',
        }
        for name, contents in modules.items():
            (package / name).write_text(contents, encoding="utf-8")
        self.env = dict(os.environ, PYTHONPATH=str(package.parent))

    def command(self, *arguments):
        return subprocess.run(
            [sys.executable, "-B", "-m", *arguments], cwd=self.project, env=self.env,
            capture_output=True, text=True, timeout=30,
        )

    def make_recording(self, recording_id, start=1):
        directory = self.root / recording_id
        directory.mkdir()
        frames, annotations, predictions = [], [], []
        for frame_id in range(start, start + 3):
            image = directory / f"{frame_id}.png"
            Image.new("RGB", (40, 30), (frame_id, 0, 0)).save(image)
            frames.append({
                "recording_id": recording_id, "frame_id": frame_id, "timestamp_s": (frame_id-start)/10,
                "width": 40, "height": 30, "image_path": str(image),
            })
            common = {"recording_id": recording_id, "frame_id": frame_id, "class_name": "Bus", "bbox_xyxy": [3, 4, 20, 22]}
            annotations.append(dict(common, source="annotated", track_id=7))
            predictions.append(dict(common, source="predicted", track_id=99))
        for name, data in (("frames", frames), ("annotations", annotations), ("predictions", predictions)):
            (directory / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
        question = {"question_id": "q1", "recording_id": recording_id, "query_frame": start + 2, "question": "Which bus?"}
        path = directory / "questions.jsonl"
        path.write_text(json.dumps(question) + "\n", encoding="utf-8")
        return path

    def review(self, recording_id, questions, source, output):
        return self.command(
            "trackrag", "review", "--data-root", str(self.root), "--recording", recording_id,
            "--questions", str(questions), "--source", source, "--answerer", "gemini", "--out", str(output),
        )

    def test_annotated_then_predicted_then_offline_scoring(self):
        questions = self.make_recording("clip-a")
        annotated = self.root / "annotated-run"
        completed = self.review("clip-a", questions, "annotated", annotated)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        # Predicted review must still work after the human labels are removed.
        (self.root / "clip-a" / "annotations.json").unlink()
        predicted = self.root / "predicted-run"
        completed = self.review("clip-a", questions, "predicted", predicted)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        saved = json.loads((predicted / "results.jsonl").read_text())
        self.assertEqual(saved["source"], "predicted")
        self.assertEqual(saved["answer"]["cited_evidence_ids"], ["E1"])
        self.assertEqual(saved["evidence"][0]["interval"]["track_id"], 99)
        for image in saved["evidence"][0]["images"]:
            with Image.open(predicted / image["image_path"]) as marked:
                self.assertEqual(marked.getpixel((3, 4)), (255, 0, 0))
        gold = self.root / "gold.jsonl"
        gold.write_text(json.dumps({
            "question_id": "q1", "recording_id": "clip-a", "query_frame": 3, "answerable": True,
            "acceptable_evidence": [{"track_id": 7, "start_frame": 1, "end_frame": 3}],
        }) + "\n", encoding="utf-8")
        completed = self.command(
            "evaluation", "--results", str(predicted / "results.jsonl"), "--gold", str(gold),
            "--predicted-tracks", str(predicted / "tracks.jsonl"),
            "--annotated-tracks", str(annotated / "tracks.jsonl"),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["counts"]["top1_correct"], 1)

    def test_same_command_handles_a_second_recording_and_frame_origin(self):
        for recording_id, first_frame in (("one", 1), ("two", 10)):
            questions = self.make_recording(recording_id, first_frame)
            output = self.root / f"run-{recording_id}"
            completed = self.review(recording_id, questions, "annotated", output)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            saved = json.loads((output / "results.jsonl").read_text())
            self.assertEqual(saved["question"]["recording_id"], recording_id)
            self.assertEqual(saved["evidence"][0]["images"][0]["frame_id"], first_frame)

    def test_adapter_check_finds_the_callable_connections(self):
        completed = self.command("trackrag", "check-adapters", "--source", "predicted", "--answerer", "gemini")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(json.loads(completed.stdout)["ready"])


if __name__ == "__main__":
    unittest.main()
