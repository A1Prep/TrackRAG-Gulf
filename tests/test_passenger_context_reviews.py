"""Local software fixtures only. Original photographs, reviewer rows, and model results stay private."""
from copy import deepcopy
import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image

from tools import passenger_context_resolve as returns
from tools import passenger_context_reviews as packet
from tools.passenger_reviewed_baseline import Reference


def fixture_sources(root):
    inputs = root / "fixture-inputs"
    recording = "video_122233"  # The original source-path validator accepts only registered recording names.
    requests, originals, signed, bindings, scenes = [], {}, [], [], {}
    for i, status in enumerate(("proposed", "needs_clarification", "insufficient_visual_evidence"), 1):
        fid = i + 9000
        identity = f"software-fixture-{i}"
        relative = Path("images", "test", recording, f"{fid:05d}.jpg")
        photo = inputs / "source" / relative
        photo.parent.mkdir(exist_ok=True, parents=True)
        Image.new("RGB", (64, 48), (30*i, 20, 140)).save(photo, "JPEG")
        digest = sha256(photo.read_bytes()).hexdigest()
        frame = {"recording_id": recording, "frame_id": fid, "timestamp_s": i*2.0,
                 "width": 64, "height": 48, "image_path": str(photo)}
        track = {"recording_id": recording, "frame_id": fid, "track_id": i, "source": "predicted",
                 "class_name": "software fixture", "bbox_xyxy": [3., 4., 9., 15.]}
        req = {"request_id": identity, "text": "Synthetic, quoted <script> & reference " + str(i),
               "dialogue": [{"role": "user", "text": "A synthetic earlier turn.", "recording_id": recording, "frame_id": fid}]
               if i == 2 else [], "input_mode": "typed",
               "scene": {"frame": frame, "source": "predicted", "tracks": [track], "provenance": {"image_sha256": digest}}}
        old = deepcopy(req)
        old["scene"]["tracks"] = []
        old["scene"]["frame"]["timestamp_s"] = float(i)
        old["scene"]["provenance"] = {"detector_run": False, "mode": "image_only_local_annotation"}
        target = {"status": status, "destination_point_xy": [6., 8.] if i == 1 else None,
                  "destination_box_xyxy": [3., 4., 9., 15.] if i == 1 else None,
                  "clarification_question": 'Software fixture: "which mock object"?' if i == 2 else None}
        fields = sorted(target)
        target_hash = packet.supervision_sha256(target, fields, frame)
        prior = packet.inference_input_sha256(old, digest)
        current = packet.inference_input_sha256(req, digest)
        scene_hash = sha256(f"fixture-scene-{i}".encode()).hexdigest()
        scenes[scene_hash] = {"scene_sha256": scene_hash, "model_input": {
            "recording_id": recording, "frame_id": fid, "width": 64, "height": 48,
            "timestamp_s": frame["timestamp_s"], "image_sha256": digest, "observations": [deepcopy(track)]}}
        binding = {"case_id": identity, "original_git_image": packet.PACKAGE + "/" + relative.as_posix(),
                   "reviewed_input_sha256": prior, "saved_detector_input_sha256": current, "image_sha256": digest,
                   "recording_id": recording, "frame_id": fid, "width": 64, "height": 48, "reviewed_timestamp_s": float(i),
                   "reconstructed_timestamp_s": frame["timestamp_s"], "saved_detector_conditions":
                       {"finetuned": {"scene_sha256_from_original_hashes": scene_hash, "current_observations": 1}}}
        earlier = {"status": "human_reviewed", "supervision_sha256": target_hash, "input_sha256": prior,
                   "reviewers": ["fixture-sentinel-A", "fixture-sentinel-B"]}
        source = {"dataset": "Software fixture", "source_split": "test", "recording_id": recording,
                  "image_sha256": digest, "frame_id": fid, "scene_group_verified": True, "grouping_level": "recording",
                  "landmarks": ["old-landmark-fixture-sentinel"], "paraphrases": ["old-paraphrase-fixture-sentinel"]}
        signed.append({"schema_version": 1, "case_id": identity, "partition": "test", "scene_group": recording,
                       "source": source, "target": deepcopy(target), "supervised_fields": fields,
                       "review": earlier, "request": deepcopy(old)})
        originals[identity] = Reference(old, target, digest)
        requests.append(req)
        bindings.append(binding)
    return requests, originals, signed, bindings, scenes, inputs


def write_fixture(root):
    request, old, signed, bindings, scenes, inputs = fixture_sources(root)
    out = root / "packet"
    (out / "tools").mkdir(parents=True)
    resolver = packet.git(packet.TASK2, packet.PACKAGE + "/tools/dropoff_resolve_reviews.py")
    (out / "tools/dropoff_resolve_reviews.py").write_bytes(resolver)
    cases = packet.private_cases(request, old, signed[::-1], bindings[::-1], scenes, inputs, out)
    packet.put(out / "review_cases.jsonl", "")
    packet.put(out / "reconfirmation_cases.jsonl", "".join(packet.compact(c) + "\n" for c in cases))
    packet.write_forms(out, cases, packet.indexed(bindings, "case_id"))
    packet.put(out / "resolve_context.py", (packet.HERE / "passenger_context_resolve.py").read_text(encoding="utf-8"))
    inventory = {p.relative_to(out).as_posix(): packet.file_sha256(p) for p in out.rglob("*") if p.is_file()}
    packet.put(out / "packet-files.json", packet.compact(inventory))
    return out


def form(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_form(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class InputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sources = fixture_sources(self.root)

    def prepare(self):
        return packet.private_cases(*self.sources, self.root / "packet")

    def test_reordered_mixed_sources_match_original_targets_and_all_input(self):
        requests, old, signed, bindings, _, _ = self.sources
        signed.reverse()
        bindings[:] = [bindings[1], bindings[2], bindings[0]]
        result = self.prepare()
        self.assertEqual([c["case_id"] for c in result], [q["request_id"] for q in requests])
        self.assertEqual([c["target"]["status"] for c in result],
                         ["proposed", "needs_clarification", "insufficient_visual_evidence"])
        for case in result:
            self.assertEqual(case["review"]["status"], "pending_reconfirmation")
            self.assertIs(case["review"]["new_human_review"], False)
            self.assertEqual(case["request"]["dialogue"], old[case["case_id"]].request["dialogue"])
            self.assertNotIn("sentinel", packet.compact(case))

    def test_materially_changed_detector_boxes_are_rejected(self):
        self.sources[0][0]["scene"]["tracks"][0]["bbox_xyxy"][0] = 2.
        with self.assertRaisesRegex(ValueError, "detector binding"):
            self.prepare()

    def test_original_photo_changed_even_if_same_dimension(self):
        image = self.sources[0][0]["scene"]["frame"]["image_path"]
        Image.new("RGB", (64, 48), (2, 91, 81)).save(image)
        with self.assertRaisesRegex(ValueError, "photograph"):
            self.prepare()

    def test_input_inconsistent_with_task1_fails_at_full_precision(self):
        canonical = next(iter(self.sources[4].values()))
        canonical["model_input"]["observations"][0]["bbox_xyxy"][0] += 0.00000005
        with self.assertRaisesRegex(ValueError, "observations"):
            self.prepare()

    def test_case_duplication_missing_source_and_external_scene_refused(self):
        self.sources[0].append(self.sources[0][0])
        with self.assertRaisesRegex(ValueError, "Repeated"):
            self.prepare()
        self.sources[0].pop()
        missing = self.sources[2].pop()
        with self.assertRaisesRegex(ValueError, "identities"):
            self.prepare()
        self.sources[2].append(missing)
        self.sources[4]["unreviewed-scene"] = deepcopy(next(iter(self.sources[4].values())))
        with self.assertRaisesRegex(ValueError, "scene"):
            self.prepare()

    def test_prior_supervision_cannot_change_under_its_old_hash(self):
        old = self.sources[1][self.sources[0][0]["request_id"]]
        old.target["destination_point_xy"][0] += 1.
        with self.assertRaisesRegex(ValueError, "candidate"):
            self.prepare()


class OriginalResolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ordinary shallow CI checkouts may not include the teammate's separate unmerged source commit.
        if subprocess.run(["git","cat-file","-e",packet.TASK2+":"+packet.PACKAGE+"/tools/dropoff_resolve_reviews.py"],
                          cwd=packet.ROOT,capture_output=True).returncode:
            raise unittest.SkipTest("Fetch Task 2 commit to exercise its byte-exact original resolver")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.packet = write_fixture(self.root)

    def test_headers_blank_roles_and_raw_input_escape(self):
        baseline = form(self.packet / "reviewer_1.csv")
        self.assertEqual(baseline, form(self.packet / "reviewer_2.csv"))
        self.assertEqual(len({r["input_sha256"] for r in baseline}), 3)
        for row in baseline:
            self.assertFalse(any(row[k] for k in set(row) - set(returns.FIXED)))
        text = (self.packet / "review.html").read_text(encoding="utf-8")
        embedded = text.split('id="packetData">')[1].split("</script>", 1)[0]
        self.assertNotIn("<", embedded)
        self.assertEqual(json.loads(embedded)["scenes"][1]["request"]["dialogue"][0]["text"], "A synthetic earlier turn.")

    def test_blank_and_reordered_forms_entirely_pending(self):
        rows = form(self.packet / "reviewer_2.csv")
        reordered = self.root / "reordered.csv"
        save_form(reordered, rows[::-1])
        dest = self.root / "resolved"
        result = returns.resolve(self.packet, self.packet/"reviewer_1.csv", reordered, dest)
        self.assertEqual((result["current_cases"], result["pending_cases"], result["approved_cases_written"]), (3, 3, 0))
        self.assertFalse((dest / "approved_cases.jsonl").exists())
        with self.assertRaises(FileExistsError):
            returns.resolve(self.packet, self.packet/"reviewer_1.csv", reordered, dest)

    def test_two_mock_return_roles_pass_through_upstream_with_disagreement(self):
        archive = self.root / "fixture-private.zip"
        packet.archive(self.packet, archive)
        extracted = self.root / "extracted" / "packet"
        with ZipFile(archive) as zipped:
            zipped.extractall(extracted)
        self.packet = extracted
        paths = []
        for role in (1, 2):
            rows = form(self.packet / f"reviewer_{role}.csv")
            for i, row in enumerate(rows[:2]):
                row.update(reviewer_id=f"SOFTWARE-FIXTURE-{role}", reviewed_at="2001-01-01T09:00:00+00:00",
                           decision="accept" if i == 0 or role == 1 else "reject")
            row = rows[2]
            row.update(reviewer_id=f"SOFTWARE-FIXTURE-{role}", reviewed_at="2001-01-01T09:00:00+00:00",
                       decision="correct", corrected_status="needs_clarification", corrected_clarification_question="Which fixture?")
            path = self.root / f"return-{role}.csv"
            save_form(path, rows[::-1])
            paths.append(path)
        dest = self.root / "resolution-sibling"
        proc = subprocess.run([sys.executable, "-B", str(self.packet / "resolve_context.py"),
                               "--reviewer-1", str(paths[0]), "--reviewer-2", str(paths[1]),
                               "--output", str(dest)], capture_output=True, check=True, text=True)
        result = json.loads(proc.stdout)
        self.assertEqual((result["approved_cases_written"], result["unresolved_disagreements"]), (2, 1))
        approved = packet.json_rows((dest / "approved_cases.jsonl").read_bytes())
        self.assertEqual(approved[1]["target"]["clarification_question"], "Which fixture?")
        for case in approved:
            image = dest / case["request"]["scene"]["frame"]["image_path"]
            self.assertEqual(packet.file_sha256(image), case["source"]["image_sha256"])
            self.assertEqual(case["review"]["reviewers"], ["SOFTWARE-FIXTURE-1", "SOFTWARE-FIXTURE-2"])
            self.assertEqual(case["review"]["input_sha256"], packet.inference_input_sha256(case["request"], case["source"]["image_sha256"]))

    def test_aliased_ids_cannot_resolve_as_two_people_even_if_rows_do_not_overlap(self):
        for alias in ("  software  fixture ", "ＳＯＦＴＷＡＲＥ  FIXTURE", "SoFTwaRe\tFiXTURE"):
            with self.subTest(alias=alias):
                files = []
                for i, name in enumerate(("SOFTWARE FIXTURE", alias)):
                    rows = form(self.packet/"reviewer_1.csv")
                    rows[i].update(decision="accept", reviewer_id=name, reviewed_at="2001-01-01T09:00:00+00:00")
                    returned = self.root / f"row-{i}.csv"
                    save_form(returned, rows)
                    files.append(returned)
                with self.assertRaisesRegex(ValueError, "different people"):
                    returns.resolve(self.packet, *files, self.root/"should-refuse")

    def test_multiple_reviewers_within_one_form_refused_and_partial_return_preserved(self):
        rows = form(self.packet/"reviewer_1.csv")
        rows[0].update(decision="accept", reviewer_id="SOFTWARE fixture A", reviewed_at="2001-01-01T09:00:00+00:00")
        path = self.root/"partial.csv"
        save_form(path, rows)
        res = returns.resolve(self.packet, path, self.packet/"reviewer_2.csv", self.root/"pending")
        self.assertEqual((res["pending_cases"], res["approved_cases_written"]), (3, 0))
        rows[1].update(decision="accept", reviewer_id="SOFTWARE fixture B", reviewed_at="2001-01-01T09:00:00+00:00")
        save_form(path, rows)
        with self.assertRaisesRegex(ValueError, "only one"):
            returns.resolve(self.packet, path, self.packet/"reviewer_2.csv", self.root/"refused")

    def test_changed_return_columns_and_legacy_columns_are_rejected(self):
        first = self.packet / "reviewer_1.csv"
        for key, bad in (("destination_point_xy","[10,10]"), ("input_sha256","a"*64), ("image_path","another.png"),
                         ("legacy_decision","accept")):
            with self.subTest(field=key):
                rows = form(first)
                rows[0][key] = bad
                altered = self.root / "edited.csv"
                save_form(altered, rows)
                with self.assertRaisesRegex(ValueError, "metadata"):
                    returns.resolve(self.packet, first, altered, self.root/"no-results")
                self.assertFalse((self.root/"no-results").exists())

    def test_corrupted_picture_prevents_resolution(self):
        path = self.packet / form(self.packet/"reviewer_1.csv")[0]["image_path"]
        path.write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "contents changed"):
            returns.resolve(self.packet, self.packet/"reviewer_1.csv", self.packet/"reviewer_2.csv", self.root/"refused")

    def test_preserve_preexisting_output(self):
        with patch.object(packet, "reviewed_bank", side_effect=AssertionError("should not read")):
            with self.assertRaises(FileExistsError):
                packet.build(self.root/"missing", self.packet)


if __name__ == "__main__":
    unittest.main()
