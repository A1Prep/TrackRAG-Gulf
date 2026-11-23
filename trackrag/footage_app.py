"""Run the existing cited footage reviewer from a scene in the passenger application."""
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import secrets
from threading import Lock

from trackrag.dropoff.inputs import MAX_IMAGE_BYTES, _inspect_image


@dataclass
class FootageReview:
    images: Path
    predictions: Path
    vision_config: Path
    out: Path
    frame_rate: float | None = None
    _clips: dict = field(default_factory=dict, init=False, repr=False)
    _subjects: dict = field(default_factory=dict, init=False, repr=False)
    _clip_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def _clip(self, scene):
        """Cache original file locations and the release clock, never labels or generated frames."""
        from trackrag.emt import open_recording
        from tools.vision_saved_adapters import validated_frame_ids

        name = scene["frame"]["recording_id"]
        provenance_sha = scene["provenance"]["prediction_provenance_sha256"]
        source = self.predictions / f"{name}.provenance.json"
        stat = source.stat()
        key = (name, provenance_sha, stat.st_mtime_ns, stat.st_size)
        with self._clip_lock:
            if key not in self._clips:
                raw = source.read_bytes()
                stored = json.loads(raw)
                if sha256(raw).hexdigest() != provenance_sha or stored.get("recording_id") != name:
                    raise ValueError("The clip and selected recording use different prediction provenance.")
                rec = open_recording(self.images, name, frame_rate=self.frame_rate, headers_only=True)
                clock, paths = [], {}
                for original, predicted in zip(rec.frame_ids, validated_frame_ids(stored["frame_ids"])):
                    if original != predicted:
                        break  # The saved-prediction reader accepts only its exact original prefix.
                    stamp = (original - rec.first_frame_id) / rec.frame_rate
                    clock.append({"id": original, "time": stamp})
                    paths[original] = rec.frame_index[original]
                if not clock:
                    raise ValueError("No original preview frames match the saved footage.")
                self._clips[key] = (clock, paths)
            return self._clips[key]

    def timeline(self, scene):
        clock, paths = self._clip(scene)
        if scene["frame"]["frame_id"] not in paths:
            raise ValueError("The selected frame is unavailable for this recording.")
        key = (scene["frame"]["recording_id"], scene["provenance"]["tracks_sha256"],
               scene["provenance"]["prediction_provenance_sha256"])
        with self._clip_lock:
            if key not in self._subjects:
                _, tracks, _ = self._source(scene, clock[-1]["id"])
                largest = {}
                for track in tracks:
                    left, top, right, bottom = track["bbox_xyxy"]
                    area = (right-left)*(bottom-top)
                    fid = track["frame_id"]
                    if fid not in largest or area > largest[fid][0]:
                        largest[fid] = (area, track["track_id"], track["class_name"])
                self._subjects[key] = {fid: {"id": tid, "class_name": label} for fid, (_,tid,label) in largest.items()}
            subjects = self._subjects[key]
        return {"recording_id": key[0], "frames": clock, "subjects": subjects}

    def preview_frame(self, scene, frame_id, if_none_match=None):
        _, paths = self._clip(scene)
        path = paths.get(frame_id)
        if path is None:
            raise ValueError("The original frame is unavailable.")
        stat = path.stat()
        if not 0 < stat.st_size <= MAX_IMAGE_BYTES:
            raise ValueError("The original frame is unavailable.")
        etag = f'W/"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
        mime = {".jpg":"image/jpeg", ".jpeg":"image/jpeg", ".png":"image/png"}[path.suffix.lower()]
        if if_none_match == etag:
            return mime, None, etag
        with path.open("rb") as stream:
            data = stream.read(MAX_IMAGE_BYTES + 1)
        if len(data) != stat.st_size:
            raise ValueError("The original changed while reading this frame.")
        return mime, data, etag

    def _source(self, template, frame_id):
        from tools.dropoff_prepare_scene import _RecordingPrefix
        from tools.vision_saved_adapters import predict_tracks_with_receipt
        from trackrag.emt import open_recording
        from trackrag.vision.config import config_from_dict

        if type(frame_id) is not int or frame_id < 0:
            raise ValueError("Enter an original nonnegative whole frame number.")
        recording = open_recording(self.images,template["frame"]["recording_id"],
                                   frame_rate=self.frame_rate,headers_only=True)
        prefix = _RecordingPrefix(recording,frame_id)
        config = config_from_dict(json.loads(self.vision_config.read_bytes()),source_path=str(self.vision_config))
        tracks, receipt = predict_tracks_with_receipt(prefix,directory=self.predictions,config=config)
        for recorded, selected in (("vision_identity_hash","vision_identity_hash"),("tracks_sha256","tracks_sha256"),
                                   ("provenance_sha256","prediction_provenance_sha256")):
            if receipt[recorded] != template["provenance"].get(selected):
                raise ValueError("The selected scene and the footage search use different saved predictions.")
        return prefix, tracks, receipt

    def frame_scene(self, template, frame_id):
        """Construct the same passenger input as offline preparation at any available original frame."""
        from trackrag.dropoff.inputs import build_instruction_request
        prefix, tracks, receipt = self._source(template, frame_id)
        return build_instruction_request(prefix, tracks, frame_id=frame_id, request_id="open-recorded-frame",
            text="Open this recorded scene.", provenance={
                "vision_source":"validated_saved_predictions", "vision_identity_hash":receipt["vision_identity_hash"],
                "tracks_sha256":receipt["tracks_sha256"], "prediction_provenance_sha256":receipt["provenance_sha256"],
                "prediction_image_hash_binding":"not_verified_in_historical_artifact"})["scene"]

    def _bound_source(self, scene):
        original = scene["frame"]
        prefix, tracks, receipt = self._source(scene, original["frame_id"])
        current = prefix.get_frame(original["frame_id"])
        _, image_hash = _inspect_image(current)
        if (image_hash != scene["provenance"].get("image_sha256")
                or not math.isclose(current["timestamp_s"], original["timestamp_s"], abs_tol=1e-6)):
            raise ValueError("The selected photo differs from the review recording.")
        def observations(rows):
            return sorted((row["recording_id"], row["frame_id"], row["track_id"], row["source"],
                           row["class_name"], tuple(row["bbox_xyxy"])) for row in rows)
        selected = (row for row in tracks if row["frame_id"] == original["frame_id"])
        if observations(scene["tracks"]) != observations(selected):
            raise ValueError("The selected scene's observations differ from the saved predictions at this frame.")
        return prefix, tracks, receipt

    def covers(self, scene):
        """A passenger still can belong to a different run or fall outside the available footage."""
        try:
            self._bound_source(scene)
        except (OSError, KeyError, ValueError):
            return False
        return True

    def ask(self, scene, text, *, hosted=False, image_guard=None):
        from trackrag.answering import answer_question, spatial_reference_from_tracks
        from trackrag.footage_facts import answer_from_tracks, supports_track_question
        from trackrag.playback_time import playback_scope
        from trackrag.evidence_time import requested_frame_ids
        from trackrag.retrieval_temporal import build_index, rank_evidence
        from trackrag.review import review_recording

        if not isinstance(text,str) or not 1 <= len(text.strip()) <= 4096 or type(hosted) is not bool:
            raise ValueError("Enter a recording question and select an available answer mode.")
        original = scene["frame"]
        prefix, tracks, receipt = self._bound_source(scene)
        key = secrets.token_hex(16)
        folder = self.out/key
        question = {"question_id":key,"recording_id":prefix.recording_id,
                    "query_frame":original["frame_id"],"question":text.strip()}
        clock = playback_scope(text, prefix, original["frame_id"])
        clarification = None if hosted else spatial_reference_from_tracks(question, scene["tracks"], original["width"])
        def local_answer(q, evidence, _folder):
            if clarification and any(item["interval"]["track_id"] is None and
                                     any(image["frame_id"] == q["query_frame"] for image in item["images"])
                                     for item in evidence):
                return {"answer_status":"insufficient_evidence","answer":clarification,"cited_evidence_ids":[]}
            exact = requested_frame_ids(text)
            moments = clock.get("moments", []) if clock else []
            fid = moments[0]["frame_id"] if moments else exact[0] if len(exact) == 1 else original["frame_id"]
            target = prefix.get_frame(fid) if 0 <= fid <= original["frame_id"] and fid in prefix.frame_ids else original
            factual = answer_from_tracks(q, evidence, tracks, playback_scope=clock, frame_size=(target["width"], target["height"]))
            return factual or {"answer_status":"insufficient_evidence",
                              "answer":"The saved tracks and the retrieved frames do not establish that detail at the requested moment.",
                              "cited_evidence_ids":[]}

        local = not hosted and (clarification is not None or supports_track_question(text))
        answerer = "gemini" if hosted else "local_tracks" if local else "evidence_only"
        result = review_recording(prefix,[question],source="predicted",load_tracks=lambda _: tracks,
                                  build_index=build_index,rank_evidence=rank_evidence,
                                  answer_question=answer_question if hosted else local_answer if local else None,output_dir=folder,
                                  image_guard=image_guard,
                                  run_metadata={"vision":"validated_saved_predictions","prediction_receipt":receipt,
                                                "retrieval":"trackrag.retrieval_temporal","boundary_answer":"visual-candidate-v1",
                                                "answerer":answerer})
        row = json.loads(result.read_bytes())
        can_answer_from_tracks = (hosted and row["answer"]["answer_status"] == "error"
                                  and supports_track_question(text)
                                  and local_answer(question, row["evidence"], folder)["answer_status"] == "answered")
        evidence = []
        for number, item in enumerate(row["evidence"]):
            images = [{"frame_id":im["frame_id"],"url":f"/api/footage/{key}/{number}/{i}"}
                      for i,im in enumerate(item["images"])]
            evidence.append({"id":item["evidence_id"],"interval":item["interval"],"images":images})
        return {"question":question,"answer":row["answer"],"evidence":evidence,"hosted_requested":hosted,
                "answer_source":answerer,"can_answer_from_tracks":can_answer_from_tracks,"playback_scope":clock,
                "vision":"validated_saved_predictions"}

    def evidence_image(self,key,rank,index):
        if re.fullmatch(r"[0-9a-f]{32}",key) is None or rank not in range(3) or index not in range(32):
            raise ValueError("Unknown evidence image.")
        folder = (self.out/key).resolve()
        row = json.loads((folder/"results.jsonl").read_bytes())
        try:
            relative = row["evidence"][rank]["images"][index]["image_path"]
        except IndexError as exc:
            raise ValueError("Unknown evidence image.") from exc
        path = (folder/relative).resolve()
        if not path.is_relative_to(folder/"evidence") or not path.is_file():
            raise ValueError("The saved evidence image is unavailable.")
        return ("image/png" if path.suffix.lower()==".png" else "image/jpeg"), path.read_bytes()
