"""Run the existing cited footage reviewer from a scene in the passenger application."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import secrets

from trackrag.dropoff.inputs import _inspect_image


@dataclass
class FootageReview:
    images: Path
    predictions: Path
    vision_config: Path
    out: Path
    frame_rate: float | None = None

    def ask(self, scene, text, *, hosted=False):
        from tools.dropoff_prepare_scene import _RecordingPrefix
        from tools.vision_saved_adapters import predict_tracks_with_receipt
        from trackrag.answering import answer_question
        from trackrag.emt import open_recording
        from trackrag.retrieval import build_index, rank_evidence
        from trackrag.review import review_recording
        from trackrag.vision.config import config_from_dict

        if not isinstance(text,str) or not 1 <= len(text.strip()) <= 4096 or type(hosted) is not bool:
            raise ValueError("Enter a recording question and select an available answer mode.")
        original = scene["frame"]
        recording = open_recording(self.images,original["recording_id"],frame_rate=self.frame_rate)
        prefix = _RecordingPrefix(recording,original["frame_id"])
        current = prefix.get_frame(original["frame_id"])
        _, image_hash = _inspect_image(current)
        if (image_hash != scene["provenance"].get("image_sha256")
                or not math.isclose(current["timestamp_s"],original["timestamp_s"],abs_tol=1e-6)):
            raise ValueError("The selected photo differs from the review recording.")
        config = config_from_dict(json.loads(self.vision_config.read_bytes()),source_path=str(self.vision_config))
        tracks, receipt = predict_tracks_with_receipt(prefix,directory=self.predictions,config=config)
        for recorded, selected in (("vision_identity_hash","vision_identity_hash"),("tracks_sha256","tracks_sha256"),
                                   ("provenance_sha256","prediction_provenance_sha256")):
            if receipt[recorded] != scene["provenance"].get(selected):
                raise ValueError("The selected scene and the footage search use different saved predictions.")
        key = secrets.token_hex(16)
        folder = self.out/key
        question = {"question_id":key,"recording_id":recording.recording_id,
                    "query_frame":original["frame_id"],"question":text.strip()}
        result = review_recording(prefix,[question],source="predicted",load_tracks=lambda _: tracks,
                                  build_index=build_index,rank_evidence=rank_evidence,
                                  answer_question=answer_question if hosted else None,output_dir=folder,
                                  run_metadata={"vision":"validated_saved_predictions","prediction_receipt":receipt,
                                                "retrieval":"trackrag.retrieval","answerer":"gemini" if hosted else "none"})
        row = json.loads(result.read_bytes())
        evidence = []
        for number, item in enumerate(row["evidence"]):
            images = [{"frame_id":im["frame_id"],"url":f"/api/footage/{key}/{number}/{i}"}
                      for i,im in enumerate(item["images"])]
            evidence.append({"id":item["evidence_id"],"interval":item["interval"],"images":images})
        return {"question":question,"answer":row["answer"],"evidence":evidence,"hosted_requested":hosted,
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
