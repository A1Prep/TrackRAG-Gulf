"""Local passenger grounding with an explicit landmark stage and learned regional extent."""
import hashlib
import json
from pathlib import Path
import re
import threading

import numpy as np

from .anchor_geometry import AnchorGeometry
from .contracts import error_outcome, validate_box, validate_request, validate_target
from .data import inference_input_sha256
from .landmarks import LandmarkResolver, landmark_messages, parse_landmark, resolve_image
from .model import ArtifactError, GroundingConfig, ResourceError, verify_adapter


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def unresolved(question=None):
    return {"status": "needs_clarification" if question else "insufficient_visual_evidence",
            "destination_point_xy": None, "destination_box_xyxy": None,
            "clarification_question": question}


def proposed(point, box, anchors=()):
    result = {"status": "proposed", "destination_point_xy": point,
              "destination_box_xyxy": box, "clarification_question": None}
    if anchors:
        result["anchor_boxes_xyxy"] = anchors
    return result


class RoadEvidence:
    """SegFormer road support from this exact photograph, at a bounded resolution."""
    def __init__(self, cache):
        self.cache = Path(cache)
        self.model = None
        self.image_hash = None
        self.road = None

    def mask(self, image, image_hash):
        if self.image_hash == image_hash and self.road is not None:
            return self.road
        from .road_cache import read_mask, save_mask
        cached = read_mask(self.cache,image_hash,image.size)
        if cached is not None:
            self.image_hash, self.road = image_hash,cached
            return cached
        import torch
        import torch.nn.functional as F
        from tools.dropoff_ground_candidates import prepare_model
        if self.model is None:
            for name, expected in {
                "config.json":"649ba4afbaa11ee730b14ad5e31bd50f2e2d5be661a291e31a91512510dba8d9",
                "preprocessor_config.json":"d96ab2cb112985c19d5ea32f1d1f2369104fc78b1379f5ce26e33ca8646202a7",
            }.items():
                if hashlib.sha256((self.cache/name).read_bytes()).hexdigest() != expected:
                    raise ArtifactError("Road segmentation configuration changed")
            if not (self.cache/"pytorch_model.bin").is_file():
                raise FileNotFoundError("Prepare the pinned road segmentation weights before starting the app")
            self.model, self.processor = prepare_model(self.cache)
        w = 512
        h = max(1, round(w*image.height/image.width))
        if h > 768:
            h, w = 768, max(1, round(768*image.width/image.height))
        inputs = self.processor(images=image, size={"height":384,"width":640}, return_tensors="pt")
        with torch.inference_mode():
            logits = F.interpolate(self.model(**inputs).logits, (h,w), mode="bilinear", align_corners=False)
            self.road = (logits.softmax(1)[0,0] >= .5).cpu().numpy()
        self.image_hash = image_hash
        save_mask(self.cache,image_hash,image.size,self.road)
        return self.road


def supported_region(mask, size, desired_point, box, unavailable, between=()):
    """Choose within the predicted coarse region, excluding current road-user silhouettes.

    Preserve the learned regional extent. Only the proposed point is certified
    as supported by this segmentation. No safe or legal stop is inferred.
    """
    box = validate_box(box, *size)
    h, w = mask.shape
    xs = (np.arange(w)+.5)*size[0]/w
    ys = (np.arange(h)+.5)*size[1]/h
    valid = mask.copy()
    valid &= (xs >= box[0]) & (xs < box[2])
    valid &= ((ys >= box[1]) & (ys < box[3]))[:,None]
    for x1,y1,x2,y2 in unavailable:
        valid &= ~(((ys >= y1) & (ys < y2))[:,None] & ((xs >= x1) & (xs < x2)))
    if between:
        # Both referenced objects must bound the selected road in the image.
        # The gap constrains support; the learned head ranks eligible pixels.
        a, b = between
        gaps = [max(b[0]-a[2],a[0]-b[2])/size[0], max(b[1]-a[3],a[1]-b[3])/size[1]]
        axis = int(np.argmax(gaps))
        if gaps[axis] <= 0:
            return None
        lo, hi = sorted((a,b),key=lambda z:z[axis])
        line = xs if axis == 0 else ys
        gap = (line >= lo[axis+2]) & (line < hi[axis])
        valid &= gap if axis == 0 else gap[:,None]
        if axis == 0:
            margin = max(a[3]-a[1], b[3]-b[1], size[1]/h)
            depth = (ys >= min(a[3],b[3])-margin) & (ys <= max(a[3],b[3])+margin)
            valid &= depth[:,None]
        else:
            lateral = (xs >= min(a[0],b[0])) & (xs < max(a[2],b[2]))
            valid &= lateral
    iy, ix = np.nonzero(valid)
    if not len(ix):
        return None
    distances = ((xs[ix]-desired_point[0])/(box[2]-box[0]))**2 + (
                 (ys[iy]-desired_point[1])/(box[3]-box[1]))**2
    best = int(distances.argmin())
    return [float(xs[ix[best]]),float(ys[iy[best]])], box


def instruction_context(request):
    turns = [t["text"] for t in request["dialogue"] if t["role"] == "user"]+[request["text"]]
    active = []
    for turn in turns:
        if re.search(r"\b(?:instead|actually|forget (?:that|it)|change (?:that|it)|not between)\b",turn,re.I):
            active.clear()
        active.append(turn)
    return " ".join(active)


def pair_required(text):
    return bool(re.search(r"\bbetween\b",text,re.I)) and not bool(
        re.search(r"\b(?:not|never|don't|do not)\s+(?:(?:stop|park|go)\s+)?between\b",text,re.I))


def operational_code(exc):
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ArtifactError):
        return "artifact_invalid"
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "dependency_unavailable"
    if isinstance(exc, FileNotFoundError):
        return "checkpoint_or_image_unavailable"
    if isinstance(exc, ResourceError) or "out of memory" in str(exc).lower():
        return "resource_unavailable"
    if isinstance(exc, ValueError):
        return "invalid_input_or_output"
    return "grounding_failed"


class PassengerGrounder:
    def __init__(self, config, geometry, road, *, verify_landmarks=False):
        self.config, self.geometry, self.road = config, geometry, road
        self.resolver = LandmarkResolver(config)
        self.verify_landmarks = verify_landmarks
        self.lock = threading.RLock()

    def _evidence(self, image, request, image_hash, desired_point, box, between=()):
        from .road_cache import ROAD_RECIPE_SHA256
        frame = request["scene"]["frame"]
        exclusions = [t["bbox_xyxy"] for t in request["scene"]["tracks"] if t["frame_id"] == frame["frame_id"]]
        mask = self.road.mask(image, image_hash)
        geometry = supported_region(mask, image.size, desired_point, box, exclusions, between)
        return geometry, {"road_recipe_sha256":ROAD_RECIPE_SHA256,
                          "road_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                          "mask_shape":list(mask.shape), "current_exclusion_boxes":exclusions}

    def interpret(self, request):
        with self.lock:
            try:
                request = validate_request(request)
                image, binding = resolve_image(request, self.config)
                with image:
                    decision = self.resolver.resolve(request,image)
                    return self.from_decision(request,image,binding,decision)
            except Exception as exc:
                return error_outcome(operational_code(exc), str(exc)[:4000] or type(exc).__name__,
                                     request.get("request_id","") if isinstance(request,dict) else "",
                                     {"method":"qwen_landmark_region"})

    def from_decision(self, request, image, binding, record):
        """Shared decision-to-proposal path. The application always calls interpret() live."""
        from tools.dropoff_ground_candidates import MODEL_ID, MODEL_REVISION, WEIGHTS_SHA256
        messages, index = landmark_messages(request)
        records_image = record.get("image_sha256") or record.get("image",{}).get("sha256")
        if (record.get("request_sha256") != digest(request) or records_image != binding["sha256"]
                or record.get("messages") != messages
                or record.get("messages_sha256") != hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()
                or record.get("model_id") != self.config.model_id or record.get("revision") != self.config.revision
                or record.get("adapter") != verify_adapter(self.config)):
            raise ValueError("The saved landmark decision differs from its scene, request or selected model")
        provenance = {"method":"qwen_landmark_region", "geometry_sha256":self.geometry.sha256,
                      "image":binding, "request_sha256":digest(request), "resolver":record,
                      "inference_input_sha256":inference_input_sha256(request,binding["sha256"]),
                      "road_model":MODEL_ID, "road_revision":MODEL_REVISION, "road_checkpoint_sha256":WEIGHTS_SHA256,
                      "road_class":"road", "road_probability_minimum":.5}
        try:
            decision = parse_landmark(record.get("raw_text"),index,request["scene"]["frame"])
        except ValueError as exc:
            return error_outcome("invalid_model_output", str(exc),
                                 request["request_id"],provenance)
        if self.verify_landmarks and decision["kind"] in ("object","landmark"):
            from .landmark_check import check_landmark
            try:
                decision, inspection = check_landmark(self.resolver,request,image,decision)
            except Exception:
                self.resolver.close()
                raise
            provenance["visual_verification"] = inspection
            if decision is None:
                return error_outcome("invalid_verification_output",inspection["parse_error"],
                                     request["request_id"],provenance)
        text = instruction_context(request)
        normalized = " ".join(re.findall(r"[a-z]+",text.lower()))
        bare = re.fullmatch(r"(?:(?:can|could|would) you )?(?:please )?(?:(?:drop|let) (?:me|us) (?:off |out )?|"
                            r"(?:stop|pull over|park) )?(?:right |over )?(?:here|there)(?: please)?",normalized)
        # These two constraints act on missing evidence. Neither assigns coordinates.
        if bare:
            provenance["constraint"] = "unpointed_deictic"
            result = unresolved("Which visible landmark is beside your drop-off, or can you mark the area on the photo?")
        elif decision["kind"] == "ask":
            result = unresolved(decision["question"])
        elif decision["kind"] == "missing":
            result = unresolved()
        else:
            boxes = decision["boxes"]
            require_pair = pair_required(text)
            if require_pair and len(boxes) != 2:
                provenance["constraint"] = "unresolved_pair"
                result = unresolved("I couldn't identify both landmarks. Which two objects bound the area you mean?")
            else:
                point, region, neighbors = self.geometry.predict_with_evidence(boxes, image.size, text)
                support, evidence = self._evidence(image,request,binding["sha256"],point,region,
                                         between=boxes if len(boxes) == 2 else ())
                provenance.update(unconstrained_point=point, unconstrained_region=region,
                                  selected_track_ids=decision["track_ids"],road_evidence=evidence,
                                  retrieved_training_examples=neighbors)
                result = (proposed(*support, anchors=boxes) if support else unresolved(
                    "I can identify the landmark, but I can't place a road area beside it. Can you mark the intended patch?"))
        validate_target(result,list(result),request["scene"]["frame"])
        return {"ok":True, "request_id":request["request_id"], "result":result, "provenance":provenance}

    def passenger_area(self, request, box):
        with self.lock:
            try:
                return self._passenger_area(request, box)
            except Exception as exc:
                return error_outcome(operational_code(exc), str(exc)[:4000] or type(exc).__name__,
                                     request.get("request_id","") if isinstance(request,dict) else "",
                                     {"method":"explicit_passenger_region"})

    def _passenger_area(self, request, box):
        request = validate_request(request)
        image, binding = resolve_image(request,self.config)
        with image:
            region = validate_box(box,*image.size)
            point = [(region[0]+region[2])/2,(region[1]+region[3])/2]
            support, evidence = self._evidence(image,request,binding["sha256"],point,region)
        result = proposed(*support) if support else unresolved(
            "The selected area has no visible road support. Can you outline a different patch?")
        return {"ok":True,"request_id":request["request_id"],"result":result,
                "provenance":{"method":"explicit_passenger_region","image":binding,"selected_box":region,
                              "road_evidence":evidence,"request_sha256":digest(request),
                              "inference_input_sha256":inference_input_sha256(request,binding["sha256"]),
                              "independent_model_proposal":False}}

    def close(self):
        with self.lock:
            self.resolver.close()


def load_passenger_model(config_path, *, geometry_path, vision_cache, verify_landmarks=False):
    config = GroundingConfig.from_mapping(json.loads(Path(config_path).read_text(encoding="utf-8-sig")))
    model = PassengerGrounder(config,AnchorGeometry.load(geometry_path),RoadEvidence(vision_cache),
                              verify_landmarks=verify_landmarks)
    return model
