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
from .pair_resolution import PairResolutionError, collect_pair_evidence, replay_pair_decision


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
    if isinstance(exc, PairResolutionError):
        return exc.code
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ArtifactError):
        return "artifact_invalid"
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "dependency_unavailable"
    if isinstance(exc, FileNotFoundError):
        return "checkpoint_or_image_unavailable"
    message = str(exc).lower()
    gpu_failure = isinstance(exc, RuntimeError) and any(detail in message for detail in (
        "cublas_status_not_supported", "cuda error: unknown error", "bad allocation"
    ))
    if isinstance(exc, (ResourceError, MemoryError)) or "out of memory" in message or gpu_failure:
        return "resource_unavailable"
    if isinstance(exc, ValueError):
        return "invalid_input_or_output"
    return "grounding_failed"


def interactive_timeout(request):
    return error_outcome("timeout", "The automatic photo check took too long. You can mark your spot or try a different description.",
                         request["request_id"], {"method":"passenger_interactive_deadline", "request_sha256":digest(request),
                                                 "image":{"sha256":request["scene"]["provenance"]["image_sha256"]}})


class PassengerGrounder:
    requires_verified_manual_surface = True

    def __init__(self, config, geometry, road, *, verify_landmarks=False, referent_ranker=None, between_road_mode="edge"):
        from .between_policy import check_mode
        self.config, self.geometry, self.road = config, geometry, road
        self.referent_ranker = referent_ranker
        self.between_road_mode = check_mode(between_road_mode)
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
            binding = None
            try:
                request = validate_request(request)
                image, binding = resolve_image(request, self.config)
                with image:
                    decision = self.resolver.resolve(request,image)
                    pair = collect_pair_evidence(request,image,self.resolver)
                    if pair is not None:
                        decision["pair_resolution"] = pair
                    return self.from_decision(request,image,binding,decision)
            except Exception as exc:
                code = operational_code(exc)
                provenance = {"method":"qwen_landmark_region"}
                if code == "resource_unavailable" and binding:
                    provenance.update(runtime_resource_failure=True, request_sha256=digest(request),
                                      image={"sha256":binding["sha256"]})
                return error_outcome(code, str(exc)[:4000] or type(exc).__name__,
                                     request.get("request_id","") if isinstance(request,dict) else "",
                                     provenance)

    def from_decision(self, request, image, binding, record):
        """Shared decision-to-proposal path. The application always calls interpret() live."""
        from . import between_road
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
            if "pair_resolution" in record:
                decision = replay_pair_decision(request,binding,record["pair_resolution"],self.config)
                provenance["pair_decision"] = decision
            else:
                decision = parse_landmark(record.get("raw_text"),index,request["scene"]["frame"])
        except PairResolutionError as exc:
            return error_outcome(exc.code,str(exc),request["request_id"],provenance)
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
        if self.referent_ranker is not None and decision["kind"] in ("object","landmark") and not bare and not pair_required(text):
            from .referent_policy import clarify_reference
            scene = request["scene"]
            ranking = self.referent_ranker.rank(image,text,scene["tracks"],frame=scene["frame"])
            clarification = clarify_reference(text,ranking,image.size)
            provenance["referent_ranking"] = ranking
            if clarification:
                provenance["referent_clarification"] = clarification
                decision = {"kind":"ask","question":clarification["question"]}
        if bare:
            provenance["constraint"] = "unpointed_deictic"
            result = unresolved("Which visible landmark is beside your drop-off, or can you mark the area on the photo?")
        elif decision["kind"] == "ask":
            result = unresolved(decision["question"])
        elif decision["kind"] == "missing":
            result = unresolved()
        elif between_road.requested(request):
            from .landmarks import InvalidLandmarkIDs
            try:
                result, check = between_road.native_result(self, request, image, binding, decision, record, index)
            except InvalidLandmarkIDs as exc:
                # A valid subquery pair cannot rescue a malformed whole-sentence reading.
                # Preserve the same error and all three real replies for inspection.
                provenance["rejected_landmark_stage"] = "whole_instruction_pair_consistency"
                return error_outcome(operational_code(exc), str(exc), request["request_id"], provenance)
            provenance["between_road"] = check
            provenance["selected_track_ids"] = check["resolved_track_ids"]
        elif between_road.unbound_vehicle_selection(request,decision):
            result = unresolved(between_road.WHICH_PAIR)
            provenance["constraint"] = "unbound_vehicle_between_reference"
            provenance["selected_track_ids"] = decision["track_ids"]
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
        from . import manual_offlane
        try:
            return manual_offlane.select(request, box)
        except Exception as exc:
            return error_outcome(operational_code(exc), str(exc)[:4000] or type(exc).__name__,
                                 request.get("request_id","") if isinstance(request,dict) else "",
                                 {"method":manual_offlane.METHOD})

    def close(self):
        with self.lock:
            self.resolver.close()


def load_passenger_model(config_path, *, geometry_path, vision_cache, verify_landmarks=False,referent_path=None,
                         between_road_mode="edge"):
    config = GroundingConfig.from_mapping(json.loads(Path(config_path).read_text(encoding="utf-8-sig")))
    ranker = None
    if referent_path is not None:
        from .referent import ReferentRanker
        ranker = ReferentRanker.load(referent_path)
    model = PassengerGrounder(config,AnchorGeometry.load(geometry_path),RoadEvidence(vision_cache),
                              verify_landmarks=verify_landmarks,referent_ranker=ranker,between_road_mode=between_road_mode)
    return model
