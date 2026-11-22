"""Offer the only supported visible vehicle gap when an initial collective request is ambiguous."""

from itertools import combinations

from . import between_policy, between_road
from .between_audit import _native_selection
from .contracts import validate_request
from .data import inference_input_sha256
from .landmarks import resolve_image
from .pair_dialogue import VEHICLE_CLASSES, dark_pair_evidence, unpointed_pair, unpointed_reference
from .passenger import digest, proposed
from .passenger_pair_choice import current_tracks
from .projected_pair import GEOMETRY, Ineligible
from .referent import ReferentRanker


METHOD = "unique_road_qualified_vehicle_pair"
ACTOR = "road_qualified_pair"
RANKER = "d96a9a69e2e52ddbc21851d5ff6f79b344de6c8558600e54be5cb2712480a07d"
MAX_CANDIDATES = 8  # At most 28 complete source-bound pair checks; never discard rivals to fit the bound.


def _original_selection(model, request, original, image, binding):
    """Retain a real clarification and, if Qwen identified one vehicle, bind that identity to its raw reply."""
    info = original["provenance"]
    if any(section.get(flag) not in (None,False) for section in (original,info)
           for flag in ("injected","substitute","software_fixture","fixture","test_fixture")):
        return None
    if info.get("method") == "deictic_pair_clarification":
        if original != unpointed_pair(model, request):
            raise ValueError("The original pair clarification cannot be reproduced")
        return []
    if (info.get("method") != "qwen_landmark_region" or info.get("between_road", {}).get("withheld_reason")
            != "two_distinct_current_vehicles_required"):
        return None
    decision, record, index = _native_selection(model, request, original, image, binding)
    ids = decision.get("track_ids", [])
    if decision.get("kind") != "landmark" or len(ids) != 1:
        return None
    expected, audit = between_road.native_result(model,request,image,binding,decision,record,index)
    if expected != original["result"] or audit != info["between_road"] or ids != info.get("selected_track_ids"):
        raise ValueError("The original one-vehicle reading differs from the native response")
    return ids


def _appearance(model, request, image, tracks, noun, colour):
    candidates = [t for t in tracks if t["class_name"] in VEHICLE_CLASSES[noun]]
    ids = {t["track_id"] for t in candidates}
    if not colour:
        return candidates, ids, {"colour":None, "class_candidates":sorted(ids)}
    brightness = dark_pair_evidence(image, candidates)
    possible = set(brightness["possible_tracks"])
    definite = {v["track_id"] for v in brightness["readings"] if v["appearance"] == "dark"}
    evidence = {"colour":colour, "brightness":brightness}
    if colour == "black":
        # Keep rivals with uncertain paint. Only a verified bright vehicle can be eliminated as too light.
        rank = model.referent_ranker.rank(image,request["text"],tracks,frame=request["scene"]["frame"])
        definite &= {v["track_id"] for v in rank["alternatives"] if
                     min(v["colour_fractions"]["black"],v["central_colour_fractions"]["black"]) >= .20}
        evidence["colour_ranking_sha256"] = digest(rank)
    return [t for t in candidates if t["track_id"] in possible], definite, evidence


def search(model, request, original):
    """Return a complete proposed outcome, or None when zero or several pairs could ground the request."""
    if (between_policy.mode(model) != between_policy.EDGE_OR_BAY or not isinstance(original,dict)
            or original.get("ok") is not True or original.get("result",{}).get("status") != "needs_clarification"
            or not isinstance(request,dict) or request.get("dialogue") or request.get("scene",{}).get("source") != "predicted"
            or original.get("request_id") != request.get("request_id")):
        return None
    request = validate_request(request)
    words = unpointed_reference(request["text"])
    if not words:
        return None
    colour, noun = (words["colour"] or "").lower(), words["noun"].lower()
    # Current detector categories cannot establish sedan vs SUV or a vehicle's parking state.
    if noun not in {"cars", "vehicles", "buses"} or colour not in {"", "dark", "black"}:
        return None
    origin=original.get("provenance",{})
    if not (origin.get("method")=="deictic_pair_clarification" or (origin.get("method")=="qwen_landmark_region"
            and origin.get("between_road",{}).get("withheld_reason")=="two_distinct_current_vehicles_required")):
        return None
    if not isinstance(getattr(model,"referent_ranker",None),ReferentRanker):
        raise FileNotFoundError("A verified referring-object ranker is required to compare the complete candidate set")
    if model.geometry.sha256 != GEOMETRY or model.referent_ranker.sha256 != RANKER:
        raise ValueError("Vehicle-pair evidence uses unregistered region or referring-object weights")
    image, binding = resolve_image(request,model.config)
    with image:
        native_ids = _original_selection(model,request,original,image,binding)
        if native_ids is None:
            return None
        tracks = current_tracks(request)
        candidates, definite, appearance = _appearance(model,request,image,tracks,noun,colour)
        if not (2 <= len(candidates) <= MAX_CANDIDATES) or len(definite) < 2:
            return None
        mask = model.road.mask(image,binding["sha256"])
        qualified, attempts = [], []
        for pair in combinations(candidates,2):
            ids = [row["track_id"] for row in pair]
            try:
                between_policy.preflight(model,pair,image.size)
                patch = between_policy.estimate(model,mask,image,pair,tracks,request["text"],image_sha=binding["sha256"])
            except Ineligible as error:
                if str(error) in {"unsupported_road_mask","unsupported_eligibility_mask","no_prediction_from_trained_pair_head"}:
                    raise ValueError("Cannot compare candidate pairs using an invalid road mask or fitted estimate") from error
                attempts.append({"track_ids":ids, **between_policy.refusal(error)})
            else:
                # An uncertain-looking pair still competes. Never promote the first paint match or highest-ranked rival.
                qualified.append((pair, patch))
                attempts.append({"track_ids":ids, "road_guard":patch["guard"], "patch_sha256":digest(patch)})
        if len(qualified) != 1:
            return None
        selected, patch = qualified[0]
        selected_ids = [t["track_id"] for t in selected]
        if not set(selected_ids) <= definite or not set(native_ids) <= set(selected_ids):
            return None
        target = proposed(patch["point"],patch["box"],[t["bbox_xyxy"] for t in selected])
        info = {"method":METHOD, "between_road_mode":between_policy.EDGE_OR_BAY, "image":binding,
                "request_sha256":digest(request), "inference_input_sha256":inference_input_sha256(request,binding["sha256"]),
                "current_observations_sha256":digest(tracks), "geometry_sha256":model.geometry.sha256,
                "ranker_sha256":model.referent_ranker.sha256, "appearance":appearance,
                "definite_appearance_ids":sorted(definite), "possible_candidate_ids":[t["track_id"] for t in candidates],
                "maximum_candidates":MAX_CANDIDATES, "evaluated_pairs":attempts, "selected_track_ids":selected_ids,
                "road_geometry":patch, "retained_native_track_ids":native_ids, "original_outcome":original,
                "original_outcome_sha256":digest(original), "additional_native_calls":0, "independent_native_proposal":False}
        return {"ok":True,"request_id":request["request_id"],"result":target,"provenance":info}


def review(model, request, original):
    """Preserve the existing clarification when exhaustive, verified geometric disambiguation is unavailable."""
    if between_policy.mode(model) != between_policy.EDGE_OR_BAY:
        return original
    try:
        with model.lock:
            return search(model,request,original) or original
    except (AttributeError, ImportError, OSError, ValueError, TypeError, KeyError, IndexError, RuntimeError) as exc:
        if not isinstance(original,dict) or original.get("result",{}).get("status") != "needs_clarification":
            raise
        # A disk/model/source error is auditable separately from an ordinary absent or ambiguous gap.
        code = ("optional_evidence_unavailable" if isinstance(exc,(ImportError,OSError,RuntimeError))
                else "optional_evidence_invalid")
        details = {"code":code,"exception_type":type(exc).__name__,"message":str(exc)[:300],
                   "original_outcome_sha256":digest(original),"additional_native_calls":0}
        return {**original,"provenance":{**original["provenance"],"qualified_pair_failure":details}}


def original_after_failure(outcome):
    """Recover an unchanged initial clarification for exact legacy replay while retaining its optional failure."""
    source=outcome.get("provenance",{})
    if "qualified_pair_failure" not in source:
        return outcome
    failure=source["qualified_pair_failure"]
    original={**outcome,"provenance":{k:v for k,v in source.items() if k != "qualified_pair_failure"}}
    if (not isinstance(failure,dict) or set(failure)!={"code","exception_type","message","original_outcome_sha256","additional_native_calls"}
            or failure.get("code") not in {"optional_evidence_unavailable","optional_evidence_invalid"}
            or not isinstance(failure.get("message"),str) or len(failure["message"])>300
            or not isinstance(failure.get("exception_type"),str) or not 0<len(failure["exception_type"])<=80
            or type(failure.get("additional_native_calls")) is not int or failure["additional_native_calls"]!=0
            or failure.get("original_outcome_sha256") != digest(original) or original.get("result",{}).get("status")!="needs_clarification"):
        raise ValueError("Optional road-pair failure does not preserve its original clarification")
    return original


def validate(model, request, outcome):
    """Before a confirmation token, reproduce the original reading, every competitor and the complete winning area."""
    origin = outcome["provenance"].get("original_outcome")
    if not isinstance(origin,dict) or outcome["provenance"].get("original_outcome_sha256") != digest(origin):
        raise ValueError("Road-qualified proposal has a different original response")
    with model.lock:
        expected = search(model,request,origin)
    if expected is None or expected != outcome:
        raise ValueError("Road-qualified vehicle pair did not reproduce from its current photo, alternatives and original response")


def replay_files(request, outcome, geometry_path, road_cache, referent_path, *, config=None):
    """Cold-verify without GPU or generation, using only the released local weights and original-bound segmentation caches."""
    import json
    from pathlib import Path
    from threading import RLock
    from types import SimpleNamespace
    from .anchor_geometry import AnchorGeometry
    from .model import GroundingConfig
    from .road_cache import read_mask

    geometry_path, road_cache = Path(geometry_path), Path(road_cache)
    request = validate_request(request)
    image_sha = request["scene"]["provenance"]["image_sha256"]

    def saved_road(image, fingerprint):
        if fingerprint != image_sha:
            raise ValueError("Cold verification tried to use a different original image")
        mask = read_mask(road_cache, fingerprint, image.size)
        if mask is None:
            raise FileNotFoundError("The original-bound saved road mask is required to verify the selected pair")
        return mask

    origin = outcome.get("provenance",{}).get("original_outcome",{}).get("provenance",{})
    if config is None and origin.get("method") == "qwen_landmark_region":
        config = geometry_path.with_name("grounding.json")
        if not config.is_file():
            raise FileNotFoundError("Cold verification of the earlier native turn requires its exact grounding config")
    if not isinstance(config,GroundingConfig):
        config = (GroundingConfig.from_mapping(json.loads(Path(config).read_text(encoding="utf-8-sig")))
                  if config is not None else GroundingConfig())
    model = SimpleNamespace(config=config, geometry=AnchorGeometry.load(geometry_path,expected_sha=GEOMETRY),
                            referent_ranker=ReferentRanker.load(referent_path,expected_sha256=RANKER),
                            road=SimpleNamespace(cache=road_cache,image_hash=image_sha,mask=saved_road),
                            between_road_mode=between_policy.EDGE_OR_BAY,lock=RLock())
    validate(model,request,outcome)
    info = outcome["provenance"]
    return {"current_image_and_tracks_checked":True,"all_candidate_pairs_recomputed":len(info["evaluated_pairs"]),
            "selected_track_ids":info["selected_track_ids"],"learned_geometry_and_mask_replayed":True,
            "decision_reproduced":True,"ranker_sha256":model.referent_ranker.sha256,
            "fitted_geometry_sha256":model.geometry.sha256,"additional_native_calls":0}
