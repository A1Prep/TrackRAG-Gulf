"""Reproduce an offered two-vehicle region from its original sensor and fitted-model evidence."""
from hashlib import sha256
import json

from . import between_policy, between_road, projected_pair
from .landmarks import landmark_messages, parse_landmark
from .model import verify_adapter
from .pair_resolution import replay_pair_decision
from .passenger import digest
from .structured import LANDMARK_CONTRACT


def _native_selection(model, request, outcome, image, binding):
    info = outcome["provenance"]
    record = info["resolver"]
    messages, index = landmark_messages(request)
    frame = request["scene"]["frame"]
    if (info.get("image") != binding or record.get("image_sha256") != binding["sha256"]
            or record.get("request_sha256") != digest(request) or record.get("messages") != messages
            or record.get("messages_sha256") != sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()
            or record.get("model_id") != model.config.model_id or record.get("revision") != model.config.revision
            or record.get("adapter") != verify_adapter(model.config) or record.get("contract") != LANDMARK_CONTRACT):
        raise ValueError("Between-vehicle resolver receipt differs from the request, original or selected model")
    projected_pair.native_record(record, model.config)
    selected = parse_landmark(record["raw_text"], index, frame)
    if selected != record.get("decision") or "parse_error" in record:
        raise ValueError("Recorded pair differs from the raw landmark reply")
    if "pair_resolution" in record:
        pair = record["pair_resolution"]
        selected = replay_pair_decision(request,binding,pair,model.config)
        if selected != info.get("pair_decision"):
            raise ValueError("Between-vehicle description subqueries disagree with the stored decision")
        for item in pair["subqueries"]:
            projected_pair.native_record(item["resolver"], model.config)
    check = info.get("visual_verification")
    if check is not None:
        chosen = {key:selected[key] for key in ("kind","boxes","track_ids")}
        projected_pair.verify_crop_receipt(request,image,chosen,check)
        projected_pair.native_record(check,model.config)
        raw = check["raw_text"].strip()
        if raw != "MATCH":
            context = json.loads(check["messages"][1]["content"][1]["text"])
            candidates = {key:t for key,t in index.items() if key in context["visible_alternative_ids"]
                          or t["track_id"] in selected["track_ids"]}
            selected = parse_landmark(raw,candidates,frame)
    return selected, record, index


def reproduce(model, request, outcome, actor, image, binding, mask, selected, tracks, patch):
    if actor == "model":
        decision, record, index = _native_selection(model,request,outcome,image,binding)
        target, evidence = between_road.native_result(model,request,image,binding,decision,record,index)
        if target != outcome["result"] or evidence != outcome["provenance"]["between_road"]:
            raise ValueError("The between-vehicle offer is not the fitted winner for the raw native selection")
    else:
        expected = between_policy.estimate(model,mask,image,selected,tracks,request["text"],image_sha=binding["sha256"])
        if expected != patch:
            raise ValueError("The passenger-selected between-vehicle offer is not the fitted winner")
