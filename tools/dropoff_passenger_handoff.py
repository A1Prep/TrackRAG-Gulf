"""Export composite passenger-run evidence without creating a native Task 1 model condition."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from tools.dropoff_predict import read_requests
from trackrag.dropoff.contracts import FULL_FIELDS, validate_request, validate_target
from trackrag.dropoff.data import file_sha256, inference_input_sha256
from trackrag.dropoff.passenger import digest


def need(ok, message):
    if not ok:
        raise ValueError(message)


def records(path):
    """Keep the original line hash as well as its decoded contents."""
    return [(sha256(line).hexdigest(), json.loads(line)) for line in path.read_bytes().splitlines() if line.strip()]


def bound_outcome(outcome, request, *, passenger_selected=False, passenger_landmarks=False, dialogue_policy=False, requested_box=None,
                  referent_path=None, pair_geometry=None, road_cache=None, passenger_requested_pointing=False,
                  passenger_requested_area=False, parking_artifacts=None, selected_bay_id=None, bay_source_sha=None):
    need(isinstance(outcome, dict) and outcome.get("request_id") == request["request_id"], "Mismatched outcome ID")
    source = outcome.get("provenance", {})
    need(isinstance(source, dict), "Missing outcome provenance")
    from trackrag.dropoff import manual_mark, manual_offlane, whole_bay
    if passenger_requested_area or source.get("method") == manual_mark.REQUEST_METHOD:
        need(passenger_requested_area and dialogue_policy and
             not (passenger_selected or passenger_landmarks or passenger_requested_pointing) and
             outcome == manual_mark.question(request), "Area-mark request differs from the passenger action or original frame")
        return {"ok":True, "request_id":request["request_id"], "method":manual_mark.REQUEST_METHOD,
                "image_sha256":source["image"]["sha256"], "inference_input_sha256":source["inference_input_sha256"],
                "saved_provenance":source, "result":outcome["result"], "independent_model_proposal":False,
                "native_model_calls":0}
    method = source.get("method")
    if method in whole_bay.MANUAL_METHODS:
        need(passenger_selected and not (passenger_landmarks or dialogue_policy),
             "An off-lane mark requires a passenger selection")
        if outcome.get("ok") is True:
            if method == whole_bay.CHOICE:
                need(selected_bay_id and selected_bay_id == source.get("bay_id") and bay_source_sha and
                     bay_source_sha == digest(source.get("choice_source")), "Bay selection differs from the preceding pictured options")
            else:
                need(requested_box is not None and selected_bay_id is None, "An off-lane mark requires the passenger's submitted rectangle")
            check = manual_offlane.verify if method == manual_offlane.METHOD else whole_bay.verify
            checked = check(request, outcome, requested_box=requested_box)
            return {"ok":True, "request_id":request["request_id"], "method":method,
                    "image_sha256":source["image"]["sha256"], "inference_input_sha256":source["inference_input_sha256"],
                    "saved_provenance":source, "result":outcome["result"], "passenger_mark_verification":checked,
                    "native_model_calls":0, "independent_model_proposal":False}
    from trackrag.dropoff import passenger_parking as parking
    from trackrag.dropoff import parking_clarification
    from trackrag.dropoff.interaction import signed_place_verifier
    place_verifier = signed_place_verifier(source.get("method"))
    if place_verifier is not None:
        need(dialogue_policy and not (passenger_selected or passenger_landmarks or passenger_requested_pointing),
             "A place-model decision belongs to a normal submitted request")
    if outcome.get("ok") and place_verifier is not None:
        checked = place_verifier(request, outcome)
        return {"ok":True,"method":source["method"],"request_id":request["request_id"],"result":outcome["result"],
                "image_sha256":source["image"]["sha256"], "inference_input_sha256":source["inference_input_sha256"],
                "saved_provenance":source, "place_verification":checked, "native_model_calls":checked["language_calls"],
                "independent_model_proposal":checked["independent_model_proposal"]}
    if source.get("method") == parking_clarification.METHOD:
        need(dialogue_policy and not (passenger_selected or passenger_landmarks or passenger_requested_pointing),
             "A reference clarification belongs to a normal submitted passenger request")
        checks = parking_clarification.verify(request, outcome)
        return {"ok":True, "method":source["method"], "result":outcome["result"],
                "request_id":request["request_id"], "image_sha256":source["image"]["sha256"],
                "inference_input_sha256":source["inference_input_sha256"], "saved_provenance":source,
                "reference_verification":checks, "native_model_calls":0, "independent_model_proposal":False}
    if source.get("method") in {parking.METHOD, parking.FAILURE_METHOD}:
        need(dialogue_policy and not (passenger_selected or passenger_landmarks or passenger_requested_pointing),
             "A parking decision belongs only to a normal submitted passenger instruction")
        checks = parking.verify_response(request, outcome, artifacts=parking_artifacts)
        data = {"ok":outcome["ok"], "request_id":request["request_id"], "method":source["method"],
                "image_sha256":source["image"]["sha256"], "inference_input_sha256":source["inference_input_sha256"],
                "independent_model_proposal":checks["independent_model_proposal"], "parking_verification":checks,
                "saved_provenance":source}
        field = "result" if outcome["ok"] else "error"
        return {**data, field:outcome[field]}
    if outcome.get("ok") is False:
        from tools.passenger_place_call import UNAVAILABLE
        if source.get("method") == UNAVAILABLE:
            from trackrag.dropoff.place_evidence import verify_unavailable
            need(dialogue_policy and not (passenger_selected or passenger_landmarks or passenger_requested_pointing),
                 "A place service failure belongs to the automatic submission")
            return {"ok":False, "error":outcome["error"], "request_id":request["request_id"], "method":UNAVAILABLE,
                    "saved_provenance":source, "place_verification":verify_unavailable(request, outcome),
                    "independent_model_proposal":False}
        if source.get("method") in {"explicit_passenger_landmark_pair", "explicit_passenger_region"}:
            need((source["method"] == "explicit_passenger_landmark_pair" and passenger_landmarks) or
                 (source["method"] == "explicit_passenger_region" and passenger_selected),
                 "An assisted failure was supplied as a different interaction role")
        return {"ok": False, "error": outcome.get("error"), "method": source.get("method"),
                "saved_provenance": source}
    need(outcome.get("ok") is True, "Unrecognized outcome type")
    frame = request["scene"]["frame"]
    image_sha = request["scene"]["provenance"]["image_sha256"]
    need(source.get("image", {}).get("sha256") == image_sha and file_sha256(frame["image_path"]) == image_sha,
         "Outcome or original image does not match the requested scene")
    need(source.get("request_sha256") == digest(request), "Outcome does not match the exact request")
    portable = inference_input_sha256(request, image_sha)
    if "inference_input_sha256" in source:
        need(source["inference_input_sha256"] == portable, "Outcome has a mismatched portable input binding")
    target = outcome["result"]
    need(FULL_FIELDS.issubset(target), "Incomplete passenger interpretation")
    target = validate_target(target, list(target), frame)
    method = source.get("method")
    record = {"ok": True, "method": method, "result": target, "request_id": request["request_id"],
              "inference_input_sha256": portable, "image_sha256": image_sha, "source_scene": request["scene"]["provenance"]}
    if passenger_landmarks:
        from trackrag.dropoff.passenger_pair_choice import replay_choice_evidence, validate_choice_evidence
        selection = source.get("passenger_selection") or {}
        validate_choice_evidence(request, outcome, selection.get("track_ids"), selection.get("placement"),
                                 passenger_requested=passenger_requested_pointing)
        need((pair_geometry is None) == (road_cache is None), "Supply both the fitted pair geometry and current road-mask cache for full replay")
        record["passenger_geometry_verification"] = (replay_choice_evidence(request, outcome, pair_geometry, road_cache,
                passenger_requested=passenger_requested_pointing)
                if pair_geometry is not None else {"current_image_and_tracks_checked":True,
                    "learned_geometry_and_mask_replayed":False,"scope":"Journal evidence only; road-mask bytes and fitted patch were not independently replayed."})
    elif passenger_selected:
        from trackrag.dropoff.manual_mark import verify
        need(method == "explicit_passenger_region" and source.get("independent_model_proposal") is False,
             "Passenger marking lacks its manual-intervention declaration")
        record["passenger_mark_verification"] = verify(request,outcome,requested_box=requested_box,road_cache=road_cache)
    elif dialogue_policy and passenger_requested_pointing:
        from trackrag.dropoff.passenger_pair_choice import pointing_question
        need(outcome == pointing_question(request), "Pointing prompt differs from the passenger action or original frame")
        record["native_model_calls"] = 0
    elif method == "unique_road_qualified_vehicle_pair" and dialogue_policy:
        from trackrag.dropoff.qualified_pair import replay_files
        need(pair_geometry is not None and road_cache is not None,
             "Supply the fitted geometry and current segmentation cache to reproduce the unique vehicle-pair proposal")
        weights = referent_path if referent_path is not None else Path(pair_geometry).with_name("referent.json")
        record["pair_policy_verification"] = replay_files(request, outcome, pair_geometry, road_cache, weights)
        record["independent_native_proposal"] = False
    elif method == "deictic_pair_clarification" and dialogue_policy:
        from types import SimpleNamespace
        from trackrag.dropoff.model import GroundingConfig
        from trackrag.dropoff.pair_dialogue import unpointed_pair
        from trackrag.dropoff.qualified_pair import original_after_failure
        from trackrag.dropoff.referent import ReferentRanker
        weights = Path(referent_path) if referent_path is not None else Path(__file__).resolve().parents[1] / "models" / "passenger-feature" / "referent.json"
        ranker = ReferentRanker.load(weights, expected_sha256=source.get("ranker_sha256"))
        reproduced = unpointed_pair(SimpleNamespace(config=GroundingConfig(), referent_ranker=ranker), request)
        need(reproduced == original_after_failure(outcome) and target["status"] == "needs_clarification" and source.get("native_model_calls") == 0,
             "Automatic pair-identification question differs from the exact original image, passenger words, ranks or current tracks")
        record["native_model_calls"] = 0
        record["policy_recomputed_from_original"] = True
    else:
        need(method == "qwen_landmark_region" and source.get("independent_model_proposal") is not False
             and source.get("injected") is not True,
             "An assisted, injected or unknown method was supplied as a model attempt")
        resolver = source.get("resolver", {})
        raw = resolver.get("raw_text")
        need(isinstance(raw, str) and resolver.get("request_sha256") == source["request_sha256"],
             "Missing or mismatched raw landmark response")
        need((resolver.get("image_sha256") or resolver.get("image", {}).get("sha256")) == image_sha,
             "Raw landmark response belongs to another image")
        model = resolver.get("backend_details", {})
        need(isinstance(resolver.get("model_id"), str) and isinstance(resolver.get("revision"), str)
             and model.get("model_id") == resolver["model_id"] and model.get("revision") == resolver["revision"]
             and model.get("runtime") == "streamed", "Incomplete local resolver runtime identity")
        messages_sha = sha256(json.dumps(resolver["messages"], sort_keys=True).encode()).hexdigest()
        need(resolver.get("messages_sha256") == messages_sha, "Resolver prompt differs from its hash")
        record["landmark_raw_text_sha256"] = sha256(raw.encode()).hexdigest()
        if "visual_verification" in source:
            inspection_raw = source["visual_verification"].get("raw_text")
            need(isinstance(inspection_raw, str), "Missing visual-verifier response")
            record["verification_raw_text_sha256"] = sha256(inspection_raw.encode()).hexdigest()
    # Full original provenance includes weights, prompts, raw response, optional visual check and road evidence.
    record["saved_provenance"] = source
    return record


def evidence_head():
    return {"schema_version": 1, "kind": "passenger_composite_evidence",
            "native_task1_condition": False, "scoring_references_read": False}


def batch(run_dir, requests_path, replay_path=None):
    settings_path, manifest_path, outcomes_path = (run_dir / name for name in ("settings.json", "run.json", "outcomes.jsonl"))
    settings = json.loads(settings_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    need(settings == {k: v for k, v in manifest.items() if k not in ("errors", "outcomes_sha256")},
         "Saved run settings and completion manifest disagree")
    input_hash = file_sha256(requests_path)
    need(manifest.get("requests_sha256") == input_hash and manifest.get("outcomes_sha256") == file_sha256(outcomes_path)
         and manifest.get("reference_access") is False, "Saved inputs, outcomes or reference-access record do not match")
    requests = {r["request_id"]: r for r in read_requests(requests_path)}
    raws = records(outcomes_path)
    need(len(raws) == len(requests) == manifest["requested"] and
         {v["request_id"] for _, v in raws} == requests.keys(), "Missing, extra or duplicate run outcomes")
    kind = settings["resolver_execution"]
    saved = {}
    if kind == "saved_decisions":
        need(replay_path is not None and file_sha256(replay_path) == manifest["saved_decisions_sha256"],
             "Supply the exact saved raw landmark batch named by this run")
        rows = records(replay_path)
        saved = {row["request_id"]: (h, row) for h, row in rows}
        need(len(saved) == len(rows) and saved.keys() == requests.keys(), "Saved raw landmark cohort differs")
    else:
        need(kind == "live" and replay_path is None and manifest["saved_decisions_sha256"] is None,
             "Invalid declaration of live or saved resolver execution")
    events = []
    for raw_sha, outcome in raws:
        key = outcome["request_id"]
        request = requests[key]
        item = bound_outcome(outcome, request)
        need(outcome["provenance"].get("resolver_execution") == kind, "Outcome and batch disagree on resolver execution")
        if outcome["ok"]:
            resolver = outcome["provenance"]["resolver"]
            cfg = settings["config"]
            need(resolver["model_id"] == cfg["model_id"] and resolver["revision"] == cfg["revision"]
                 and bool(resolver.get("adapter")) == bool(cfg.get("adapter_dir"))
                 and outcome["provenance"]["geometry_sha256"] == settings["geometry_sha256"],
                 "Model or geometry in the recorded answer differs from the batch")
            if saved:
                need(saved[key][1] == resolver, "Embedded landmark receipt differs from the original raw decision")
        item.update(request_id=key, phase="model_after_prior_dialogue" if request["dialogue"] else "initial_model",
                    input_mode=request["input_mode"], raw_outcome_line_sha256=raw_sha,
                    raw_saved_decision_line_sha256=saved[key][0] if saved else None)
        events.append(item)
    need(sum(not e["ok"] for e in events) == manifest["errors"], "Completed run error count disagrees with outcomes")
    return dict(evidence_head(), origin="passenger_batch", settings=settings, run_sha256=file_sha256(manifest_path),
                settings_sha256=file_sha256(settings_path), requests_sha256=input_hash,
                outcomes_sha256=manifest["outcomes_sha256"], saved_decisions_sha256=manifest["saved_decisions_sha256"],
                events=events)


def session(journal_path, *, referent_path=None, pair_geometry=None, road_cache=None, parking_artifacts=None):
    from trackrag.dropoff import passenger_parking as parking
    from trackrag.dropoff import parking_clarification
    from trackrag.dropoff import vehicle_identity_reply
    raw = records(journal_path)
    need(raw and raw[0][1]["action"] == "opened", "Session lacks its opening record")
    scene = raw[0][1]["detail"]["scene"]
    request, previous, original_submission, native = None, {}, None, None
    passenger_requested_pointing = passenger_requested_area = False
    offered_tokens = set()
    parking_offer_sha = None
    place_offer_sha = None
    manual_offer_sha = None
    bay_menu_sha = None
    events = []
    for line_sha, event in raw:
        action, detail, view = event["action"], event["detail"], event["view"]
        frame = scene["frame"]
        need((view["recording_id"], view["frame_id"], view["image_sha256"]) ==
             (frame["recording_id"], frame["frame_id"], scene["provenance"]["image_sha256"]), "Session changed scene")
        item = {"action": action, "timestamp": event["timestamp"], "request_id": view["request_id"],
                "proposal_id": view["proposal_id"], "selected_by": view["selected_by"],
                "state": view["state"], "raw_event_line_sha256": line_sha}
        token = view["proposal_id"]
        if view["state"] not in {"proposed", "confirmed"}:
            need(view.get("bay_outline") is None, "Unproposed interaction contains a pictured destination")
        if view["state"] != "needs_clarification":
            need(not view.get("bay_choices"), "Inactive bay choices appear outside clarification")
        if view["state"] in {"proposed", "confirmed"}:
            need(isinstance(token, str) and re.fullmatch(r"[0-9a-f]{32}", token), "Confirmation requires a real proposal token")
            if view["state"] == "proposed":
                need(action in {"submitted", "passenger_marked", "passenger_selected_pair", "passenger_selected_bay"} and token not in offered_tokens,
                     "An offered passenger proposal token was reused")
                offered_tokens.add(token)
        else:
            need(token is None, "An inactive passenger interaction exposed a proposal token")
        if action in ("submitted", "passenger_marked", "passenger_selected_pair", "passenger_selected_bay"):
            parking_offer_sha = place_offer_sha = manual_offer_sha = None
            pick = action == "passenger_selected_bay"
            manual = action == "passenger_marked" or pick
            pair = action == "passenger_selected_pair"
            if not manual and not pair:
                # The application can journal a rejected submission before it reaches the model.
                # Keep that error in the export even if the submitted text failed validation.
                prior_request = request
                request = detail["request"] if detail["outcome"] is None else validate_request(detail["request"])
                need(isinstance(request, dict), "Submitted request is missing")
                need(request["scene"] == scene, "Submitted request belongs to another scene")
                if "replaces_request_id" in detail:
                    need("retries_waiting_request_id" not in detail, "A replacement cannot also retry a waiting result")
                    replaced = detail["replaces_request_id"]
                    need(isinstance(replaced,str) and replaced and previous.get("request_id") == replaced
                         and previous.get("state") not in {"confirmed","cancelled"} and replaced != request["request_id"]
                         and not request["dialogue"], "Replacement retained prior words or targeted a different request")
                    item["replaces_request_id"] = replaced
                elif "retries_waiting_request_id" in detail:
                    prior_id = detail["retries_waiting_request_id"]
                    need(prior_request and events and events[-1].get("outcome",{}).get("method")=="public_place_lookup_wait_v1"
                         and previous.get("state")=="needs_clarification" and previous.get("request_id")==prior_id
                         and prior_request["request_id"]==prior_id and request["request_id"]!=prior_id and
                         all(prior_request.get(k)==request.get(k) for k in ("text","dialogue","scene","input_mode","original_transcript")),
                         "A repeated waiting check changed its original words, context or mode")
                    item["retries_waiting_request_id"]=prior_id
                else:
                    expected_dialogue = [] if previous.get("state") in {"confirmed","cancelled"} else previous.get("turns", [])
                    need(request["dialogue"] == expected_dialogue, "Submitted history differs from the active passenger conversation")
                passenger_requested_pointing = detail.get("passenger_requested_pointing", False)
                passenger_requested_area = detail.get("passenger_requested_area", False)
                need(type(passenger_requested_pointing) is bool and type(passenger_requested_area) is bool
                     and not (passenger_requested_pointing and passenger_requested_area), "Invalid passenger pointing action")
                source = (detail.get('outcome') or {}).get('provenance') or {}
                if source.get('method') == vehicle_identity_reply.METHOD:
                    need(previous.get('state') == 'needs_clarification' and previous.get('reference_kind') == 'possible_vehicle'
                         and previous.get('request_id') == source['reference_request']['request_id']
                         and native == source.get('reference_outcome') and not (passenger_requested_area or passenger_requested_pointing),
                         'The passenger answered a different vehicle question from the one displayed in this session')
                native = detail["outcome"]
                origin_state = ("original_area_request_state" if passenger_requested_area else
                                "original_pointing_state" if passenger_requested_pointing else "original_automatic_state")
                original_submission = {"request_id":request["request_id"],"request_sha256":digest(request),
                    "original_outcome_sha256":digest(native) if native else None, origin_state:view["state"]}
            need(request is not None and request["request_id"] == view["request_id"], "Mismatched interaction request")
            if pair or manual:
                need(not (pair and passenger_requested_area), "A vehicle pair requires its own passenger action")
                from trackrag.dropoff.passenger_pair_choice import RECOVERABLE, landmark_error_guidance
                active = previous.get("state") in {"proposed", "needs_clarification", "insufficient_visual_evidence"}
                if previous.get("state") == "error" and events:
                    prior = events[-1]
                    checked_parking = prior.get("outcome", {}).get("parking_verification", {})
                    recovery = (prior["action"] == "submitted" and (
                        previous.get("parking_recovery") is True and checked_parking.get("input_verified") is True
                        or bool(previous.get("landmark_error_guidance")) and previous["landmark_error_guidance"] ==
                        landmark_error_guidance(request, native)))
                    active = recovery or (pair and (previous.get("error") or {}).get("code") in RECOVERABLE)
                need(active and previous.get("request_id") == request["request_id"] and original_submission,
                     "Assisted selection followed an inactive or different passenger request")
            item["phase"] = ("passenger_selected_bay" if pick else "passenger_selected_landmarks" if pair else "passenger_marked" if manual else
                             "explicit_passenger_area_request" if passenger_requested_area else
                             "model_after_prior_dialogue" if request["dialogue"] else "initial_model")
            item["submitted_text"] = request["text"]
            item["input_mode"] = request["input_mode"]
            item["original_transcript"] = request.get("original_transcript")
            if pick:
                need(bay_menu_sha and detail.get("prior_outcome_sha256") == bay_menu_sha
                     and previous.get("state") == "needs_clarification" and detail.get("bay_id") in
                     {b["id"] for b in previous.get("bay_choices", [])}, "Selection requires the preceding offered bays")
                item["bay_id"] = detail["bay_id"]
            elif manual:
                if (detail.get("outcome") or {}).get("ok") is True:
                    need(detail.get("requested_box") is not None, "Passenger mark lacks its submitted rectangle")
                    confirmed = (native or {}).get('provenance', {})
                    marked_source = detail['outcome']['provenance']
                    if confirmed.get('method') == vehicle_identity_reply.METHOD and confirmed.get('reply') == 'yes':
                        need(marked_source.get('confirmed_vehicle_reply') == native,
                             'A mark after vehicle confirmation must retain the chosen vehicle and the requested relation')
                    else:
                        need('confirmed_vehicle_reply' not in marked_source, 'This request did not confirm that vehicle')
                item["requested_box"] = detail.get("requested_box")
            elif pair:
                item["track_ids"],item["placement"] = detail["track_ids"],detail["placement"]
            else:
                item["speech"] = detail.get("speech")
                if item["speech"] and detail["outcome"] is not None:
                    audio_sha = item["speech"].get("audio_sha256")
                    need(isinstance(audio_sha, str) and len(audio_sha) == 64 and
                         all(c in "0123456789abcdef" for c in audio_sha), "Invalid original audio digest")
                    wav = journal_path.parent / "audio" / (audio_sha + ".wav")
                    need(wav.is_file() and file_sha256(wav) == audio_sha and
                         item["speech"].get("transcript") == request.get("original_transcript"),
                         "Saved speech receipt differs from the original WAV or recognized transcript")
                    item["original_wav_verified"] = True
            if detail["outcome"] is None:
                need(view["state"] == "error", "Missing outcome for a successful passenger turn")
                item["outcome"] = {"ok": False, "error": view["error"], "adapter_outcome_missing": True}
            else:
                item["outcome"] = bound_outcome(detail["outcome"], request, passenger_selected=manual, passenger_landmarks=pair,
                                                dialogue_policy=not manual and not pair, referent_path=referent_path,
                                                pair_geometry=pair_geometry,road_cache=road_cache,
                                                requested_box=detail.get("requested_box") if manual else None,
                                                selected_bay_id=detail.get("bay_id") if pick else None,
                                                bay_source_sha=bay_menu_sha if pick else None,
                                                passenger_requested_pointing=passenger_requested_pointing,
                                                passenger_requested_area=passenger_requested_area and not (manual or pair),
                                                parking_artifacts=parking_artifacts)
                if item["outcome"].get("method") in {parking.METHOD, parking.FAILURE_METHOD}:
                    item["phase"] = "parking_operational_error"
                    if item["outcome"]["ok"]:
                        item["phase"] = "automatic_parking_response"
                        if item["outcome"]["result"]["status"] == "proposed":
                            item["phase"] = "automatic_parking_proposal"
                            parking_offer_sha = digest(detail["outcome"])
                if (item["outcome"].get("place_verification") and item["outcome"]["ok"]
                        and item["outcome"]["result"]["status"] == "proposed"):
                    place_offer_sha = digest(detail["outcome"])
                if (item["outcome"].get("passenger_mark_verification", {}).get("offlane_mask_replayed")
                        and item["outcome"]["result"]["status"] == "proposed"):
                    manual_offer_sha = digest(detail["outcome"])
                if item["outcome"].get("method") == "deictic_pair_clarification" and item["outcome"]["ok"]:
                    item["phase"] = "automatic_pair_identification_question"
                if item["outcome"].get("method") == parking_clarification.METHOD:
                    item["phase"] = "reference_type_clarification"
                if item["outcome"].get("method") == "unique_road_qualified_vehicle_pair" and item["outcome"]["ok"]:
                    item["phase"] = "automatic_road_qualified_pair"
                if not manual and not pair and passenger_requested_pointing:
                    item["phase"] = "explicit_passenger_pointing_request"
            if item["outcome"]["ok"]:
                source = detail["outcome"]["provenance"]
                need(view.get("bay_outline") == source.get("bay_outline") and
                     view.get("bay_choices", []) == source.get("bay_choices", []), "Displayed bay differs from the verified offer")
                if pair:
                    receipt = detail["outcome"]["provenance"]
                    origin = "pointing_submission" if passenger_requested_pointing else "automatic_submission"
                    other = "automatic_submission" if passenger_requested_pointing else "pointing_submission"
                    need(receipt.get(origin)==original_submission and other not in receipt and
                         (receipt["passenger_selection"]["track_ids"],receipt["passenger_selection"]["placement"])==
                         (item["track_ids"],item["placement"]), "Assisted choice differs from the original automatic turn or passenger action")
                selected_by = ("passenger_landmarks" if pair else "passenger"
                               if manual or passenger_requested_pointing or passenger_requested_area else "model")
                if item["phase"] == "automatic_road_qualified_pair":
                    selected_by = "road_qualified_pair"
                if item["phase"] in {"automatic_parking_proposal", "automatic_parking_response"}:
                    selected_by = parking.ACTOR
                if item["phase"] == "reference_type_clarification":
                    selected_by = parking_clarification.ACTOR
                need(view["selected_by"] == selected_by and
                     view["result"] == item["outcome"]["result"] and view["state"] == view["result"]["status"],
                     "Saved view changed the grounding result or who selected it")
                bay_menu_sha = digest(detail["outcome"]) if view.get("bay_choices") else None
            else:
                bay_menu_sha = None
                need(view["state"] == "error" and view.get("result") is None and view.get("selected_by") is None
                     and view.get("error") == item["outcome"].get("error"),
                     "Failed interaction differs from the error presented to the passenger")
        elif action == "confirmed":
            need(view.get("bay_outline") == previous.get("bay_outline") and not view.get("bay_choices"),
                 "Confirmed bay differs from the displayed outline")
            need(previous.get("state") == "proposed" and view["state"] == "confirmed" and
                 (previous["request_id"], previous["proposal_id"], previous["selected_by"], previous["result"]) ==
                 (view["request_id"], detail["proposal_id"], detail["selected_by"], view["result"]) and
                 (view["proposal_id"], view["selected_by"]) == (detail["proposal_id"], detail["selected_by"]),
                 "Confirmation differs from the currently offered proposal")
            item["phase"] = "confirmed_previous_selection"
            if manual_offer_sha or "verified_manual_outcome_sha256" in detail:
                need(manual_offer_sha and detail.get("verified_manual_outcome_sha256") == manual_offer_sha,
                     "Passenger confirmation differs from the verified parking evidence for their mark")
                item["verified_manual_outcome_sha256"] = manual_offer_sha
            if place_offer_sha or "verified_place_outcome_sha256" in detail:
                need(place_offer_sha and detail.get("verified_place_outcome_sha256") == place_offer_sha,
                     "Place confirmation differs from the signed decision for this proposal")
                item["verified_place_outcome_sha256"] = place_offer_sha
            if view["selected_by"] == parking.ACTOR:
                need(parking_offer_sha and detail.get("verified_parking_outcome_sha256") == parking_offer_sha,
                     "Parking confirmation differs from the verified decision for this proposal")
                item["verified_parking_outcome_sha256"] = parking_offer_sha
        else:
            bay_menu_sha = None
            need(not view.get("bay_choices") and view.get("bay_outline") is None, "Inactive response contains a pictured bay")
            need(action in ("opened", "cancelled", "confirmation_blocked"), "Unsupported interaction action")
            state = {"opened": "idle", "cancelled": "cancelled", "confirmation_blocked": "error"}[action]
            need(view["state"] == state and view.get("selected_by") is None and view.get("result") is None,
                 "An inactive interaction cannot contain a destination or selector")
            if action == "opened":
                need(not events and view.get("request_id") is None, "Opening record must start the session")
            elif action == "confirmation_blocked":
                need(previous.get("state") == "proposed" and view["request_id"] == previous["request_id"]
                     and view.get("error"), "Blocked confirmation lacks an active offer")
            item["phase"] = action
        events.append(item)
        previous = view
    return dict(evidence_head(), origin="passenger_interaction_journal", journal_sha256=file_sha256(journal_path),
                application_source_commit_recorded_by_journal=False, actor_identity_verified=False,
                confirmation_scope="A client explicitly accepted the current proposal. Human identity and consent require separate evidence.",
                source_scene=scene["provenance"], events=events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    b = modes.add_parser("batch", help="Export a completed live or saved-landmark batch")
    b.add_argument("--run", required=True, type=Path)
    b.add_argument("--requests", required=True, type=Path)
    b.add_argument("--replay", type=Path)
    j = modes.add_parser("session", help="Export submitted, manually marked, and confirmed turns separately")
    j.add_argument("--journal", required=True, type=Path)
    j.add_argument("--referent", type=Path, help="Exact fitted ranker for replaying any zero-Qwen pair clarification")
    j.add_argument("--geometry", type=Path, help="Released learned geometry for independent replay of passenger-selected pairs")
    j.add_argument("--road-cache", type=Path, help="Verified original-image road masks for independent replay; use with --geometry")
    j.add_argument("--parking-artifacts", type=Path, help="Optional private artifacts for offline verification of parking-model decisions")
    for sub in (b, j):
        sub.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    data = (batch(args.run, args.requests, args.replay) if args.mode == "batch" else
            session(args.journal, referent_path=args.referent,pair_geometry=args.geometry,road_cache=args.road_cache,
                    parking_artifacts=args.parking_artifacts))
    with args.out.open("x", encoding="utf8", newline="\n") as target:
        json.dump(data, target, allow_nan=False)
        target.write("\n")


if __name__ == "__main__":
    main()
