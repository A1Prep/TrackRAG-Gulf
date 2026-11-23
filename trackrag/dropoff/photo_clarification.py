"""Explain what a source-bound photo observation can establish when no spot matches."""
from pathlib import Path

from . import bay_intent, bay_landmarks, curb_point_language, manual_surfaces
from .parking_evidence import source_input


POLICY = "visible_landmark_without_destination_v1"


def visible_category(text, labels):
    intent=bay_intent.parse(bay_intent.destination_words(text))
    if not intent or intent.blocked or intent.uncertain or intent.vehicles or len(intent.frontage_phrases)!=1:
        return None
    terms=intent.frontage_phrases[0]
    category=next((word for word in terms if word in {"restaurant","hotel","cafe","shop","pharmacy","supermarket",
                                                       "garage","school","mosque","tree","tower","building","entrance"}),None)
    if not category:
        return None
    matches=[i for i,phrase in enumerate(labels) if
             curb_point_language.observed_landmark(terms,{"frontage":phrase,"roadside":""}) is True]
    if intent.side=="right":
        matches=[i for i in matches if i==2]
    elif intent.side=="left":
        matches=[i for i in matches if i==0]
    if not matches:
        return None
    return "trees" if category=="tree" else "the "+category


def rebuild(request,photo,surface,landmarks,*,local):
    from . import local_surfaces, whole_bay
    clean, key, portable=source_input(request,photo)
    if not bay_intent.permitted_destination(clean["text"]):
        return None
    scene=clean["scene"]
    manual_surfaces.decode(surface,scene["frame"],key)
    if local and not local_surfaces.matches_scene(scene):
        return None
    category=None
    if landmarks:
        labels=bay_landmarks.verify(landmarks,photo,(scene["frame"]["width"],scene["frame"]["height"]))
        category=visible_category(clean["text"],labels)
    if category:
        nearby="them" if category=="trees" else "it"
        question=(f"I can see {category} in the photo. I couldn’t verify an open pull-in beside {nearby}. "
                  "Could you tap the off-lane area you meant, or name another landmark?")
    elif local:
        question=("I couldn’t match that description to a visible drop-off area. "
                  "Tap an off-lane spot to check the surface, or describe another landmark.")
    else:
        return None
    return whole_bay._make(clean,portable,[],surface,method=whole_bay.AUTO,reason="pictured_place_space_unverified",
                           question=question,explanation_policy=POLICY,local_photo_check=local,
                           landmark_evidence=landmarks,clearance_policy=whole_bay.CLEARANCE_POLICY)


def explain(request,prior,*,local=False):
    from . import curb_points, whole_bay
    if (whole_bay._has_location(prior) or (prior and prior.get("result",{}).get("anchor_boxes_xyxy")) or
            not bay_intent.permitted_destination(request["text"])):
        return prior
    key=request["scene"]["provenance"]["image_sha256"]
    surface=manual_surfaces.get_packet(key)
    if surface is None:
        return prior
    photo=Path(request["scene"]["frame"]["image_path"]).read_bytes()
    if local:
        # Native whole-bay handling remains first. On newly prepared photos a single
        # physical point can also carry independently inspected close-up observations.
        automatic=whole_bay.automatic(request,photo)
        if whole_bay._has_location(automatic):
            return automatic
        points=curb_points.get_packet(key)
        if points is not None:
            answer=whole_bay.point_outcome(request,photo,surface,points,conversational=True,expanded=True)
            if whole_bay._has_location(answer):
                return answer
    return rebuild(request,photo,surface,bay_landmarks.get_packet(key),local=local) or prior
