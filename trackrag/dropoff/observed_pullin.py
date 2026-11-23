"""Use original photo geometry and source-bound descriptions for a visible pull-in."""
from io import BytesIO
from pathlib import Path
import re

from PIL import Image

from . import bay_intent, curb_point_language as words, manual_offlane, manual_surfaces
from . import paver_pullin, visual_space
from .curb_utterance import destination_clause
from .parking_evidence import source_input


POLICY = "pictured_paved_pullin_v1"
_RELATIVE = re.compile(r"\b(behind|(?:to (?:the )?)?(?:left|right) of)\s+")
_CLOSE = re.compile(r"\s+(?:please|would (?:be great|be fine|work)|thanks)\s*[.,!?]*$")
_STREET_SIDE = re.compile(r"\s+on (?:the |my |our )?(left|right)(?:[- ]hand)?(?: side)?\b")
_VEHICLE_ATTRIBUTE = {"advertising","commercial","branded","bright"}


def _vehicle_label(raw):
    # The reviewer describes the vehicle first and may separately report absent lettering.
    # Keep negative colour/type statements intact. Only discard the irrelevant livery qualifier.
    return re.split(r"(?:,|\s+with)\s*no (?:visible )?(?:commercial )?(?:livery|branding|lettering)\b",raw,
                    maxsplit=1,flags=re.I)[0].strip()


def _relative(text, options, size):
    """Bind the relation to its vehicle. A colour or business name must have pictured evidence."""
    text = _CLOSE.sub("", words._normal(destination_clause(text))).strip(" .,?!")
    relation = _RELATIVE.search(text)
    if not relation or set(words._terms(text[:relation.start()])) - words.FILLER - {"we","let","straight","just"}:
        return [], []
    tail = text[relation.end():]
    sides = _STREET_SIDE.findall(tail)
    if len(set(sides)) > 1:
        return [], []
    tail = _STREET_SIDE.sub(" ",tail).strip()
    parts = re.split(r"\s+(?:outside(?: of)?|in front of|near|next to|beside|by)\s+", tail, maxsplit=1)
    qualifier = set(words._terms(parts[0])) & _VEHICLE_ATTRIBUTE
    phrase = " ".join(v for v in words._terms(parts[0]) if v not in _VEHICLE_ATTRIBUTE)
    spec = words._vehicle(phrase,requested=True)
    if not spec:
        return [], []
    landmark = tuple(w for w in words._terms(parts[1]) if w not in words.ARTICLES) if len(parts)>1 else ()
    if len(parts)>1 and not landmark:
        return [], []
    certain, partial = [], []
    for option in options:
        desc = option["descriptions"]
        if landmark and not words.observed_landmark(landmark,desc):
            continue
        along = relation[1].strip()
        fields = ("nearer_vehicle","farther_vehicle") if along=="behind" else (
            ("farther_vehicle",) if "left" in along else ("nearer_vehicle",))
        states = []
        for field in fields:
            raw = desc[field]
            state = words._vehicle_match(spec,raw)
            if state is False:
                continue
            index = 0 if field=="nearer_vehicle" else 1
            track = option["vehicle_crops"][index][1]
            x = option["point"][0]
            if along!="behind" and not (x < track[0] if "left" in along else x > track[2]):
                continue
            # Side modifies the chosen vehicle, not the image position of the free ground.
            anchor_x = (track[0]+track[2])/(2*size[0])
            if sides and ((sides[0]=="left" and anchor_x>.56) or (sides[0]=="right" and anchor_x<.44)):
                continue
            if sides and .44<=anchor_x<=.56:
                state = None
            if along=="behind" and words._vehicle_match(spec,option["rear_vehicle"]) is not True:
                state = None
            if qualifier - set(words._terms(raw)):
                state = None
            states.append(state)
        if states:
            (certain if True in states else partial).append(option)
    return certain, partial


def visual_outcome(request, photo, surface, packet):
    from . import whole_bay
    if not bay_intent.permitted_destination(request["text"]):
        return None
    clean, _, portable = source_input(request,photo)
    scene, text = clean["scene"],clean["text"]
    size = [scene["frame"]["width"],scene["frame"]["height"]]
    options = visual_space.verified(scene,photo,surface,packet,paved=True)
    # This recipe reconstructs an entrance and paving. Its box is only the inspected
    # interior patch. The seat of a complete parking bay cannot be inferred from it.
    for option in options:
        desc = option["descriptions"]
        desc["roadside"] = (desc["roadside"] + " | paved").strip(" |")
        for field in ("nearer_vehicle","farther_vehicle"):
            desc[field] = _vehicle_label(desc[field])
    if _RELATIVE.search(words._normal(text)):
        supported, tentative = _relative(text,options,size)
    else:
        supported, tentative = words.match_points(text,options,size,conversational=True)
    # An uncertain visual clearance inspection can only offer a question, never an automatic pin.
    tentative = [o for o in supported if o["evidence"]["status"]!="OPEN"] + tentative
    supported = [o for o in supported if o["evidence"]["status"]=="OPEN"]
    offered = supported or tentative
    if not offered:
        return None
    if supported:
        question = "I found more than one pictured pull-in matching your description. Which one did you mean?"
    elif any(o["evidence"]["status"]!="OPEN" for o in offered):
        question = ("Is the highlighted point where you meant? The paving and nearby vehicles help locate it. "
                    "I couldn’t verify clearance there from the photo.")
    else:
        question = "Is the highlighted point where you meant? I couldn’t verify all the landmarks in your description."
    return whole_bay._make(clean,portable,offered,surface,method=whole_bay.AUTO,
                           reason="observed_paved_pullin" if supported else "unverified_bay_reference",question=question,
                           require_choice=not supported,visual_space_evidence=packet,pullin_policy=POLICY,
                           clearance_policy=whole_bay.CLEARANCE_POLICY,language_policy=whole_bay.INDEXED_DIALOGUE_POLICY)


def supplement(request, prior):
    from . import whole_bay
    if whole_bay._has_location(prior) or not bay_intent.permitted_destination(request["text"]):
        return prior
    key = request["scene"]["provenance"]["image_sha256"]
    packet = visual_space.get_packet(key,paved=True)
    if packet is None:
        return prior
    surface = manual_surfaces.get_packet(key)
    if surface is None:
        return prior
    try:
        photo = Path(request["scene"]["frame"]["image_path"]).read_bytes()
        return visual_outcome(request,photo,surface,packet) or prior
    except (OSError,TypeError,KeyError,ValueError):
        return prior


def marked_outcome(request, photo, surface, box):
    from . import whole_bay
    clean, key, portable = source_input(request,photo)
    _, _, mark, center = manual_offlane.mark_input(clean,box)
    scene = clean["scene"]
    classes = manual_surfaces.decode(surface,scene["frame"],key)
    reason, _ = manual_offlane.surface_check(scene["frame"],scene["tracks"],mark,center,classes)
    if reason in {"observed_object","broad_area"}:
        return None
    with Image.open(BytesIO(photo)) as original:
        native = paver_pullin.marked(original,classes,visual_space.native_context(scene),mark)
    if native is None:
        return None
    option = {"box":native["box"],"point":native["point"],"polygon":native["support_patch"],
              "evidence":{"recipe":paver_pullin.RECIPE,"coverage":"interior_point_only"}}
    return whole_bay._make(clean,portable,[option],surface,method=whole_bay.MARK,reason="passenger_mark_in_paved_pullin",
                           question="",tap=center,selected_box=mark,native_mark_geometry=native["evidence"],
                           pullin_policy=POLICY,clearance_policy=whole_bay.CLEARANCE_POLICY)
