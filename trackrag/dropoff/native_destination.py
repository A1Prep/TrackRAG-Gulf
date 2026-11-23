"""Passenger-mark and language decisions for a measured point in a native road image."""
from io import BytesIO
from pathlib import Path
import re

from PIL import Image

from . import bay_intent, curb_point_language as language, driveway_apron, manual_offlane
from . import manual_surfaces, paver_pullin, row_extension, visual_space
from .curb_utterance import destination_clause
from .parking_evidence import source_input


POLICY = "pictured_native_destination_v2"
GEOMETRIES = (paver_pullin, driveway_apron, row_extension)
DRIVE = re.compile(r"\b(?:driveways?|drive-up|entrance drives?|vehicle entrances?|forecourts?)\b")
SHOULDER = re.compile(r"\b(?:shoulders?|(?:roadside|curbside|kerbside|parking) strips?)\b")
RELATIVE = re.compile(r"\bbehind\b")
LINK = re.compile(r"\b(?:near|by|beside|next to|outside|at|in front of)\b")
SIDE = re.compile(r"\b(?:on|toward|towards) (?:the |our |my )?(left|right)(?:[- ]hand)?(?: side)?\b")
EXCLUDED = re.compile(r"\b(?:not|never|no|except|instead|either|or|beyond|after|before|past|"
                      r"across|opposite|between|upper|lower|next one|first|second|third|farthest|closest|"
                      r"left of|right of|traffic lane|bus lane|bike lane|cycle lane|motorway|highway)\b")
IGNORED = language.FILLER | {"would","we","us","at","in","on","the","our","dropoff","alight","from","please"}
MATERIAL = {"paved","pavement","paving","asphalt","brick","cobblestone","cobblestones","tarmac",
            "gravel","sand","sandy","unpaved","dirt","painted","marked"}
SIZE_WORDS = {"large","small","big","little","huge","oversized"}
BIN = re.compile(r"\b(?:(?:industrial|rubbish|waste|trash|garbage|refuse)\s+)*(?:skip|dumpster|bin|container)s?\b")


def _sentence(text):
    text=language._normal(destination_clause(re.sub(r"\bSt\.(?=\s)", "st", text, flags=re.I)))
    return re.sub(r"\boff (?:of )?(?:the )?(?:main |through )?(?:road|carriageway|street)\b"," ",text)


def _parts(text):
    if not bay_intent.permitted_destination(text):
        return None
    text=_sentence(text)
    if EXCLUDED.search(text):
        return None
    drive,strip=bool(DRIVE.search(text)),bool(SHOULDER.search(text))
    rear=RELATIVE.search(text)
    if drive and (strip or rear) or not (drive or strip or rear):
        return None
    sides=SIDE.findall(text)
    if len(set(sides))>1:
        return None
    text=SIDE.sub(" ",text)
    # Preserve unknown instructions as constraints. Only the passenger action and
    # the actual category established by geometry may be discarded.
    if drive:
        head=DRIVE.sub(" ",text)
        head=re.sub(r"\b(?:entrance|driveway|forecourt|at|of|to|into|outside|near|beside|by)\b"," ",head)
        terms=[v for v in language._terms(head) if v not in IGNORED]
        return {"kind":"entrance","side":next(iter(sides),None),"place":terms,"vehicle":None,"material":set()}
    before,rear_word,tail=text.partition("behind")
    before=SHOULDER.sub(" ",before)
    place=[]
    vehicle=None
    if rear_word:
        bits=LINK.split(tail,maxsplit=1)
        vehicle=language._vehicle(bits[0].strip(" ,.!?"),requested=True)
        if vehicle is None:
            return None
        if len(bits)==2:
            place=[v for v in language._terms(bits[1]) if v not in language.ARTICLES]
            if not place:
                return None
    else:
        bits=LINK.split(before,maxsplit=1)
        before=bits[0]
        if len(bits)==2:
            place=[v for v in language._terms(bits[1]) if v not in language.ARTICLES]
            if not place:
                return None
    qualifiers=set(language._terms(before))-IGNORED
    if qualifiers-MATERIAL:
        return None
    return {"kind":"shoulder","side":next(iter(sides),None),"place":place,
            "vehicle":vehicle,"material":qualifiers}


def _safe_seen(raw):
    """Keep the appearance attached to this pictured vehicle, apart from any printed livery."""
    return re.split(r"\s+with\b|[,;]|\b(?:displaying|advertising|bearing|reading)\b",raw,1,flags=re.I)[0].strip()


def _words(text):
    text=re.sub(r"\b(?:saint|st\.)\b", "st", text.casefold())
    return language._terms(BIN.sub("dumpster",text))


def _place_state(terms, desc):
    if not terms:
        return True
    normalized=set(_words(" ".join(terms)))
    substantial=normalized-SIZE_WORDS
    if not substantial:
        return False
    for raw in re.split(r"[|;,]",desc):
        if language.UNCERTAIN.search(raw) or language.SEPARATED.search(raw):
            continue
        seen=set(_words(raw))
        if substantial <= seen:
            if normalized <= seen:
                return True
            if normalized & {"large","big","huge","oversized"} and seen & {"small","little"}:
                return False
            if normalized & {"small","little"} and seen & {"large","big","huge","oversized"}:
                return False
            return None
    return False


def _physical_option(native):
    return {"point":native["point"],"box":native["box"],"polygon":native["support_patch"],
            "evidence":{"recipe":native["evidence"]["recipe"],"coverage":"interior_point_only"}}


def _row_anchor(parts, opt, width):
    evidence=opt["native_option"]["evidence"]
    if evidence["recipe"]!=row_extension.RECIPE or not parts["vehicle"]:
        return None
    point_x=opt["native_option"]["point"][0]/width
    if (parts["side"]=="left" and point_x>.56) or (parts["side"]=="right" and point_x<.44):
        return None
    desc=opt["descriptions"]
    matches=[]
    for (key,box),caption in zip(opt["vehicle_crops"],(desc["nearer_vehicle"],desc["farther_vehicle"]),strict=True):
        if language._vehicle_match(parts["vehicle"],_safe_seen(caption)) is not True:
            continue
        pos=(box[0]+box[2])/(2*width)
        if (parts["side"]=="left" and pos>=.56) or (parts["side"]=="right" and pos<=.44):
            continue
        matches.append((key,box))
    if len(matches)==1 and matches[0][0]==evidence["nearest_row_track_id"]:
        return matches[0][1]
    return None


def _matching(parts, opt, width):
    native=opt["native_option"]
    geometry=native["evidence"]
    desc=opt["descriptions"]
    kind=parts["kind"]
    if geometry["recipe"]!=(driveway_apron.RECIPE if kind=="entrance" else row_extension.RECIPE):
        return False
    sides=parts["side"]
    x=native["point"][0]/width
    if sides=="left" and x>.56 or sides=="right" and x<.44:
        return False
    states=[None if sides and .44<=x<=.56 else True]
    if kind=="entrance":
        states.append(_place_state(parts["place"],desc["frontage"]))
    else:
        # This geometry measures neutral surfaced ground. It cannot establish a
        # narrower material, paint designation or rights of way without the trained observation.
        materials=parts["material"]
        detailed=materials-{"paved","pavement","paving"}
        if detailed:
            states.append(_place_state(detailed,desc["roadside"]))
        states.append(_place_state(parts["place"],desc["roadside"]+" | "+desc["frontage"]))
        if parts["vehicle"]:
            labels=dict((row[0],caption) for row,caption in zip(opt["vehicle_crops"],
                            (desc["nearer_vehicle"],desc["farther_vehicle"]),strict=True))
            caption=labels.get(geometry["nearest_row_track_id"])
            if not caption:
                return False
            anchor_box=dict(opt["vehicle_crops"])[geometry["nearest_row_track_id"]]
            anchor_x=(anchor_box[0]+anchor_box[2])/(2*width)
            if (sides=="left" and anchor_x>.56) or (sides=="right" and anchor_x<.44):
                return False
            if sides and .44<=anchor_x<=.56:
                states.append(None)
            anchor_state=language._vehicle_match(parts["vehicle"],_safe_seen(caption))
            if anchor_state is True and _row_anchor(parts,opt,width) is None:
                anchor_state=None
            states.append(anchor_state)
            # The near vehicle is the anchor. The farther witness is not a vehicle
            # bracketing the other side of the requested point.
            observed=opt["rear_vehicle"]
            if observed.strip().casefold() in {"none","unclear","unknown"}:
                states.append(None)
            else:
                states.append(language._vehicle_match(parts["vehicle"],_safe_seen(observed)))
    if False in states:
        return False
    return None if None in states else True


def marked_outcome(request, photo, surface, box):
    from . import whole_bay
    clean,key,portable=source_input(request,photo)
    _,_,mark,center=manual_offlane.mark_input(clean,box)
    scene=clean["scene"]
    classes=manual_surfaces.decode(surface,scene["frame"],key)
    reason,_=manual_offlane.surface_check(scene["frame"],scene["tracks"],mark,center,classes)
    if reason in {"observed_object","broad_area"}:
        return None
    options=[]
    with Image.open(BytesIO(photo)) as original:
        ctx=visual_space.native_context(scene)
        for module in GEOMETRIES:
            native=module.marked(original,classes,ctx,mark)
            if native is not None:
                options.append(native)
    if len(options)!=1:
        return None
    native=options[0]
    return whole_bay._make(clean,portable,[_physical_option(native)],surface,method=whole_bay.MARK,
        reason="passenger_mark_in_native_destination",question="",tap=center,selected_box=mark,
        native_mark_geometry=native["evidence"],pullin_policy=POLICY,clearance_policy=whole_bay.CLEARANCE_POLICY)


def language_outcome(request, photo, surface, signed_observations):
    from . import whole_bay
    clean,_,portable=source_input(request,photo)
    parts=_parts(clean["text"])
    if parts is None:
        return None
    scene=clean["scene"]
    options=visual_space.verified_extension(scene,photo,surface,signed_observations)
    relevant_recipe=driveway_apron.RECIPE if parts["kind"]=="entrance" else row_extension.RECIPE
    related=[o for o in options if o["native_option"]["evidence"]["recipe"]==relevant_recipe]
    if not related:
        return None
    strong,partial,vetoed,identified=[],[],[],[]
    for opt in related:
        status=opt["evidence"]["status"]
        if status in {"LANE","BLOCKED","SEPARATED","UNCLEAR"}:
            actor=_row_anchor(parts,opt,scene["frame"]["width"])
            if actor is not None and actor not in identified:
                identified.append(actor)
        state=_matching(parts,opt,scene["frame"]["width"])
        if state is False:
            continue
        if status in {"LANE","BLOCKED","SEPARATED","UNCLEAR"}:
            vetoed.append(opt)
        elif status=="OPEN" and state is True:
            strong.append(opt)
        elif status=="OPEN":
            partial.append(opt)
    meta={"pullin_policy":POLICY,"native_location_observations":signed_observations,
          "clearance_policy":whole_bay.CLEARANCE_POLICY,"language_policy":whole_bay.INDEXED_DIALOGUE_POLICY}
    if not (strong or partial or vetoed or identified):
        return whole_bay._make(clean,portable,[],surface,method=whole_bay.AUTO,reason="native_reference_unmatched",
            question="I couldn’t match those details to a verified location in this photo. Could you mark the spot you mean?",**meta)
    if not strong and not partial:
        if len(identified)==1:
            answer=whole_bay._make(clean,portable,[],surface,method=whole_bay.AUTO,reason="vehicle_seen_space_withheld",
                question="The outline shows the vehicle you described. I couldn’t verify a clear spot behind it. Could you mark another visible spot?",**meta)
            answer["result"]["anchor_boxes_xyxy"]=identified
            return answer
        return whole_bay._make(clean,portable,[],surface,method=whole_bay.AUTO,reason="native_destination_access_withheld",
            question="I couldn’t verify an open space at that location. Could you mark a different spot?",**meta)
    used=strong or partial
    display=[_physical_option(o["native_option"]) for o in used]
    if strong:
        question="More than one visible location matches. Which do you mean?"
    else:
        question="I found a visible point in the requested area. Some details are unclear from this angle. Is this where you meant?"
    return whole_bay._make(clean,portable,display,surface,method=whole_bay.AUTO,
        reason="observed_native_destination" if strong else "unverified_bay_reference",question=question,
        require_choice=not strong,**meta)


def supplement(request, prior):
    from . import whole_bay
    if whole_bay._has_location(prior) or _parts(request["text"]) is None:
        return prior
    key=request["scene"]["provenance"]["image_sha256"]
    evidence=visual_space.get_extension_packet(key)
    if evidence is None:
        return prior
    surface=manual_surfaces.get_packet(key)
    if surface is None:
        return prior
    try:
        photo=Path(request["scene"]["frame"]["image_path"]).read_bytes()
        return language_outcome(request,photo,surface,evidence) or prior
    except (KeyError,TypeError,ValueError,OSError):
        return prior
