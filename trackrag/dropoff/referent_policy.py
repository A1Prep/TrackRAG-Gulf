"""Ask for a distinction when several real, current detected vehicles fit a single reference."""
from __future__ import annotations

import math
import re

from .referent import COLORS, overlap


VERSION = "visible-single-vehicle-clarification-v2"
NOUN = re.compile(r"\b(cars?|vans?|buses|bus|trucks?|lorr(?:y|ies)|motor(?:cycles?|bikes?)|bicycles?|vehicles?|"
                  r"suvs?|sedans?|pickups?|taxis?|cabs?|hatchbacks?|coupes?|jeeps?|minivans?|"
                  r"pedestrians?|people|person|cyclists?|scooters?|trailers?|tractors?)\b", re.I)
STATIC = re.compile(r"\b(?:building|entrance|crossing|sign|lamp|tree|fence|gate|curb cut|door|shop|store|"
                    r"garage|station|arrow|divider|junction|corner|panels?|shelter|stop sign|booth|bus stop|"
                    r"car park|parking lot|car wash|bike lane|truck stop|bicycle rack)\b", re.I)
DEICTIC = re.compile(r"(?:(?:hey|hi|please|okay|ok|can|could|would|you|we|i|just|want|like|to|please|"
                     r"let|leave|take|drop|pull|over|off|out|us|me|stop|at|in|on|park|right|up|get|meant|mean|"
                     r"this|that|place|spot|one|here|there|now|is|it)\s*)+",re.I)
UNMODELED = re.compile(r"\b(?:behind|ahead of|in front of|next to|beside|under|opposite|past|with|whose|"
                       r"wearing|turning|moving|turns|marked|painted|logo|lights?|plate|second|third|fourth|"
                       r"nearest|closest|farthest|furthest|frontmost|rearmost|first|last)\b",re.I)
VEHICLES = {"car","truck","bus","motorcycle","bicycle","trailer","construction"}
COARSE = {"large_vehicle","medium_vehicle","small_vehicle","heavy_vehicle","light_vehicle",
          "small_motorised_vehicle","emergency_vehicle"}
BROAD_SETTING = re.compile(r"\b(?:under (?:this |that |the |an? )?(?:bridge|overpass|flyover)|"
                           r"(?:on|along) (?:this|that|the) (?:road|street))\b",re.I)
MODIFIERS = set(("leftmost rightmost left right black white gray grey silver red blue green yellow "
                 "big large small larger smaller largest smallest parked waiting stationary hand side far").split())
TAIL = MODIFIERS | set(("please to on at in near the a an our your my this that just right "
                       "there here over ahead directly side of me us picture image one").split())


def _noun(token):
    if token.startswith("car"):
        return "car"
    if token.startswith("van"):
        return "van"
    if token.startswith("bus"):
        return "bus"
    if token.startswith(("truck","lorr")):
        return "truck"
    if token.startswith("motor"):
        return "motorcycle"
    if token.startswith("bicycl"):
        return "bicycle"
    return "vehicle" if token.startswith("vehicle") else "unverified_subtype"


def _candidate(row, name):
    cls = row["class_name"].lower().replace(" ", "_")
    category = row["category"]
    if name == "vehicle":
        return category in VEHICLES or cls in COARSE
    if name == "van":
        # The source detector folds vans into cars. No van subtype can be confirmed.
        return cls == "van"
    if name=="truck":
        # A coarse road-user detection can support a neutral choice about which
        # vehicle, but cannot establish that either one is specifically a truck.
        return category=="truck" or cls in {"medium_vehicle","large_vehicle","heavy_vehicle"}
    return category == name


def _undirected(text):
    spoken = re.sub(r"[^a-z ]", " ", text.lower())
    return (bool(re.search(r"\b(?:here|there|this (?:one|spot|place)|that (?:one|spot|place))\b",spoken))
            and DEICTIC.fullmatch(" ".join(spoken.split())) is not None)


def _single_reference(text):
    changes = list(re.finditer(r"\b(?:actually|i mean|forget that|instead(?!\s+of))\b", text, re.I))
    if changes:
        revised = text[changes[-1].end():].strip(" ,.!?")
        if revised and (NOUN.search(revised) or STATIC.search(revised) or _undirected(revised)):
            text = revised
    if _undirected(text):
        return "deictic", None, None
    if re.search(r"\b(?:between|both|two|either|other than|except|not|never|rather than|instead of)\b|n['’]t\b", text, re.I) or STATIC.search(text):
        return None, None, None
    nouns = list(NOUN.finditer(text))
    if len(nouns) != 1:
        return None, None, None
    match = nouns[0]
    prefix, suffix = text[:match.start()].lower(), text[match.end():].lower().strip()
    if _noun(match.group(0).lower())=="unverified_subtype":
        return None,None,None
    # A gesture plus a broad setting ('that car under the bridge') still leaves
    # an unresolved identity when several cars are plausible. The detector
    # cannot establish which ones are under the bridge, so the question never claims that.
    broad = BROAD_SETTING.search(suffix) if re.search(r"\b(?:that|this)\s+$",prefix) else None
    if broad:
        suffix = (suffix[:broad.start()]+" "+suffix[broad.end():]).strip()
    # A second reference/attribute outside this policy's sensors remains the visual model's job.
    if UNMODELED.search(suffix) or re.search(r"\b(?:nearest|closest|farthest|furthest|first|second|third|last)\b", prefix):
        return None, None, None
    # Open-ended descriptions ('the damaged car', 'the car under the awning')
    # may already distinguish the object in the image. Do not flatten those to 'car'.
    cut = re.split(r"\b(?:the|an?|this|that|our|my|your|beside|near|by|of|behind|past)\b|[,;.!?]",prefix)[-1]
    if set(re.findall(r"[a-z]+", cut)) - MODIFIERS or set(re.findall(r"[a-z]+",suffix)) - TAIL:
        return None, None, None
    # 'Stop to the left of the car' specifies a destination relation. Limit side/colour
    # tests to adjacent modifiers of the object, and to a trailing 'car on the left'.
    before = re.search(r"(?:(?:\b(?:leftmost|rightmost|left|right|black|white|gray|grey|silver|red|blue|green|yellow|"
                       r"big|large|small|larger|smaller|largest|smallest|parked|waiting|stationary|dark|light|"
                       r"hand|side|far|the|that|this|a|an)\b[\s-]+)*)$", prefix)
    modifiers = (before.group(0)+" "+suffix).lower()
    return _noun(match.group(0).lower()), modifiers, bool(broad)


def _center(row, axis, size):
    b = row["bbox_xyxy"]
    return (b[axis]+b[axis+2])/(2*size[axis])


def _area(row):
    l,t,r,b = row["bbox_xyxy"]
    return (r-l)*(b-t)


def _question(rows, noun, frame_size, reason):
    first = rows[0]  # Highest-ranked still-plausible current observation.
    pairs = [(v, abs(_center(v,0,frame_size)-_center(first,0,frame_size)),
                  abs(_center(v,1,frame_size)-_center(first,1,frame_size))) for v in rows[1:]
             if overlap([v["bbox_xyxy"]], first["bbox_xyxy"])[0] < .35]
    if not pairs:
        return None
    # Prefer the best-ranked, clearly separated competitor. No depth or movement
    # is inferred from a screen position.
    rival,dx,dy = next((p for p in pairs if max(p[1:])>=.075),pairs[0])
    if dx >= .075:
        descriptions = ["farther left","farther right"]
        options = sorted((first,rival),key=lambda r:_center(r,0,frame_size))
    elif dy >= .075:
        descriptions = ["higher in the picture","lower in the picture"]
        options = sorted((first,rival),key=lambda r:_center(r,1,frame_size))
    else:
        descriptions,options=[],[first,rival]
    if not descriptions:
        prompt = f"Which {noun} do you mean? Can you describe where it is in the picture?"
    elif len(rows)==2:
        prompt = f"Do you mean the {noun} {descriptions[0]} or the one {descriptions[1]}?"
    else:
        def region(row):
            x=_center(row,0,frame_size)
            return "toward the left" if x<1/3 else "toward the right" if x>2/3 else "near the middle"
        locations=[region(r) for r in options]
        if locations[0]!=locations[1]:
            prompt=(f"Which {noun} do you mean? Is it {locations[0]} of the picture, "
                    f"{locations[1]}, or somewhere else?")
        else:
            prompt=(f"Which {noun} do you mean? Can you describe its position "
                    "or mark it on the picture?")
    return {"policy":VERSION, "reason":reason,"question":prompt,"category":noun,
            "candidate_count":len(rows), "track_ids":[r["track_id"] for r in options],
            "ranked_plausible_track_ids":[r["track_id"] for r in rows],
            "descriptions":descriptions if len(rows)==2 else []}


def clarify_reference(text, ranking, frame_size):
    """Return an ask-only decision or None. Call with rank() from this verified input.

    No decision certifies a referent, stopping area, motion, distance or absence.
    The verbatim current instruction plus relevant passenger dialogue may be supplied.
    """
    if not isinstance(text,str) or not text.strip():
        raise ValueError("A nonempty passenger instruction is required")
    if len(frame_size)!=2 or any(not math.isfinite(float(v)) or v<=0 for v in frame_size):
        raise ValueError("Valid original-frame dimensions are required")
    name, modifiers, broad_setting = _single_reference(text)
    if name=="deictic":
        return {"policy":VERSION, "reason":"unpointed_deictic","category":None,"candidate_count":0,
                "track_ids":[],"ranked_plausible_track_ids":[],"descriptions":[],
                "question":"Which visible landmark is beside your drop-off, or can you mark the area on the picture?"}
    if not name:
        return None
    candidates = []
    for row in ranking["alternatives"]:
        if not _candidate(row,name):
            continue
        l,t,r,b = row["bbox_xyxy"]
        if (r-l)/frame_size[0] < .012 or (b-t)/frame_size[1]< .012:
            continue
        if any(overlap([row["bbox_xyxy"]], prev["bbox_xyxy"])[0] >= .55 for prev in candidates):
            continue
        candidates.append(row)
    if len(candidates)<2:
        return None
    mention = set(re.findall(r"[a-z]+",modifiers))
    color = (mention & set(COLORS)) | ({"gray"} if mention & {"grey","silver"} else set())
    if len(color)>1:
        return None
    reason = ("demonstrative_and_unlocalized_broad_setting" if broad_setting
              else "multiple_current_objects_fit_single_reference")
    if color:
        c = next(iter(color))
        limit = .20 if c in {"black","white","gray"} else .16
        def strength(row):
            broad = row.get("colour_fractions",{}).get(c)
            middle = row.get("central_colour_fractions",{}).get(c)
            return (.5*float(broad)+.5*float(middle) if broad is not None and middle is not None else 0.)
        measured = [r for r in candidates if strength(r)>=limit]
        ranked_strengths=sorted((strength(r) for r in candidates),reverse=True)
        # A few bright pixels in a dark frame are weak evidence of white paint.
        if (len(measured)==1 and ranked_strengths[0]>=.45
                and ranked_strengths[0]-ranked_strengths[1]>=.17):
            return None
        if len(measured)>1:
            candidates=measured
            reason="multiple_current_objects_share_measured_colour"
        else:
            # Shadow, specular reflections, and broad detector crops can hide the
            # requested colour. Uncertain appearance cannot prove that a vehicle is absent.
            reason="multiple_current_objects_colour_unresolved"
    single_side = (mention & {"left","right","leftmost","rightmost"})
    if len(single_side)>1:
        return None
    if single_side:
        which = next(iter(single_side))
        if (which.endswith("most") or re.search(r"\bfar[\s-]+(?:left|right)\b",modifiers)):
            # Extreme horizontal position can be a genuine distinguishing descriptor.
            ordered = sorted(candidates,key=lambda row:_center(row,0,frame_size),reverse=which.startswith("right"))
            edge = _center(ordered[0],0,frame_size)
            candidates = [r for r in candidates if abs(_center(r,0,frame_size)-edge) <= .045]
        else:
            candidates = [r for r in candidates if (_center(r,0,frame_size) <= .55 if which=="left"
                                                      else _center(r,0,frame_size) >= .45)]
        if len(candidates)<2:
            return None
        reason="multiple_current_objects_on_requested_side"
    size_words = mention & {"big","large","larger","largest","small","smaller","smallest"}
    if size_words:
        small = bool(size_words & {"small","smaller","smallest"})
        if small and bool(size_words & {"big","large","larger","largest"}):
            return None
        extreme = min(map(_area,candidates)) if small else max(map(_area,candidates))
        candidates = [r for r in candidates if max(_area(r),extreme)/min(_area(r),extreme) < 1.6]
        if len(candidates)<2:
            return None
        reason="multiple_current_objects_of_similar_projected_size"
    label = "vehicle" if name=="truck" and any(row["category"]!="truck" for row in candidates) else name
    result = _question(candidates,label,frame_size,reason)
    if result is not None:
        result["requested_category"] = name
    return result
