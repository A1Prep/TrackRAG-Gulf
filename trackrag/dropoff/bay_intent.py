"""Match an explicit parking-space request to observable attributes of verified outlines."""
from dataclasses import dataclass
from itertools import permutations
import math
import re

from .passenger import digest


PARKING = re.compile(r"\bbays?\b|\b(?:(?:parking|empty|open|vacant|available|unoccupied|marked|painted|"
                     r"parallel|angled|diagonal|perpendicular)\s+)+(?:spaces?|spots?|stalls?)\b", re.I)
NAMED_GAP = re.compile(r"\b(?:in|into|at|to)\s+(?:(?:the|that|this|a|an|visible|open|empty)\s+)*"
                       r"(?:space|spot|gap|opening)\s+(?:right\s+)?between\b",re.I)
RELATION = re.compile(r"\b(?:on the opposite side of|across (?:the road|the street) from|across from|opposite|"
                      r"to (?:the )?(?:left|right) of|in front of|next to|(?:nearest|closest) to|"
                      r"(?:furthest|farthest) from|(?:closest|nearest)(?= (?:the|that|this|a|an)\b)|"
                      r"between|behind|beyond|before|past|after|"
                      r"alongside|beside|outside|along|near|by|at)\b")
VEHICLE = re.compile(r"\b(pick[ -]?ups?(?: trucks?)?|suvs?|sedans?|hatchbacks?|coupes?|minivans?|"
                     r"vans?|cars?|autos?|vehicles?|trucks?|lorr(?:y|ies)|buses|bus|taxis?|motorbikes?|motorcycles?)\b")
COLORS = {"black","white","silver","gray","red","blue","green","yellow","brown","orange","beige","gold"}
ALIASES = {"grey":"gray","automobile":"car","auto":"car","lorry":"truck","lorries":"truck","motorbike":"motorcycle",
           "hedgerow":"hedge","bushes":"bush","shrubs":"shrub","trees":"tree","palms":"palm","doors":"door",
           "buildings":"building","towers":"tower","restaurants":"restaurant","garages":"garage",
           "shops":"shop","store":"shop","stores":"shop","pharmacies":"pharmacy","entrances":"entrance",
           "cafes":"cafe","café":"cafe","cafés":"cafe","offices":"office","gardens":"garden","hedges":"hedge"}
FILLER = set(("a an the to by near outside of at in on into from with along and or my our your me us you i we "
              "can could would will let get drop stop take leave put pull park find use choose want please just "
              "some any one two both those that these this available idle open clear empty free unoccupied vacant "
              "parking marked painted spot spots space spaces bay bays stall stalls side hand left right leftmost "
              "rightmost nearest closest nearer farther furthest farthest biggest largest widest smallest narrowest "
              "parallel angled diagonal perpendicular forward off out next beside alongside beyond past after before "
              "behind in front facing between parked visible shown picture photo curbside street road carpark of "
              "there here over toward towards go get would like may end first are is either area somewhere "
              "kindly dropoff help pick passenger driver").split())
UNCLEAR_OBS = re.compile(r"\b(?:none|unknown|unclear|possibly|maybe|uncertain|unidentifiable|no|not)\b|[?]")
PLACE_CATEGORIES = set(("building tower residence residential commercial office entrance door shop restaurant "
                        "cafe pharmacy supermarket market garage hotel school mall clinic kiosk house garden "
                        "hedge shrub bush tree palm sidewalk fence wall storefront sign street green glass "
                        "large small tall visible front al el the").split())
UNSUPPORTED = re.compile(r"\b(?:not|except|excluding|avoid|instead of|across from|opposite|across the "
                         r"(?:(?:main )?road|street)|other side of|no stopping|bus bay|loading bay|"
                         r"wheelchair accessible|disabled|accessible|permit.only|reserved|legal|legally)\b|n['’]t\b")
SIDE = re.compile(r"\b(?:(?:on|to)\s+(?:the\s+|our\s+|my\s+|your\s+)?(left|right)(?!\s+of)\b|"
                  r"(left|right)(?:[- ]hand)?\s+(?:side|bay|space|spot|stall)s?\b)")
ORIENT = {"parallel":"parallel","perpendicular":"perpendicular","diagonal":"angled","angled":"angled"}
RANK = {"nearest":"nearest","closest":"nearest","farthest":"farthest","furthest":"farthest",
        "biggest":"largest","largest":"largest","widest":"widest","smallest":"smallest","narrowest":"narrowest",
        "leftmost":"leftmost","rightmost":"rightmost"}


def _tokens(s):
    return [ALIASES.get(w,w) for w in re.findall(r"[^\W_]+",s.casefold())]


def _kind(name):
    name = name.replace("-", " ").strip()
    if name.startswith("pick"):
        return "pickup"
    if name in {"bus","buses"}:
        return "bus"
    return ALIASES.get(name, ALIASES.get(name.removesuffix("s"), name.removesuffix("s")))


@dataclass(frozen=True)
class Vehicle:
    kind: str
    color: str | None = None


@dataclass(frozen=True)
class BayIntent:
    side: str | None
    orientation: str | None
    vehicle_relation: str | None
    vehicles: tuple[Vehicle, ...]
    frontage: tuple[str, ...]
    rank: str | None
    uncertain: bool
    blocked: bool
    frontage_phrases: tuple[tuple[str, ...], ...]


def parse(words):
    """Return None outside the explicit parking task, including ordinary vehicle-only references."""
    if not isinstance(words,str):
        return None
    text = words.casefold().replace("’","'").replace("grey","gray")
    if not PARKING.search(text) and not (NAMED_GAP.search(text) and VEHICLE.search(text)
            and re.search(r"\b(?:drop|stop|park|pull|leave me|get out|get off)\b",text)):
        return None
    text = re.sub(r"\b(i|we)'d\b",r"\1 would",text)
    tokens = _tokens(text)
    orientations = {ORIENT[t] for t in tokens if t in ORIENT}
    ranks = {RANK[t] for t in tokens if t in RANK}
    sides = {next(x for x in m.groups() if x) for m in SIDE.finditer(text)}
    blocked = bool(UNSUPPORTED.search(text) or len(sides)>1 or len(orientations)>1 or len(ranks)>1)
    relative = list(RELATION.finditer(text))
    vehicles,frontage = [],[]
    vehicle_relation = None
    consumed = set()
    uncertain = False
    for index, match in enumerate(relative):
        relation = match.group()
        value = text[match.end():relative[index+1].start() if index+1<len(relative) else len(text)]
        value = re.split(r"[,;.!?]|\b(?:on (?:the |our |my |your )?(?:right|left)|please)\b",value,maxsplit=1)[0]
        if not value.strip() or PARKING.search(value):
            continue
        nouns = list(VEHICLE.finditer(value))
        if not nouns:
            terms = [t for t in _tokens(value) if t not in FILLER]
            if terms:
                frontage.append(tuple(dict.fromkeys(terms)))
                consumed.update(terms)
            if relation in {"behind","beyond","past","after","before"} or relation.endswith(" of"):
                uncertain = True
            continue
        kind_of_relation = ("between" if relation=="between" else "beside" if relation in
                            {"by","near","next to","beside","alongside","at"} else relation)
        if vehicle_relation is not None and vehicle_relation != kind_of_relation:
            uncertain = True
        vehicle_relation = kind_of_relation
        if kind_of_relation not in {"between","beside"}:
            uncertain = True  # Crops and a same-row pair cannot establish road direction or longitudinal ordering.
        prior = 0
        current = []
        for noun in nouns:
            prefix = _tokens(value[prior:noun.start()])
            colors = [t for t in prefix if t in COLORS]
            typ = _kind(noun.group())
            if len(colors)>1:
                if kind_of_relation=="between" and "and" in prefix and len(nouns)==1 and len(colors)==2:
                    current.extend(Vehicle(typ,c) for c in colors)
                else:
                    uncertain = True
                    current.append(Vehicle(typ))
            else:
                current.append(Vehicle(typ,colors[0] if colors else None))
            consumed.update(colors)
            consumed.update(_tokens(noun.group()))
            prior=noun.end()
        if kind_of_relation=="between" and len(current)==1:
            if nouns[0].group().endswith("s") or "two" in _tokens(value) or "both" in _tokens(value):
                current*=2
            else:
                uncertain=True
        vehicles.extend(current)
    if ((vehicle_relation=="between" and len(vehicles)!=2)
            or (vehicle_relation not in {None,"between"} and len(vehicles)!=1)):
        uncertain = True
    unconsumed=set(tokens)-FILLER-consumed
    if (unconsumed or re.search(r"\b(?:driver|passenger)['’]s? side\b",text) or re.search(r"\b(?:left|right) of\b",text)
            or re.search(r"\b(?:first|second|third|last)\b",text)):
        uncertain=True
    return BayIntent(min(sides,default=None),min(orientations,default=None),vehicle_relation,tuple(vehicles),
                     tuple(dict.fromkeys(t for part in frontage for t in part)),min(ranks,default=None),
                     uncertain,blocked,tuple(frontage))


def _category(wanted, actual):
    if not actual:
        return None
    if wanted=="vehicle":
        return True
    cars={"car","sedan","suv","hatchback","coupe","taxi"}
    if wanted=="car":
        return actual in cars
    if wanted=="truck":
        return actual in {"truck","pickup"}
    if wanted=="van":
        return actual in {"van","minivan"}
    if (actual=="car" and wanted in cars or actual=="vehicle"
            or actual=="truck" and wanted=="pickup" or actual=="van" and wanted=="minivan"):
        return None
    return wanted==actual


def _vehicle_state(spec, appearance, detector_type):
    """False is explicit contradiction; None is an observation too coarse to decide."""
    text=str(appearance or "").casefold()
    tokens=_tokens(text)
    visible_types={_kind(m.group()) for m in VEHICLE.finditer(text)} if not UNCLEAR_OBS.search(text) else set()
    visible_colors=set(tokens)&COLORS if not UNCLEAR_OBS.search(text) else set()
    # 'pickup truck' is one type, while two conflicting alternative body labels are inconclusive.
    coarse = {"car":"car","sedan":"sedan","suv":"suv","bus":"bus","truck":"truck","van":"van",
              "pickup":"pickup","taxi":"taxi","motorcycle":"motorcycle","motorbike":"motorcycle"}
    actual=next(iter(visible_types)) if len(visible_types)==1 else coarse.get(detector_type)
    typ=_category(spec.kind,actual)
    if typ is False or (spec.color and visible_colors and spec.color not in visible_colors):
        return False
    if typ is not True or spec.color and visible_colors!={spec.color}:
        return None
    return True


def _nearby(box, option, frame):
    l,t,r,b=box
    px,py=option["point"]
    x0,y0,x1,y1=option["box"]
    if not 0<=l<r<=frame["width"] or not 0<=t<b<=frame["height"]:
        return False
    gap_x=max(x0-r,l-x1,0)
    gap_y=max(y0-b,t-y1,0)
    return (gap_x<=min(.18*frame["width"],max((r-l)*.65,(x1-x0)*.48))
            and gap_y<=min(.15*frame["height"],max((b-t)*.45,(y1-y0)*.5))
            and not (l<px<r and t<py<b))


def _brackets(first, second, option, frame):
    a,b=first,second
    w,h=frame["width"],frame["height"]
    p,q=((a[0]+a[2])/(2*w),a[3]/h),((b[0]+b[2])/(2*w),b[3]/h)
    cx,cy=option["point"][0]/w,option["point"][1]/h
    vx,vy=q[0]-p[0],q[1]-p[1]
    distance=vx*vx+vy*vy
    if distance<.0003:
        return False
    along=((cx-p[0])*vx+(cy-p[1])*vy)/distance
    if not .025<along<.975:
        return False
    l,t,r,d=option["box"]
    width,height=(r-l)/w,(d-t)/h
    perpendicular=math.hypot(cx-p[0]-along*vx,cy-p[1]-along*vy)
    return perpendicular<=max(.035,.78*math.hypot(width,height),.24*math.sqrt(distance))


def _vehicle_constraint(intent, option, scene, observation):
    from .bay_anchors import vehicles

    frame=scene["frame"]
    current=vehicles(scene)
    nearby={tid:(kind.casefold(),box) for tid,kind,box in current if _nearby(box,option,frame)}
    if intent.vehicle_relation in {"beyond","past","after","before","to the left of","to the right of"}:
        px,py=option["point"];width,height=frame["width"],frame["height"]
        def compatible(box):
            l,t,r,b=box
            cx=(l+r)/2
            if not (0<=l<r<=width and 0<=t<b<=height and abs(cx-px)<.28*width and abs(b-py)<.42*height):
                return False
            if intent.vehicle_relation.endswith("left of"):
                return px<cx
            if intent.vehicle_relation.endswith("right of"):
                return px>cx
            if (cx<.45*width and px>.55*width) or (cx>.55*width and px<.45*width):
                return False
            return py>=b-.025*height if intent.vehicle_relation=="before" else py<=b+.025*height
        nearby={tid:(kind.casefold(),box) for tid,kind,box in current if compatible(box)}
    signed=[]
    for row in observation.get("pair") or []:
        key=row.get("track_id")
        if key not in nearby or list(row.get("box",[]))!=nearby[key][1] or any(s[0]==key for s in signed):
            continue
        kind,box=nearby[key]
        signed.append((key,kind,box,row.get("appearance")))
    genuine_pair=len(signed)==2 and _brackets(signed[0][2],signed[1][2],option,frame)
    if intent.vehicle_relation=="between":
        if len(intent.vehicles)!=2:
            return False
        candidates=signed if genuine_pair else [(key,kind,box,None) for key,(kind,box) in nearby.items()]
        states=[]
        for first,second in permutations(candidates,2):
            if not _brackets(first[2],second[2],option,frame):
                continue
            match=[_vehicle_state(spec,c[3],c[1]) for spec,c in zip(intent.vehicles,(first,second))]
            if False not in match:
                states.append(genuine_pair and all(m is True for m in match))
        return True if any(states) else None if states else False
    if len(intent.vehicles)!=1:
        return False
    # A genuine per-vehicle crop is usable for 'beside' without requiring both boxes to bracket.
    observed_ids={v[0] for v in signed}
    pool=signed+[(key,kind,box,None) for key,(kind,box) in nearby.items() if key not in observed_ids]
    states=[_vehicle_state(intent.vehicles[0],appearance,kind) for _,kind,_,appearance in pool]
    if any(v is True for v in states):
        return True
    return None if any(v is None for v in states) else False


def _frontage_state(terms, parts):
    if not terms:
        return True
    wanted=set(terms)
    if any(wanted<=part for part in parts):
        return True
    requested_name = wanted - PLACE_CATEGORIES
    for part in parts:
        # Reading a different name on the same category of building cannot confirm this passenger's name.
        if (wanted & part & PLACE_CATEGORIES and requested_name and part-PLACE_CATEGORIES
                and not requested_name & part):
            continue
        if (wanted & part) - {"al","el","the"}:
            return None
    if not parts:
        return None
    return False


def _side(option, frame):
    x=option["point"][0]/frame["width"]
    return "left" if x<.47 else "right" if x>.53 else None


def _area(polygon):
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(polygon,polygon[1:]+polygon[:1])))/2


def _rank(rank, options, frame):
    if rank in {"nearest","farthest"}:
        if len({_side(o,frame) for o in options})!=1:
            return None
        axis=1
        key=lambda o:o["point"][axis]/frame["height"]
        minimum=.05
    elif rank in {"leftmost","rightmost"}:
        key=lambda o:o["point"][0]/frame["width"]
        minimum=.06
    else:
        ys=[o["point"][1] for o in options]
        if max(ys)-min(ys)>.08*frame["height"] or len({_side(o,frame) for o in options})!=1:
            return None
        key=(lambda o:_area(o["polygon"])) if rank in {"largest","smallest"} else (lambda o:o["box"][2]-o["box"][0])
        minimum=.2*max(key(o) for o in options)
    ordered=sorted(options,key=key,reverse=rank in {"nearest","rightmost","largest","widest"})
    return ordered[0] if abs(key(ordered[0])-key(ordered[1]))>minimum else None


def match(words, options, scene, observations, *, require_observed_landmark=True):
    """Return fully supported options and separate alternatives that require passenger clarification."""
    from .curbside_bays import RECIPE as parallel_geometry

    intent=parse(words)
    if intent is None or intent.blocked or scene.get("source")!="predicted":
        return [],[]
    frame=scene["frame"]
    supported,plausible=[],[]
    for option in options:
        side=_side(option,frame)
        if intent.side and side and side!=intent.side:
            continue
        evidence=option["evidence"]
        orientation=evidence.get("orientation")
        if evidence.get("recipe")==parallel_geometry:
            orientation="parallel"
        if intent.orientation and orientation and orientation!=intent.orientation:
            continue
        obs=(observations or {}).get(digest(option["polygon"]),{})
        phrases=[set(_tokens(part)) for part in str(obs.get("frontage") or "").split("|")
                 if part.strip() and not UNCLEAR_OBS.search(part.casefold())]
        states=[_frontage_state(terms,phrases) for terms in intent.frontage_phrases]
        frontage=False if False in states else None if None in states else True
        anchors=_vehicle_constraint(intent,option,scene,obs) if intent.vehicles else True
        if frontage is False or anchors is False:
            continue
        # A photo side and an unidentified nearby detection cannot locate a named frontage
        # and its neighbouring vehicle. One identified landmark can justify asking about the other.
        if (require_observed_landmark and intent.frontage and intent.vehicle_relation=="beside"
                and len(intent.vehicles)==1 and anchors is None and not any(state is True for state in states)):
            continue
        if intent.frontage and frontage is None and not phrases and not intent.vehicles:
            continue
        if intent.uncertain and not intent.frontage and not intent.vehicles and not intent.orientation:
            continue
        if (intent.uncertain or intent.side and not side or intent.orientation and not orientation
                or frontage is None or anchors is None):
            plausible.append(option)
        else:
            supported.append(option)
    if intent.rank and (intent.frontage or intent.vehicles):
        return [],[o for o in options if o in supported or o in plausible]
    if intent.rank and len(supported)+len(plausible)>1:
        if not plausible:
            first=_rank(intent.rank,supported,frame)
            if first is not None:
                return [first],[]
        return [],[o for o in options if o in supported or o in plausible]
    return supported,plausible
