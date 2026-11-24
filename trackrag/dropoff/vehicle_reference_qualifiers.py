"""Resolve a single described vehicle without dropping a second colour, name or comparison."""
from hashlib import sha256
from io import BytesIO
import math
import re

from PIL import Image

from . import bay_intent, single_vehicle


POLICY = "source_bound_vehicle_qualifiers_v1"
_LEGACY_RELATION = re.compile(r"\b(behind|to the rear of|next to|beside|alongside|in front of|by|near)\b")
_RELATION = re.compile(r"\b(behind|to the rear of|next to|beside|alongside|in front of|by|near|"
                       r"(?:on |to )?(?:the )?(left|right)(?: hand)?(?: side)? of)\b")
_SIDE = re.compile(r"\b(?:on|over on|to) (?:the |our |my |your )?(left|right)(?:[- ]hand)?(?: side)?\b")
_GROUND_SIDE = re.compile(r"\b(?:(?:and |just |specifically )*)(?:on|to) (?:its (left|right)(?: hand)?(?: side)?|"
                          r"(?:the )?(left|right)(?: hand)?(?: side)? of (?:it|that vehicle|that car))\b")
_GROUND_CONTEXT = re.compile(r"\b(?:in|into|at) (?:the |that |a |an )?((?:[a-z]+ ){0,3}"
                             r"(?:gap|pull in|driveway|recess|verge|shoulder|apron|forecourt))\b")
_DETAIL = re.compile(r"\b(with|bearing|sporting|showing|nearest(?: to)?|closest(?: to)?|outside(?: of)?|"
                     r"in front of|beneath|under|alongside|next to|beside|near|by)\b")
_KINDS = {"vehicle":"vehicle","car":"car","auto":"car","automobile":"car","sedan":"sedan","saloon":"sedan",
          "suv":"suv","crossover":"suv","pickup":"pickup","van":"van","minivan":"van","lorry":"truck",
          "truck":"truck","bus":"bus","hatchback":"car","coupe":"car","taxi":"car"}
_MAKES = set("mini jeep toyota nissan mercedes bmw ford honda kia hyundai lexus audi volkswagen chevrolet "
             "tesla porsche suzuki mitsubishi dodge mazda subaru peugeot renault volvo landrover jaguar".split())
_COLOURS = single_vehicle.COLORS | {"cream","beige","gold","purple"}
_LIVERY = {"design":"design","graphic":"design","graphics":"design","pattern":"design","motif":"design",
           "artwork":"design","stripe":"stripe","stripes":"stripe","striping":"stripe","decal":"decal",
           "decals":"decal","sticker":"decal","stickers":"decal","logo":"logo","logos":"logo",
           "lettering":"lettering","branding":"branding","livery":"livery","marking":"mark","markings":"mark"}
_PARTS = {"door":"door","doors":"door","hood":"bonnet","bonnet":"bonnet","roof":"roof","side":"side",
          "sidewall":"side","bumper":"bumper","window":"window","tailgate":"tailgate","boot":"boot","trunk":"boot"}
_COMPARATORS = {"motorcycles":"motorcycle","motorcycle":"motorcycle","motorbikes":"motorcycle",
                "motorbike":"motorcycle","mopeds":"moped","moped":"moped","buses":"bus","bus":"bus",
                "trucks":"truck","truck":"truck","vans":"van","van":"van","bicycles":"bicycle",
                "bicycle":"bicycle","cyclists":"cyclist","cyclist":"cyclist","pedestrians":"pedestrian",
                "pedestrian":"pedestrian","people":"pedestrian","person":"pedestrian"}
_DETERMINERS = {"a","an","the","that","this","there","one","please","my","our","its","it","of","on","with","just"}
_BAD = re.compile(r"\b(?:none|unknown|unreadable|unclear|unidentified|unverified|possibly|maybe|cannot|"
                  r"without|no|not|isn't|can't|either|or)\b")


def _text(text):
    s=str(text).casefold().replace("’","'").replace("grey","gray")
    s=re.sub(r"\bpick[- ]up\b","pickup",s)
    s=re.sub(r"\bland[ -]rover\b","landrover",s)
    return " ".join(re.findall(r"[a-z0-9']+",s))


def _content(text):
    return [w for w in _text(text).split() if w not in _DETERMINERS]


def parse(request, *, expanded=True):
    """Return only a qualified, direct one-vehicle reference; ordinary vehicle/place routes remain theirs."""
    text=_text(request.get("text",""))
    rel=(_RELATION if expanded else _LEGACY_RELATION).search(text)
    if (not rel or not bay_intent.permitted_destination(text) or bay_intent.PARKING.search(text)
            or re.search(r"\b(?:between|both|either|instead|except|not|never|two)\b",text)
            or expanded and re.search(r"\bor\b",text)
            or single_vehicle.PLACE_OF_VEHICLE.search(text)):
        return None
    before=text[:rel.start()]
    if set(before.split())-bay_intent.FILLER-{"just","immediately","directly","right","offload","out","please","u","can","could"}:
        return None
    phrase=text[rel.end():].strip()
    ground_sides=[next(side for side in match.groups() if side) for match in _GROUND_SIDE.finditer(phrase)] if expanded else []
    if expanded and rel.group(2):
        ground_sides.append(rel.group(2))
    if len(set(ground_sides))>1:
        return None
    if expanded:
        phrase=_GROUND_SIDE.sub(" ",phrase)
    grounds=_GROUND_CONTEXT.findall(phrase) if expanded else []
    if grounds:
        phrase=_GROUND_CONTEXT.sub(" ",phrase)
    if expanded and "and" in phrase.split():
        # A bounded single-car graphic may contain a conjunction. A coordinated second referent cannot.
        detail_start=_DETAIL.search(phrase)
        remainder=phrase.split(" and ",1)[1].split()
        if (not detail_start or detail_start.group() not in {"with","bearing","sporting","showing"}
                or "and" in phrase[:detail_start.start()].split() or set(remainder)&(_KINDS.keys()|_MAKES|{"one","another"})):
            return None
    first_detail=_DETAIL.search(phrase)
    if expanded and first_detail:
        sides=_SIDE.findall(phrase[:first_detail.start()])
        ambiguous_side=bool(_SIDE.search(phrase[first_detail.end():]))
    else:
        sides=_SIDE.findall(phrase)
        ambiguous_side=False
    phrase=_SIDE.sub(" ",phrase)
    clips=list(_DETAIL.finditer(phrase))
    head=_content(phrase[:clips[0].start()] if clips else phrase)
    head=[re.sub(r"^(left|right)most$",r"\1",t) for t in head]
    sides.extend(t for t in head if t in ("left","right"))
    if len(set(sides))>1:
        return None
    makes=list(dict.fromkeys(t for t in head if t in _MAKES))
    shapes=list(dict.fromkeys(t for t in head if t in _KINDS))
    if shapes==["pickup","truck"]:
        shapes=["pickup"]
        head=[w for w in head if w!="truck"]
    if (not makes and not shapes) or len(makes)>1 or len(shapes)>1:
        return None
    kinds=[_KINDS[t] for t in shapes]
    body_colours=list(dict.fromkeys(t for t in head if t in _COLOURS))
    qualifiers=[]
    if ambiguous_side:
        qualifiers.append({"field":"unknown","value":"which side the extra detail refers to"})
    if makes:
        qualifiers.append({"field":"make","value":makes[0]})
    if len(body_colours)>1:
        qualifiers.append({"field":"unknown","value":"the vehicle’s colour combination"})
    additional=[w for w in head if w not in _KINDS and w not in _MAKES and w not in _COLOURS
                and w not in {"left","right","parked","ahead","vehicle","please"}]
    if additional:
        qualifier={"field":"appearance","value":" ".join(additional)}
        if expanded:
            qualifier["source"]="body_identity"
        qualifiers.append(qualifier)
    if shapes and shapes[0] in ("taxi","hatchback","coupe","crossover"):
        qualifiers.append({"field":"appearance","value":shapes[0]})
    for i,part in enumerate(clips):
        connector=part.group()
        raw=phrase[part.end():clips[i+1].start() if i+1<len(clips) else None]
        ts=_content(raw)
        if not ts:
            qualifiers.append({"field":"unknown","value":"the extra detail"})
        elif connector in {"with","bearing","sporting","showing"}:
            marks=list(dict.fromkeys(_LIVERY[v] for v in ts if v in _LIVERY))
            colours=list(dict.fromkeys(v for v in ts if v in _COLOURS))
            parts=list(dict.fromkeys(_PARTS[v] for v in ts if v in _PARTS))
            leftover=set(ts)-_LIVERY.keys()-_COLOURS-_PARTS.keys()-{"paint","painted","body","panel","panels","on","its","it"}
            if len(marks)==1 and len(colours)<=1 and len(parts)<=1 and not leftover:
                qualifiers.append({"field":"livery","value":marks[0],"colour":next(iter(colours),None),"part":next(iter(parts),None)})
            else:
                qualifiers.append({"field":"appearance","value":" ".join(ts)[:65]})
        elif connector.startswith(("nearest","closest")):
            anchor=_COMPARATORS.get(ts[-1])
            simple=len(ts)==1
            qualifiers.append({"field":"comparison" if anchor and simple else "unknown",
                               "value":anchor if anchor and simple else "the relative position"})
        else:
            qualifiers.append({"field":"place","value":" ".join(ts)[:65],"relation":connector})
    qualifiers.extend({"field":"ground","value":g} for g in grounds)
    if not qualifiers and not ground_sides:
        return None
    relation=("behind" if rel.group() in {"behind","to the rear of"} else "front" if rel.group()=="in front of" else "beside")
    ref={"kind":kinds[0] if kinds else "car","relation":relation,"color":next(iter(body_colours),None),
         "side":next(iter(sides),None),"unsupported":False,"qualified":True,"qualifiers":qualifiers}
    if expanded:
        ref["ground_side"]=next(iter(ground_sides),None)
    return ref


def packets_for(key):
    """Collect available existing receipts without any model calls or network access."""
    from . import bay_anchors,roadside_context,surface_stalls,visual_space
    raw={"anchor":bay_anchors.get_packet(key),"row_anchor":bay_anchors.get_row_packet(key),
         "roadside":roadside_context.get_packet(key),"gravel":roadside_context.get_packet(key,gravel_only=True),
         "paved":visual_space.get_packet(key,paved=True),"extended":visual_space.get_extension_packet(key)}
    if raw["roadside"] or raw["gravel"]:
        raw["stall_review"]=surface_stalls.get_packet(key)
    return {name:packet for name,packet in raw.items() if packet is not None}


def read_observations(request,photo,surface,*,packets=None):
    """Rebuild only descriptions attached to an original crop of the same current track."""
    from . import bay_anchors,roadside_context,visual_space
    from .parking_evidence import source_input
    clean,key,_=source_input(request,photo)
    data=packets_for(key) if packets is None else packets
    if not isinstance(data,dict) or set(data)-{"anchor","row_anchor","roadside","gravel","stall_review","paved","extended"}:
        raise ValueError("Unknown vehicle-detail evidence source.")
    scene=clean["scene"]
    observations=[]
    with Image.open(BytesIO(photo)) as image:
        for bank in ("anchor","row_anchor"):
            if bank not in data:
                continue
            body,tracks=bay_anchors.checked_context(data[bank],scene,image.size)
            by_id={str(track_id):(track_id,box) for track_id,_,box in tracks}
            for tag,entry in body["appearance"].items():
                if tag not in by_id:
                    raise ValueError("Vehicle appearance does not belong to a current track.")
                track_id,box=by_id[tag];caption=entry.get("reply") if isinstance(entry,dict) else None
                if (not isinstance(caption,str) or len(caption)>55
                        or sha256(bay_anchors.crop_car(image,box)).hexdigest()!=entry.get("view")):
                    raise ValueError("Vehicle appearance differs from its original crop.")
                observations.append({"track_id":track_id,"box":list(box),"appearance":caption})
    options=[]
    for name in ("roadside","gravel"):
        if name in data:
            options.extend(roadside_context.verified(clean,photo,surface,data.get("stall_review"),data[name],gravel_only=name=="gravel"))
    if "paved" in data:
        for blocked in (False,True):
            options.extend(visual_space.verified(scene,photo,surface,data["paved"],paved=True,references=blocked))
    if "extended" in data:
        options.extend(visual_space.verified_extension(scene,photo,surface,data["extended"]))
    for option in options:
        crops=option.get("vehicle_crops",[])
        if len(crops)!=2:
            continue
        desc=option["descriptions"]
        for (track_id,box),field in zip(crops,("nearer_vehicle","farther_vehicle"),strict=True):
            observations.append({"track_id":track_id,"box":list(box),"appearance":desc[field],
                "frontage":desc["frontage"],"roadside":desc["roadside"],"point":option["point"]})
    return observations


def _safe_words(caption):
    clean=_text(caption or "")
    return clean.split() if clean and not _BAD.search(clean) else []


def _has(need, words):
    phrase=" "+" ".join(words)+" "
    return bool(need) and " "+" ".join(need)+" " in phrase


def _body_words(appearance):
    if not _safe_words(appearance):
        return []
    head=re.split(r"[,;|()]|\b(?:with|bearing|sporting|featuring|displaying|showing|beside|next|"
                  r"near|by|behind|alongside|outside|adjacent|decorated|branded|and|in front of)\b",
                  appearance.casefold(),maxsplit=1)[0]
    words=_text(head).split()
    graphic=next((i for i,w in enumerate(words) if w in _LIVERY),len(words))
    if graphic<len(words):
        # An unpunctuated 'Jeep logo' is evidence for an emblem, including when it follows a body noun.
        graphic=max(0,graphic-1)
    return words[:graphic]


def _identity(appearance):
    """Read the described vehicle itself, excluding markings, neighbouring objects and logos."""
    words=_body_words(appearance)
    identity_end=next((i for i,w in enumerate(words) if w in _KINDS or w in _MAKES),len(words))
    colours={w for w in words[:identity_end] if w in _COLOURS}
    kinds={_KINDS[w] for w in words if w in _KINDS}-{"vehicle"}
    if kinds=={"pickup","truck"}:
        kinds={"pickup"}
    if len(kinds)>1:
        kinds.discard("car")
    return (next(iter(colours)) if len(colours)==1 else None,
            next(iter(kinds)) if len(kinds)==1 else None,set(words)&_MAKES)


def _body_reading(answer, records):
    """An independent crop may refine generic body labels. Orientation and ground always come from the original reader."""
    original_color,original_kind=answer[:2]
    labels=[_identity(row.get("appearance")) for row in records]
    colours={c for c,_,_ in labels if c}
    kinds={k for _,k,_ in labels if k}
    if original_color!="unknown":
        colours.add(original_color)
    if original_kind!="unknown":
        kinds.add(original_kind)
    if len(kinds)>1 and kinds<={"car","suv","sedan"}:
        kinds.discard("car")
    conflict=len(colours)>1 or len(kinds)>1
    color=next(iter(colours)) if len(colours)==1 else original_color
    kind=next(iter(kinds)) if len(kinds)==1 else original_kind
    return [color,kind,*answer[2:]],conflict


def _part_on_mark(words,index,expected):
    # Tie the graphic to its own panel rather than to another decoration in the same caption.
    if not expected:
        return True
    tail=[]
    for w in words[index+1:index+8]:
        if w in {"and","but","beside","near","with","while"} or w in _LIVERY:
            break
        tail.append(w)
    return (any(_PARTS.get(w)==expected for w in tail) or index>=1 and _PARTS.get(words[index-1])==expected)


def _detail_confirmed(q, receipts, spec, frame):
    field=q["field"]
    for r in receipts:
        caption=r.get("appearance") or ""
        own=re.split(r"\b(?:beside|next to|in front of|behind|alongside|opposite|near|adjacent to)\b",caption,maxsplit=1,flags=re.I)[0]
        words=_safe_words(own)
        if field=="make" and _identity(r.get("appearance"))[2]=={q["value"]}:
            return True
        if field=="appearance" and _has(q["value"].split(),_body_words(own) if q.get("source")=="body_identity" else words):
            return True
        if field=="livery":
            for j,token in enumerate(words):
                sign=_LIVERY.get(token)
                if sign!=q["value"] and not (q["value"] in {"design","mark","livery"} and sign is not None):
                    continue
                colour=q["colour"]
                if colour and not (j>=1 and words[j-1]==colour or j>=2 and words[j-2]==colour and
                                   (words[j-1] in ("painted","body") or _PARTS.get(words[j-1])==q["part"])):
                    continue
                part=q["part"]
                if _part_on_mark(words,j,part):
                    return True
        if field=="place" and r.get("point"):
            x,y=r["point"];l,t,rr,b=spec["box"]
            if (max(l-x,x-rr,0)>.65*(rr-l) or abs(y-b)>.70*(b-t)):
                continue
            location=q["value"].split()
            for caption in (r.get("frontage"),r.get("roadside")):
                observed=_safe_words(caption)
                if set(observed)&{"across","opposite","distant","advert","billboard","reflection"} or not _has(location,observed):
                    continue
                if q["relation"] in {"under","beneath"} and not set(observed)&{"overhead","under","above","beneath"}:
                    continue
                return True
    return False


def _comparative(scene, matching, whole_specs, anchor_type):
    frame=scene["frame"];width,height=frame["width"],frame["height"]
    current=[t for t in scene["tracks"] if t["frame_id"]==frame["frame_id"]]
    targets=[t["bbox_xyxy"] for t in current if _COMPARATORS.get(_text(t["class_name"]),_text(t["class_name"]))==anchor_type]
    if not targets:
        return None
    def distance(box):
        middle=lambda b:((b[0]+b[2])/(2*width),b[3]/height)
        x,y=middle(box)
        return min(math.hypot(x-a,y-b) for a,b in map(middle,targets))
    ranked=sorted((distance(s["box"]),s["track_id"]) for s,_ in matching)
    if not ranked or ranked[0][0]>.28:
        return None
    best,tid=ranked[0]
    margin=max(.035,.23*best)
    if len(ranked)>1 and ranked[1][0]-best<margin:
        return None
    identified={s["track_id"] for s in whole_specs}
    for row in current:
        if row["track_id"] not in identified and _text(row["class_name"]) in {"car","suv","sedan","truck","van","pickup","bus","light vehicle","medium vehicle"}:
            if distance(row["bbox_xyxy"])<best+margin:
                return None
    return tid


def _current_comparison(scene,matching,described,anchor_type):
    """Compare visible envelopes at similar projected depth; a distant speck by a large bike cannot prove proximity."""
    frame=scene["frame"]
    width,height=frame["width"],frame["height"]
    current=[r for r in scene["tracks"] if r["frame_id"]==frame["frame_id"]]
    targets=[(r["track_id"],r["bbox_xyxy"]) for r in current
             if _COMPARATORS.get(_text(r["class_name"]),_text(r["class_name"]))==anchor_type]
    if not targets:
        return None
    def distance(box,identity):
        left,top,right,bottom=box
        h=bottom-top
        scores=[]
        for target,(x0,y0,x1,y1) in targets:
            if target==identity:
                continue
            other_h=y1-y0
            if h<=0 or other_h<=0 or not .40<=h/other_h<=3.0 or abs(bottom-y1)>.80*max(h,other_h):
                continue
            dx=max(left-x1,x0-right,0)/width
            scores.append(math.hypot(dx,(bottom-y1)/height))
        return min(scores,default=math.inf)
    ranked=sorted((distance(s["box"],s["track_id"]),s["track_id"]) for s,_ in matching)
    if not ranked or ranked[0][0]>.28:
        return None
    best,track_id=ranked[0]
    margin=max(.035,.23*best)
    if len(ranked)>1 and ranked[1][0]-best<margin:
        return None
    classified={s["track_id"] for s in described}
    for r in current:
        if (r["track_id"] not in classified and _text(r["class_name"]) in
                {"car","suv","sedan","truck","van","pickup","bus","light vehicle","medium vehicle"}
                and distance(r["bbox_xyxy"],r["track_id"])<best+margin):
            return None
    return track_id


def _description(q):
    if q["field"]=="livery":
        return " ".join(v for v in (q["colour"],q["value"]) if v)+(f" on the {q['part']}" if q["part"] else "")
    if q["field"]=="comparison":
        return "which vehicle is nearest the "+q["value"]
    if q["field"]=="place":
        return "its position "+q["relation"]+" "+q["value"]
    if q["field"]=="make":
        return "the "+q["value"].title()+" make"
    if q["field"]=="ground":
        return "the requested "+q["value"]
    return q["value"]


def possible_side_reference(request, ref, specs, answers, *, observations=()):
    """Offer a unique body as a question when road-side and image-side descriptions may differ."""
    side=ref.get("side")
    words=_text(request["text"]).split()
    if (side not in {"left","right"} or ref.get("ground_side") or ref.get("unsupported")
            or set(words)&{"screen","photo","picture","image","frame","leftmost","rightmost"}):
        return None
    details=ref.get("qualifiers",[])
    if any(q.get("field")!="appearance" or q.get("value")!="dark" for q in details):
        return None
    wants_dark=bool(details)
    width=request["scene"]["frame"]["width"]
    plausible=[]
    for spec,row in zip(specs,answers,strict=True):
        color,kind=row[:2]
        if (kind=="unknown" or color=="unknown" or not single_vehicle._compatible(ref["kind"],kind)
                or ref.get("color") and color!=ref["color"]):
            continue
        linked=[r for r in observations if r.get("track_id")==spec["track_id"] and r.get("box")==spec["box"]]
        explicit_dark=False
        conflict=False
        for r in linked:
            caption=r.get("appearance")
            shade,observed_kind,_=_identity(caption)
            if (shade and shade!=color or observed_kind and not
                    (single_vehicle._compatible(kind,observed_kind) or single_vehicle._compatible(observed_kind,kind))):
                conflict=True
            names=_body_words(caption)
            if _has(["dark",color],names) and observed_kind and single_vehicle._compatible(ref["kind"],observed_kind):
                explicit_dark=True
        if conflict or wants_dark and color!="black" and not explicit_dark:
            continue
        if single_vehicle._side(spec["box"],width,side):
            return None
        label=(("dark " if explicit_dark and color!="black" else "")+color+" "+("SUV" if kind=="suv" else kind))
        plausible.append((spec["box"],label))
    if len(plausible)!=1:
        return None
    box,label=plausible[0]
    mid=(box[0]+box[2])/2
    position="left of centre" if mid<width/2 else "right of centre" if mid>width/2 else "in the centre"
    return [box],f"I can see a {label} {position} in the photo. Is that the vehicle you meant by “on the {side}”?"


def resolve(request,ref,specs,answers,*,observations=(),expanded=True):
    """Return qualified matches, drawable anchor boxes, and a clarification unless a unique match is proved."""
    if not expanded:
        return _legacy_resolve(request,ref,specs,answers,observations=observations)
    frame=request["scene"]["frame"]
    if len(specs)!=len(answers):
        raise ValueError("Vehicle readings and current crops differ.")
    evidence={}
    for record in observations:
        evidence.setdefault((record.get("track_id"),tuple(record.get("box",[]))),[]).append(record)
    subjects=[]
    contradictions=set()
    visible_identities={}
    for spec,answer in zip(specs,answers,strict=True):
        records=evidence.get((spec["track_id"],tuple(spec["box"])),[])
        answer,conflicting=_body_reading(answer,records)
        visible_identities[spec["track_id"]]={(answer[0],answer[1]),
                 *((c,k) for c,k,_ in (_identity(r.get("appearance")) for r in records) if c and k)}
        if conflicting:
            contradictions.add(spec["track_id"])
        subjects.append((spec,answer))
    def matches_body(spec,color):
        return any(k!="unknown" and single_vehicle._compatible(ref["kind"],k) and (not color or c==color)
                   for c,k in visible_identities[spec["track_id"]])
    sided=[(s,a) for s,a in subjects if single_vehicle._side(s["box"],frame["width"],ref["side"])]
    base=[(s,a) for s,a in sided if matches_body(s,ref["color"])]
    shade={"gray":"silver","silver":"gray"}.get(ref["color"])
    alternate=[(s,a) for s,a in sided if shade and matches_body(s,shade) and (s,a) not in base]
    if base and all(s["track_id"] in contradictions for s,_ in base):
        base.extend(alternate)
    if not base:
        if alternate:
            similar=[s["box"] for s,_ in alternate]
            extras=[_description(q) for q in ref["qualifiers"] if q["field"]!="unknown"]
            detail=" or "+" and ".join(extras[:2]) if extras else ""
            return [],similar,f"The outlined vehicle{'s look' if len(similar)>1 else ' looks'} {shade} in this photo. I couldn’t confirm the {ref['color']} shade{detail}. Is the vehicle you mean outlined? You can mark the ground you want."
        return [],[], "I couldn’t verify the described vehicle in this photo. Could you give another visible detail or mark where you mean?"
    qualifiers=ref["qualifiers"]
    comparisons={q["value"]:_current_comparison(request["scene"],base,
                 [s for s,a in subjects if a[0]!="unknown" and a[1]!="unknown" and s["track_id"] not in contradictions],q["value"])
                 for q in qualifiers if q["field"]=="comparison"}
    results=[]
    for spec,answer in base:
        records=evidence.get((spec["track_id"],tuple(spec["box"])),[])
        observed_makes=set().union(*(_identity(r.get("appearance"))[2] for r in records))
        wanted_make=next((q["value"] for q in qualifiers if q["field"]=="make"),None)
        if wanted_make and observed_makes and wanted_make not in observed_makes:
            continue
        unresolved=[]
        excluded=False
        for i,q in enumerate(qualifiers):
            if q["field"]=="comparison":
                winner=comparisons[q["value"]]
                if winner is not None and winner!=spec["track_id"]:
                    excluded=True
                    break
                ok=winner==spec["track_id"]
            else:
                ok=_detail_confirmed(q,records,spec,frame)
            if not ok:
                unresolved.append(i)
        if not excluded:
            results.append((spec,answer,unresolved))
    if not results:
        return [],[],"The vehicle descriptions I can verify differ from the requested one. Could you give another detail or mark where you mean?"
    boxes=[s["box"] for s,_,_ in results]
    remaining=sorted({i for _,_,unresolved in results for i in unresolved})
    where={"behind":"behind it","beside":"beside it","front":"in front of it"}[ref["relation"]]
    if ref.get("ground_side"):
        where=(where+" and " if ref["relation"]!="beside" else "")+"on its "+ref["ground_side"]
    if any(s["track_id"] in contradictions for s,_,_ in results):
        landmark=next((q["value"] for i,q in enumerate(qualifiers) if q["field"]=="place" and i not in remaining),None)
        place=f"I can see the {landmark.title()} frontage. " if landmark else ""
        makeup=next((q["value"] for i,q in enumerate(qualifiers) if q["field"]=="make" and i in remaining),None)
        limit=f" I couldn’t confirm the {makeup.title()} make." if makeup else ""
        return [],boxes,(place+"The close-up readings disagree about the outlined vehicle"+
                         ("s’ colours or body types." if len(boxes)>1 else "’s colour or body type.")+
                         limit+f" Could you identify which vehicle you mean or mark the ground {where}?")
    if len(results)==1 and not results[0][2] and ref["relation"]!="front" and not ref.get("ground_side"):
        return [(results[0][0],results[0][1])],[results[0][0]["box"]],None
    if remaining:
        details=" and ".join(_description(qualifiers[i]) for i in remaining[:2])
        return [],boxes,f"I can outline the vehicle{'s' if len(boxes)>1 else ''} matching the basic description. I couldn’t verify {details}. Could you identify it another way or mark the ground {where}?"
    if ref["relation"]=="front":
        return [],boxes,"I can outline the matching vehicle. Could you mark the ground in front of it so I can check that location?"
    if ref.get("ground_side") and len(results)==1:
        return [],boxes,f"I can outline the vehicle. Could you mark the ground {where} so I can check that location?"
    return [],boxes,"Several outlined vehicles match those details. Which one did you mean? You can mark its location on the photo."


# Archived qualifier policy for signed sessions that predate the explicit ground-side field.
def _legacy_detail_confirmed(q, receipts, spec, frame):
    field=q["field"]
    for r in receipts:
        words=_safe_words(r.get("appearance"))
        if field in {"make","appearance"} and _has(q["value"].split(),words):
            return True
        if field=="livery":
            for j,token in enumerate(words):
                sign=_LIVERY.get(token)
                if sign!=q["value"] and not (q["value"] in {"design","mark","livery"} and sign is not None):
                    continue
                colour=q["colour"]
                if colour and not (j>=1 and words[j-1]==colour or j>=2 and words[j-2:j] in ([colour,"painted"],[colour,"body"])):
                    continue
                part=q["part"]
                if not part or any(_PARTS.get(w)==part for w in words[max(0,j-2):j+7]):
                    return True
        if field=="place" and r.get("point"):
            x,y=r["point"];l,t,rr,b=spec["box"]
            if (max(l-x,x-rr,0)>.65*(rr-l) or abs(y-b)>.70*(b-t)):
                continue
            location=q["value"].split()
            for caption in (r.get("frontage"),r.get("roadside")):
                observed=_safe_words(caption)
                if not _has(location,observed):
                    continue
                if q["relation"] in {"under","beneath"} and not set(observed)&{"overhead","under","above","beneath"}:
                    continue
                return True
    return False


def _legacy_resolve(request,ref,specs,answers,*,observations=()):
    """Return qualified matches, drawable anchor boxes, and a clarification unless a unique match is proved."""
    frame=request["scene"]["frame"]
    if len(specs)!=len(answers):
        raise ValueError("Vehicle readings and current crops differ.")
    base=[(s,a) for s,a in zip(specs,answers,strict=True)
          if a[1]!="unknown" and single_vehicle._compatible(ref["kind"],a[1])
          and (not ref["color"] or a[0]==ref["color"]) and single_vehicle._side(s["box"],frame["width"],ref["side"])]
    shown=[s["box"] for s,_ in base]
    if not base:
        return [],[], "I couldn’t verify the described vehicle in this photo. Could you give another visible detail or mark where you mean?"
    qualifiers=ref["qualifiers"]
    comparisons={q["value"]:_comparative(request["scene"],base,
                 [s for s,a in zip(specs,answers,strict=True) if a[0]!="unknown" and a[1]!="unknown"],q["value"])
                 for q in qualifiers if q["field"]=="comparison"}
    results=[]
    for spec,answer in base:
        records=[r for r in observations if r.get("track_id")==spec["track_id"] and list(r.get("box",[]))==list(spec["box"])]
        unresolved=[]
        excluded=False
        for i,q in enumerate(qualifiers):
            if q["field"]=="comparison":
                winner=comparisons[q["value"]]
                if winner is not None and winner!=spec["track_id"]:
                    excluded=True
                    break
                ok=winner==spec["track_id"]
            else:
                ok=_legacy_detail_confirmed(q,records,spec,frame)
            if not ok:
                unresolved.append(i)
        if not excluded:
            results.append((spec,answer,unresolved))
    if len(results)==1 and not results[0][2] and ref["relation"]!="front":
        return [(results[0][0],results[0][1])],[results[0][0]["box"]],None
    remaining=sorted({i for _,_,unresolved in results for i in unresolved})
    boxes=[s["box"] for s,_,_ in results] or shown
    where={"behind":"behind it","beside":"beside it","front":"in front of it"}[ref["relation"]]
    if remaining:
        details=" and ".join(_description(qualifiers[i]) for i in remaining[:2])
        return [],boxes,f"I can outline the vehicle{'s' if len(boxes)>1 else ''} matching the basic description. I couldn’t verify {details}. Could you identify it another way or mark the ground {where}?"
    if ref["relation"]=="front":
        return [],boxes,"I can outline the matching vehicle. Could you mark the ground in front of it so I can check that location?"
    return [],boxes,"Several outlined vehicles match those details. Which one did you mean? You can mark its location on the photo."
