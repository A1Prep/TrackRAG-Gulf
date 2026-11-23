"""Resolve a single described vehicle without dropping a second colour, name or comparison."""
from hashlib import sha256
from io import BytesIO
import math
import re

from PIL import Image

from . import bay_intent, single_vehicle


POLICY = "source_bound_vehicle_qualifiers_v1"
_RELATION = re.compile(r"\b(behind|to the rear of|next to|beside|alongside|in front of|by|near)\b")
_SIDE = re.compile(r"\b(?:on|over on|to) (?:the |our |my |your )?(left|right)(?:[- ]hand)?(?: side)?\b")
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


def parse(request):
    """Return only a qualified, direct one-vehicle reference; ordinary vehicle/place routes remain theirs."""
    text=_text(request.get("text",""))
    rel=_RELATION.search(text)
    if (not rel or not bay_intent.permitted_destination(text) or bay_intent.PARKING.search(text)
            or re.search(r"\b(?:between|both|either|instead|except|not|never|two)\b",text)
            or single_vehicle.PLACE_OF_VEHICLE.search(text)):
        return None
    before=text[:rel.start()]
    if set(before.split())-bay_intent.FILLER-{"just","immediately","directly","right","offload","out","please","u","can","could"}:
        return None
    phrase=text[rel.end():].strip()
    sides=_SIDE.findall(phrase)
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
    if makes:
        qualifiers.append({"field":"make","value":makes[0]})
    if len(body_colours)>1:
        qualifiers.append({"field":"unknown","value":"the vehicle’s colour combination"})
    additional=[w for w in head if w not in _KINDS and w not in _MAKES and w not in _COLOURS
                and w not in {"left","right","parked","ahead","vehicle","please"}]
    if additional:
        qualifiers.append({"field":"appearance","value":" ".join(additional)})
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
    if not qualifiers:
        return None
    relation=("behind" if rel.group() in {"behind","to the rear of"} else "front" if rel.group()=="in front of" else "beside")
    return {"kind":kinds[0] if kinds else "car","relation":relation,"color":next(iter(body_colours),None),
            "side":next(iter(sides),None),"unsupported":False,"qualified":True,"qualifiers":qualifiers}


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
                track_id,box=by_id[tag];caption=entry.get("reply")
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


def _detail_confirmed(q, receipts, spec, frame):
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


def _description(q):
    if q["field"]=="livery":
        return " ".join(v for v in (q["colour"],q["value"]) if v)+(f" on the {q['part']}" if q["part"] else "")
    if q["field"]=="comparison":
        return "which vehicle is nearest the "+q["value"]
    if q["field"]=="place":
        return "its position "+q["relation"]+" "+q["value"]
    if q["field"]=="make":
        return "the "+q["value"].title()+" make"
    return q["value"]


def resolve(request,ref,specs,answers,*,observations=()):
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
                ok=_detail_confirmed(q,records,spec,frame)
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
