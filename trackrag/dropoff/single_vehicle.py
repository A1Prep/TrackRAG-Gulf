"""Ground 'behind one vehicle' in a current original, with separate evidence for the rear and parking surface."""
import base64
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import zlib

from PIL import Image, ImageDraw

from .parking_evidence import _authenticate, _decode_map, source_input
from .passenger import digest, proposed, unresolved
from .place_grounding import call_identity
from .referent import ReferentRanker, overlap
from .single_vehicle_surface import behind_patch, beside_patches

METHOD = "single_vehicle_behind_parking_v1"
RECIPE = "single_vehicle_rear_observations_v1"
PREVIOUS_POLICY = "single_vehicle_rear_and_side_v2"
POLICY = "single_vehicle_visible_rear_v3"
RANKER = "d96a9a69e2e52ddbc21851d5ff6f79b344de6c8558600e54be5cb2712480a07d"
MODELS = {"ranker_sha256": RANKER,"base":"10d842936325905934ea24e64325d774693f166e",
          "adapter_sha256":"76bfb3230b17a86248b52ca6a9f8d2ffca324aa80216d6b93845ccb12ad86743",
          "processor":"0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
          "parking_segmenter":"4772b6bf101d91f2534c106dc524d906aeb3c68a"}
COLORS = {"white","black","silver","gray","red","blue","green","yellow","orange","brown"}
TYPES = {"car","truck","bus","van","suv","sedan"}
RESPONSES = {(k+" "+c).upper() for k in TYPES for c in COLORS|{"unknown"}}
GUIDE = (
    "Inspect this street photograph. The left panel shows the scene and the right panel magnifies the same location. "
    "The magenta outline identifies ONE vehicle. The cyan ring is a proposed patch of ground. "
    "Check that you see the rear of the outlined vehicle facing the camera (trunk or rear doors and rear lights), "
    "and the cyan ring immediately behind it is empty parking or a distinct off-lane pull-in. "
    "If the vehicle faces the camera front-first, you see mainly its side, the ring is in a traffic or gate approach lane, "
    "any of these conditions is uncertain, or the area is occupied, answer NONE. "
    "Otherwise report two words only: the outlined vehicle's type (SEDAN, SUV, CAR, VAN, TRUCK, BUS) "
    "and its visible body paint (WHITE, BLACK, SILVER, GRAY, RED, BLUE, GREEN, YELLOW, ORANGE, BROWN, UNKNOWN). "
    "Judge the actual paint in context, including shade. Do not use a nearby object's colour."
)
WORDS = "Check the outlined vehicle and circled parking position."
NOUN = re.compile(r"\b(pick[- ]?up(?:\s+truck)?|car|sedan|suv|truck|lorry|bus|van|vehicle)\b",re.I)
RELATION = re.compile(r"\b(behind|next\s+to|beside|alongside)\b")
PLACE_OF_VEHICLE = re.compile(r"\b(?:bus stop|bus shelter|truck stop|car park|car wash|bus station)\b",re.I)
ASK = {
    "ambiguous":"Which vehicle do you mean? Tell me its colour or side, or mark the spot on the photo.",
    "unsupported":"I can check a spot behind one vehicle. Could you name its colour and side, or mark the spot?",
    "no_supported_space":"I can’t see a clear off-lane space immediately behind that vehicle. Could you mark a different spot or name another landmark?",
    "unverified_vehicle":"I couldn’t match that vehicle and its rear to a clear off-lane space. Could you mark the spot or describe another vehicle?",
    "rear_view_unclear":"I can’t tell exactly where the space behind that vehicle begins from this angle. Could you mark the area you mean?",
    "service_unavailable":"I couldn’t finish checking the vehicle and the space behind it. You can mark a spot on this photo or describe another landmark.",
    "side_service_unavailable":"I couldn’t finish checking the parking space beside that vehicle. Could you mark a spot on the photo?",
    "side_unsupported":"Which vehicle and which side do you mean? You can also mark the spot beside it on the photo.",
    "no_supported_side_space":"I can’t see a clear off-lane space beside that vehicle. Could you mark a different spot or name another landmark?",
    "side_needs_mark":"Which spot beside that vehicle do you mean? Please mark the area on the photo so you can check it.",
    "side_unverified_vehicle":"I couldn’t verify a vehicle with that description beside an open parking space. Could you mark the spot you mean?",
    "pending":"I’m checking this photo. You can mark the spot now, or try your request again in a moment.",
    "busy":"The photo check is busy. You can mark the spot now, or try your request again in a moment.",
}


def request_reference(request):
    """Recognize the direct spatial relation to one vehicle in the passenger's own words."""
    words=" ".join(request["text"].casefold().split())
    relations=list(RELATION.finditer(words))
    if (len(relations)!=1 or PLACE_OF_VEHICLE.search(words)
            or re.search(r"\b(?:between|both|two|either|instead of|not|never|except)\b|n['’]t\b",words)):
        return None
    relation=relations[0]
    before,phrase=words[:relation.start()],words[relation.end():]
    if (NOUN.search(before) or len(NOUN.findall(phrase))!=1 or re.search(r"\band\b",phrase)
            or re.search(r"\b(?:nearest|beside|opposite|near|across from|by the|at the)\b",before)):
        return None
    tokens=re.findall(r"[a-z]+",phrase)
    kind=NOUN.search(phrase)[1]
    kind="pickup" if kind.startswith("pick") else "truck" if kind=="lorry" else kind
    colors={("gray" if t=="grey" else t) for t in tokens if t in COLORS|{"grey"}}
    sides={t.removesuffix("most") for t in tokens if t in {"left","right","leftmost","rightmost"}}
    allowed=COLORS|{"grey","a","an","the","that","this","on","to","my","your","our","side","left","right","please","ahead","parked","rightmost","leftmost",*TYPES,"vehicle","lorry","pick","up","pickup"}
    return {"kind":kind,"relation":"behind" if relation.group()=="behind" else "beside",
            "color":next(iter(colors),None),"side":next(iter(sides),None),
            "unsupported":len(colors)>1 or len(sides)>1 or bool(set(tokens)-allowed)}


def context(request):
    f=request["scene"]["frame"]
    tracks=sorted(([t["track_id"],t["class_name"],t["bbox_xyxy"]] for t in request["scene"]["tracks"] if t["frame_id"]==f["frame_id"]),key=lambda t:t[0])
    return {"recording_id":f["recording_id"],"frame_id":f["frame_id"],"size":[f["width"],f["height"]],
            "image_sha256":request["scene"]["provenance"]["image_sha256"],"tracks":tracks}


def motor_rows(ctx):
    size=ctx["size"]
    rows=[]
    for tid,cls,box in sorted(ctx["tracks"],key=lambda t:-(t[2][2]-t[2][0])*(t[2][3]-t[2][1])):
        name=cls.casefold().replace(" ","_")
        if (name not in {"car","sedan","suv","van","truck","bus","medium_vehicle","large_vehicle","heavy_vehicle","light_vehicle"}
                or (box[2]-box[0])/size[0]<.025 or (box[3]-box[1])/size[1]<.025):
            continue
        if any(other[1]==name and overlap([other[2]],box)[0]>=.60 for other in rows):
            continue
        rows.append((tid,name,box))
    return rows


def opportunities(ctx,classes):
    occupied=[t[2] for t in ctx["tracks"]]
    options=[]
    for tid,cls,box in motor_rows(ctx):
        patch=behind_patch(classes,box,occupied,ctx["size"])
        if patch is not None:options.append({"track_id":tid,"detector_class":cls,"anchor_box":box,**patch})
    return options


def inspection(original,option):
    w,h=original.size
    image=original.convert("RGB")
    b=option["anchor_box"];px,py=option["point"];bw,bh=b[2]-b[0],b[3]-b[1]
    left,top=max(0,int(min(b[0],px)-.85*bw)),max(0,int(b[1]-.70*bh))
    right,bottom=min(w,int(max(b[2],px)+.85*bw)),min(h,int(max(b[3],py)+.56*bh))
    wide=image.copy();wide.thumbnail((448,440),Image.Resampling.LANCZOS)
    crop=image.crop((left,top,right,bottom));crop.thumbnail((620,440),Image.Resampling.LANCZOS)
    def paint(view,x0,y0,sw,sh):
        sx,sy=view.width/sw,view.height/sh
        x1,y1,x2,y2=(b[0]-x0)*sx,(b[1]-y0)*sy,(b[2]-x0)*sx,(b[3]-y0)*sy
        d=ImageDraw.Draw(view);d.rectangle([x1,y1,x2,y2],outline="magenta",width=2)
        x,y=(px-x0)*sx,(py-y0)*sy
        d.ellipse((x-6,y-6,x+6,y+6),outline="black",width=3);d.ellipse((x-5,y-5,x+5,y+5),outline="cyan",width=2)
    paint(wide,0,0,w,h);paint(crop,left,top,right-left,bottom-top)
    sheet=Image.new("RGB",(wide.width+crop.width+5,max(wide.height,crop.height)),"#202020")
    sheet.paste(wide,(0,(sheet.height-wide.height)//2));sheet.paste(crop,(wide.width+5,(sheet.height-crop.height)//2))
    out=BytesIO();sheet.save(out,format="JPEG",quality=92,subsampling=0)
    return out.getvalue()


def observation_calls(original,options,read):
    calls=[]
    if len(options)>8:return calls
    for option in options:
        pixels=inspection(original,option)
        observation=read(GUIDE,WORDS,pixels).strip().upper()
        if observation!="NONE" and observation not in RESPONSES:
            raise ValueError("Unexpected trained single-vehicle visual description")
        calls.append({**call_identity("single_vehicle_rear",GUIDE,WORDS,pixels),"track_id":option["track_id"],"response":observation})
    return calls


@lru_cache(maxsize=1)
def ranker_from(archive):
    if not isinstance(archive,str) or len(archive)>110000:
        raise ValueError("Invalid attached referent model")
    dec=zlib.decompressobj();data=dec.decompress(base64.b64decode(archive,validate=True),150000)
    if not dec.eof or dec.unused_data or sha256(data).hexdigest()!=RANKER:
        raise ValueError("Unexpected referring-object checkpoint")
    return ReferentRanker(json.loads(data),RANKER)


def _compatible(kind,spoken):
    return kind=="vehicle" or spoken==kind or kind=="car" and spoken in {"sedan","suv","car"}


def _coarse(kind,cls):
    return (kind=="vehicle" or kind=="pickup" and cls in {"car","truck","light_vehicle","medium_vehicle","large_vehicle"}
            or kind=="truck" and cls in {"truck","large_vehicle","medium_vehicle","heavy_vehicle"}
            or kind=="bus" and cls=="bus" or kind in {"car","suv","sedan","van"} and cls in {"car","sedan","suv","van","light_vehicle"})


def _side(box,w,side):
    c=(box[0]+box[2])/(2*w)
    return not side or (c<.46 if side=="left" else c>.54)


def visible_vehicle_end(box, size):
    """A cropped or strongly elongated silhouette cannot establish a camera-facing bumper."""
    left, top, right, bottom = box
    width, height = size
    # Width here protects against an obvious broadside view misclassified by the crop reader.
    # A tighter shape only permits further checks. It does not establish rear orientation.
    return (min(left, width - right) > max(1, .002 * width)
            and min(top, height - bottom) > max(1, .002 * height)
            and right - left <= 2 * (bottom - top))


def decide(request,original,packet,*,policy=POLICY):
    ref=request_reference(request);ctx=context(request)
    _authenticate(packet);body=packet["body"]
    if (not ref or body.get("recipe")!=RECIPE or body.get("models")!=MODELS or body.get("context_sha256")!=digest(ctx)
            or body.get("image_sha256")!=ctx["image_sha256"]):
        raise ValueError("Single-vehicle evidence belongs to another image, observation set or checkpoint")
    classes=_decode_map(body["class_map"],request["scene"]["frame"])
    options=opportunities(ctx,classes);raw=body["calls"]
    if len(raw)!=(len(options) if len(options)<=8 else 0):raise ValueError("Missing or extra vehicle observations")
    matching=[];observed=[];width=ctx["size"][0]
    for opt,call in zip(options,raw):
        key=call_identity("single_vehicle_rear",GUIDE,WORDS,inspection(original,opt))
        rep=call.get("response")
        if (call.get("track_id")!=opt["track_id"] or any(call.get(k)!=v for k,v in key.items())
                or rep!="NONE" and rep not in RESPONSES):
            raise ValueError("Vehicle model call does not match the displayed anchor and parking position")
        if rep=="NONE":continue
        name,color=rep.lower().split()
        color_matches=not ref["color"] or ref["color"]==color or {ref["color"],color}=={"gray","silver"}
        if not color_matches or not _side(opt["anchor_box"],width,ref["side"]):
            continue
        if _compatible(ref["kind"],name):
            matching.append(opt)
            observed.append(opt["track_id"])
        elif ref["kind"]=="pickup" and name=="truck":
            # A positive truck observation can narrow a no-space check. It can't certify a pickup subtype.
            observed.append(opt["track_id"])
    rank=ranker_from(body["ranker_zlib"]).rank(original,request["text"],request["scene"]["tracks"],frame=request["scene"]["frame"])
    coarse=[v for v in motor_rows(ctx) if _coarse(ref["kind"],v[1]) and _side(v[2],width,ref["side"])]
    if ref["relation"]=="beside":
        # These signed visual answers concern the rear. Side geometry is checked
        # separately; without an independently verified side view we ask the passenger.
        occupied=[row[2] for row in ctx["tracks"]]
        described=[row for row in coarse if row[0] in observed]
        has_side=lambda vehicles:any(beside_patches(classes,box,occupied,ctx["size"]) for _,_,box in vehicles)
        if policy is None:
            reason="side_unsupported" if ref["unsupported"] else "side_needs_mark" if has_side(coarse) else "no_supported_side_space"
        elif ref["unsupported"] or len(described)>1 or not ref["color"] and len(coarse)>1:
            reason="side_unsupported"
        elif len(described)==1 and not has_side(described):
            reason="no_supported_side_space"
        elif described or not observed and has_side(coarse):
            reason="side_needs_mark" if described else "side_unverified_vehicle"
        else:
            reason="no_supported_side_space" if coarse else "side_unverified_vehicle"
        return unresolved(ASK[reason]),reason,None,rank["checkpoint_sha256"]
    chosen=None
    if ref["unsupported"]:reason="unsupported"
    elif len(options)>8 or len(matching)>1 or not ref["color"] and len(coarse)>1:reason="ambiguous"
    elif not options:reason="no_supported_space"
    elif len(matching)!=1:reason="unverified_vehicle"
    elif policy==POLICY and not visible_vehicle_end(matching[0]["anchor_box"],ctx["size"]):
        reason="rear_view_unclear"
    else:
        opt=matching[0];scores=[s for s in rank["alternatives"] if any(s["track_id"]==row[0] for row in coarse)]
        # The fitted referring-object model must independently rank the candidate in its leading two alternatives.
        if any(s["track_id"]==opt["track_id"] and s["score"]>rank["null_score"] for s in scores[:2]):
            chosen=opt;reason="matched"
        else:reason="unverified_vehicle"
    result=proposed(chosen["point"],chosen["box"],[chosen["anchor_box"]]) if chosen else unresolved(ASK[reason])
    return result,reason,chosen,rank["checkpoint_sha256"]


def outcome(request,photo,packet=None,*,waiting=None,policy=POLICY):
    req,key,input_sha=source_input(request,photo)
    ref=request_reference(req)
    if not ref:raise ValueError("A single-vehicle relation is required")
    if policy not in (None,PREVIOUS_POLICY,POLICY):
        raise ValueError("Unknown single-vehicle passenger policy")
    if waiting is not None and (packet is not None or waiting not in {"pending","busy"}):
        raise ValueError("Only a check without model evidence may be marked pending")
    if packet is None:
        reason=waiting or ("service_unavailable" if ref["relation"]=="behind" else "side_service_unavailable")
        result=unresolved(ASK[reason]);opt=None;rank_sha=None
    else:
        with Image.open(BytesIO(photo)) as image:result,reason,opt,rank_sha=decide(req,image,packet,policy=policy)
    return {"ok":True,"request_id":req["request_id"],"result":result,"provenance":{
        "method":METHOD,"image":{"sha256":key},"request_sha256":digest(req),"inference_input_sha256":input_sha,
        **({} if policy is None else {"passenger_policy":policy}),
        "reason":reason,"operational_status":waiting or ("unavailable" if packet is None else "complete"),"single_vehicle_evidence":packet,
        "selected_track_ids":[opt["track_id"]] if opt else [],"surface":opt,"ranker_sha256":rank_sha,
        "independent_model_proposal":opt is not None}}


SURFACE_METHOD="single_vehicle_visible_space_check_v1"


def empty_visible_space(request,photo,packet):
    from . import manual_surfaces
    req,key,input_sha=source_input(request,photo)
    ref=request_reference(req)
    if not ref or ref["unsupported"]:
        return None
    ctx=context(req)
    classes=manual_surfaces.decode(packet,req["scene"]["frame"],key)
    seen=[row for row in motor_rows(ctx) if _coarse(ref["kind"],row[1]) and _side(row[2],ctx["size"][0],ref["side"])]
    occupied=[row[2] for row in ctx["tracks"]]
    if ref["relation"]=="beside":
        has_space=any(beside_patches(classes,box,occupied,ctx["size"]) for _,_,box in seen)
    else:
        has_space=any(behind_patch(classes,box,occupied,ctx["size"]) is not None for _,_,box in seen)
    if has_space:
        return None
    words=("I can’t see an open off-lane area "+("behind" if ref["relation"]=="behind" else "beside")+
           " a vehicle matching that description here. Is there another visible space you meant?")
    return {"ok":True,"request_id":req["request_id"],"result":unresolved(words),
            "provenance":{"method":SURFACE_METHOD,"request_sha256":digest(req),"image":{"sha256":key},
                          "inference_input_sha256":input_sha,"selected_track_ids":[],"independent_model_proposal":False,
                          "surface_evidence":packet,"reason":"no_visible_vehicle_space"}}


def verify_empty(request,out):
    photo=Path(request["scene"]["frame"]["image_path"]).read_bytes()
    if empty_visible_space(request,photo,out["provenance"]["surface_evidence"])!=out:
        raise ValueError("The visible roadside evidence differs from the selected vehicle and photograph.")
    return {"input_verified":True,"decision_reproduced":True,"language_calls":0,"independent_model_proposal":False}


def verify(request,out):
    photo=Path(request["scene"]["frame"]["image_path"]).read_bytes()
    packet=out["provenance"]["single_vehicle_evidence"]
    status=out["provenance"]["operational_status"]
    waiting=status if status in {"pending","busy"} else None
    if outcome(request,photo,packet,waiting=waiting,policy=out["provenance"].get("passenger_policy"))!=out:
        raise ValueError("Single-vehicle result differs from original model and surface evidence")
    return {"input_verified":True,"decision_reproduced":True,"model_images_reproduced":packet is not None,
            "language_calls":len(packet["body"]["calls"]) if packet else 0,"surface_model_calls":int(packet is not None),
            "operational_status":out["provenance"]["operational_status"],
            "independent_model_proposal":out["provenance"]["independent_model_proposal"]}


def verify_current(request, out):
    """Require the current passenger policy before displaying or confirming a new outcome."""
    if out.get("provenance", {}).get("passenger_policy") != POLICY:
        raise ValueError("The single-vehicle decision uses a superseded passenger policy")
    return verify(request, out)
