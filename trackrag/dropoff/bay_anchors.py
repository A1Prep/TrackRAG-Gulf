"""Bind each verified bay to the parked vehicles and frontage actually visible around it."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re

from PIL import Image,ImageDraw,ImageFont

from . import bay_landmarks
from .bay_packets import read_photo_packet
from .parking_evidence import _authenticate
from .passenger import digest

LEGACY_RECIPE="public_bay_anchor_observations_v1"
RECIPE="public_bay_anchor_observations_v2"
PAIR=("The cyan outline is one candidate parking space. Numbered boxes identify actual detected vehicles. "
      "Which two parked vehicles immediately flank this empty space on opposite ends within the SAME row of parking? "
      "Check that the space lies between them. A nearby moving car or two vehicles on the same end do not qualify. "
      "Reply with exactly two box numbers or NONE.")
FRONT=("Read the visible frontage directly beside the outlined space. Give its short ordinary category and its "
       "printed business or building name if legible. Include an adjacent hedge or tree if prominent. "
       "Give up to two phrases separated by |. Reply NONE if you cannot see the frontage.")
APPEARANCE=("Describe only the vehicle enclosed by the numbered rectangle in this crop. Give its body colour "
            "and type. Colours include black, white, silver, grey, blue, green, red, brown, and yellow. "
            "Types include sedan, SUV, pickup, van, bus and truck. Reply with just the colour and type or NONE.")
FRONT_WORDS="Describe this bay's frontage."
PAIR_WORDS="Give the two box numbers, or NONE."
APPEARANCE_WORDS="Describe this cropped vehicle."
BANK=Path(__file__).resolve().parents[2]/"experiments/dropoff/passenger-bay-anchors-v1.zip"


def vehicles(scene):
    classes={"car","sedan","suv","light_vehicle","medium_vehicle","heavy_vehicle","large_vehicle","taxi",
             "van","minivan","pickup","truck","bus","motorcycle","motorbike"}
    return sorted([[t["track_id"],t["class_name"],list(map(float,t["bbox_xyxy"]))] for t in scene["tracks"]
                   if t["frame_id"]==scene["frame"]["frame_id"] and t["class_name"].casefold() in classes])


def closest_tracks(option, tracks):
    x,y=option["point"]
    def distance(row):
        l,t,r,b=row[2]
        return (max(l-x,0,x-r)**2+max(y-b,0,b-y)**2, row[0])
    return sorted(tracks,key=distance)[:8]


def view(image, option, tracks):
    im=image.convert("RGB");sz=im.size;im.thumbnail((1220,850),Image.Resampling.LANCZOS)
    sx,sy=im.width/sz[0],im.height/sz[1]
    draw=ImageDraw.Draw(im);font=ImageFont.load_default(size=18)
    p=[(x*sx,y*sy) for x,y in option["polygon"]]
    draw.line(p+p[:1],fill="black",width=5);draw.line(p+p[:1],fill="cyan",width=3)
    for i,(_,_,box) in enumerate(tracks,1):
        l,t,r,b=box[0]*sx,box[1]*sy,box[2]*sx,box[3]*sy
        draw.rectangle((l,t,r,b),outline="#ffb739",width=2)
        draw.rectangle((l,t,min(im.width-1,l+18),min(im.height-1,t+21)),fill="black")
        draw.text((l+2,t),str(i),fill="white",font=font)
    out=BytesIO();im.save(out,"JPEG",quality=91,subsampling=0)
    return out.getvalue()


def crop_car(im,box):
    l,t,r,b=box
    image=im.crop((max(0,int(l)),max(0,int(t)),min(im.width,int(r)+1),min(im.height,int(b)+1))).convert("RGB")
    image.thumbnail((560,440),Image.Resampling.LANCZOS)
    data=BytesIO();image.save(data,"JPEG",quality=91,subsampling=0)
    return data.getvalue()


def pair_indices(text, n):
    match=re.fullmatch(r"\s*(?:BOX(?:ES)?\s*)?[#\[]?(\d+)[\s,;&]+(?:AND\s*)?[#]?(\d+)\]?\.?\s*",text,re.I)
    if not match:return []
    a,b=map(int,match.groups())
    return [a-1,b-1] if a!=b and 1<=a<=n and 1<=b<=n else []


def get_packet(key):
    return read_photo_packet(BANK, key, 30000)


def prompt_digest(recipe=RECIPE):
    if recipe == LEGACY_RECIPE:
        return digest([PAIR,FRONT,APPEARANCE])
    if recipe == RECIPE:
        return digest([[PAIR,PAIR_WORDS],[FRONT,FRONT_WORDS],[APPEARANCE,APPEARANCE_WORDS]])
    raise ValueError("Unknown parking-bay observation recipe")


def checked_context(packet, scene, size):
    """Authenticate the model, original photograph and exact current vehicle detections."""
    _authenticate(packet)
    body=packet["body"];tracklist=vehicles(scene)
    if (body.get("model")!=bay_landmarks.MODELS or body.get("prompt")!=prompt_digest(body.get("recipe"))
            or body.get("image_sha256")!=scene["provenance"]["image_sha256"] or body.get("size")!=list(size)
            or body.get("current_vehicles_sha256")!=digest(tracklist)):
        raise ValueError("Bay context is for another photo, vehicle set, or model.")
    rows,appearance=body.get("rows"),body.get("appearance")
    if not isinstance(rows,list) or not 1<=len(rows)<=18 or not isinstance(appearance,dict):
        raise ValueError("Invalid bay observation records")
    return body,tracklist


def observations(image, options, scene, packet):
    if packet is None:return {}
    body,tracklist=checked_context(packet,scene,image.size)
    rows,appearance=body["rows"],body["appearance"]
    shapes=[digest(opt["polygon"]) for opt in options]
    if any(not isinstance(row,dict) or not isinstance(row.get("shape"),str) for row in rows):
        raise ValueError("Invalid signed bay observation row")
    by_shape={r["shape"]:r for r in rows}
    if len(by_shape)!=len(rows) or len(shapes)!=len(rows) or set(shapes)!=set(by_shape):
        raise ValueError("Bay observations must cover each current candidate exactly once")
    result,crops={},{}
    for key,opt in zip(shapes,options,strict=True):
        row=by_shape[key]
        if (not isinstance(row.get("frontage"),str) or len(row["frontage"])>240
                or not isinstance(row.get("pair_reply"),str) or len(row["pair_reply"])>60):
            raise ValueError("Invalid bay observation text")
        local=closest_tracks(opt,tracklist)
        if sha256(view(image,opt,local)).hexdigest()!=row.get("view"):
            raise ValueError("Bay observation differs from the current photo and numbered vehicle boxes")
        indices=pair_indices(row["pair_reply"],len(local))
        pair=[]
        for idx in indices:
            tid,_,box=local[idx]
            track_key=str(tid)
            if track_key not in crops:
                record=appearance.get(track_key)
                if (not isinstance(record,dict) or not isinstance(record.get("reply"),str) or len(record["reply"])>55
                        or sha256(crop_car(image,box)).hexdigest()!=record.get("view")):
                    raise ValueError("Vehicle description differs from its original crop")
                crops[track_key]=record["reply"].lower()
            pair.append({"track_id":tid,"box":box,"appearance":crops[track_key]})
        result[key]={"frontage":row["frontage"],"pair":pair}
    if set(crops)!=set(appearance):
        raise ValueError("Unused vehicle appearance evidence")
    return result
