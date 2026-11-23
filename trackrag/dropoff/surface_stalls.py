"""Partition the learned parking surface into visible individual spaces for visual review."""
from io import BytesIO
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile, BadZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import painted_bays

RECIPE = "visually_checked_parking_components_v1"
CHECK = ("Only assess the single outlined region in this driver-view photo. Does the outline follow "
         "one vacant vehicle-sized parking bay or curbside pull-in? Check paint, adjoining stalls, kerb, vehicles, "
         "and the traffic aisle. Reject an outline that spans multiple stalls, crosses the travel lane or sidewalk, "
         "captures a small scrap of pavement, an occupied stall, or the space across a barrier. "
         "Reply BAY, MULTIPLE, OCCUPIED, LANE, FRAGMENT, or UNCLEAR.")
WORDS = "Does the outlined shape correspond to one visible, empty parking space?"
BANK = Path(__file__).resolve().parents[2]/"experiments/dropoff/passenger-stall-reviews-v1.zip"


@lru_cache(maxsize=96)
def _read_bank(path, modified, photo_sha):
    with ZipFile(path) as bank:
        name=photo_sha+".json"
        if bank.namelist().count(name)!=1:
            raise ValueError("Missing or duplicate parking review packet")
        if bank.getinfo(name).file_size>20000:
            raise ValueError("Oversize signed parking review.")
        return bank.read(name)


def get_packet(photo_sha):
    if not isinstance(photo_sha,str) or len(photo_sha)!=64 or any(c not in "0123456789abcdef" for c in photo_sha):
        return None
    try:
        return json.loads(_read_bank(str(BANK),BANK.stat().st_mtime_ns,photo_sha))
    except (OSError,KeyError,ValueError,BadZipFile):
        return None


def proposals(original, classes, context):
    """Generate candidates without passenger words, study references or scene identities."""
    height,width=classes.shape
    scale=np.array(original.size)/[width,height]
    rgb=np.asarray(original.convert("RGB").resize((width,height),Image.Resampling.BOX)).astype("float32")
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    neutral=np.ptp(rgb,axis=2)<55
    ground=np.isin(classes,(10,13,24))
    parking=classes==10
    min_area=max(45,round(classes.size*.00040))
    if parking.sum()<min_area:
        return []
    distance=cv2.distanceTransform((~parking).astype("uint8"),cv2.DIST_L2,5)
    lines=painted_bays._dividers(gray,neutral,ground,classes,distance)
    separated=cv2.morphologyEx(parking.astype("uint8"),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    separated[~ground]=0
    for line in lines:
        a,b=line["ends"]
        length=np.linalg.norm(b-a)
        if length<.025*width:
            continue
        # Carry genuine divider fragments through a break in paint inside the *parking mask*.
        # A kerb, obstacle or a vehicle splits the mask regardless of a line extension.
        length_limit=min(.40*length,.06*width)
        axis=(b-a)/length
        ends=np.rint([a-axis*length_limit,b+axis*length_limit]).astype(int)
        cv2.line(separated,tuple(ends[0]),tuple(ends[1]),0,max(2,round(line["width"])))
    boxes=np.asarray(context["current_boxes"],dtype=float).reshape(-1,4)/np.tile(scale,2)
    for l,t,r,b in boxes:
        cv2.rectangle(separated,(int(l),int(t)),(int(np.ceil(r)),int(np.ceil(b))),0,-1)
    _,labels,stats,_=cv2.connectedComponentsWithStats(separated)
    choices=[]
    for idx in range(1,len(stats)):
        x,y,w,h,area=stats[idx]
        if (area<min_area or area>.17*classes.size or min(w,h)<5 or w<width*.025
                or w>width*.56 or y+h<height*.47):
            continue
        component=(labels[y:y+h,x:x+w]==idx).astype("uint8")
        # Robust against isolated tendrils reaching into an adjacent stall.
        component=cv2.morphologyEx(component,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        contours=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        if not contours:
            continue
        boundary=max(contours,key=cv2.contourArea)
        hull=cv2.convexHull(boundary)
        poly=cv2.approxPolyDP(hull,max(1.4,.019*cv2.arcLength(hull,True)),True).reshape(-1,2)
        if not 3<=len(poly)<=9:
            continue
        inside=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(inside,poly,1)
        region=classes[y:y+h,x:x+w][inside.astype(bool)]
        if (region.size<min_area or (region==10).mean()<.65 or
                not np.isin(region,(10,13,24,36)).mean()>=.96 or np.isin(region,(8,23)).any()):
            continue
        p=np.float32(poly+[x,y])
        if any(cv2.intersectConvexConvex(p,np.float32([[l,t],[r,t],[r,b],[l,b]]))[0]>.5 for l,t,r,b in boxes):
            continue
        moment=cv2.moments(p)
        if not moment["m00"]:
            continue
        center=np.array([moment["m10"],moment["m01"]])/moment["m00"]
        if classes[round(center[1]),round(center[0])]!=10:
            continue
        observed=[]
        for line in lines:
            a,b=line["ends"]
            # Attribute only the portion of a real stripe abutting this component.
            samples=np.linspace(a,b,31)
            good=[s for s in samples if abs(cv2.pointPolygonTest(p,tuple(s),True))<=max(3.5,line["width"]+1.8)]
            if len(good)<5 or np.linalg.norm(good[-1]-good[0])<6:
                continue
            observed.append((np.round(np.array([good[0],good[-1]])*scale,3).tolist(),
                             round(float(line["width"]*scale.mean()),3)))
        p0=p*scale
        top=p0[np.argsort(p0[:,1])[:2]].mean(axis=0)
        choices.append({"polygon":p0.round(3).tolist(),"point":(center*scale).round(3).tolist(),
            "box":np.r_[p0.min(axis=0),p0.max(axis=0)].round(3).tolist(),
            "evidence":{"recipe":RECIPE,"frontage_xy":top.round(3).tolist(),
                "painted_boundaries":[p for p,_ in observed],"boundary_widths":[w for _,w in observed],
                "boundary_opposite_closed":[False]*len(observed),"visible_parking_cells":int((region==10).sum()),
                "interior_cells":int(len(region))}})
    return sorted(choices,key=lambda row:(-row["evidence"]["visible_parking_cells"],*row["point"]))[:8]


def review_view(image, option, close=False):
    """The trained reviewer sees the actual perimeter and enough unmarked context to assess it."""
    full_w,full_h=image.size
    x0,y0,x1,y1=option["box"]
    if close:
        sw=min(full_w,max(.43*full_w,2.3*(x1-x0))); sh=min(full_h,max(.48*full_h,2.5*(y1-y0)))
        left=round(max(0,min(full_w-sw,(x0+x1-sw)/2)));top=round(max(0,min(full_h-sh,(y0+y1-sh)/2)))
        image=image.crop((left,top,min(full_w,left+round(sw)),min(full_h,top+round(sh))))
    else:
        left=top=0
    image=image.convert("RGB")
    w,h=image.size;image.thumbnail((1000,750),Image.Resampling.LANCZOS)
    polygon=[((x-left)*image.width/w,(y-top)*image.height/h) for x,y in option["polygon"]]
    d=ImageDraw.Draw(image)
    d.line(polygon+polygon[:1],fill="black",width=4)
    d.line(polygon+polygon[:1],fill="#00e8f2",width=2)
    data=BytesIO();image.save(data,"JPEG",quality=90,subsampling=0)
    return data.getvalue()


def approved(original, classes, context, surface, packet):
    from .bay_landmarks import MODELS
    from .parking_evidence import _authenticate
    from .passenger import digest
    _authenticate(packet)
    body=packet["body"]
    if (body.get("recipe")!=RECIPE or body.get("models")!=MODELS or body.get("image_sha256")!=context["image_sha256"]
            or body.get("size")!=list(original.size) or body.get("context_sha256")!=digest(context)
            or body.get("surface_signature")!=surface["signature"]
            or body.get("prompt_sha256")!=sha256((CHECK+"\n"+WORDS).encode()).hexdigest()):
        raise ValueError("The parking review does not match the current scene, tracks or trained model.")
    rows=body.get("options")
    if (not isinstance(rows,list) or not 1<=len(rows)<=8 or any(not isinstance(row,dict)
            or not isinstance(row.get("shape_sha256"),str) or "reviews" not in row for row in rows)):
        raise ValueError("The parking review has invalid candidate records.")
    options=proposals(original,classes,context)
    expected=[digest(option) for option in options]
    reviews={row["shape_sha256"]:row["reviews"] for row in rows}
    if len(reviews)!=len(rows) or len(options)!=len(rows) or set(expected)!=set(reviews):
        raise ValueError("Parking reviews must cover each current candidate exactly once")
    found=[]
    for key,option in zip(expected,options,strict=True):
        calls=reviews[key]
        if not isinstance(calls,list) or not 1<=len(calls)<=2:
            raise ValueError("Invalid number of model views for a parking region")
        answers=[]
        for index,call in enumerate(calls):
            if (not isinstance(call,dict) or not isinstance(call.get("reply"),str) or len(call["reply"])>45
                    or sha256(review_view(original,option,close=bool(index))).hexdigest()!=call.get("view_sha256")):
                raise ValueError("Parking review does not reproduce its model-visible photo")
            answers.append(call["reply"].strip().rstrip(".!")=="BAY")
        if len(answers)!=(2 if answers[0] else 1):
            raise ValueError("Parking review does not follow the recorded full-view and crop decisions")
        if all(answers):
            found.append(option)
    return found
