"""Original-view and local-surface checks for passenger place-point experiments."""
from io import BytesIO
import re
from PIL import Image, ImageDraw
import numpy as np

SCENE = (
    "You are viewing a paused forward road photograph. Read the requested landmark. "
    "Is there a visible, vacant, car-sized bay, entrance apron, parking strip or vehicle driveway serving it? "
    "A shop across an uninterrupted sidewalk, chained forecourt, closed gate, live through lane or occupied row "
    "has no such opening. Ignore whether stopping is legally permitted. "
    "Answer only OPEN if you see the requested opening, BLOCKED if obstructed, or UNCLEAR."
)
SURFACE = (
    "A small cyan ring marks an exact point within this close-up of the street surface. "
    "Classify the pavement at its centre in the scene: BAY (vacant parking space, pull-in, accessible driveway "
    "or drop-off apron); LANE (traffic lane or junction); WALK (walkway, island or kerb); "
    "BLOCKED (vehicle, chain, barrier, or closed gate); UNKNOWN. Respond with one word."
)


def describes_place(request):
    from .place_grounding import VEHICLE, has_landmark_terms, place_reference
    words=request["text"]
    if VEHICLE.search(words):
        return False
    if place_reference(request):
        return True
    # A scene noun after a spatial relation can name an unfamiliar sign or object.
    # Do not turn a gesture ("just there", "on the right") into a named place.
    relation=re.search(r"\b(?:of|under|past|opposite|before|after|beside|by|near|toward|towards|at)\s+"
                       r"(?:the |that |this |an? )?(.+)",words,re.I)
    if not relation:
        return False
    return has_landmark_terms(relation[1])


def photo_view(photo, point=None):
    out = BytesIO()
    if point is None:
        view = photo.copy()
        view.thumbnail((1280,900), Image.Resampling.LANCZOS)
    else:
        if len(point) != 2 or any(type(v) is not int or not 0 <= v <= 1000 for v in point):
            raise ValueError("Invalid specialist coordinate")
        w,h=photo.size
        sw,sh=max(1,round(w*.51)),max(1,round(h*.63))
        x,y=[min(n-1, round(a*n/1000)) for a,n in zip(point, (w,h))]
        # Keep the marked point at its true relative position when a crop touches an edge.
        left,top=max(0,min(w-sw,x-sw//2)),max(0,min(h-sh,y-sh//2))
        view=photo.crop((left,top,left+sw,top+sh))
        view.thumbnail((850,650),Image.Resampling.LANCZOS)
        cx,cy=(x-left)*view.width/sw,(y-top)*view.height/sh
        draw=ImageDraw.Draw(view)
        for radius,color in ((9,"black"),(7,"cyan")):
            draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline=color,width=2)
    view.save(out,format="JPEG",quality=92,subsampling=0)
    return out.getvalue()


def verified_point(classes, point, boxes, size, scene_reply, surface_reply):
    """Require independent evidence for the specialist's exact pixel, without moving it."""
    if point is None or surface_reply != "BAY" or scene_reply not in {"OPEN","BLOCKED","UNCLEAR"}:
        return False
    width,height=size
    x,y=[min(dim-1,v*dim/1000) for v,dim in zip(point,size)]
    if any(left<=x<=right and top<=y<=bottom for left,top,right,bottom in boxes):
        return False
    rows,cols=classes.shape
    cx=min(cols-1,int(x*cols/width));cy=min(rows-1,int(y*rows/height))
    # A small interior neighbourhood suppresses ambiguous boundary pixels.
    dx=max(2,round(cols*.009));dy=max(2,round(rows*.009))
    region=classes[max(0,cy-dy):min(rows,cy+dy+1),max(0,cx-dx):min(cols,cx+dx+1)]
    center=int(classes[cy,cx])
    if center==10:
        return float(np.mean(region==10))>=.75
    return center==13 and scene_reply=="OPEN" and float(np.mean(region==13))>=.85
