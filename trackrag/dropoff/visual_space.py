"""Generate review locations on visible ground, with evidence bound to the source photograph."""
from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from itertools import combinations
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from PIL import Image

from . import bay_clearance, manual_surfaces, roadside_board, visual_space_board as board
from .bay_packets import read_photo_packet
from .parking_evidence import _authenticate
from .passenger import digest


BANK = Path(__file__).resolve().parents[2] / "experiments/dropoff/passenger-ground-view-observations-v1.zip"
PAVED_BANK = BANK.with_name("passenger-paved-row-observations-v1.zip")
EXTENSION_BANK = BANK.with_name("passenger-entrance-row-observations-v1.zip")
_cache = OrderedDict()
_lock = Lock()


def context(scene):
    frame = scene["frame"]
    return {"size":[frame["width"],frame["height"]],
            "tracks":sorted([[t["track_id"],t["class_name"],t["bbox_xyxy"]] for t in scene["tracks"]
                             if t["frame_id"]==frame["frame_id"]],key=lambda t:t[0])}


def samples(scene, classes):
    # These are questions for the visual reviewer. Their generation alone can never produce a pin.
    ctx=context(scene)
    h,w=classes.shape
    native=np.asarray(ctx["size"],dtype=float)/[w,h]
    ground=np.isin(classes,(10,13,26)).astype("uint8")
    vehicles=[]
    for _,name,box in ctx["tracks"]:
        l,t,r,b = np.asarray(box)/np.tile(native,2)
        ground[max(0,int(t)):min(h,int(np.ceil(b))+1), max(0,int(l)):min(w,int(np.ceil(r))+1)] = 0
        if name.casefold() in {"car","light_vehicle","medium_vehicle","heavy_vehicle","bus","truck","van","taxi",
                              "suv","pickup","motorcycle","motorbike","bicycle"} and r-l>.017*w and b-t>.016*h:
            vehicles.append((l,t,r,b))
    distance=cv2.distanceTransform(ground,cv2.DIST_L2,5)
    proposed={}
    def add(x,y,priority):
        x,y=round(float(x)),round(float(y))
        if not (.025*w < x < .975*w and .405*h < y < .91*h) or distance[y,x] < max(3,.009*h):
            return
        label=int(classes[y,x])
        if label==13:
            if abs(x/w-.50) < .035+max(0,y/h-.45)*.25:
                return
            if not any(max(l-x,0,x-r) < max(.09*w,.95*(r-l)) and abs(y-b)<max(.08*h,.8*(b-t))
                       for l,t,r,b in vehicles):
                return
        score=priority+(.7 if label==10 else .30 if label==26 else 0)+min(1,float(distance[y,x])/(.06*h))*.30
        proposed[x,y]=max(proposed.get((x,y),-1),score)
    for l,t,r,b in vehicles:
        bw,bh=r-l,b-t
        for x,y in (((l+r)/2,b+.22*bh),(l-.34*bw,b-.06*bh),(r+.34*bw,b-.06*bh)):
            add(x,y,2)
    for a,b in combinations(sorted(vehicles,key=lambda box:box[0]),2):
        gap=b[0]-a[2]
        if .012*w<gap<.37*w and abs(b[3]-a[3]) < max(.08*h,.55*max(a[3]-a[1],b[3]-b[1])):
            for fraction in (.35,.65):
                x=a[2]+fraction*gap
                foot=a[3]+fraction*(b[3]-a[3])
                add(x,foot,3)
                add(x,foot-.07*min(a[3]-a[1],b[3]-b[1]),2.8)
    for x in np.linspace(.035*w,.965*w,34):
        for y in np.linspace(.445*h,.895*h,17):
            add(x,y,0)
    result={}
    for side in ("left","right"):
        available={p:s for p,s in proposed.items() if (p[0]<.5*w)==(side=="left")}
        chosen=[]
        while available and len(chosen)<20:
            point=max(available,key=lambda p:(available[p],p[1],p[0]))
            x,y=point
            chosen.append(point)
            # Space the tags at the original perspective. Repeated points at a single
            # bumper would hide other areas of the same frontage from the reviewer.
            available={p:s for p,s in available.items()
                       if ((p[0]-x)/(.055*w))**2+((p[1]-y)/(.066*h))**2>1}
        rows=[]
        for i,(x,y) in enumerate(sorted(chosen,key=lambda xy:(xy[1],xy[0])),1):
            p=np.round((np.asarray([x,y])+.5)*native-.5,3)
            radius=np.maximum(1,np.minimum(.25*distance[y,x],.008*w)*native)
            box=[*np.maximum(0,p-radius),*np.minimum(np.asarray(ctx["size"])-1,p+radius)]
            rows.append({"id":i,"point":p.tolist(),"box":[round(float(v),3) for v in box],"class":int(classes[y,x])})
        if rows:
            result[side]=rows
    return result


def native_context(scene):
    from . import bay_anchors,curb_points
    ctx=curb_points.source_context(scene)
    return {"size":ctx["size"],"current_boxes":ctx["boxes"],"current_vehicles":bay_anchors.vehicles(scene)}


def paved_samples(scene,classes,original):
    from . import paver_pullin
    ctx=native_context(scene)
    boxes={i:b for i,_,b in ctx["current_vehicles"]}
    width,height=ctx["size"]
    result={}
    for option in paver_pullin.candidates(original,classes,ctx):
        x,y=option["point"]
        if not .34*height < y < height-1:
            continue
        row=result.setdefault("left" if x<width/2 else "right",[])
        cell_x,cell_y=int(x*classes.shape[1]/width),int(y*classes.shape[0]/height)
        ids=option["evidence"]["flanking_track_ids"]
        vehicles=sorted([[i,boxes[i]] for i in ids],key=lambda v:v[1][0]+v[1][2])
        row.append({"id":len(row)+1,"point":list(map(float,option["point"])),"box":list(map(float,option["box"])),
                    "class":int(classes[cell_y,cell_x]),"vehicles":vehicles,"source_shape":digest(option)})
    return result


def get_packet(key, *, paved=False):
    return read_photo_packet(PAVED_BANK if paved else BANK,key,90000)

def extension_samples(scene, classes, original):
    from . import driveway_apron, row_extension
    ctx = native_context(scene)
    boxes = {i:b for i,_,b in ctx["current_vehicles"]}
    width, height = ctx["size"]
    result = {}
    for model, field in ((driveway_apron,"supporting_entrance_track_ids"),(row_extension,"supporting_row_track_ids")):
        for option in model.candidates(original,classes,ctx):
            x, y = option["point"]
            rows = result.setdefault("left" if x < width/2 else "right", [])
            vehicles = sorted([[i,boxes[i]] for i in option["evidence"][field]],key=lambda v:v[1][0]+v[1][2])
            cell = int(classes[int(y*classes.shape[0]/height),int(x*classes.shape[1]/width)])
            rows.append({"id":len(rows)+1, "point":option["point"], "box":option["box"], "class":cell,
                         "vehicles":vehicles, "native_option":option, "source_shape":digest(option)})
    return result


def get_extension_packet(key):
    return read_photo_packet(EXTENSION_BANK,key,90000)


def verified_extension(scene,photo,surface,packet):
    return verified(scene,photo,surface,packet,extension=True)


def verified(scene, photo, surface, packet, *, paved=False, references=False, extension=False):
    key=scene["provenance"]["image_sha256"]
    if sha256(photo).hexdigest()!=key:
        raise ValueError("Visual location evidence requires its original photograph.")
    stamp=digest([key,context(scene),surface,packet,paved,references,extension])
    with _lock:
        if stamp in _cache:
            _cache.move_to_end(stamp)
            return deepcopy(_cache[stamp])
    _authenticate(packet)
    body=packet["body"]
    recipe=board.PAVED_RECIPE if paved else board.RECIPE
    guides=[board.REVIEW,board.PAVED_GUIDE] if paved else [board.SELECT,board.REVIEW]
    if extension:
        recipe=board.EXTENSION_RECIPE
        guides=[board.REVIEW,board.EXTENSION_GUIDE]
    expected=(recipe,roadside_board.REVISION,key,digest(context(scene)),digest(guides))
    if tuple(body.get(k) for k in ("recipe","revision","source","context","guides")) != expected:
        raise ValueError("The visual location review differs from this scene or model.")
    classes=manual_surfaces.decode(surface,scene["frame"],key)
    options=[]
    with Image.open(BytesIO(photo)) as original:
        points=(extension_samples(scene,classes,original) if extension else
                paved_samples(scene,classes,original) if paved else samples(scene,classes))
        if set(points)!=set(body["sides"]):
            raise ValueError("The reviewed photo sides changed.")
        for side, rows in points.items():
            receipt=body["sides"][side]
            if (receipt["spec"]!=digest(rows) or receipt["view"]!=sha256(board.selection(original,rows,side)).hexdigest()
                    or len(receipt["reviewed"])>(len(rows) if paved or extension else 2)):
                raise ValueError("A labelled view of the original changed.")
            selected=set()
            for review in receipt["reviewed"]:
                i=review["id"]
                if type(i) is not int or not 1<=i<=len(rows) or i in selected:
                    raise ValueError("Invalid reviewed ground point.")
                selected.add(i)
                sample=rows[i-1]
                if review["view"]!=sha256(board.inspection(original,sample)).hexdigest():
                    raise ValueError("The visual review shows a different ground location.")
                obs=review["description"]
                statuses={"OPEN","BLOCKED","LANE","SEPARATED","UNCLEAR"}
                edges={"painted bay","recessed curb","different surface","parked row","none"}
                fields=("left_vehicle","right_vehicle","rear_vehicle","frontage","roadside")
                if (obs.get("status") not in statuses or obs.get("edge") not in edges or
                        any(not isinstance(obs.get(k),str) or len(obs[k])>180 for k in fields)):
                    raise ValueError("Invalid labels in visual inspection.")
                blocked=obs["status"] in {"BLOCKED","LANE","SEPARATED"}
                if not extension and ((references and (not paved or not blocked)) or (not references and blocked)):
                    continue
                if not (paved or extension) and (obs["status"]!="OPEN" or obs["edge"]=="none"):
                    continue
                box=sample["box"]
                l,t,r,b=box
                options.append({"kind":"paved_pullin" if paved else "visual_point","point":sample["point"], "box":box,
                                "polygon":[[l,t],[r,t],[r,b],[l,b]], "surface_class":sample["class"],
                                "vehicle_crops":sample.get("vehicles",[]),
                                "evidence":{"recipe":recipe,"coverage":"interior_point_only","painted_boundaries":[],
                                            "boundary_widths":[],"boundary_opposite_closed":[],"edge":obs["edge"],"status":obs["status"]},
                                "descriptions":{"nearer_vehicle":obs["left_vehicle"],"farther_vehicle":obs["right_vehicle"],
                                                "frontage":obs["frontage"],"roadside":obs["roadside"]},
                                "rear_vehicle":obs["rear_vehicle"]})
                if extension:
                    options[-1].update(kind="native_entrance_or_row", native_option=sample["native_option"])
    with _lock:
        _cache[stamp]=deepcopy(options)
        if len(_cache)>96:
            _cache.popitem(last=False)
    return options
