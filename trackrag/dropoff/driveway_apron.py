"""Locate an interior point of a visibly distinct vehicle entrance through a footway.

A marked curb cut on its own is commonly a pedestrian ramp. Require a broad
material corridor between two raised returns, free access from the currently
visible carriageway, and parked vehicles on the far end of the same surface.
"""
import cv2
import numpy as np
from PIL import Image

from .bay_clearance import visible_road_component
from .curb_recess import _vehicles
from .row_extension import MOTOR_CLASSES, _locally_planted, _overlap


RECIPE = "contrasting_footways_vehicle_entrance_v1"
MATERIAL_CLASSES = (9, 10, 14, 15)


def _dilate(mask, width):
    return cv2.dilate(mask.astype(np.uint8), np.ones((width,width),np.uint8)).astype(bool)


def _material(lab, labels, cut, free):
    chroma = np.median(lab[cut,1:], axis=0)
    if np.any(np.percentile(lab[cut,1:],85,axis=0) - np.percentile(lab[cut,1:],15,axis=0) > 10):
        return []
    permitted = np.isin(labels,MATERIAL_CLASSES) & free
    core = permitted & (np.linalg.norm(lab[:,:,1:] - chroma,axis=2) <= 3.4)
    core = cv2.morphologyEx(core.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)).astype(bool)
    count,groups,stats,_ = cv2.connectedComponentsWithStats((core & permitted).astype(np.uint8),connectivity=8)
    touching = _dilate(cut,5)
    result=[]
    h,w=labels.shape
    for i in range(1,count):
        x,y,cw,ch,area=stats[i]
        if not (.10*w <= cw <= .62*w and .060*h <= ch <= .39*h and area >= .006*h*w):
            continue
        component = groups == i
        if (np.count_nonzero(component & touching) < .12 * cut.sum()
                or np.count_nonzero(component & (labels==10)) < max(20,.003*area)):
            continue
        result.append((component,(int(x),int(y),int(cw),int(ch)),chroma))
    return result


def _rows_of_returns(component, bounds, chroma, labels, lab):
    """Compare the two outer pavements and require actual kerb labels at both boundaries."""
    lx,ty,cw,ch = bounds
    w=labels.shape[1]
    matched=[]
    left_kerbs=[]
    right_kerbs=[]
    for y in range(ty,ty+ch):
        row=component[y,lx:lx+cw]
        ends=np.diff(np.r_[False,row,False].astype(np.int8))
        intervals=list(zip(np.flatnonzero(ends==1)+lx,np.flatnonzero(ends==-1)+lx))
        if not intervals:
            continue
        left,right = max(intervals,key=lambda pair:pair[1]-pair[0])
        width=right-left
        if width < .38*cw:
            continue
        margin=max(4,round(.075*width))
        at_left=slice(max(0,left-margin),min(w,left+margin+1))
        at_right=slice(max(0,right-margin),min(w,right+margin+1))
        if np.any(labels[y,at_left]==2):
            left_kerbs.append(y)
        if np.any(labels[y,at_right]==2):
            right_kerbs.append(y)
        # Sample beyond the concrete kerb and its shadow, on both pedestrian pavements.
        d1,d2=round(.18*width),round(.27*width)
        if left-d2<0 or right+d2>=w:
            continue
        margins=(slice(left-d2,left-d1),slice(right+d1,right+d2))
        if any(np.mean(np.isin(labels[y,v],(2,10,15))) < .60 for v in margins):
            continue
        colors=[np.median(lab[max(0,y-1):y+2,v,1:],axis=(0,1)) for v in margins]
        shifts=[float(np.linalg.norm(v-chroma)) for v in colors]
        if min(shifts)>=5.0 and np.linalg.norm(colors[0]-colors[1])<=.62*min(shifts):
            matched.append((y,left,right))
    span=max(8,round(.10*ch))
    if (len(matched)<span or matched[-1][0]-matched[0][0]<span
            or len(left_kerbs)<span or len(right_kerbs)<span):
        return None
    return {"paired_pavement_rows":[matched[0][0],matched[-1][0]],
            "paired_pavement_count":len(matched),
            "kerb_return_rows":[len(left_kerbs),len(right_kerbs)]}


def _contacts(cars, labels, component, cut):
    ends=[]
    cut_y=float(np.median(np.nonzero(cut)[0]))
    h,w=labels.shape
    for car in cars:
        l,t,r,b=car.box
        if b>=cut_y-.22*car.height:
            continue
        body=labels[round(t+.52*car.height):int(np.ceil(b)),int(l):int(np.ceil(r))]
        if not body.size or np.mean(np.isin(body,MOTOR_CLASSES))<.33:
            continue
        a,z = round(l+.16*car.width),round(r-.16*car.width)
        v,u = min(h-1,round(b)+1),min(h,round(b)+max(4,round(.23*car.height)))
        if z<=a or u<=v:
            continue
        foot=np.s_[v:u,a:z]
        coverage=component[foot]
        # A wide area of visually matching material can appear immediately
        # beyond a raised kerb or low wall. Follow separate wheel columns from
        # the bounding-box foot; pavement behind a sustained barrier cannot
        # establish that the observed vehicle actually sits on our apron.
        barriers=np.isin(labels[foot],(2,3,4,5,6,26,29,30))
        sustained=np.zeros_like(barriers)
        sustained[1:]=barriers[1:] & barriers[:-1]
        contact=coverage & ~np.maximum.accumulate(sustained,axis=0)
        direct_columns=int(contact.any(axis=0).sum())
        if direct_columns<max(.25*(z-a),.65*coverage.any(axis=0).sum()) or contact.mean()<.20:
            continue
        parked=labels[foot]==10
        if coverage.mean()>=.29:
            # Save the actual part of the photographed pavement that contacts this tyre line.
            yy,xx=np.nonzero(coverage)
            supported=coverage & parked
            parking_fraction=float(supported.sum()/coverage.sum()) if supported.sum()>=5 else 0.
            ends.append((car,parking_fraction,[a+int(np.median(xx)),v+int(np.median(yy))]))
    pairs=[]
    for i,first in enumerate(ends):
        for second in ends[i+1:]:
            c1,c2=first[0],second[0]
            centres=[(c.box[0]+c.box[2])/2 for c in (c1,c2)]
            if (max(first[1],second[1])>=.40 and _overlap(c1.box,c2.box)<.26
                    and abs(centres[1]-centres[0])>.72*min(c1.width,c2.width)):
                pairs.append((first,second))
    return sorted(pairs,key=lambda pair:-max(v[1] for v in pair))


def _reachable(component, cut, road, labels, free, car_width, car_height):
    # Restrict exits to THIS original cut. Touching another piece of matching
    # pavement elsewhere in the picture cannot bypass a chain, bollard or kerb.
    aisle=road & np.isin(labels,(13,24))
    admissible_aisle=aisle & (~_dilate(component,7) | _dilate(cut,11))
    passable=(component | cut | admissible_aisle) & free
    if np.any(_dilate(cut,11) & np.isin(labels,(7,8,12,23))):
        return None
    rx=max(5,round(.39*car_width))
    ry=max(4,round(.20*car_height))
    vehicle=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rx+1,2*ry+1))
    centerline=cv2.erode(passable.astype(np.uint8),vehicle,borderType=cv2.BORDER_CONSTANT,borderValue=0)
    _,groups=cv2.connectedComponents(centerline,connectivity=8)
    seeds=(labels==13) & aisle & ~_dilate(component,25)
    ids,areas=np.unique(groups[seeds],return_counts=True)
    valid=[]
    for i,area in zip(ids,areas,strict=True):
        if i and area>labels.size*.002 and np.count_nonzero((groups==i)&cut)>=max(5,.35*car_width):
            valid.append(i)
    if not valid:
        return None
    reachable=np.isin(groups,valid)
    return reachable,[rx,ry],int(np.count_nonzero(reachable&cut))


def _run(original, labels, context, mark=None):
    if (not isinstance(labels,np.ndarray) or labels.ndim!=2 or min(labels.shape)<64
            or not isinstance(context,dict) or tuple(context.get("size",()))!=original.size
            or "current_vehicles" not in context or "current_boxes" not in context):
        return []
    h,w=labels.shape
    if (np.count_nonzero(labels==9)<max(40,labels.size*.0003)
            or np.mean(labels==17)<.025 or np.mean(labels==15)<.022 or np.mean(labels==10)<.0010):
        return []
    scale=np.array([w,h])/np.asarray(original.size)
    try:
        cars,boxes=_vehicles(context,scale,np.asarray(original.size),h,w)
    except (TypeError,ValueError):
        return []
    if len(cars)<2:
        return []
    road=visible_road_component(labels)
    if road is None:
        return []
    free=np.ones((h,w),bool)
    for l,t,r,b in boxes:
        free[int(t):int(np.ceil(b))+1,int(l):int(np.ceil(r))+1]=False
    rgb=np.asarray(original.convert("RGB").resize((w,h),Image.Resampling.BOX))
    lab=cv2.cvtColor(cv2.medianBlur(rgb,5),cv2.COLOR_RGB2LAB).astype(np.float32)
    number,groups,stats,_=cv2.connectedComponentsWithStats((labels==9).astype(np.uint8),connectivity=8)
    options=[]
    for idx in range(1,number):
        _,_,cw,ch,area=stats[idx]
        if max(cw/w,ch/h)<.065 or area<max(40,labels.size*.00035):
            continue
        cut=groups==idx
        near=_dilate(cut,5)
        if (np.count_nonzero(near & (labels==13) & road)<.18*max(cw,ch)
                or np.count_nonzero(near & (labels==2))<.22*max(cw,ch)):
            continue
        for component,bounds,chroma in _material(lab,labels,cut,free):
            if mark is not None:
                mx,my=np.rint((np.asarray(mark)+.5)*scale-.5).astype(int)
                if not component[my,mx]:
                    continue
            borders=_rows_of_returns(component,bounds,chroma,labels,lab)
            if not borders:
                continue
            pairs=_contacts(cars,labels,component,cut)
            distance=cv2.distanceTransform(component.astype(np.uint8),cv2.DIST_L2,5)
            for pair in pairs:
                car_width=min(v[0].width for v in pair)
                car_height=min(v[0].height for v in pair)
                opening=_reachable(component,cut,road,labels,free,car_width,car_height)
                if opening is None:
                    continue
                reachable,footprint,gate_cells=opening
                # Both parked vehicles genuinely touch this drivable material.
                # Some terminal wheel pixels sit immediately under the detected box,
                # so look for the neighbouring reachable corridor behind the contact.
                if any(not _dilate(reachable,max(7,2*footprint[1]+5))[p[2][1],p[2][0]] for p in pair):
                    continue
                if mark is None:
                    yy,xx=np.nonzero(component)
                    x,y=round(float(xx.mean())),round(float(yy.mean()))
                else:
                    x,y=int(mx),int(my)
                radius=max(2,min(6,round(.08*car_width)))
                if (not reachable[y,x] or labels[y,x] not in (10,14,15)
                        or distance[y,x]<max(radius+2,.18*car_width)):
                    continue
                def native(x,y):
                    return [round((x+.5)/scale[0]-.5,3),round((y+.5)/scale[1]-.5,3)]
                point=native(x,y) if mark is None else list(mark)
                l,t=native(x-radius,y-radius)
                r,b=native(x+radius,y+radius)
                if _locally_planted(original,(int(l),int(t),int(np.ceil(r)),int(np.ceil(b)))):
                    continue
                evidence={"recipe":RECIPE,"extent":"interior_point_only","signed_center_class":int(labels[y,x]),
                          "supporting_entrance_track_ids":[v[0].track for v in pair],
                          "vehicle_contact_native":[native(*v[2]) for v in pair],"material_parking_cells":int(np.sum(component&(labels==10))),
                          "cut_cells":int(area),"cut_vehicle_clear_cells":gate_cells,"clearance_grid_radii":footprint,**borders}
                options.append({"point":point,"box":[l,t,r,b],"support_patch":[[l,t],[r,t],[r,b],[l,b]],"evidence":evidence})
                break
    return sorted(options,key=lambda o:o["point"])


def candidates(original, classes, context):
    return _run(original,classes,context)


def marked(original, classes, context, box):
    if not isinstance(box,(list,tuple)) or len(box)!=4:
        return None
    try:
        l,t,r,b=map(float,box)
    except (TypeError,ValueError):
        return None
    w,h=original.size
    if not (0<=l<r<=w and 0<=t<b<=h and r-l<=.012*w and b-t<=.012*h):
        return None
    options=[o for o in _run(original,classes,context,[(l+r)/2,(t+b)/2])
             if o["box"][0]<=l<=r<=o["box"][2] and o["box"][1]<=t<=b<=o["box"][3]]
    return options[0] if len(options)==1 else None
