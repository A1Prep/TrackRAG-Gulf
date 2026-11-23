"""Detect forward-facing parallel stalls with two painted T/L ends and a visible kerb."""
from itertools import combinations

import cv2
import numpy as np
from PIL import Image


# The Vistas labels distinguish parking, road, paint and flush manhole covers.
PAVEMENT = (10, 13, 24, 36)
KERB = (2, 15)
RECIPE = "painted_curbside_stall_ends_v1"


def _sample(array, points):
    x,y = np.rint(points).astype(int).T
    return array[y.clip(0,array.shape[0]-1), x.clip(0,array.shape[1]-1)]


def _ends(gray, classes):
    h,w = gray.shape
    ground = np.isin(classes,PAVEMENT)
    curb = np.isin(classes,KERB)
    y0 = int(.43*h)
    contrast = cv2.createCLAHE(2.5,(12,8)).apply(gray)
    segments = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(
        cv2.GaussianBlur(contrast[y0:],(3,3),0))[0]
    if segments is None:
        return []
    # All differences of uint8 intensities fit exactly in a signed 16-bit
    # integer. Keep at most one pair of neighbouring rows and one accumulator.
    intensity = gray.astype(np.int16)
    # At this perspective a distant stripe can be one pixel thick. Test its light ridge against
    # both flanks at two scales; an asphalt/sidewalk edge alone does not supply a painted stripe.
    ridge = None
    for gap in (3,6):
        flanks = np.roll(intensity,gap,0)
        np.maximum(flanks,np.roll(intensity,-gap,0),out=flanks)
        np.subtract(intensity,flanks,out=flanks)
        if ridge is None:
            ridge = flanks
        else:
            np.maximum(ridge,flanks,out=ridge)
    cap_list = []
    for segment in segments.reshape(-1,2,2):
        a,b = sorted(segment + [0,y0],key=lambda p:p[0])
        dx,dy = b-a
        if dx <= 0 or not .22 <= dy/dx <= 3.5:
            continue
        length = np.linalg.norm(b-a)
        if not .009*w <= length <= .11*w or (a[0]+b[0])/2 < .29*w or (a[1]+b[1])/2 < .52*h:
            continue
        probes = np.linspace(a,b,11)[1:-1]
        if _sample(ground,probes).mean() < .85:
            continue
        axis = (b-a)/length
        n = np.array([-axis[1],axis[0]])
        core = np.maximum.reduce([_sample(intensity,probes+n*delta) for delta in (-2,0,2)])
        flanks = np.maximum(_sample(intensity,probes-n*6),_sample(intensity,probes+n*6))
        if np.median(core-flanks) < 6:
            continue
        cap_list.append((a,b,axis))
    # Nearby edge fragments are common on lettering or a bumper. Their image classes and the
    # two ground stripes ending at the same curb independently constrain the candidates.
    results = []
    for a,b,axis in cap_list:
        best = None
        for t in (-.10,.15,.45,.75,1.05,1.25):
            tip = a + t*(b-a)
            x,y = np.rint(tip).astype(int)
            if y<2 or y>=h-3 or x<0 or x>=w-8 or not ground[y,x]:
                continue
            curb_indices = np.flatnonzero(curb[y,x+5:])
            if len(curb_indices)==0:
                continue
            reach = int(curb_indices[0])+5
            if not .041*w < reach < .315*w:
                continue
            # Horizontal parking ends run across the stall, towards its curb.
            offsets=np.arange(6,max(8, min(reach-4,int(.18*w))),3)
            if len(offsets)<12:
                continue
            slopes=np.linspace(-.16,.16,17)
            xx=x+offsets
            yy=np.rint(y+slopes[:,None]*offsets).astype(int).clip(2,h-3)
            values=np.maximum.reduce([ridge[yy+d,xx] for d in (-1,0,1)])
            # Require a long, predominantly continuous stroke, including near the junction.
            good=(values>=6) & ground[yy,xx]
            scores = .5*good.mean(axis=1) + .3*good[:,:len(offsets)//2].mean(axis=1) + .2*good[:,:8].mean(axis=1)
            i=int(np.argmax(scores))
            if scores[i]<.66 or np.median(values[i])<7:
                continue
            ry=np.rint(y+slopes[i]*np.arange(0,min(w-x,int(.35*w)))).astype(int).clip(0,h-1)
            rx=x+np.arange(len(ry))
            meet=np.flatnonzero(curb[ry,rx])
            if not len(meet) or not .04*w<meet[0]<.32*w:
                continue
            k=int(meet[0])
            if ground[ry[3:k-2],rx[3:k-2]].mean()<.91:
                continue
            item=(float(scores[i]),np.array([x,y],float),np.array([rx[k-1],ry[k-1]],float),axis)
            if best is None or item[0]>best[0]:
                best=item
        if best is not None:
            results.append(best)
    distinct=[]
    for item in sorted(results,key=lambda r:-r[0]):
        if all(np.linalg.norm(item[1]-other[1])>.017*w for other in distinct):
            distinct.append(item)
    return sorted(distinct,key=lambda row:row[1][1])


def _stalls(gray, classes, boxes, colored_markings):
    h,w=gray.shape
    ends=_ends(gray,classes)
    result=[]
    for far,near in combinations(ends,2):
        af,bf,an,bn=far[1],far[2],near[1],near[2]
        dx,dy=an-af
        if not .033*h<dy<.36*h or not .52<dx/dy<4.8:
            continue
        row=(an-af)/np.linalg.norm(an-af)
        if min(row@far[3],row@near[3])<.89:
            continue
        wf,wn=bf[0]-af[0],bn[0]-an[0]
        if not 1.10<wn/wf<3.5 or not .19<dy/((wf+wn)/2)<1.45:
            continue
        normal = np.array([-row[1], row[0]])
        if any(af[1]+4<e[1][1]<an[1]-4 and
               abs(normal @ (e[1]-af))<.12*wf for e in ends):
            continue
        p=np.float32([af,bf,bn,an])
        if not cv2.isContourConvex(p):
            continue
        # Check the complete curb-facing side and all visible pavement enclosed by the markings.
        curb_side=np.linspace(bf,bn,25)[2:-2]
        nearby_curb=np.maximum.reduce([np.isin(_sample(classes,curb_side+[d,0]),KERB) for d in (-6,0,5,10)])
        if nearby_curb.mean()<.8:
            continue
        x,y,pw,ph=cv2.boundingRect(p); mask=np.zeros((ph,pw),np.uint8)
        cv2.fillConvexPoly(mask,np.rint(p-[x,y]).astype(np.int32),1)
        core=cv2.erode(mask,np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
        interior=classes[y:y+ph,x:x+pw][core]
        if not interior.size or np.isin(interior,PAVEMENT).mean()<.975 or np.isin(interior,(8,23)).any():
            continue
        # A continuous coloured edge stripe passing through the suggested footprint can look
        # like two stall ends where it crosses paving joints. Keep it out of the parking offer.
        colored = colored_markings[y:y+ph,x:x+pw] & core
        if colored.sum() >= .025*len(interior):
            _,_,stats,_ = cv2.connectedComponentsWithStats(colored.astype("uint8"))
            if any(s[cv2.CC_STAT_HEIGHT] >= .56*ph and s[cv2.CC_STAT_AREA] >= .018*len(interior) for s in stats[1:]):
                continue
        occupied=False
        for left,top,right,bottom in boxes:
            rect=np.float32([[left,top],[right,top],[right,bottom],[left,bottom]])
            if cv2.intersectConvexConvex(p,rect)[0]>1:
                occupied=True
                break
        if occupied:
            continue
        result.append((p,far,near,int((interior==10).sum()),int(interior.size)))
    return result


def candidates(original, classes, context):
    """Return positive parallel stalls, from both sides of the original image."""
    w=min(1200,original.width)
    h=round(original.height*w/original.width)
    rgb=np.array(original.convert("RGB").resize((w,h),Image.Resampling.BOX))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    hue,sat,value=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV).transpose(2,0,1)
    colored=(hue>=10)&(hue<=42)&(sat>85)&(value>112)
    classes=cv2.resize(np.asarray(classes),(w,h),interpolation=cv2.INTER_NEAREST)
    scale=np.array(original.size,dtype=float)/[w,h]
    boxes=np.asarray(context["current_boxes"],dtype=float).reshape(-1,4)/np.tile(scale,2)
    options=[]
    for mirror in (False,True):
        cb=boxes.copy()
        if mirror:
            cb[:,[0,2]]=w-1-cb[:,[2,0]]
        for p,far,near,parking,count in _stalls(gray[:,::-1] if mirror else gray,classes[:,::-1] if mirror else classes,cb,
                                                colored[:,::-1] if mirror else colored):
            p=p.astype(float)
            if mirror:
                p[:,0]=w-1-p[:,0]
            p*=scale
            polygon=p.round(3).tolist()
            moment=cv2.moments(p.astype(np.float32))
            center=np.array([moment["m10"],moment["m01"]])/moment["m00"]
            options.append({"polygon":polygon,"point":center.round(3).tolist(),
                            "box":np.r_[p.min(axis=0),p.max(axis=0)].round(3).tolist(),
                            "evidence":{"recipe":RECIPE,"painted_boundaries":[polygon[:2],[polygon[3],polygon[2]]],
                                        "boundary_widths":[round(float(3*scale.mean()),3)]*2,"boundary_opposite_closed":[False,False],
                                        "paint_contrast_support":[round(far[0],3),round(near[0],3)],
                                        "visible_parking_cells":parking,"interior_cells":count}})
    return sorted(options,key=lambda option:tuple(option["point"]))
