"""Measure a short paved pocket beyond the projected, continuous kerb on both approaches."""
import cv2
import numpy as np
from PIL import Image

from . import bay_clearance

RECIPE = "two_approach_kerb_recess_v1"
KERB, ROAD, SIDEWALK = 2, 13, 15
KERBSIDE = (2, 9, 15, 36)


def _runs(mask):
    changes=np.diff(np.r_[False,mask,False].astype(np.int8))
    return zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))


def _edge(classes, connected):
    """The first visible outside edge across a substantial strip of the camera's roadway."""
    h,w=classes.shape
    road=(classes==ROAD)&connected
    # A one-cell outer kerb tip can disappear when the reachable-road mask is opened.
    # Prove connectivity across the strip leading to it, then use the actual measured edge.
    both=(classes[:,:-1]==ROAD) & np.isin(classes[:,1:],KERBSIDE)
    profile=np.full(h,np.nan)
    inward=max(8,round(.035*w))
    for y,x in zip(*np.nonzero(both)):
        if x<w*.5 or x>=w-8 or np.isfinite(profile[y]):
            continue
        if road[y,max(0,x-inward):x+1].mean()<.90:
            continue
        back=classes[y,x+1:min(w,x+max(8,round(.030*w)))]
        if not np.any(np.isin(back,(KERB,SIDEWALK))) or np.isin(back,KERBSIDE).mean()<.55:
            continue
        profile[y]=x+.5
    return profile


def _fits(profile, w):
    """A perspective straight edge must return on both sides of a localised outward deflection."""
    h=len(profile)
    after=max(13,round(.087*h))
    tick=max(2,round(.007*h))
    result=[]
    for start in range(int(.43*h),int(.92*h)-after,tick):
        yy=np.arange(start,start+after)
        valid=np.isfinite(profile[yy])
        if valid.mean()<.90:
            continue
        slope,b=np.polyfit(yy[valid],profile[yy][valid],1)
        if not .26<=slope<=5.5:
            continue
        baseline=np.arange(h)*slope+b
        error=profile-baseline
        if np.quantile(abs(error[yy][valid]),.85)>.005*w:
            continue
        lower=max(0,start-round(.30*h))
        for left,right in _runs(np.isfinite(error[lower:start]) & (error[lower:start]>.014*w)):
            lo,hi=left+lower,right+lower
            if not .05*h<=hi-lo<=.25*h or start-hi>.055*h or lo<7:
                continue
            far=np.arange(max(0,lo-round(.028*h)),lo)
            inside=far[np.isfinite(error[far]) & (abs(error[far])<.018*w)]
            if len(inside)<max(5,round(.009*h)) or inside[-1]<lo-3:
                continue
            if len(inside)>1 and np.any(np.diff(inside)>2):
                continue
            lobe=error[lo:hi]
            peak=float(lobe.max())
            if (peak<.057*w or peak>.36*w or np.mean(lobe>.45*peak)<.30 or
                    np.median(lobe[:max(2,len(lobe)//6)])>.78*peak or
                    np.median(lobe[-max(2,len(lobe)//6):])>.72*peak):
                continue
            # Fit quality on both measured approaches ranks competing fits to the same kerb.
            quality=float(np.median(abs(error[yy][valid]))+np.median(abs(error[inside])))
            result.append((quality,lo,hi,baseline,peak))
    return sorted(result,key=lambda item:item[0])


def _optical_separator(gray, baseline, profile, lo, hi, peak):
    """Look across the whole entry corridor for a persistent cast edge or raised/painted stripe."""
    h,w=gray.shape
    rows=np.arange(lo,hi)
    rows=rows[profile[rows]-baseline[rows]>.55*peak]
    if len(rows)<8:
        return True
    lateral=np.arange(-round(.085*w),max(4,min(round(.05*w),round(.37*peak)))+1)
    # A curb, fence base or longitudinal marking can lie on either side of the
    # projected line. It can converge towards the camera at a different angle.
    for tilt in (-.7,-.35,0,.35,.7):
        x=np.rint(baseline[rows,None]+lateral+tilt*(rows[:,None]-np.median(rows))).astype(int)
        usable=(x>=7)&(x<w-7)&(x<profile[rows,None]-.20*(profile[rows,None]-baseline[rows,None]))
        x=np.clip(x,7,w-8)
        y=rows[:,None]
        center=(gray[y,x-1]+gray[y,x]+gray[y,x+1])/3
        left=(gray[y,x-6]+gray[y,x-4])/2
        right=(gray[y,x+4]+gray[y,x+6])/2
        stripe=center-np.maximum(left,right)>15
        face=(np.maximum(left,right)-np.minimum(left,right)>35)&(np.abs(center-(left+right)/2)>12)
        if ((usable & (stripe|face)).sum(axis=0)>.64*len(rows)).any():
            return True
    return False


def _open_region(classes, rgb, occupied, profile, fit):
    _,lo,hi,baseline,peak=fit
    h,w=classes.shape
    green=(rgb[:,:,1].astype(int)>rgb[:,:,0].astype(int)*1.12+4) & (rgb[:,:,1].astype(int)>rgb[:,:,2].astype(int)*1.10+4)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)
    if _optical_separator(gray,baseline,profile,lo,hi,peak):
        return None
    allowed=[]
    contrasts=[]
    lip_paint=[]
    for y in range(lo,hi):
        mouth=baseline[y]
        outer=profile[y]
        depth=outer-mouth
        if depth<.35*peak:
            continue
        start=max(0,int(mouth-.085*w)); lip=max(start,int(mouth-.025*w)); stop=min(w,int(outer-1))
        if stop-start<10:
            continue
        segment=classes[y,lip:stop]
        # The wider strip can include the neighbouring lane's paint before the
        # entrance. Its pavement must stay connected, with no fence, curb or island.
        approach=classes[y,start:lip]
        clear=(np.isin(approach,(ROAD,10,24,36)).all()
               and np.mean(segment==ROAD)>=.97 and np.isin(segment,(ROAD,10)).all()
               and not occupied[y,start:stop].any() and green[y,start:stop].mean()<.035)
        if not clear:
            continue
        m=int(round(mouth)); x=int(outer)
        if m<5 or x<4 or x+5>=w:
            continue
        contrasts.append(abs(float(gray[y,x+1:x+5].mean()-gray[y,x-4:x].mean())))
        lip_paint.append(float(gray[y,m-1:m+2].mean()-max(gray[y,m-5:m-2].mean(),gray[y,m+3:m+6].mean())))
        # Keep a measured margin on both the main lane and the raised outer kerb.
        allowed.append((y,float(mouth+.22*depth),float(outer-.13*depth)))
    eligible=np.flatnonzero((profile[lo:hi]-baseline[lo:hi])>=.35*peak)+lo
    if (len(allowed)<max(9,.064*h) or len(allowed)<.92*len(eligible) or
            np.quantile(contrasts,.62)<8 or np.mean(np.asarray(lip_paint)>16)>.27):
        return None
    rows=np.array([a[0] for a in allowed])
    if np.max(np.diff(rows))>max(1,.004*h):
        return None
    return allowed


def _regions(original, classes, context):
    if not isinstance(original,Image.Image) or original.size!=tuple(context["size"]):
        raise ValueError("The kerb check requires the original photograph and dimensions")
    if classes.ndim!=2 or classes.dtype!=np.uint8:
        raise ValueError("Expected the original signed semantic map")
    h,w=classes.shape
    road=bay_clearance.visible_road_component(classes)
    if road is None or not np.any(classes==KERB) or not np.any(classes==SIDEWALK):
        return
    rgb=np.asarray(original.convert("RGB").resize((w,h),Image.Resampling.BOX))
    scale=np.array([w/original.width,h/original.height])
    occupied=np.zeros((h,w),np.uint8)
    for box in context["current_boxes"]:
        x0,y0,x1,y1=np.array(box,dtype=float)*np.tile(scale,2)
        occupied[max(0,int(y0)):min(h,int(np.ceil(y1))+1),max(0,int(x0)):min(w,int(np.ceil(x1))+1)]=1
    for side in ("left","right"):
        labels,connected,colour,objects=(tuple(np.fliplr(v) for v in (classes,road,rgb,occupied))
                                          if side=="left" else (classes,road,rgb,occupied))
        profile=_edge(labels,connected)
        accepted=[]
        for fit in _fits(profile,w):
            _,lo,hi,baseline,peak=fit
            if any(max(lo,a)<min(hi,b) for a,b in accepted):
                continue
            allowed=_open_region(labels,colour,objects,profile,fit)
            if allowed is None:
                continue
            accepted.append((lo,hi))
            yield side,allowed,scale,w,{"recipe":RECIPE,"coverage":"interior_point_only","side":side,
                "map_size":[w,h],"recess_rows":[int(lo),int(hi)],"approach_line_x_at_endpoints":
                [round(float(baseline[lo]),3),round(float(baseline[hi-1]),3)],
                "maximum_setback_map_pixels":round(peak,3),"clear_opening_rows":len(allowed)}


def _result(p,mark,evidence):
    return {"point":list(p),"box":list(mark),
            "polygon":[[mark[0],mark[1]],[mark[2],mark[1]],[mark[2],mark[3]],[mark[0],mark[3]]],
            "evidence":evidence}


def marked(original,classes,context,mark):
    width,height=original.size
    x0,y0,x1,y1=mark
    if not (0<=x0<x1<=width and 0<=y0<y1<=height and x1-x0<.055*width and y1-y0<.055*height):
        return None
    p=[(x0+x1)/2,(y0+y1)/2]
    choices=[]
    for side,rows,scale,w,evidence in _regions(original,classes,context):
        left,top,right,bottom=np.array(mark)*np.tile(scale,2)
        if side=="left":
            left,right=w-1-right,w-1-left
        envelope={y:(a,b) for y,a,b in rows}
        covered=range(int(np.floor(top)),int(np.ceil(bottom))+1)
        if all(y in envelope and envelope[y][0]<=left<=right<=envelope[y][1] for y in covered):
            choices.append(_result(p,mark,evidence))
    return choices[0] if len(choices)==1 else None


def candidates(original,classes,context):
    options=[]
    for side,rows,scale,w,evidence in _regions(original,classes,context):
        # Choose a central interior of the measured open section, away from its narrower ends.
        q=rows[len(rows)//2]
        x=(q[1]+q[2])/2
        xy=np.array([w-1-x if side=="left" else x,q[0]],dtype=float)/scale
        dx,dy=2/scale
        mark=np.round([xy[0]-dx,xy[1]-dy,xy[0]+dx,xy[1]+dy],3).tolist()
        options.append(_result(np.round(xy,3).tolist(),mark,evidence))
    return options


def marked_outcome(request,photo,surface,box):
    from io import BytesIO
    from . import manual_offlane,manual_surfaces,visual_space,whole_bay
    from .parking_evidence import source_input
    clean,key,portable=source_input(request,photo)
    _,_,mark,center=manual_offlane.mark_input(clean,box)
    scene=clean["scene"]
    classes=manual_surfaces.decode(surface,scene["frame"],key)
    reason,_=manual_offlane.surface_check(scene["frame"],scene["tracks"],mark,center,classes)
    if reason!="through_lane":
        return None
    with Image.open(BytesIO(photo)) as original:
        option=marked(original,classes,visual_space.native_context(scene),mark)
    if option is None:
        return None
    return whole_bay._make(clean,portable,[option],surface,method=whole_bay.MARK,
        reason="passenger_mark_inside_measured_kerb_recess",question="",tap=center,selected_box=mark,
        kerb_geometry=option["evidence"],kerb_policy=RECIPE,clearance_policy=whole_bay.CLEARANCE_POLICY)
