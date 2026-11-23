"""Locate empty interiors between three observed dividers in a photographed parking row."""
import cv2
import numpy as np
from PIL import Image

from . import bay_clearance, parking_divider_choice

RECIPE = "three_visible_dividers_with_open_aisle_v1"
POINT_COVERAGE = "interior_point_only"


def _sample(arr, pixels):
    xy = np.rint(pixels).astype(int)
    h, w = arr.shape[:2]
    return arr[np.clip(xy[..., 1], 0, h - 1), np.clip(xy[..., 0], 0, w - 1)]


def _stripes(gray, classes):
    """Recover the centre of an optical stripe and require its corresponding learned marking."""
    h, w = classes.shape
    raw = cv2.createLineSegmentDetector().detect(gray)[0]
    if raw is None:
        return []
    found = []
    for a, b in raw.reshape(-1, 2, 2).astype(float):
        if a[0] > b[0]:
            a, b = b, a
        length = np.linalg.norm(b-a)
        if length < .038*w:
            continue
        axis = (b-a)/length
        normal = np.array([-axis[1], axis[0]])
        ray = a + np.linspace(.06, .94, 25)[:,None]*(b-a)
        if not np.any(_sample(classes, ray)==24):
            checks = _sample(classes, ray[:,None,:]+np.array([-2,2])[None,:,None]*normal)
            if not np.any(checks==24):
                continue
        offsets = np.arange(-6,7)
        values = _sample(gray, ray[:,None,:]+offsets[None,:,None]*normal).astype(float)
        levels = np.median(values,axis=0)
        peak = np.argmax(levels)
        shift = offsets[peak]
        if abs(shift)>4 or levels[peak]-max(levels[:2].mean(),levels[-2:].mean())<7:
            continue
        middle = ray+shift*normal
        on_paint = _sample(classes,middle[:,None,:]+np.arange(-1,2)[None,:,None]*normal)
        if np.any(on_paint==24,axis=1).mean()<.62:
            continue
        sides = [_sample(classes,middle + step*normal) for step in (-6,6)]
        if max(np.mean(s==10) for s in sides)<.68:
            continue
        found.append((a + shift*normal, b + shift*normal, axis))
    # Combine fragments belonging to one continuous piece of signed and photographed paint.
    stripes=[]
    for a,b,axis in sorted(found,key=lambda r:-np.linalg.norm(r[1]-r[0])):
        combined = False
        for index,(c,d,other) in enumerate(stripes):
            if abs(axis@other)<.995:
                continue
            normal = np.array([-other[1],other[0]])
            if max(abs((a-c)@normal),abs((b-c)@normal))>3.5:
                continue
            projection = (np.array([a,b,c,d])-c)@other
            if min(projection[1],projection[3])-max(projection[0],projection[2]) < -max(4,.010*w):
                continue
            start=c+projection.min()*other
            end=c+projection.max()*other
            stripes[index]=(start,end,other)
            combined=True
            break
        if not combined:
            stripes.append((a,b,axis))
    return stripes


def _cross(stripe, origin, normal, *, limit=0):
    a,b,axis = stripe
    n=np.array([-axis[1],axis[0]])
    cosine=normal@n
    if abs(cosine)<.97:
        return None
    d=float(((a-origin)@n)/cosine)
    projection=(origin+d*normal-a)@axis
    if not -limit<=projection<=np.linalg.norm(b-a)+limit:
        return None
    return d


def _context(classes,scene,original_size):
    h,w=classes.shape
    scale=np.array([w/original_size[0],h/original_size[1]])
    occupied=np.zeros_like(classes)
    cars=[]
    for obs in scene["tracks"]:
        if obs["frame_id"]!=scene["frame"]["frame_id"]:
            continue
        x0,y0,x1,y1=np.array(obs["bbox_xyxy"])*np.tile(scale,2)
        occupied[max(0,int(y0)):min(h,int(np.ceil(y1))+1),max(0,int(x0)):min(w,int(np.ceil(x1))+1)]=1
        if obs["class_name"].casefold() in {"car","suv","sedan","van","pickup","truck","taxi","light_vehicle"}:
            cars.append([x0,y0,x1,y1])
    radius=max(3,round(.012*w))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    open_ground=np.isin(classes,(10,13,24,36)) & (occupied==0)
    eroded=cv2.erode(open_ground.astype(np.uint8),kernel).astype(bool)
    road=bay_clearance.visible_road_component(np.where(eroded,classes,15).astype(np.uint8))
    if road is not None:
        step=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        for _ in range(radius):
            road=cv2.dilate(road.astype(np.uint8),step).astype(bool)&open_ground
    return scale,occupied,cars,road,radius


def _open_aisle(p,axis,normal,first,second,classes,gray,occupied,access):
    """Prove an unobstructed entry of substantial width within the two measured sides."""
    h,w=classes.shape
    for direction in (-1,1):
        seen_road=0
        step=1.25
        for distance in np.arange(0,.35*w,step):
            at=p+direction*distance*axis
            ds=[_cross(s,at,normal,limit=.135*w) for s in (first,second)]
            if None in ds or abs(ds[1]-ds[0])<max(6,.012*h):
                break
            cross=at + np.linspace(ds[0],ds[1],13)[2:-2,None]*normal
            if ((cross[:,0]<1)|(cross[:,0]>=w-1)|(cross[:,1]<1)|(cross[:,1]>=h-1)).any():
                break
            labels=_sample(classes,cross)
            if (not np.isin(labels,(10,13,24)).all() or _sample(occupied,cross).any()
                    or not _sample(access,cross).all()):
                break
            near=_sample(gray,cross-4*axis).astype(float)
            far=_sample(gray,cross+4*axis).astype(float)
            centre=_sample(gray,cross).astype(float)
            # A full-width fence, kerb edge or cross-hatching in the original vetoes
            # an entrance even when the semantic map labels the underlying pavement.
            step_edges=np.abs(near-far)>30
            ridge=(centre-np.maximum(near,far)>18)|(np.minimum(near,far)-centre>23)
            if np.mean(step_edges|ridge)>.76:
                break
            if np.mean(labels==13)>=.75:
                seen_road+=1
                if step*seen_road>=.012*w:
                    return [direction,round(float(distance),3)]
            else:
                seen_road=0
    return None


def _reference_length(cars, point, spacing, axis):
    matches=[]
    for x0,y0,x1,y1 in cars:
        nearest_x=np.clip(point[0],x0,x1)
        distance=np.hypot(nearest_x-point[0],y1-point[1])
        if (distance<=4.5*spacing
                and (x1-x0)>=2*spacing and (y1-y0)>.7*spacing
                and abs(axis[0])>.65 and (x1-x0)>1.35*(y1-y0)):
            matches.append((distance,(x1-x0)*abs(axis[0])+(y1-y0)*abs(axis[1])))
    return float(min(matches)[1]) if matches else None


def _interiors_at(at, axis, triplet, classes, gray, occupied, access, cars, scale, radius):
    h,w=classes.shape
    normal=np.array([-axis[1],axis[0]])
    offsets=[_cross(s,at,normal) for s in triplet]
    if None in offsets:
        return []
    gaps=np.diff(offsets)
    if min(gaps)<max(8,.014*h) or max(gaps)>.16*h or min(gaps)<.32*max(gaps):
        return []
    vehicle_length=_reference_length(cars,at,float(np.median(gaps)),axis)
    if vehicle_length is None:
        return []
    found=[]
    for i in (0,1):
        point=at+(offsets[i]+offsets[i+1])/2*normal
        # Test a two-dimensional piece of the bay, rather than one isolated pixel.
        # A free sliver at the nose of an occupied stall does not make a parking
        # space. Check a length measured from a nearby comparably oriented vehicle.
        half_length=max(.027*w,.34*vehicle_length)
        along=np.linspace(-half_length,half_length,max(9,round(half_length/2)))
        across=np.linspace(offsets[i]+gaps[i]*.22,offsets[i+1]-gaps[i]*.22,7)
        footprint=at+along[:,None,None]*axis+across[None,:,None]*normal
        if ((footprint[...,0]<1)|(footprint[...,0]>=w-1)|(footprint[...,1]<1)|(footprint[...,1]>=h-1)).any():
            continue
        if (np.mean(_sample(classes,footprint)==10)<.94 or _sample(occupied,footprint).any()
                or not _sample(access,footprint).all()):
            continue
        route=_open_aisle(at,axis,normal,triplet[i],triplet[i+1],classes,gray,occupied,access)
        if route is None:
            continue
        xy=np.round(point/scale,3).tolist()
        dx,dy=1/scale
        x,y=xy
        patch=[[x-dx,y-dy],[x+dx,y-dy],[x+dx,y+dy],[x-dx,y+dy]]
        evidence={"recipe":RECIPE,"coverage":POINT_COVERAGE,"aisle_direction":route[0],
                  "measured_distance_to_open_road_cells":route[1],"minimum_entrance_diameter_cells":2*radius+1,
                  "observed_vehicle_length_cells":round(vehicle_length,3),"verified_empty_length_cells":round(2*half_length,3),
                  "divider_segments_xy":[np.round(np.array([s[0],s[1]])/scale,3).tolist() for s in triplet],
                  "adjacent_space_index":i}
        found.append({"point":xy,"polygon":patch,"box":[x-dx,y-dy,x+dx,y+dy],"evidence":evidence})
    return found


def _measured(original,classes,scene,tap=None):
    if original.size!=(scene["frame"]["width"],scene["frame"]["height"]):
        raise ValueError("Parking measures require the current original dimensions")
    h,w=classes.shape
    if np.count_nonzero(classes==10)<.004*h*w or not np.any(classes==24):
        return []
    scale,occupied,cars,access,radius=_context(classes,scene,original.size)
    if access is None or not cars:
        return []
    gray=np.asarray(original.convert("L").resize((w,h),Image.Resampling.BOX))
    if tap is not None and parking_divider_choice._stripe(gray,classes,np.asarray(tap)*scale) is None:
        return []
    stripes=_stripes(gray,classes)
    results=[]
    for middle in stripes:
        a,b,axis=middle;normal=np.array([-axis[1],axis[0]])
        length=np.linalg.norm(b-a)
        if tap is not None:
            p=np.asarray(tap)*scale
            t=float((p-a)@axis)/length
            if not 0<=t<=1 or abs((p-a)@normal)>2.5:
                continue
            locations=[t]
        else:
            locations=[.5,.33,.67,.20,.80]
        for t in locations:
            at=a+t*(b-a)
            sides=[(_cross(o,at,normal),j) for j,o in enumerate(stripes) if o is not middle]
            lower=[(v,j) for v,j in sides if v is not None and v<-4]
            upper=[(v,j) for v,j in sides if v is not None and v>4]
            if not lower or not upper:
                continue
            tri=(stripes[max(lower)[1]],middle,stripes[min(upper)[1]])
            choices=_interiors_at(at,axis,tri,classes,gray,occupied,access,cars,scale,radius)
            if choices:
                if tap is not None:
                    return choices
                results.extend(choices)
                break
    # The same bay can be witnessed from either one of its two boundaries.
    keep={}
    for opt in results:
        evidence=opt["evidence"];i=evidence["adjacent_space_index"]
        endpoints=evidence["divider_segments_xy"][i:i+2]
        key=tuple(sorted(tuple(tuple(p) for p in seg) for seg in endpoints))
        keep.setdefault(key,opt)
    return sorted(keep.values(),key=lambda o:o["point"])


def candidates(original,classes,scene):
    return _measured(original,classes,scene)


def matched(original,classes,scene,mark):
    width,height=original.size
    x0,y0,x1,y1=mark
    if not (0<=x0<x1<=width and 0<=y0<y1<=height and x1-x0<.055*width and y1-y0<.055*height):
        return []
    p=[(x0+x1)/2,(y0+y1)/2]
    row,col=int(p[1]*classes.shape[0]/height),int(p[0]*classes.shape[1]/width)
    if int(classes[row,col]) not in (10,24):
        return []
    for obs in scene["tracks"]:
        if obs["frame_id"]==scene["frame"]["frame_id"]:
            a,b,c,d=obs["bbox_xyxy"]
            if max(x0,a)<min(x1,c) and max(y0,b)<min(y1,d):
                return []
    return _measured(original,classes,scene,p)


def marked_outcome(request,photo,surface,box):
    from io import BytesIO
    from . import manual_offlane,manual_surfaces,whole_bay
    from .parking_evidence import source_input
    clean,key,portable=source_input(request,photo)
    _,_,mark,center=manual_offlane.mark_input(clean,box)
    scene=clean["scene"]
    classes=manual_surfaces.decode(surface,scene["frame"],key)
    with Image.open(BytesIO(photo)) as original:
        choices=matched(original,classes,scene,mark)
    if not choices:
        return None
    one=len(choices)==1
    return whole_bay._make(clean,portable,choices,surface,method=whole_bay.MARK,require_choice=True,
        reason="single_visible_bay_at_painted_edge" if one else "passenger_mark_between_measured_interiors",
        question=("I can verify the marked point on one side of this painted divider. Is that the side you meant?"
                  if one else "Your tap is on the painted divider. Which of the two marked points did you mean?"),
        tap=center,selected_box=mark,measured_row_policy=RECIPE,
        measured_row_geometry=[o["evidence"] for o in choices],clearance_policy=whole_bay.CLEARANCE_POLICY)
