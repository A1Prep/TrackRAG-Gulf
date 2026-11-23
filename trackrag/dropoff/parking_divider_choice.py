"""Offer two independently checked interiors when the passenger taps their painted divider."""
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from . import bay_clearance, curb_points, manual_offlane, manual_surfaces
from .parking_evidence import source_input

POLICY = "observed_painted_divider_and_adjacent_points_v1"
PARKING, ROAD, PAINT = 10, 13, 24


def _pixels(array, xy):
    ij = np.rint(xy).astype(int)
    h, w = array.shape[:2]
    return array[np.clip(ij[..., 1], 0, h - 1), np.clip(ij[..., 0], 0, w - 1)]


def near_paint(classes, frame, center):
    """A cheap eligibility check before reading and replaying per-point observations."""
    h,w=classes.shape
    x,y=np.floor(np.asarray(center)*[w/frame["width"],h/frame["height"]]).astype(int)
    return (int(classes[y,x]) in (PARKING,PAINT)
            and np.any(classes[max(0,y-3):min(h,y+4),max(0,x-3):min(w,x+4)]==PAINT))


def _stripe(gray, classes, p):
    """Fit the photographed bright centreline, with signed paint and parking on both sides."""
    h, w = classes.shape
    span = max(15, round(.082 * w))
    x, y = np.rint(p).astype(int)
    left, top = max(0, x - span), max(0, y - span)
    crop = np.ascontiguousarray(gray[top:y + span + 1, left:x + span + 1])
    detected = cv2.createLineSegmentDetector().detect(crop)[0]
    if detected is None:
        return None
    lines = []
    for edge in detected[:, 0].reshape(-1, 2, 2):
        a, b = edge + [left, top]
        length = np.linalg.norm(b - a)
        if length < max(17, .033*w):
            continue
        axis = (b-a) / length
        if axis[0] < 0:
            axis = -axis
        normal = np.array([-axis[1], axis[0]])
        along = (p-a) @ (b-a) / length
        if not -.07*length <= along <= 1.07*length or abs((p-a) @ normal) > 5:
            continue
        # Restrict the evidence to a continuous section around the submitted point.
        ray = p + np.linspace(-min(.035*w, span/2), min(.035*w, span/2), 27)[:,None] * axis
        offsets = np.arange(-5, 6)
        trace = ray[:,None,:] + offsets[None,:,None]*normal
        values = _pixels(gray, trace).astype(float)
        median = np.median(values, axis=0)
        bright = int(np.argmax(median))
        if abs(offsets[bright]) > 3 or min(median[bright]-median[:2].mean(), median[bright]-median[-2:].mean()) < 8:
            continue
        shift = offsets[bright]
        centre = ray + shift*normal
        # The click itself must hit the visible paint or its subpixel edge in the signed raster.
        if abs(shift) > 2 and int(_pixels(classes, p)) != PAINT:
            continue
        near = _pixels(classes, centre[:,None,:] + np.arange(-1,2)[None,:,None]*normal)
        if np.any(near==PAINT,axis=1).mean()<.58:
            continue
        parking_fractions = []
        for side in (-1, 1):
            band = centre[:,None,:]+side*np.arange(6,14)[None,:,None]*normal
            ground = _pixels(classes, band)
            if np.any(~np.isin(ground,(PARKING,ROAD,PAINT))):
                break
            parking_fractions.append(float(np.mean(ground==PARKING)))
        if len(parking_fractions)!=2 or min(parking_fractions)<.57:
            continue
        contrast = float(median[bright] - max(median[:2].mean(), median[-2:].mean()))
        lines.append((contrast, length, p+shift*normal, axis, normal, centre[[0,-1]], parking_fractions))
    return max(lines, key=lambda r:(r[0],r[1])) if lines else None


def _adjacent(point, origin, axis, normal, classes, occupied, reachable):
    """A proposed interior must be in the very next space with no object or second stripe between."""
    h, w = classes.shape
    delta=point-origin
    side=float(delta@normal)
    # Oblique rows can be very foreshortened along their photographed dividers. The
    # approach must nevertheless cross the tapped line, rather than run along the row.
    if (not max(4,.010*h)<=abs(side)<=.12*h or abs(delta@axis)>4.5*abs(side)
            or np.linalg.norm(delta)>.18*w):
        return None
    xy=origin+np.linspace(.01,1.0,61)[:,None]*delta
    checked=_pixels(classes,xy)
    if (_pixels(occupied,xy).any() or not _pixels(reachable,xy).all()
            or not np.isin(checked,(PARKING,ROAD,PAINT)).all() or int(checked[-1])!=PARKING):
        return None
    after=abs(np.einsum("ij,j->i",xy-origin,normal))>=4.5
    inner=checked[after]
    if not inner.size or np.any(inner==PAINT) or float(np.mean(inner==PARKING))<.70:
        return None
    return -1 if side<0 else 1


def adjacent_points(original, classes, scene, mark, options):
    """Select two supplied source-verified points; returns no choice on a paint/road boundary."""
    frame=scene["frame"]
    width,height=original.size
    x0,y0,x1,y1=mark
    if (original.size!=(frame["width"],frame["height"]) or not options or
            not (0<=x0<x1<=width and 0<=y0<y1<=height) or
            (x1-x0)>width*.055 or (y1-y0)>height*.055):
        return None
    center=[(x0+x1)/2,(y0+y1)/2]
    reason,_=manual_offlane.surface_check(frame,scene["tracks"],mark,center,classes)
    if reason in {"broad_area","observed_object"}:
        return None
    h,w=classes.shape
    scale=np.array([w/width,h/height]); p=np.asarray(center)*scale
    if int(_pixels(classes,p)) not in (PARKING,PAINT):
        return None
    reachable=bay_clearance.visible_road_component(classes)
    if reachable is None or not _pixels(reachable,p):
        return None
    gray=np.asarray(original.convert("L").resize((w,h),Image.Resampling.BOX))
    stripe=_stripe(gray,classes,p)
    if stripe is None:
        return None
    contrast,_,origin,axis,normal,ends,fractions=stripe
    occupied=np.zeros((h,w),np.uint8)
    for track in scene["tracks"]:
        if track["frame_id"]==frame["frame_id"]:
            a,b,c,d=np.asarray(track["bbox_xyxy"])*np.tile(scale,2)
            occupied[max(0,int(b)):min(h,int(np.ceil(d))+1),max(0,int(a)):min(w,int(np.ceil(c))+1)]=1
    # A pixel path can leak through a fence, curb corner or pedestrian gate. Prove a
    # second connection to the driving aisle with clearance on every side of its path.
    radius=max(3,round(.012*w))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    pavement=np.isin(classes,(PARKING,ROAD,PAINT,36)) & ~occupied.astype(bool)
    widened=cv2.erode(pavement.astype(np.uint8),kernel).astype(bool)
    vehicle_access=bay_clearance.visible_road_component(np.where(widened,classes,15).astype(np.uint8))
    if vehicle_access is None:
        return None
    # A point immediately beside a kerb or gate post need not be the geometric centre
    # of a vehicle. Restore only the terminal clearance removed by the erosion.
    # Grow along open cells, so the reconstruction cannot jump across a wall.
    step=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    for _ in range(radius):
        vehicle_access=cv2.dilate(vehicle_access.astype(np.uint8),step).astype(bool) & pavement
    ranked={-1:[],1:[]}
    for option in options:
        point=np.asarray(option["point"])*scale
        if not _pixels(vehicle_access,point):
            continue
        side=_adjacent(point,origin,axis,normal,classes,occupied,reachable)
        if side is not None:
            ranked[side].append((float(np.linalg.norm(point-origin)),option))
    if not all(ranked.values()):
        return None
    choices=[]
    for side in (-1,1):
        option=min(ranked[side],key=lambda p:(p[0],p[1]["point"]))[1]
        x,y=option["point"]
        dx,dy=1/scale
        patch=[[x-dx,y-dy],[x+dx,y-dy],[x+dx,y+dy],[x-dx,y+dy]]
        samples=np.asarray([*patch,[x,y]])*scale
        if (not np.all(_pixels(classes,samples)==PARKING) or _pixels(occupied,samples).any()
                or not _pixels(reachable,samples).all()):
            return None
        choices.append({**option,"polygon":patch,"box":[x-dx,y-dy,x+dx,y+dy],
                        "evidence":{**option["evidence"],"coverage":"interior_point_only"}})
    measured={"centerline_original_xy":np.round(ends/scale,3).tolist(),"paint_contrast":round(contrast,3),
              "parking_fraction_per_side":[round(v,3) for v in fractions],
              "connected_opening_minimum_diameter_cells":2*radius+1}
    return choices,measured


def marked_outcome(request, photo, surface, mark, observations):
    from . import whole_bay
    clean,key,portable=source_input(request,photo)
    _,_,mark,center=manual_offlane.mark_input(clean,mark)
    scene=clean["scene"]
    classes=manual_surfaces.decode(surface,scene["frame"],key)
    options=curb_points.verified_options(scene,photo,surface,observations)
    with Image.open(BytesIO(photo)) as original:
        matched=adjacent_points(original,classes,scene,mark,options)
    if matched is None:
        return None
    choices,geometry=matched
    return whole_bay._make(clean,portable,choices,surface,method=whole_bay.MARK,
        reason="passenger_mark_between_checked_interiors",require_choice=True,
        question="Your mark touches a painted divider. Which of the two marked points on either side did you mean?",
        tap=center,selected_box=mark,divider_geometry=geometry,divider_point_observations=observations,
        divider_policy=POLICY,clearance_policy=whole_bay.CLEARANCE_POLICY)
