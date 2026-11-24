"""Measure the parking pavement immediately past a visibly sideways tail."""
import numpy as np

from . import single_vehicle, traffic_arrow, vehicle_rear_boundaries


RECIPE = "pictured_tail_end_and_separated_parking_apron_v1"


def _lane_boundary(view, x, y, direction):
    """Find actual road adjoining a parking patch, with an optical seam or a contrasting painted edge."""
    width, _ = view.vehicle_size
    limit = int(min(1.6*width, (view.classes.shape[1]-1-x) if direction>0 else x))
    if limit < 12:
        return None
    positions = x+direction*np.arange(limit+1)
    labels = view.classes[y,positions]
    end = np.flatnonzero(labels!=10)
    if not len(end) or end[0]<.12*width:
        return None
    a = b = int(end[0])
    while b<len(labels) and labels[b]==24:
        b += 1
    reach = max(4,round(.055*width))
    if (b-a>.13*width or b+reach>len(labels) or not np.all(labels[b:b+reach]==13)
            or not view.road[y,positions[:b+reach]].all() or view.occupied[y,positions[:b+reach]].any()):
        return None
    pixels = view.rgb[y,positions].astype(float)
    parking = np.median(pixels[a-reach:a-1],axis=0)
    asphalt = np.median(pixels[b+1:b+reach],axis=0)
    ridge = b>a and np.max(pixels[a:b].mean(axis=1)) >= max(asphalt.mean(),parking.mean())+8
    if not ridge and np.linalg.norm(parking-asphalt)<26:
        return None
    return int(positions[a]), bool(ridge)


def _wheels_on_parking(view, box):
    left, top, right, bottom = box
    h, w = view.classes.shape
    # Examine each end's wheel area down to its first identifiable surface. A kerb or
    # pavement step at the tyre cannot be crossed to count parking from a row farther down.
    lo = max(0,int(np.floor(bottom)))
    hi = min(h,int(np.ceil(bottom+.14*(bottom-top)))+1)
    if hi<=lo:
        return False
    for a,b in ((.16,.39),(.61,.84)):
        x0,x1 = max(0,round(left+a*(right-left))),min(w,round(left+b*(right-left))+1)
        if x1<=x0:
            return False
        labels = view.classes[lo:hi,x0:x1]
        surface = ~np.isin(labels,(54,55,59,61))
        first = labels[np.argmax(surface,axis=0),np.arange(x1-x0)]
        if np.mean(surface.any(axis=0) & (first==10)) < .55:
            return False
    return True


def candidates(photo, classes, scene, spec, *, facing, tail):
    if tail not in {"LEFT","RIGHT"}:
        return []
    view = vehicle_rear_boundaries._view(photo, classes, scene, spec, facing, profile=True)
    if view is None or not _wheels_on_parking(view, view.box):
        return []
    width,height = view.vehicle_size
    left,_,right,bottom = view.box
    direction = -1 if tail=="LEFT" else 1
    end = left if direction<0 else right
    h,w = classes.shape
    # Another vehicle on the same parking surface supplies independent visual evidence of its use.
    companions = []
    for track, _, box in single_vehicle.motor_rows(single_vehicle.context({"scene":scene})):
        if track==spec["track_id"]:
            continue
        b = np.asarray(box)*np.tile(view.scale,2)
        gap = direction*((b[0]+b[2])/2-end)
        if (.12*width < gap < 1.85*width and abs(b[3]-bottom) < .82*height
                and b[2]-b[0] > .25*width and _wheels_on_parking(view,b)):
            companions.append(track)
    if not companions:
        return []
    x = round(end+direction*.19*width)
    y = round(bottom+.09*height)
    radius = max(2,round(.065*height))
    if (not vehicle_rear_boundaries._clear_patch(view,x,y,radius) or
            not np.all(classes[y-radius:y+radius+1,x-radius:x+radius+1]==10)):
        return []
    # Require a connected corridor from the observed bumper's base to the candidate's neighbourhood.
    x0,x1 = sorted((x,round(end-direction*.05*width)))
    if x0 < 1 or x1 >= w or not y < h or not np.all(classes[y,x0:x1+1]==10) or view.occupied[y,x0:x1+1].any():
        return []
    for lane_side in (-1,1):
        samples = []
        for py in range(max(1,y-round(.46*height)), min(h-1,y+round(.55*height)+1)):
            if classes[py,x] != 10:
                continue
            hit = _lane_boundary(view,x,py,lane_side)
            if hit:
                samples.append((py,*hit))
        if len(samples)<max(8,.42*height) or not samples[0][0]<y<samples[-1][0]:
            continue
        # A narrow pedestrian break in an otherwise continuous divider cannot establish
        # a parking entrance. Require a continuous pictured opening around the target depth.
        runs = np.split(np.asarray(samples),np.where(np.diff([s[0] for s in samples])>1)[0]+1)
        band = next((r for r in runs if r[0,0]<=y<=r[-1,0] and len(r)>=max(8,.42*height)),None)
        if band is None:
            continue
        samples = [(int(py),int(edge),bool(painted)) for py,edge,painted in band]
        ys,edges = np.array([s[:2] for s in samples]).T
        slope,intercept = np.polyfit(ys-y,edges,1)
        if np.quantile(np.abs(edges-(slope*(ys-y)+intercept)),.9)>max(3,.035*width):
            continue
        sections = [vehicle_rear_boundaries._Section(py, edge, x-lane_side*.45*width) for py,edge,_ in samples]
        if traffic_arrow.in_measured_strip(view.rgb,classes,sections,spec["box"],view.original.size,-lane_side):
            continue
        centre,a,b = np.round((np.array([[x,y],[x-radius,y-radius],[x+radius,y+radius]])+.5)/view.scale,3).tolist()
        return [{"point":centre, "box":[*a,*b], "polygon":[a,[b[0],a[1]],b,[a[0],b[1]]],
                 "evidence":{"recipe":RECIPE,"coverage":"interior_point_only","tail_edge":tail,
                             "anchor_id":spec["track_id"],"nearby_parked_track_id":companions[0],
                             "parking_side":"right" if lane_side<0 else "left","point_surface":10,
                             "current_tracks_clear":True,"travel_arrow_checked":True,
                             "optical_boundary_rows":len(samples),"painted_boundary_rows":sum(s[2] for s in samples),
                             "source_sha256":scene["provenance"]["image_sha256"]}}]
    return []
