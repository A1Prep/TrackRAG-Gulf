"""Locate a rear point on a parking strip marked between two aligned vehicles."""
import numpy as np

from . import single_vehicle, traffic_arrow, vehicle_rear_boundaries, vehicle_rear_stall


RECIPE = "rear_parking_strip_crossbar_and_parked_neighbour_v1"


def _open_street_edge(view, rows, side, target_y):
    width, height = view.vehicle_size
    _, bottom = view.bumper
    h, w = view.classes.shape
    support = np.array([(r.y,r.lane) for r in rows])
    slope, intercept = np.polyfit(support[:,0]-bottom,support[:,1],1)
    if np.max(np.abs(support[:,1]-intercept-slope*(support[:,0]-bottom)))>max(4,.12*width):
        return False
    length = max(3,round(.17*width))
    opening = []
    for y in range(max(0,round(bottom-.1*height)),min(h,round(target_y+1.2*height)+1)):
        estimated = round(intercept+slope*(y-bottom))
        for x in range(max(length+1,estimated-round(.32*width)),min(w-length,estimated+round(.32*width)+1)):
            a = x-side*np.arange(1,length+1)
            b = x+side*np.arange(length)
            if (np.all(view.classes[y,a]==13) and np.all(np.isin(view.classes[y,b],(10,24)))
                    and np.any(view.classes[y,b]==10) and not view.occupied[y,np.r_[a,b]].any()
                    and view.road[y,np.r_[a,b]].all()):
                opening.append(y)
                break
    if not opening:
        return False
    runs = np.split(opening,np.where(np.diff(opening)>1)[0]+1)
    return any(len(r)>=max(8,.52*width) and r[0]<=target_y<=r[-1] for r in runs)


def _parking_crossbar(view, point, front, back):
    """Both sides of the traced paint must reach their corresponding photographed surfaces."""
    width, height = view.vehicle_size
    direction = np.sign(back[0] - front[0])
    span = abs(back[0] - front[0])
    if not .67 * width < span < 2.75 * width or abs(back[1]-front[1]) > .42 * span:
        return None
    along = (point[0]-front[0]) / (back[0]-front[0])
    paint_y = front[1] + along*(back[1]-front[1])
    if not .14 < along < .86 or abs(point[1]-paint_y) > .32 * height:
        return None
    cx, bottom = view.bumper
    if point[1]<=bottom or not bottom + .08*height < paint_y < bottom + .95*height:
        return None
    # Every photographed cross section from the bumper to this point must stay on its parking surface.
    # A fence one image pixel thick is still a real interruption.
    radius = max(1,round(.045*width))
    for y in range(int(np.ceil(bottom)),round(point[1])+1):
        t = (y-bottom)/(point[1]-bottom)
        x = round(cx + t*(point[0]-cx))
        band = np.s_[y,x-radius:x+radius+1]
        if (view.classes[band].size != 2*radius+1 or not np.isin(view.classes[band],(10,24)).all()
                or not view.road[band].all() or view.occupied[band].any()):
            return None
    # Trace the actual strip independently of the nominal colour of the paint.
    rows = []
    h, w = view.classes.shape
    for y in range(max(1, round(bottom+.03*height)), min(h-1, round(paint_y)+1)):
        fraction = (y-bottom)/(paint_y-bottom)
        x = round(cx + fraction*(point[0]-cx))
        if not 1 <= x < w-1 or view.classes[y,x] != 10 or view.occupied[y,x]:
            continue
        row = view.classes[y]
        left = x
        while left and row[left-1] in (10,24):
            left -= 1
        right = x
        while right < w-1 and row[right+1] in (10,24):
            right += 1
        if not .64*width < right-left < 3.0*width:
            continue
        reach = max(3,round(.12*width))
        if left < reach or right+reach >= w:
            continue
        lane, outer = ((row[left-reach:left],row[right+1:right+1+reach]) if direction>0
                       else (row[right+1:right+1+reach],row[left-reach:left]))
        if not np.all(lane==13) or np.mean(np.isin(outer,(2,15))) < .55:
            continue
        inner, kerb = (left,right) if direction>0 else (right,left)
        rows.append(vehicle_rear_boundaries._Section(y,inner,kerb))
    if (len(rows) < max(4,.20*height) or rows[-1].y-rows[0].y < .18*height
            or not _open_street_edge(view,rows,int(direction),round(point[1]))):
        return None
    if traffic_arrow.in_measured_strip(view.rgb,view.classes,rows,view.spec["box"],view.original.size,direction):
        return None
    return int(direction), len(rows)


def _neighbour(view, scene, point):
    """The point must fall in the same perspective row between two current vehicle footprints."""
    width, height = view.vehicle_size
    centre, bumper_y = view.bumper
    for track, _, original_box in single_vehicle.motor_rows(single_vehicle.context({"scene":scene})):
        if track == view.spec["track_id"]:
            continue
        left, top, right, bottom = np.asarray(original_box) * np.tile(view.scale,2)
        depth = bottom - bumper_y
        neighbour_width = right-left
        if not .65*height <= depth <= 5*height or not 1.16*width <= neighbour_width <= 6*width:
            continue
        t = (point[1]-bumper_y)/depth
        if not .08 < t < .76:
            continue
        projected = (1-t)*centre + t*(left+right)/2
        if abs(point[0]-projected) >= .28*((1-t)*width+t*neighbour_width):
            continue
        depth_y = int(np.ceil(bottom))
        lo, hi = round(left+.18*neighbour_width),round(right-.18*neighbour_width)
        if depth_y<view.classes.shape[0] and hi>lo and np.mean(view.classes[depth_y,lo:hi]==10)>=.55:
            return track
        # A foreground car can extend through the bottom image edge. In that case use
        # the visible parked flank next to this strip before the vehicle leaves the frame.
        if bottom<view.classes.shape[0]-2:
            continue
        edge = int(np.floor(left))-1 if point[0]<left else int(np.ceil(right))+1
        if not 2<=edge<view.classes.shape[1]-2:
            continue
        band = view.classes[round(top+.44*(bottom-top)):round(top+.92*(bottom-top)),edge-1:edge+2]
        if band.size and np.mean(np.all(band==10,axis=1)) >= .45:
            return track
    return None


def candidates(photo, classes, scene, spec, *, facing):
    ground = spec.get("ground")
    if not ground or ground["basis"] != "parking":
        return []
    view = vehicle_rear_boundaries._view(photo,classes,scene,spec,facing)
    if view is None:
        return []
    point = np.array(ground["point"]) * view.scale
    x,y = np.rint(point).astype(int)
    radius = max(2, round(.055*min(view.vehicle_size)))
    if (not vehicle_rear_boundaries._clear_patch(view,x,y,radius)
            or not np.all(classes[y-radius:y+radius+1,x-radius:x+radius+1] == 10)):
        return []
    companion = _neighbour(view,scene,point)
    if companion is None:
        return []
    lines = [(front,back,quality) for front,back in vehicle_rear_stall._borders(view,scene)
             if (quality := _parking_crossbar(view,point,front,back))]
    if len(lines) != 1:
        return []
    front, back, (side,count) = lines[0]
    a,b = np.round(np.array([point-radius,point+radius])/view.scale,3).tolist()
    return [{"point":list(ground["point"]), "box":[*a,*b],
             "polygon":[a,[b[0],a[1]],b,[a[0],b[1]]],
             "evidence":{"recipe":RECIPE,"coverage":"interior_point_only","anchor_id":spec["track_id"],
                         "aligned_neighbour_id":companion,"curb_side":"right" if side==1 else "left",
                         "measured_rows":count,"paint_ends_xy":np.round([front/view.scale,back/view.scale],3).tolist(),
                         "point_surface":10,"current_tracks_clear":True,"travel_arrow_checked":True,
                         "source_sha256":scene["provenance"]["image_sha256"]}}]
