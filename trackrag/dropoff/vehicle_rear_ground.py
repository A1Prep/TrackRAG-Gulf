"""Measure a point between a parked row's outer kerb and its visible road-side divider."""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

import numpy as np
from PIL import Image

from . import bay_clearance, single_vehicle


RECIPE = "rear_point_in_bounded_parked_row_v1"
GROUND = (10, 13)
KERB = (2, 15)


@dataclass(frozen=True)
class Section:
    y: int
    lane: int
    curb: int
    boundary: str
    optical: bool


def _runs(indices):
    for chunk in np.split(indices, np.where(np.diff(indices) != 1)[0] + 1):
        if len(chunk):
            yield int(chunk[0]), int(chunk[-1])


def contains_measured_point(option, point, frame):
    """Two points can occupy different depths in the same precisely observed strip."""
    evidence = option['evidence']
    if evidence.get('recipe') != RECIPE or not evidence.get('surface_map_size'):
        return False
    w, h = evidence['surface_map_size']
    map_x = point[0]*w/frame['width'] - .5
    map_y = round(point[1]*h/frame['height'] - .5)
    return any(y == map_y and min(lane,kerb)+1 <= map_x <= max(lane,kerb)-1
               for y,lane,kerb in evidence['bounds_map_y_lane_curb'])


def _white_paint(rgb, y, left, right, vehicle_width):
    """Require an optical light ridge on both sides of the model's marking."""
    margin = max(2, round(0.08 * vehicle_width))
    if left < margin + 1 or right + margin + 2 >= rgb.shape[1]:
        return False
    band = rgb[y, left:right + 1].astype(float)
    intensity = band.mean(axis=1)
    colour = band[intensity.argmax()]
    shoulders = (rgb[y,left-margin-1:left-1], rgb[y,right+2:right+margin+2])
    return (intensity.max() - max(float(np.median(side.mean(axis=1))) for side in shoulders) >= 7
            and max(colour[0], colour[1]) - colour[2] < 45)


def _appearance_change(row, edge, width):
    reach = max(4,round(.14*width))
    margin = max(2,round(.04*width))
    if edge-reach<0 or edge+reach>=len(row):
        return False
    road = np.median(row[edge-reach:edge-margin],axis=0)
    parking = np.median(row[edge+margin:edge+reach],axis=0)
    return np.linalg.norm(road-parking) >= 21


def _section(classes, rgb, access, free, y, axis_x, vehicle_width, direction):
    width = classes.shape[1]
    # Mirror the horizontal scan so increasing x always faces the outside kerb.
    step = slice(None, None, direction)
    surface, connected, clear = classes[y,step], access[y,step], free[y,step]
    if direction == -1:
        axis_x = width - 1 - axis_x
    center = int(round(axis_x))
    if not 2 <= center < width-2 or surface[center] not in GROUND:
        return None
    outside = np.flatnonzero(np.isin(surface[center+1:], KERB))
    if not len(outside):
        return None
    curb = int(center + 1 + outside[0])
    if not .12 * vehicle_width <= curb - center <= 1.7 * vehicle_width:
        return None
    exterior = surface[curb:min(width, curb+max(4, round(.23*vehicle_width)))]
    if np.mean(np.isin(exterior, (10,13,24))) > .2:
        return None
    # Pale or painted outer kerbstones can extend the marking label inward from the
    # vertical kerb. Exclude that band from the measured ground, keeping the physical
    # kerb as the independent outside boundary.
    ground_edge = curb
    while ground_edge > center and surface[ground_edge-1] == 24:
        ground_edge -= 1
    if curb-ground_edge > .34*vehicle_width:
        return None
    left = max(2, int(center - 1.7*vehicle_width))
    marked = np.flatnonzero(surface[left:center] == 24) + left
    for start, end in reversed(list(_runs(marked))):
        span = curb - end
        if not .64 * vehicle_width <= span <= 1.78 * vehicle_width:
            continue
        if end-start+1 > .27 * vehicle_width or center-end < .11 * vehicle_width:
            continue
        lane = surface[max(0, start-max(4, round(.35*vehicle_width))):start]
        interior = slice(end + 2, ground_edge - 1)
        if (not len(lane) or np.mean(lane == 13) < .88
                or not connected[interior].all() or not clear[interior].all()
                or not np.isin(surface[interior], GROUND).all()):
            continue
        lo, hi = (start, end) if direction == 1 else (width-1-end, width-1-start)
        white = _white_paint(rgb, y, lo, hi, vehicle_width)
        original = (end,curb) if direction == 1 else (width-1-end,width-1-curb)
        return Section(y, *original, "paint", white)
    # A strongly different parking apron can instead have an optical seam next to
    # the through lane. Every cell across the occupied-width strip must then be the
    # learned parking class; asphalt alone cannot use this route.
    if surface[center] == 10:
        start = center
        while start>left and surface[start-1] == 10:
            start -= 1
        before = surface[max(0,start-max(4,round(.35*vehicle_width))):start]
        inside = slice(start+1,ground_edge-1)
        if (.64*vehicle_width <= curb-start <= 1.78*vehicle_width and center-start >= .11*vehicle_width
                and len(before) and (before==13).mean() >= .90
                and connected[inside].all() and clear[inside].all() and (surface[inside]==10).all()):
            optical = _appearance_change(rgb[y,step],start,vehicle_width)
            original = (start,curb) if direction==1 else (width-1-start,width-1-curb)
            return Section(y, *original,"parking_material",optical)
    return None


def _row_neighbours(vehicles, vehicle, direction):
    """Use depth ordering and similar scales to distinguish a curb row from broadside neighbours."""
    track_id, _, (left, top, right, bottom) = vehicle
    cx, bw, bh = (left+right)/2, right-left, bottom-top
    for other_id, _, (x0,y0,x1,y1) in vehicles:
        width, height = x1-x0, y1-y0
        dy, dx = bottom-y1, cx-(x0+x1)/2
        if (other_id == track_id or not .14*bh <= dy <= 2.2*bh
                or not .22*bw <= width <= 1.05*bw or not .22*bh <= height <= 1.05*bh
                or not -.14*bw <= direction*dx <= 2.4*bw or abs(dx/dy) > 4.5):
            continue
        intersection = max(0, min(right,x1)-max(left,x0))*max(0,min(bottom,y1)-max(top,y0))
        if intersection > .55 * width * height:
            continue
        yield other_id, (x0+x1)/2, y1, dx/dy, (bw-width)/dy


def _qualified_run(sections, vehicle, neighbour):
    if not sections:
        return None
    _, _, (left, top, right, bottom) = vehicle
    _, partner_x, partner_y, _, _ = neighbour
    bw, bh = right-left, bottom-top
    rows = np.array([s.y for s in sections],float)
    if (rows[0] > bottom + .16*bh or rows[-1] < bottom + .38*bh
            or len(rows) < max(6,.27*bh) or np.mean([s.optical for s in sections]) < .52):
        return None
    edges = np.array([[s.lane,s.curb] for s in sections])
    lines = np.polyfit(rows-bottom, edges, 1)
    residual = np.abs(edges - ((rows-bottom)[:,None]*lines[0]+lines[1]))
    projected_width = np.abs(edges[:,1]-edges[:,0])[:,None]
    if np.quantile(residual / np.maximum(2.5,.085*projected_width), .90) > 1:
        return None
    for px,py in (((left+right)/2,bottom),(partner_x,partner_y)):
        lane, curb = (py-bottom)*lines[0]+lines[1]
        if abs(curb-lane) < 3 or not .05 <= (px-lane)/(curb-lane) <= .95:
            return None
    return sections


def candidates(original, classes, scene, anchor, *, facing):
    """Measure a point using a verified rear, an aligned vehicle, the kerb, and a visible parking boundary.

    The caller establishes vehicle identity and source integrity. The geometry independently
    checks the current-frame box, original dimensions, detector exclusions and connected pavement.
    """
    if facing != "REAR" or not isinstance(classes,np.ndarray) or classes.ndim != 2 or not isinstance(anchor,dict):
        return []
    frame = scene["frame"]
    size = (frame["width"],frame["height"])
    if isinstance(original, bytes):
        with Image.open(BytesIO(original)) as decoded:
            return candidates(decoded,classes,scene,anchor,facing=facing)
    if not isinstance(original,Image.Image) or original.size != size:
        return []
    ctx = single_vehicle.context({"scene":scene})
    original_vehicles = single_vehicle.motor_rows(ctx)
    if not any(tid == anchor.get("track_id") and kind == anchor.get("class") and list(box) == anchor.get("box")
               for tid,kind,box in original_vehicles):
        return []
    h,w = classes.shape
    scale = np.array([w/size[0],h/size[1]])
    vehicles = [(tid,kind,np.array(box)*np.tile(scale,2)) for tid,kind,box in original_vehicles]
    vehicle = next(v for v in vehicles if v[0]==anchor["track_id"])
    left,top,right,bottom = vehicle[2]
    bw,bh = right-left,bottom-top
    if bw < max(15,.035*w) or bh < max(10,.045*h) or not .56 < (right-left)/(bottom-top) < 2.05:
        return []
    access = bay_clearance.visible_road_component(classes)
    if access is None:
        return []
    free = np.ones_like(access)
    for _,_,box in ctx["tracks"]:
        a,b,c,d = np.asarray(box)*np.tile(scale,2)
        free[max(0,int(b)):min(h,int(np.ceil(d))),max(0,int(a)):min(w,int(np.ceil(c)))] = False
    rgb = np.asarray(original.convert("RGB").resize((w,h),Image.Resampling.BOX))
    found = []
    for direction in (-1, 1):
        for neighbour in _row_neighbours(vehicles, vehicle, direction):
            _,_,_,slope,growth = neighbour
            sections = []
            start = max(1,int(np.ceil(bottom+.045*bh)))
            for y in range(start,min(h-1,int(bottom+.91*bh)+1)):
                depth = y-bottom
                axis_x = (left+right)/2 + slope*depth
                width = bw + np.clip(growth*depth,-.05*bw,.7*bw)
                current = _section(classes,rgb,access,free,y,axis_x,width,direction)
                if current is None:
                    if sections and y-sections[-1].y > max(2,.06*bh):
                        break
                    continue
                if sections and (max(abs(current.lane-sections[-1].lane),abs(current.curb-sections[-1].curb))
                                 > max(5,.20*bw)):
                    break
                sections.append(current)
            sections = _qualified_run(sections,vehicle,neighbour)
            if sections:
                found.append((direction,neighbour[0],sections))
    if not found or len({direction for direction,_,_ in found}) != 1:
        return []
    direction,partner,sections = max(found,key=lambda result:len(result[2]))
    middle = min(sections,key=lambda s:abs(s.y - (bottom+.54*bh)))
    x,y = int(round(.5*(middle.lane+middle.curb))),middle.y
    radius = max(2, round(.07*min(bw,bh)))
    patch = np.s_[y-radius:y+radius+1,x-radius:x+radius+1]
    if (x-radius<0 or y-radius<0 or x+radius>=w or y+radius>=h or not free[patch].all()
            or not access[patch].all() or not np.isin(classes[patch], GROUND).all()):
        return []
    def pixel(x,y):
        return [round(float((x+.5)/scale[0]),3),round(float((y+.5)/scale[1]),3)]
    point = pixel(x,y)
    p0,p1 = pixel(x-radius,y-radius),pixel(x+radius,y+radius)
    box = [*p0,*p1]
    return [{"point":point,"box":box,"polygon":[p0,[p1[0],p0[1]],p1,[p0[0],p1[1]]],
             "evidence":{"recipe":RECIPE,"coverage":"interior_point_only","curb_side":"right" if direction==1 else "left",
                         "anchor_id":int(vehicle[0]),"aligned_neighbour_id":int(partner),"verified_facing":facing,
                         "section_count":len(sections),"surface_map_size":[w,h],
                         "bounds_map_y_lane_curb":[[s.y,s.lane,s.curb] for s in sections],
                         "divider_observations":{k:sum(s.boundary==k and bool(s.optical) for s in sections)
                                                 for k in ("paint","parking_material")},
                         "point_surface_class":int(classes[y,x]),"source_sha256":ctx["image_sha256"],
                         "registered_rgb_sha256":sha256(rgb.tobytes()).hexdigest(),
                         "map_sha256":sha256(np.ascontiguousarray(classes).tobytes()).hexdigest()}}]
