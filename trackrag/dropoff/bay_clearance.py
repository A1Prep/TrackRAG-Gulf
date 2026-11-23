"""Exclude visible scraps of parking pavement that belong to already occupied stalls."""
from . import surface_stalls


def visible_road_component(classes, *, rectangular_seed=False):
    """Identify open pavement continuous with a substantial region of road directly ahead."""
    import cv2
    import numpy as np

    h,w = classes.shape
    pavement = np.isin(classes,(10,13,24,36)).astype("uint8")
    pavement = cv2.morphologyEx(pavement,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    _, labels = cv2.connectedComponents(pavement,connectivity=8)
    if rectangular_seed:
        # Reproduce the fixed lower-centre rectangle used in deployed v3 reports.
        centre = np.s_[int(.65*h):int(.94*h),int(.30*w):int(.70*w)]
        seeds = labels[centre][classes[centre] == 13]
        counts = np.bincount(seeds[seeds>0])
        if not len(counts) or counts.max() < .04*labels[centre].size:
            return None
        return labels == counts.argmax()
    # Start on the nearest visible road in each central column. The bonnet can occupy
    # almost half a dashcam frame, so a fixed foreground rectangle can miss every road pixel.
    left,right = int(.25*w),int(.75*w)
    top,bottom = int(.44*h),int(.95*h)
    road = (classes[top:bottom,left:right] == 13) & (labels[top:bottom,left:right] != 0)
    columns = np.flatnonzero(road.any(axis=0))
    if columns.size < .12*(right-left) or road.sum() < .0015*w*h:
        return None
    nearest = bottom-1-np.argmax(road[::-1,columns],axis=0)
    counts = np.bincount(labels[nearest,left+columns])
    if counts.max() < .70*columns.size:
        return None
    return labels == counts.argmax()


def on_visible_road(option, road, frame):
    """Withhold an outline across an observed divider when the two pavement areas are separated."""
    import cv2
    import numpy as np

    if road is None:
        return True
    h,w = road.shape
    points = np.asarray(option["polygon"])*[w/frame["width"],h/frame["height"]]
    pixels = np.zeros((h,w),np.uint8)
    cv2.fillPoly(pixels,[np.rint(points).astype("int32")],1)
    coverage = road[pixels.astype(bool)]
    return coverage.size > 0 and coverage.mean() >= .60


def vehicle_sized(option, scene):
    """Use cars at comparable image depth as a reference only for learned surface fragments."""
    if option["evidence"]["recipe"] != surface_stalls.RECIPE:
        return True
    frame = scene["frame"]
    width, height = frame["width"], frame["height"]
    left, top, right, bottom = option["box"]
    cx, cy = option["point"]
    area_width, area_height = right-left, bottom-top
    polygon = option["polygon"]
    area = abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(polygon,polygon[1:]+polygon[:1]))) / 2

    for track in scene["tracks"]:
        if track["frame_id"] != frame["frame_id"] or track["class_name"].casefold() not in {
                "car", "sedan", "suv", "van", "minivan", "pickup", "taxi", "light_vehicle"}:
            continue
        l,t,r,b = track["bbox_xyxy"]
        car_width, car_height = r-l,b-t
        if car_width < .035*width or car_height < .035*height:
            continue
        # The tyre contact must actually be beside the patch. A much nearer foreground car
        # or a car parked in another row is not a scale reference for this piece of ground.
        if abs(cy-b) > .42*car_height or max(l-cx, cx-r, 0) > .70*car_width:
            continue
        very_small = (area_width < .55*car_width and area_height < .37*car_height
                      and area < .15*car_width*car_height)
        overlap = max(0, min(right,r)-max(left,l))
        bumper_strip = (overlap >= .65*area_width and abs(top-b) < .22*car_height
                        and area_height < .30*car_height and area < .21*car_width*car_height)
        if very_small or bumper_strip:
            return False
    return True
