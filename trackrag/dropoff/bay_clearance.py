"""Exclude visible scraps of parking pavement that belong to already occupied stalls."""
from . import surface_stalls


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
