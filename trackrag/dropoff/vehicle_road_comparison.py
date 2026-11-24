"""Ask about a visible vehicle when its distance along a side of the roadway is unresolved."""
import re

from . import single_vehicle, vehicle_reference_qualifiers as qualifiers


POLICY = "road_relative_position_identity_only_v1"
_COMPARISON = re.compile(
    r"\b(?P<distance>(?:further|farther) (?:up|down|along|ahead))"
    r"(?: (?:the|this) (?:road|street))?"
    r"(?: (?:on|along))? (?:the |our |my )?(?P<side>left|right)"
    r"(?: hand)?(?: side)?(?: of (?:the|this) (?:road|street))?(?: please)?$"
)
_EXPLICIT_IMAGE = {"screen", "photo", "photograph", "picture", "frame", "image", "leftmost", "rightmost"}


def clarification(request, specs, answers, observations=()):
    """Use colour and body only for identity. The comparative itself never establishes a location."""
    from .vehicle_ground_reference import direct_reference

    words = qualifiers._text(request['text'])
    comparison = _COMPARISON.search(words)
    if not comparison or set(words.split()) & _EXPLICIT_IMAGE:
        return None
    body_request = {**request, 'text':words[:comparison.start()].strip()}
    reference = direct_reference(body_request)
    if (not reference or reference.get('qualified') or reference.get('unsupported')
            or reference.get('side') or reference.get('ground_side') or not reference.get('color')
            or reference['relation'] not in {'behind', 'beside'}):
        return None

    colour, kind = reference['color'], reference['kind']
    article = 'an' if colour[0] in 'aeiou' else 'a'
    requested_kind = 'SUV' if kind == 'suv' else kind
    matching, uncertain_rival = [], False
    for spec, reading in zip(specs, answers, strict=True):
        linked = [r for r in observations if r.get('track_id') == spec['track_id']
                  and r.get('box') == spec['box']]
        refined, conflicts = qualifiers._body_reading(reading, linked)
        paint, body = refined[:2]
        if conflicts:
            observations_for_box = [reading[:2], *(qualifiers._identity(row.get('appearance'))[:2] for row in linked)]
            if any(p in {None, 'unknown', colour} and
                   (k in {None, 'unknown'} or single_vehicle._compatible(kind, k))
                   for p, k in observations_for_box):
                uncertain_rival = True
            continue
        if paint not in {'unknown', colour} or body != 'unknown' and not single_vehicle._compatible(kind, body):
            continue
        if paint == 'unknown' or body == 'unknown':
            uncertain_rival = True
        else:
            matching.append((spec['box'], body))
    description = f"{comparison['distance']} the {comparison['side']}-hand side of the road"
    if len(matching) != 1 or uncertain_rival:
        if matching:
            boxes = [box for box, _ in matching]
            return boxes, 'vehicle_seen_space_withheld', (
                f"I can’t tell which vehicle you meant {description}. "
                "Could you name a detail on the one you mean?"
            )
        return [], 'vehicle_reference_unmatched', (
            f"I couldn’t verify {article} {colour} {requested_kind} {description}. "
            "Could you describe another detail or mark the area you mean?"
        )
    box, body = matching[0]
    middle = (box[0] + box[2]) / (2 * request['scene']['frame']['width'])
    position = 'left of centre' if middle < .46 else 'right of centre' if middle > .54 else 'in the centre'
    label = 'SUV' if body == 'suv' else body
    return [box], 'vehicle_side_uncertain', (
        f"I can see {article} {colour} {label} {position} in the photo. "
        f"I can’t verify that it is {description}. Is this the vehicle you mean? "
        f"I haven’t verified a drop-off point {reference['relation']} it."
    )
