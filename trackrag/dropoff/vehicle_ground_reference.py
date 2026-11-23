"""Resolve the pictured vehicle before deciding what the visible ground can support."""
from pathlib import Path
import re

import cv2
import numpy as np

from . import bay_intent, manual_offlane, manual_surfaces, single_vehicle, vehicle_curb_ground
from . import vehicle_ground_view as view, vehicle_reference_qualifiers as qualifiers
from .bay_packets import read_photo_packet
from .parking_evidence import _authenticate, source_input
from .passenger import digest, unresolved


BANK = Path(__file__).resolve().parents[2] / 'experiments/dropoff/passenger-vehicle-reference-views-v1.zip'
MODEL = 'Qwen/Qwen3.6-35B-A3B-FP8'
REVISION = 'c653bba46dfa06e9658dcc0b0dc87fe4685f14a2'
CHOICE_REASON = 'choose_observed_ground_behind_vehicle'
RELATION_POLICY = 'separate_vehicle_and_measured_destination_v2'
NO_READING = 'passenger_vehicle_reading_unavailable_v1'


def direct_reference(request, *, expanded=True):
    qualified = qualifiers.parse(request) if expanded else qualifiers.parse(request, expanded=False)
    if qualified is not None:
        return qualified
    text = request['text'].lower()
    ref = single_vehicle.request_reference(request)
    if not ref or ref['unsupported'] or not bay_intent.permitted_destination(text) or bay_intent.PARKING.search(text):
        return None
    prefix = text[:single_vehicle.RELATION.search(text).start()]
    if set(re.findall(r'[a-z]+', prefix)) - bay_intent.FILLER - {'offload'}:
        return None
    return ref


def _matching(specs, answers, ref, width):
    return [(spec, row) for spec, row in zip(specs, answers, strict=True)
            if row[1] != 'unknown' and single_vehicle._compatible(ref['kind'], row[1])
            and (not ref['color'] or ref['color'] == row[0])
            and single_vehicle._side(spec['box'], width, ref['side'])]


def _accepts_mark(spec, center, frame, classes):
    """Allow transverse paint tolerance inside the curb strip, never sideways into the traffic lane."""
    shape = np.asarray(spec['ground']['polygon'], dtype=np.float32)
    x, y = center
    ix = min(classes.shape[1]-1, int(x * classes.shape[1] / frame['width']))
    iy = min(classes.shape[0]-1, int(y * classes.shape[0] / frame['height']))
    if classes[iy, ix] not in (10, 13, 24):
        return False
    if cv2.pointPolygonTest(shape, (float(x), float(y)), False) >= 0:
        return True
    if spec['ground']['basis'] != 'curb_and_lane_divider':
        return False
    top, bottom = float(shape[:,1].min()), float(shape[:,1].max())
    tol = min(.006 * frame['height'], .045 * (spec['box'][3]-spec['box'][1]))
    if not top-tol <= y <= bottom+tol or top <= y <= bottom:
        return False
    # Keep the tap within both photographed side boundaries, at the closest measured depth.
    depth = min(max(y, top+.1), bottom-.1)
    return cv2.pointPolygonTest(shape, (float(x), depth), False) == 1


def _option(spec):
    g = spec['ground']
    xs, ys = zip(*g['polygon'])
    return {'point': g['point'], 'polygon': g['polygon'], 'box':[min(xs), min(ys), max(xs), max(ys)],
            'evidence': {'recipe': view.RECIPE, 'coverage':'interior_point_only'}}


def mark_agrees_with_bay(request, vehicle_mark, bay):
    """A separately measured parking bay may support the passenger's own mark on that bay."""
    if not bay or bay['result']['status'] != 'proposed':
        return False
    ref = direct_reference(request)
    if ref is None:
        return False
    source = vehicle_mark['provenance']
    if ref['relation'] == 'behind':
        boxes = source.get('visible_rear_boxes', [])
    else:
        boxes = vehicle_mark['result'].get('anchor_boxes_xyxy', [])
    if len(boxes) != 1:
        return False
    left, top, right, bottom = boxes[0]
    width, height = right-left, bottom-top
    def aligned(point):
        x, y = point
        if ref['relation'] == 'beside':
            return (left-width <= x < left or right < x <= right+width) and bottom-.35*height <= y <= bottom+.40*height
        frame = request['scene']['frame']
        # Follow the image projection away from the road's central vanishing region.
        depth = y-bottom
        if not 0 < depth < 1.30*height or bottom <= .52*frame['height']:
            return False
        mid = (left+right)/2
        projected = mid + (mid - .5*frame['width'])*depth / max(.18*frame['height'], bottom-.5*frame['height'])
        return abs(x-projected) < .75*width
    x0, y0, x1, y1 = source['selected_box']
    return aligned([(x0+x1)/2, (y0+y1)/2]) and aligned(bay['result']['destination_point_xy'])


def decision(request, ref, specs, answers, classes, mark=None, *, observations=(), expanded=True):
    """Return the anchor(s), independently observed ground, and a passenger-facing decision."""
    scene = request['scene']
    frame = scene['frame']
    size = (frame['width'], frame['height'])
    if ref.get('qualified'):
        matches, boxes, question = qualifiers.resolve(request, ref, specs, answers, observations=observations, expanded=expanded)
        if question:
            if expanded and mark is not None:
                cx = (mark[0]+mark[2])/2
                cy = (mark[1]+mark[3])/2
                checked, detail = manual_offlane.surface_check(frame, scene['tracks'], mark, [cx,cy], classes)
                cell = detail.get('center_class_id')
                if checked in {'observed_object','broad_area'}:
                    question = manual_offlane.QUESTIONS[checked]
                elif cell in {8,23}:
                    question = manual_offlane.QUESTIONS['pedestrian_crossing']
                elif cell in {2,15}:
                    question = 'Your mark falls on the curb or footpath. I can’t verify vehicle access there. Could you choose another visible area?'
                else:
                    question = 'I checked the spot you marked. I can’t verify a separate open parking area there. Could you choose another visible area?'
            return boxes, [], 'vehicle_seen_space_withheld' if boxes else 'vehicle_reference_unmatched', question
    else:
        matches = _matching(specs, answers, ref, size[0])
    boxes = [spec['box'] for spec, row in matches]
    no_ground = 'vehicle_seen_space_withheld'
    relation = ref['relation']
    if not matches:
        if ref['kind'] == 'car' and ref['color']:
            other_bodies = [(s, a) for s, a in zip(specs, answers, strict=True) if a[0] == ref['color']
                            and a[1] in {'van', 'pickup'} and single_vehicle._side(s['box'], size[0], ref['side'])]
            if other_bodies:
                labels = ' or '.join(dict.fromkeys(a[1] for _, a in other_bodies))
                return [s['box'] for s, _ in other_bodies], [], no_ground, f'I can see a {ref["color"]} {labels}. Is one of the outlined vehicles the one you mean?'
        return [], [], 'vehicle_reference_unmatched', 'I couldn’t identify a clearly visible vehicle with that description. Could you give its colour and side or mark the spot?'
    if len(matches) > 1:
        return boxes, [], no_ground, 'More than one outlined vehicle matches. Which one do you mean? You can tell me its side or mark the ground.'
    spec, row = matches[0]
    label = ' '.join(t for t in row[:2] if t != 'unknown').replace('suv', 'SUV')
    if mark:
        center = [(mark[0]+mark[2])/2, (mark[1]+mark[3])/2]
        checked, detail = manual_offlane.surface_check(frame, scene['tracks'], mark, center, classes)
        if checked in {'observed_object','broad_area'}:
            return boxes, [], no_ground, manual_offlane.QUESTIONS[checked]
        cell = detail.get('center_class_id')
        if cell in (8,23):
            return boxes, [], no_ground, manual_offlane.QUESTIONS['pedestrian_crossing']
        if cell in (2,15):
            return boxes, [], no_ground, 'Your mark is on the curb or footpath. Could you tap the vehicle-accessible ground you mean?'
    if relation == 'beside':
        return boxes, [], no_ground, f'I found the outlined {label}. Which side of it do you mean? Please mark the open ground beside it.'
    if row[2] != 'REAR' or not single_vehicle.visible_vehicle_end(spec['box'], size):
        return boxes, [], no_ground, f'I found the outlined {label}. Its rear is hard to place from this angle. Could you mark the ground you mean?'
    ground = spec['ground']
    if not ground or row[3] not in ('CURBSIDE','UNCLEAR') or (row[3]=='UNCLEAR' and ground['basis']!='curb_and_lane_divider'):
        if mark:
            left, top, right, bottom = spec['box']
            if left <= center[0] <= right and bottom <= center[1] <= bottom + .85*(bottom-top):
                return boxes, [], no_ground, f'I see your mark behind the {label}. The photo doesn’t establish a separate open pull-in there. Could you mark another spot?'
        return boxes, [], no_ground, f'I found the outlined {label}. I can’t verify open curbside ground behind it. Could you mark the area you mean?'
    if mark and not _accepts_mark(spec, center, frame, classes):
        return boxes, [], no_ground, 'Your mark falls outside the curbside area I can see behind the outlined vehicle. Could you tap inside that area or choose another spot?'
    question = (f'Your mark is at the curbside strip behind the {label}. Is the highlighted ground where you meant?' if mark else
                f'I found the {label} and the curbside strip behind it. Is the highlighted ground where you meant?')
    if row[3] == 'UNCLEAR':
        question = f'I can trace the curb and the divider behind the {label}, but the ground is hard to verify. Is the highlighted point what you meant?'
    return boxes, [_option(spec)], CHOICE_REASON, question+' The photo doesn’t show how much room a vehicle would have.'


def _explicit_ground(scene, classes, mark, center):
    from . import bay_clearance
    frame = scene['frame']
    reason, detail = manual_offlane.surface_check(frame, scene['tracks'], mark, center, classes)
    if reason != 'parking':
        return None
    road = bay_clearance.visible_road_component(classes)
    x, y = detail['center_cell_xy']
    if road is None or not road[y,x]:
        return None
    # The selection authorizes only the actual mark. Its extent says nothing about vehicle clearance.
    left, top, right, bottom = mark
    return {'point':center, 'box':mark, 'polygon':[[left,top],[right,top],[right,bottom],[left,bottom]],
            'evidence':{'recipe':RELATION_POLICY, 'coverage':'interior_point_only'}}


def _previously_observed_rear(request, photo, surface, earlier, specs, answers, boxes):
    """Reconcile two visual readings of the same point using an independently measured stall."""
    from . import bay_clearance, vehicle_relative_bays, whole_bay
    if len(boxes) != 1:
        return None
    matches = [(s, a) for s, a in zip(specs, answers, strict=True) if s['box'] == boxes[0]]
    if len(matches) != 1:
        return None
    spec, reading = matches[0]
    frame = request['scene']['frame']
    if (not spec['ground'] or reading[2] != 'REAR' or reading[3] not in {'UNCLEAR','LANE'}
            or not single_vehicle.visible_vehicle_end(spec['box'], (frame['width'], frame['height']))):
        return None
    replay = single_vehicle.outcome(request, photo, earlier)['result']
    point = replay.get('destination_point_xy')
    if (replay['status'] != 'proposed' or replay['anchor_boxes_xyxy'] != boxes
            or not np.allclose(point, spec['ground']['point'], rtol=0, atol=1)):
        return None
    clean, _, classes, stalls = whole_bay._layout(request, photo, surface)
    stalls = whole_bay._with_rows(clean, photo, surface, stalls, whole_bay.CLEARANCE_POLICY)
    road = bay_clearance.visible_road_component(classes)
    if road is None:
        return None
    supported = [o for o in stalls if vehicle_relative_bays.is_complete(o)
                 and cv2.pointPolygonTest(np.float32(o['polygon']), tuple(point), False) == 1
                 and bay_clearance.on_visible_road(o, road, frame)]
    if len(supported) != 1:
        return None
    option = _option(spec)
    return option, ('I found the vehicle and a marked bay behind it. The image checks disagree about the lane edge. '
                    'Is the highlighted point where you meant? The driver still needs to check the available room.')


def outcome(request, photo, surface, packet, mark=None, *, qualifier_packets=None, earlier_vehicle=None, expanded=True):
    from . import whole_bay
    ref = direct_reference(request, expanded=expanded)
    if ref is None:
        return None
    clean, photo_sha, portable = source_input(request, photo)
    if clean['scene'].get('provenance',{}).get('network_policy') == 'local_only':
        return None
    _authenticate(packet)
    body = packet['body']
    classes = manual_surfaces.decode(surface, clean['scene']['frame'], photo_sha)
    specs = vehicle_curb_ground.describe(clean['scene'], classes)
    if (body.get('source') != photo_sha or body.get('recipe') != view.RECIPE or body.get('model') != MODEL
            or body.get('revision') != REVISION or body.get('prompt') != view.checksum(view.GUIDE)
            or body.get('spec') != view.checksum(specs) or body.get('count') != len(specs)
            or not re.fullmatch('[a-f0-9]{64}',body.get('view','')) or not specs):
        raise ValueError('Vehicle readings differ from the original photograph and tracked boxes.')
    answers = body['answers']
    if len(answers) != len(specs) or any(len(row)!=4 or row[0] not in view.COLORS or row[1] not in view.TYPES
                                        or row[2] not in view.ENDS or row[3] not in view.GROUND for row in answers):
        raise ValueError('Invalid vehicle readings.')
    center = None
    evidence = {'clearance_policy':whole_bay.CLEARANCE_POLICY, 'vehicle_ground_policy':view.RECIPE,
                'vehicle_observation':packet}
    if expanded:
        evidence['vehicle_relation_policy'] = RELATION_POLICY
    observations = []
    if ref.get('qualified'):
        qualifier_packets = qualifier_packets if qualifier_packets is not None else qualifiers.packets_for(photo_sha)
        observations = qualifiers.read_observations(clean, photo, surface, packets=qualifier_packets)
        evidence['vehicle_qualifier_evidence'] = qualifier_packets
    if mark is not None:
        clean, _, mark, center = manual_offlane.mark_input(clean, mark)
        evidence['selected_box'] = mark
    boxes, options, reason, question = decision(clean, ref, specs, answers, classes, mark, observations=observations, expanded=expanded)
    if (expanded and not options and mark is None and earlier_vehicle is not None
            and ref['relation'] == 'behind' and not ref.get('qualified')):
        supported = _previously_observed_rear(clean, photo, surface, earlier_vehicle, specs, answers, boxes)
        if supported is not None:
            option, question = supported
            options, reason = [option], CHOICE_REASON
            evidence['earlier_vehicle_observation'] = earlier_vehicle
    if expanded and mark is not None and not options:
        from . import vehicle_relative_bays
        marked = _explicit_ground(clean['scene'], classes, mark, center)
        if marked is not None and not vehicle_relative_bays.restricted_instruction(clean['text']):
            options = [marked]
            reason = CHOICE_REASON
            question = ('I can see open parking pavement at your exact mark. I couldn’t verify its position '
                        'relative to the vehicle you described. Do you want to use the spot you marked?')
            evidence['explicit_passenger_ground'] = True
    if mark is not None and ref['relation'] == 'behind':
        frame = clean['scene']['frame']
        evidence['visible_rear_boxes'] = [s['box'] for s, a in zip(specs, answers, strict=True)
                                          if s['box'] in boxes and a[2] == 'REAR' and
                                          single_vehicle.visible_vehicle_end(s['box'], (frame['width'], frame['height']))]
    answer = whole_bay._make(clean, portable, options, surface, method=whole_bay.MARK if mark else whole_bay.AUTO,
                             reason=reason, question=question, tap=center, require_choice=True, **evidence)
    if boxes:
        answer['result']['anchor_boxes_xyxy'] = boxes
    return answer


def available(request, photo=None, surface=None, mark=None, *, earlier_vehicle=None):
    if not direct_reference(request):
        return None
    key = request['scene']['provenance'].get('image_sha256')
    packet = read_photo_packet(BANK, key, 8192)
    if packet is None:
        return None
    surface = surface or manual_surfaces.get_packet(key)
    if surface is None:
        return None
    if photo is None:
        photo = Path(request['scene']['frame']['image_path']).read_bytes()
    try:
        return outcome(request, photo, surface, packet, mark, earlier_vehicle=earlier_vehicle)
    except (KeyError, TypeError, ValueError, OSError):
        return None


def unavailable(request):
    if direct_reference(request) is None:
        raise ValueError('An explicit vehicle reference is required.')
    frame = request['scene']['frame']
    clean, sha, portable = source_input(request, Path(frame['image_path']).read_bytes())
    return {'ok':True, 'request_id':clean['request_id'],
            'result':unresolved('I don’t have a verified view of that vehicle here. Could you mark the ground you meant so I can check that spot?'),
            'provenance':{'method':NO_READING, 'reason':'vehicle_reading_unavailable', 'image':{'sha256':sha},
                          'request_sha256':digest(clean), 'inference_input_sha256':portable, 'independent_model_proposal':False}}


def verify_unavailable(request, result):
    if unavailable(request) != result:
        raise ValueError('The unavailable vehicle reading differs from the original instruction or photograph.')
    return {'input_verified':True, 'decision_reproduced':True, 'independent_model_proposal':False}
