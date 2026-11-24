"""Resolve the pictured vehicle before deciding what the visible ground can support."""
from pathlib import Path
import re

import cv2
import numpy as np

from . import bay_intent, manual_offlane, manual_surfaces, near_vehicle, single_vehicle, vehicle_curb_ground
from . import vehicle_ground_view as view, vehicle_reference_qualifiers as qualifiers
from .bay_packets import read_photo_packet
from .parking_evidence import _authenticate, source_input
from .passenger import digest, unresolved


BANK = Path(__file__).resolve().parents[2] / 'experiments/dropoff/passenger-vehicle-reference-views-v1.zip'
MODEL = 'Qwen/Qwen3.6-35B-A3B-FP8'
REVISION = 'c653bba46dfa06e9658dcc0b0dc87fe4685f14a2'
CHOICE_REASON = 'choose_observed_ground_behind_vehicle'
SIDE_CHOICE_REASON = 'choose_observed_ground_beside_vehicle'
EXPLICIT_GROUND_REASON = 'choose_passenger_marked_parking_ground'
CHOICE_REASONS = {CHOICE_REASON, SIDE_CHOICE_REASON, EXPLICIT_GROUND_REASON}
PREVIOUS_RELATION_POLICY = 'separate_vehicle_and_measured_destination_v2'
RELATION_POLICY = 'separate_rear_lateral_and_marked_ground_v3'
LOCAL_REAR_POLICY = 'local_car_family_and_curb_strip_v1'
NO_READING = 'passenger_vehicle_reading_unavailable_v1'
LIGHT_PAINT = re.compile(r"\b(?:beige|tan|champagne|ivory|cream|off[ -]white)\b",re.I)


def direct_reference(request, *, expanded=True):
    from . import vehicle_wording
    answer = vehicle_wording.reply(request) if expanded else None
    if answer and answer[0] == 'accepted':
        # All visual receipts still bind the actual answer and its original same-photo dialogue.
        request = {**request, 'text':answer[1]}
    nearby = near_vehicle.reference(request['text']) if expanded else None
    if nearby is not None:
        return nearby
    qualified = qualifiers.parse(request) if expanded else qualifiers.parse(request, expanded=False)
    if qualified is not None:
        return qualified
    text = request['text'].lower()
    ref = single_vehicle.request_reference(request)
    shades = list(LIGHT_PAINT.finditer(text)) if expanded else []
    if len(shades) == 1 and ref and ref['unsupported']:
        # The grammar knows the vehicle/side syntax. Keep the actual shade for subsequent visual checks.
        normal = LIGHT_PAINT.sub('white',text)
        if not set(re.findall(r'[a-z]+', LIGHT_PAINT.sub(' ',text))) & single_vehicle.COLORS:
            probe = single_vehicle.request_reference({**request,'text':normal})
            if probe and not probe['unsupported']:
                ref = {**probe,'color':shades[0][0]}
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


def _generic_car_reading(spec, reading, observations):
    """When both crops say car, keep that common meaning even if the subtype differs."""
    if spec['class'] != 'car' or reading[1] != 'unknown' or reading[2] != 'REAR':
        return reading
    from . import local_vehicle_observations
    crops = [local_vehicle_observations.parse(call['reply']) for call in observations['calls']]
    if all(crop and single_vehicle._compatible('car', crop[1]) for crop in crops):
        return [reading[0], 'car', *reading[2:]]
    return reading


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


def _measured_rear(request, ref, specs, answers, observations, photo, classes, local_matches=(), *,
                   marked_strip=False, tail_packet=None, local_rear=False):
    """Pair a single verified referent with visibly separated pavement behind its own bumper."""
    from . import vehicle_rear_apron, vehicle_rear_boundaries, vehicle_rear_ground, vehicle_rear_stall
    from . import vehicle_relative_bays, vehicle_scene_details

    if ref['relation'] != 'behind' or ref.get('ground_side') or vehicle_relative_bays.restricted_instruction(request['text']):
        return None
    frame = request['scene']['frame']
    required = set()
    if ref.get('qualified'):
        matches, _, question = qualifiers.resolve(request, ref, specs, answers, observations=observations)
        if question:
            matches = []
        if not matches and len(local_matches) == 1:
            spec, row, required = local_matches[0]
            matches = [(spec,row)]
    else:
        matches = _matching(specs, answers, ref, frame['width'])

    # Parking on the right-hand kerb can project left of the picture centre. This
    # alternative needs both a unique body and an independently measured kerb on that side.
    needs_kerb_side = not matches and ref.get('side')
    if needs_kerb_side:
        plausible = qualifiers.possible_side_reference(request, ref, specs, answers, observations=observations)
        if plausible:
            matches = [(spec, row) for spec, row in zip(specs, answers, strict=True) if spec['box'] in plausible[0]]
    if len(matches) != 1:
        return None
    spec, row = matches[0]
    options = []
    if row[2] == 'REAR' and single_vehicle.visible_vehicle_end(spec['box'], (frame['width'], frame['height'])):
        options = vehicle_rear_ground.candidates(photo, classes, request['scene'], spec, facing=row[2])
        if not options and row[3] == 'NONE':
            options = vehicle_rear_apron.candidates(photo, classes, request['scene'], spec, facing=row[2])
        if not options:
            options = vehicle_rear_boundaries.candidates(photo, classes, request['scene'], spec, facing=row[2])
        if not options:
            options = vehicle_rear_stall.candidates(photo, classes, request['scene'], spec, facing=row[2])
        if not options and marked_strip:
            from . import vehicle_rear_marked_strip
            options = vehicle_rear_marked_strip.candidates(photo, classes, request['scene'], spec, facing=row[2])
    if not options and tail_packet and row[2] in {'REAR','SIDE'}:
        from . import local_vehicle_tail, vehicle_profile_ground
        edges = local_vehicle_tail.verify(request['scene'], photo, tail_packet)
        options = vehicle_profile_ground.candidates(photo, classes, request['scene'], spec, facing=row[2],
                                                    tail=edges.get(spec['track_id']))
    if (not options and local_rear and row[2] == 'REAR' and row[3] == 'NONE'
            and single_vehicle.visible_vehicle_end(spec['box'], (frame['width'], frame['height']))
            and not near_vehicle.uncertain_matches(specs, answers, ref, frame['width'])):
        options = vehicle_rear_boundaries.marked_parking_point(photo, classes, request['scene'], spec, facing=row[2])
    if len(options) != 1:
        return None
    option = options[0]
    if needs_kerb_side and option['evidence'].get('curb_side') != ref['side']:
        return None
    if not vehicle_scene_details.ground_confirms(option['evidence'], required):
        return None
    # A signed negative on an older point still applies if that point is inside the proposed patch.
    if spec['ground'] and row[3] in {'LANE', 'BLOCKED'} and cv2.pointPolygonTest(
            np.float32(option['polygon']), tuple(spec['ground']['point']), False) >= 0:
        return None
    label = ' '.join(v for v in row[:2] if v != 'unknown').replace('suv', 'SUV')
    return spec, label, option


def _unread_shop(ref):
    """A passenger may explicitly choose a uniquely coloured car with measured ground while its shop sign is unread."""
    if (not ref.get('color') or ref.get('ground_side') or len(ref.get('qualifiers', ())) != 1):
        return None
    detail = ref['qualifiers'][0]
    if detail.get('field') != 'place' or detail.get('relation') not in {'by','beside','near','next to','outside'}:
        return None
    name = detail['value']
    if not re.fullmatch(r"(?:[a-z]+ ){0,3}(?:shop|store|restaurant|pharmacy|hotel|bank|clinic|caf[eé])", name):
        return None
    return name


def _possible_light_vehicle(ref, specs, packet, width):
    """Show a body candidate only when both crops support a light/neutral paint family and the requested body."""
    from . import local_vehicle_observations as local
    if not LIGHT_PAINT.fullmatch(ref.get('color') or '') or ref.get('qualified'):
        return None
    neutral = {'cream','white','gray','silver'}
    matches = []
    for spec, observation in zip(specs,packet['body']['rows'],strict=True):
        if not single_vehicle._side(spec['box'],width,ref['side']):
            continue
        readings = [local.parse(c['reply']) for c in observation['calls']]
        if all(r and r[0] in neutral and r[1]!='unknown' and single_vehicle._compatible(ref['kind'],r[1])
               for r in readings):
            matches.append(spec)
    if len(matches) != 1:
        return None
    return matches[0]['box']


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


def outcome(request, photo, surface, packet, mark=None, *, qualifier_packets=None, earlier_vehicle=None, expanded=True,
            observe_sides=True, side_packet=None, load_side_packet=True, side_release=None, measure_rear=False,
            archived_identity=None, load_archived_identity=False, extended_rear=False, tail_packet=None,
            explain_lane_mark=False, road_comparisons=False, nearby_parking=False, local_rear=False):
    from . import whole_bay
    ref = direct_reference(request, expanded=expanded)
    if ref is None:
        return None
    clean, photo_sha, portable = source_input(request, photo)
    frame = clean['scene']['frame']
    from . import local_vehicle_observations
    local_packet = local_vehicle_observations.is_local(packet)
    if not local_packet and clean['scene'].get('provenance',{}).get('network_policy') == 'local_only':
        return None
    classes = manual_surfaces.decode(surface, frame, photo_sha)
    specs = (local_vehicle_observations.specs_for(packet, clean['scene'], classes) if local_packet
             else vehicle_curb_ground.describe(clean['scene'], classes))
    if local_packet:
        answers = local_vehicle_observations.verify(clean['scene'], photo, specs, packet)
    else:
        _authenticate(packet)
        body = packet['body']
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
    local_rear = local_rear and local_packet and ref['relation'] == 'behind'
    if local_rear:
        evidence['local_rear_policy'] = LOCAL_REAR_POLICY
        if ref['kind'] == 'car':
            broader = [_generic_car_reading(spec, reading, crops)
                       for spec, reading, crops in zip(specs, answers, packet['body']['rows'], strict=True)]
            if not near_vehicle.uncertain_matches(specs, broader, ref, frame['width']):
                answers = broader
    if expanded:
        evidence['vehicle_relation_policy'] = RELATION_POLICY if observe_sides else PREVIOUS_RELATION_POLICY
    if ref['relation']=='near':
        if not nearby_parking or mark is not None:
            return None
        current = nearby_parking is True or nearby_parking == near_vehicle.POLICY
        width = frame['width']
        matches = _matching(specs,answers,ref,width)
        uncertain = near_vehicle.uncertain_matches(specs,answers,ref,width) if current else []
        reason = 'vehicle_reference_unmatched'
        qualification = ''
        if current and not matches and not uncertain:
            possible = near_vehicle.conditional_vehicle(specs,answers,ref,width)
            if possible:
                spec, reading, qualification = possible
                matches = [(spec,reading)]
        options = []
        if not matches:
            requested = ' '.join(w for w in (ref['color'],ref['kind']) if w)
            question = f'I couldn’t match a {requested} in this photo. Could you mark the spot or name another vehicle?'
        elif len(matches)>1 or uncertain:
            reason = 'vehicle_seen_space_withheld'
            if uncertain:
                matches = matches + uncertain
                question = ('I couldn’t verify the colour or type of every outlined vehicle. '
                            'Which one do you mean? You can tell me its side or mark the spot.')
            else:
                question = 'More than one outlined vehicle matches. Which one do you mean? You can tell me its side or mark the spot.'
        else:
            spec,reading = matches[0]
            label = ' '.join(w for w in reading[:2] if w!='unknown').replace('suv','SUV')
            options = near_vehicle.parking(clean,photo,surface,classes,spec)
            if current and not options:
                options, alternative = near_vehicle.optical_parking(clean,photo,surface,classes,spec)
                if alternative:
                    qualification = (qualification + ' The parking area connects to the vehicle’s row. '
                                     'I couldn’t verify that it is the immediately adjoining bay.').strip()
            reason = near_vehicle.REASON if options else 'vehicle_seen_space_withheld'
            question = (f'Which of the highlighted parking locations near the outlined {label} do you mean?'
                        if options else f'I found the outlined {label}. I couldn’t verify open parking next to it. Could you mark a different visible spot?')
            if qualification:
                question = (qualification+' Does a highlighted parking location by the outlined vehicle match what you meant?'
                            if options else f'{qualification} Is that the vehicle you meant? I haven’t verified an open place near it.')
                reason = 'unverified_bay_reference' if options else 'vehicle_side_uncertain'
        result = whole_bay._make(clean,portable,options,surface,method=whole_bay.AUTO,reason=reason,question=question,
                                near_vehicle_policy=near_vehicle.POLICY if current else near_vehicle.ORIGINAL_POLICY,
                                require_choice=bool(qualification),
                                **({'near_vehicle_detail':qualification} if qualification else {}),**evidence)
        if matches:
            result['result']['anchor_boxes_xyxy'] = [m[0]['box'] for m in matches]
        return result
    observations = []
    if ref.get('qualified'):
        qualifier_packets = qualifier_packets if qualifier_packets is not None else qualifiers.packets_for(photo_sha)
        observations = qualifiers.read_observations(clean, photo, surface, packets=qualifier_packets)
        evidence['vehicle_qualifier_evidence'] = qualifier_packets
    if road_comparisons and expanded and mark is None:
        from . import vehicle_road_comparison
        position = vehicle_road_comparison.clarification(clean, specs, answers, observations)
        if position is not None:
            boxes, reason, question = position
            answer = whole_bay._make(clean, portable, [], surface, method=whole_bay.AUTO,
                                     reason=reason, question=question,
                                     road_comparison_policy=vehicle_road_comparison.POLICY, **evidence)
            if boxes:
                answer['result']['anchor_boxes_xyxy'] = boxes
            return answer
    if mark is not None:
        clean, _, mark, center = manual_offlane.mark_input(clean, mark)
        evidence['selected_box'] = mark
    boxes, options, reason, question = decision(clean, ref, specs, answers, classes, mark, observations=observations, expanded=expanded)
    if local_rear and mark is None and reason == 'vehicle_reference_unmatched' and not boxes:
        question = ('I couldn’t match that vehicle in this photo. Could you add a nearby landmark '
                    'or mark your intended spot?')
    if (extended_rear and local_packet and mark is None and not boxes and not options
            and (possible := _possible_light_vehicle(ref,specs,packet,clean['scene']['frame']['width']))):
        from . import vehicle_rear_marked_strip
        boxes, reason = [possible], 'vehicle_side_uncertain'
        question = (f'The outlined {ref["kind"]} looks light-coloured. I can’t verify the {ref["color"]} paint in this light. '
                    'Is this the vehicle you mean?')
        evidence['extended_rear_policy'] = vehicle_rear_marked_strip.RECIPE
    if expanded and observe_sides and not boxes and not options and mark is None:
        plausible = qualifiers.possible_side_reference(clean, ref, specs, answers, observations=observations)
        if plausible is not None:
            boxes, question = plausible
            reason = 'vehicle_side_uncertain'
    if (not local_packet and expanded and observe_sides and not options and ref['relation'] == 'beside'
            and (side_packet is not None or load_side_packet)):
        from . import vehicle_side_ground
        if side_packet is None:
            side_packet = read_photo_packet(vehicle_side_ground.BANK, photo_sha, 160_000)
        lateral = (vehicle_side_ground.side_ground_choices(clean, photo, surface, ref, specs, answers,
                                                           packet=side_packet, observations=observations,
                                                           identity_evidence=packet, expected_release=side_release)
                   if side_packet is not None else None)
        if lateral is not None:
            lateral_options, inspected, prompt = lateral
            if mark is not None:
                grade, _ = manual_offlane.surface_check(clean['scene']['frame'], clean['scene']['tracks'], mark, center, classes)
                lateral_options = [o for o in lateral_options if grade == 'parking' and
                                   cv2.pointPolygonTest(np.float32(o['polygon']), tuple(center), False) >= 0]
            if lateral_options:
                selected_tracks = {o['evidence']['track_id'] for o in lateral_options}
                boxes = [s['box'] for s in specs if s['track_id'] in selected_tracks]
                if len(selected_tracks) != 1 or len(boxes) != 1:
                    raise ValueError('Side choices must refer to one identifiable vehicle in the current photo.')
                options = lateral_options
                reason, question = SIDE_CHOICE_REASON, prompt
                evidence['side_ground_observation'] = inspected
                evidence['side_ground_release'] = vehicle_side_ground.release_id()
    if (expanded and observe_sides and mark is not None and not options and ref['relation'] == 'beside'
            and not ref.get('qualified') and reason == 'vehicle_seen_space_withheld'):
        checked, detail = manual_offlane.surface_check(clean['scene']['frame'], clean['scene']['tracks'], mark, center, classes)
        if checked == 'through_lane':
            question = 'I found the vehicle. Your mark appears to be in the traffic lane beside it. Could you choose a visible pull-in?'
        elif checked not in {'observed_object','broad_area'} and detail.get('center_class_id') not in {2,8,15,23}:
            question = 'I checked your mark beside the outlined vehicle. I couldn’t verify open parking there. Could you choose another visible spot?'
    if (expanded and not options and mark is None and earlier_vehicle is not None
            and ref['relation'] == 'behind' and not ref.get('qualified')):
        supported = _previously_observed_rear(clean, photo, surface, earlier_vehicle, specs, answers, boxes)
        if supported is not None:
            option, question = supported
            options, reason = [option], CHOICE_REASON
            evidence['earlier_vehicle_observation'] = earlier_vehicle
    measured = None
    conditional_shop = None
    if measure_rear and mark is None and ref['relation'] == 'behind':
        from . import vehicle_scene_details, vehicle_relative_bays
        local_matches = vehicle_scene_details.matching(clean,ref,specs,answers,classes,original=photo) if local_packet else []
        measured = _measured_rear(clean, ref, specs, answers, observations, photo, classes,local_matches,
                                 marked_strip=extended_rear and local_packet, tail_packet=tail_packet,
                                 local_rear=local_rear)
        if extended_rear and local_packet and not measured and not options and len(boxes) == 1:
            shop = _unread_shop(ref)
            if shop:
                candidate = _measured_rear(clean, {**ref, 'qualified':False}, specs, answers, (), photo, classes,
                                           marked_strip=True, tail_packet=tail_packet, local_rear=local_rear)
                if candidate and candidate[0]['box'] == boxes[0]:
                    measured, conditional_shop = candidate, shop
        if measured and options:
            from . import vehicle_rear_ground
            # Two observations behind the same bumper can sample different depths in its parking strip.
            same_strip = any(cv2.pointPolygonTest(np.float32(o['polygon']), tuple(measured[2]['point']), False) >= 0
                             or cv2.pointPolygonTest(np.float32(measured[2]['polygon']), tuple(o['point']), False) >= 0
                             or vehicle_rear_ground.contains_measured_point(measured[2], o['point'], clean['scene']['frame'])
                             for o in options)
            if not same_strip:
                measured = None
        if measured:
            spec, label, option = measured
            boxes, options = [spec['box']], [option]
            reason = 'measured_ground_behind_vehicle'
            question = f'Is the highlighted ground behind the {label} where you want to get out?'
            evidence['automatic_rear_ground'] = {'track_id':spec['track_id'], 'vehicle':label, 'measurement':option['evidence']}
            from . import vehicle_profile_ground, vehicle_rear_marked_strip
            if option['evidence']['recipe'] == vehicle_profile_ground.RECIPE:
                evidence['tail_orientation_observation'] = tail_packet
            if (option['evidence']['recipe'] in {vehicle_rear_marked_strip.RECIPE,vehicle_profile_ground.RECIPE}
                    or conditional_shop):
                evidence['extended_rear_policy'] = vehicle_rear_marked_strip.RECIPE
            if conditional_shop:
                reason = CHOICE_REASON
                question = (f'I found a space behind the outlined {label}. I couldn’t verify the {conditional_shop} sign. '
                            'Is this the spot you meant?')
                evidence['unverified_place_name'] = conditional_shop
        elif len(local_matches) == 1 and not options and not vehicle_relative_bays.restricted_instruction(clean['text']):
            spec, row, ground_conditions = local_matches[0]
            if not ground_conditions and row[2]=='REAR':
                label = ' '.join(x for x in row[:2] if x!='unknown').replace('suv','SUV')
                boxes, reason = [spec['box']], 'vehicle_seen_space_withheld'
                question = (f'I found the {label}. I can’t distinguish a separate open spot behind it from this view. '
                            'Can you describe another nearby spot?')
                evidence['rear_reference_check'] = {'track_id':spec['track_id'], 'vehicle':label}
    if (not local_packet and expanded and observe_sides and mark is None and not boxes and not options
            and reason == 'vehicle_reference_unmatched' and (archived_identity is not None or load_archived_identity)):
        from . import archived_vehicle_reference
        if archived_vehicle_reference.eligible(ref):
            cached = archived_identity if archived_identity is not None else archived_vehicle_reference.get_packet(photo_sha)
            if cached is not None:
                try:
                    possible = archived_vehicle_reference.clarify(clean, photo, ref, specs, answers, cached)
                except (KeyError, TypeError, ValueError, OSError):
                    if archived_identity is not None:
                        raise
                    possible = None
                if possible is not None:
                    box, question = possible
                    boxes, reason = [box], 'vehicle_side_uncertain'
                    evidence['archived_vehicle_identity'] = cached
    if expanded and mark is not None and not options:
        from . import vehicle_relative_bays
        marked = _explicit_ground(clean['scene'], classes, mark, center)
        if marked is not None and not vehicle_relative_bays.restricted_instruction(clean['text']):
            options = [marked]
            reason = EXPLICIT_GROUND_REASON if observe_sides else CHOICE_REASON
            question = ('I can see open parking pavement at your exact mark. I couldn’t verify its position '
                        'relative to the vehicle you described. Do you want to use the spot you marked?')
            evidence['explicit_passenger_ground'] = True
    if mark is not None and ref['relation'] == 'behind':
        frame = clean['scene']['frame']
        evidence['visible_rear_boxes'] = [s['box'] for s, a in zip(specs, answers, strict=True)
                                          if s['box'] in boxes and a[2] == 'REAR' and
                                          single_vehicle.visible_vehicle_end(s['box'], (frame['width'], frame['height']))]
        if explain_lane_mark and not options:
            kind, checked = manual_offlane.surface_check(frame, clean['scene']['tracks'], mark, center, classes)
            if kind == 'through_lane':
                question = 'Your mark is on the roadway. I can’t verify a separate pull-in at that point. Could you choose another spot?'
                evidence['marked_lane_cell'] = checked['center_cell_xy']
    answer = whole_bay._make(clean, portable, options, surface, method=whole_bay.MARK if mark else whole_bay.AUTO,
                             reason=reason, question=question, tap=center, require_choice=measured is None or bool(conditional_shop), **evidence)
    if boxes:
        answer['result']['anchor_boxes_xyxy'] = boxes
    return answer


def available(request, photo=None, surface=None, mark=None, *, earlier_vehicle=None):
    if not direct_reference(request):
        return None
    key = request['scene']['provenance'].get('image_sha256')
    from . import local_surfaces, local_vehicle_observations, local_vehicle_tail
    packet = (local_vehicle_observations.get_packet(key) if local_surfaces.local_only(request['scene'])
              else read_photo_packet(BANK, key, 8192))
    if packet is None:
        return None
    surface = surface or manual_surfaces.get_packet(key)
    if surface is None:
        return None
    if photo is None:
        photo = Path(request['scene']['frame']['image_path']).read_bytes()
    try:
        return outcome(request, photo, surface, packet, mark, earlier_vehicle=earlier_vehicle, measure_rear=True,
                       load_archived_identity=True, extended_rear=True, tail_packet=local_vehicle_tail.get_packet(key),
                       explain_lane_mark=True, road_comparisons=True, nearby_parking=True, local_rear=True)
    except (KeyError, TypeError, ValueError, OSError):
        return None


def example_request(scene):
    """Offer an editable example about a vehicle actually observed on this photo."""
    from . import local_surfaces, local_vehicle_observations as local
    key = scene.get('provenance',{}).get('image_sha256')
    if scene.get('source') != 'predicted' or not key:
        return None
    if local_surfaces.local_only(scene):
        packet = local.get_packet(key)
        if packet is None or not local_surfaces.matches_scene(scene):
            return None
        body = packet['body']
        if body['task1']['current_tracks_sha256'] != local.digest(local.tracks(scene)):
            return None
        rows = [r['attributes'] for r in body['rows']]
    else:
        packet = read_photo_packet(BANK,key,8192)
        if not packet:
            return None
        try:
            _authenticate(packet)
            body = packet['body']
            if (body['source'], body['model'], body['revision']) != (key, MODEL, REVISION):
                return None
            rows = body['answers']
        except (KeyError,TypeError,ValueError):
            return None
    subjects = [tuple(row[:2]) for row in rows if len(row)==4]
    for colour,kind,end,_ in rows:
        if (end=='REAR' and colour in view.COLORS-{'unknown'} and kind in view.TYPES-{'unknown'}
                and subjects.count((colour,kind))==1):
            name = 'SUV' if kind=='suv' else kind
            return f'Could you drop me behind the {colour} {name}?'
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
