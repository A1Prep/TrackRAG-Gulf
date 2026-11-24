"""Resolve a passenger's answer to one verified, same-photo vehicle-side question."""
from copy import deepcopy
import re

from .contracts import validate_request
from .data import inference_input_sha256
from .passenger import digest, unresolved


METHOD = 'passenger_vehicle_identity_reply_v1'
VEHICLE = r'(?:one|vehicle|car|sedan|suv|truck|pickup|van|bus|motorcycle)'
YES = re.compile(rf"(?:yes|yeah|yep|correct)(?: (?:that(?:'s| is) (?:it|right)|"
                 rf"(?:that(?:'s| is) the (?:same )?|that |the )({VEHICLE})(?: i (?:meant|mean))?))?")
NO = re.compile(rf"(?:no|nope)(?: (?:(?:i (?:meant|mean) )?(?:a different|another|the other) {VEHICLE}|"
                rf"not (?:that|this) {VEHICLE}))?")
REPLACEMENT = re.compile(rf"(?:no|nope) (?:i (?:meant|mean) )?the "
                         rf"((?:white|black|red|blue|brown|green|silver|grey|gray|yellow|dark) {VEHICLE})")
RELATIONS = {'behind':'behind it', 'beside':'beside it', 'front':'in front of it', 'near':'near it'}


def reply_to(text, question):
    from .single_vehicle import _compatible
    if not isinstance(text, str) or '?' in text or '？' in text:
        return None
    words = ' '.join(re.sub(r'[,!.?]+', ' ', text.casefold().replace('’', "'")).split())
    if words in {'yes please', "that's the one", 'that is the one'}:
        return 'yes', None
    if words in {'no thanks', 'no thank you', "no that's not it", 'no that is not it'}:
        return 'no', None
    yes = YES.fullmatch(words)
    if yes:
        named = yes.group(1)
        pictured = re.search(rf'\b({VEHICLE}) (?:left of centre|right of centre|in the centre) in the photo\.',
                             question.casefold())
        if named not in {None, 'one', 'vehicle'} and (not pictured or not _compatible(named, pictured[1])):
            return 'no', named
        return 'yes', None
    if NO.fullmatch(words):
        return 'no', None
    replacement = REPLACEMENT.fullmatch(words)
    if replacement:
        return 'no', replacement[1]
    return None


def is_question(request, outcome):
    from .whole_bay import AUTO
    source = outcome.get('provenance') or {}
    result = outcome.get('result') or {}
    return (outcome.get('ok') is True and outcome.get('request_id') == request['request_id']
            and source.get('method') == AUTO and source.get('reason') == 'vehicle_side_uncertain'
            and not source.get('bay_choices') and not source.get('bay_outline')
            and result.get('status') == 'needs_clarification' and bool(result.get('clarification_question'))
            and result.get('destination_point_xy') is None and result.get('destination_box_xyxy') is None
            and len(result.get('anchor_boxes_xyxy') or []) == 1)


def answer(request, prior, observed):
    """The caller holds the prior result that was verified and displayed in this session."""
    from .vehicle_ground_reference import direct_reference
    if (not is_question(prior, observed) or request['request_id'] == prior['request_id']
            or request['scene'] != prior['scene']):
        return None
    reply = reply_to(request['text'], observed['result']['clarification_question'])
    if reply is None:
        return None
    polarity, replacement = reply
    frame = request['scene']['frame']
    location = {'recording_id':frame['recording_id'], 'frame_id':frame['frame_id']}
    previous_turns = [*prior['dialogue'], {'role':'user', 'text':prior['text'], **location},
                      {'role':'assistant', 'text':observed['result']['clarification_question'], **location}]
    if request['dialogue'] != previous_turns:
        return None
    ref = direct_reference(prior)
    where = RELATIONS.get(ref['relation']) if ref else None
    if not where:
        return None
    if polarity == 'yes':
        result = unresolved(f'You confirmed the outlined vehicle. Please mark the open ground {where} where you want to get out. I’ll check that spot on the photo.')
        result['anchor_boxes_xyxy'] = deepcopy(observed['result']['anchor_boxes_xyxy'])
    elif replacement:
        result = unresolved(f'Where is the {replacement} you mean? Describe its position in the photo, or mark the open ground {where} where you want to get out.')
    else:
        result = unresolved('Which vehicle did you mean? Describe its colour, body type or position in the photo, or mark the spot where you want to get out.')
    image = request['scene']['provenance']['image_sha256']
    return {'ok':True, 'request_id':request['request_id'], 'result':result, 'provenance':{
        'method':METHOD, 'image':{'sha256':image}, 'request_sha256':digest(request),
        'inference_input_sha256':inference_input_sha256(request, image), 'reply':polarity,
        'requested_relation':ref['relation'], 'vehicle_selected_by':'passenger' if polarity == 'yes' else None,
        'reference_request':{k:deepcopy(v) for k,v in prior.items() if k not in {'scene','dialogue'}},
        'reference_outcome':deepcopy(observed), 'independent_model_proposal':False}}


def verify(request, outcome, *, live=False):
    from . import whole_bay
    clean = validate_request(request)
    source = outcome['provenance']
    prior = validate_request({**source['reference_request'], 'scene':clean['scene'], 'dialogue':clean['dialogue'][:-2]})
    observed = source['reference_outcome']
    expected = answer(clean, prior, observed)
    if expected is None or expected != outcome:
        raise ValueError('The vehicle reply must match the preceding photo, outlined vehicle and passenger instruction.')
    check = whole_bay.verify_current if live else whole_bay.verify
    check(prior, observed)
    return {'language_calls':0, 'independent_model_proposal':False, 'vehicle_selected_by':source['vehicle_selected_by'],
            'reference_request_id':prior['request_id'], 'reply':source['reply']}


def verify_current(request, outcome):
    return verify(request, outcome, live=True)


def marked(request, box, confirmed, ground):
    """A confirmed vehicle identity authorizes no ground. An eligible exact tap is a separate choice."""
    from . import manual_offlane, manual_surfaces, vehicle_ground_reference, whole_bay
    verify_current(request, confirmed)
    info = confirmed['provenance']
    if info['reply'] != 'yes':
        raise ValueError('A mark cannot use a vehicle the passenger declined.')
    source = ground['provenance']
    if source['method'] == manual_offlane.METHOD:
        manual_offlane.verify(request, ground, requested_box=box)
    elif source['method'] == whole_bay.MARK and 'confirmed_vehicle_reply' not in source:
        whole_bay.verify_current(request, ground, requested_box=box)
    else:
        raise ValueError('This mark requires an independent check of its actual pavement.')
    clean, image, tap, center = manual_offlane.mark_input(request, box)
    surface = info['reference_outcome']['provenance']['surface_evidence']
    answer = ground['result']
    option = None
    if answer['status'] == 'proposed' or source.get('bay_choices'):
        classes = manual_surfaces.decode(surface, clean['scene']['frame'], image)
        option = vehicle_ground_reference._explicit_ground(clean['scene'], classes, tap, center)
    relation = {'behind':'behind', 'beside':'beside', 'front':'in front of', 'near':'near'}[info['requested_relation']]
    if option:
        question = (f'I can see parking pavement at your mark. I couldn’t verify that it is {relation} the outlined vehicle. '
                    'Would you like to use your marked spot anyway?')
    else:
        question = (answer.get('clarification_question') if not source.get('bay_choices') else None) or manual_offlane.QUESTIONS['unverified_surface']
    result = whole_bay._make(clean, info['inference_input_sha256'], [option] if option else [], surface,
                             method=whole_bay.MARK, reason=vehicle_ground_reference.EXPLICIT_GROUND_REASON if option else 'vehicle_seen_space_withheld',
                             question=question, tap=center, require_choice=True, selected_box=tap,
                             clearance_policy=whole_bay.CLEARANCE_POLICY, confirmed_vehicle_reply=confirmed, ground_check=ground)
    result['result']['anchor_boxes_xyxy'] = deepcopy(confirmed['result']['anchor_boxes_xyxy'])
    return result
