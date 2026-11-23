"""Answer reviewer questions from bounded, privacy-processed evidence."""

import json
import re
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Protocol

from trackrag.contracts import Answer, Evidence, Question
from trackrag.privacy import anonymize_image
from trackrag.questions import validate_evidence
from trackrag.evidence_time import requested_frame_ids,referenced_track_ids
from trackrag.evidence_motion import verify_horizontal_claim


ANSWER_SCHEMA = {
    'type': 'object',
    'required': ['answer_status', 'answer', 'cited_evidence_ids'],
    'properties': {
        'answer_status': {'type': 'string', 'enum': ['answered', 'insufficient_evidence']},
        'answer': {'type': 'string'},
        'cited_evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
    },
    'additionalProperties': False,
}


class AnswerModel(Protocol):
    """An injected model receives public question fields and processed views."""

    def generate(self, question: Question, evidence: list[dict]) -> dict:
        ...


def make_answer(status, answer=None, cited_evidence_ids=None) -> Answer:
    return {'answer_status': status, 'answer': answer,
            'cited_evidence_ids': [] if cited_evidence_ids is None else cited_evidence_ids}


def unavailable_observation_source(question_text, evidence):
    """Independent annotation and detector IDs cannot substitute for each other."""
    available = {item['interval'].get('source') for item in evidence}
    labels = re.search(r'\b(?:annotated|annotations?|ground[ -]truth|manually (?:labelled|labeled)|human (?:labels?|track IDs?))\b', question_text, re.I)
    predicted = re.search(r'\b(?:predicted tracks?|detector(?:[’\']s)? (?:tracks?|IDs?|predictions?)|tracker(?:[’\']s)? IDs?)\b', question_text, re.I)
    if labels and available == {'predicted'}:
        return 'These records contain detector predictions. The human annotation and its separately assigned track IDs are unavailable.'
    if predicted and available == {'annotated'}:
        return 'These records contain human annotations. The detector predictions and their separately assigned track IDs are unavailable.'
    return None


def asks_for_unmeasured_speed(text):
    """The evidence contains image positions, without calibrated target-vehicle speeds."""
    if re.search(r'\b(?:pixels?|image[ -]plane|screen|speedometer|dashboard|overlay|telemetry|radar|'
                 r'limit|sign|printed|displayed|written|readout|reads?)\b', text, re.I):
        return False
    target = r'(?:cars?|trucks?|lorr(?:y|ies)|pickups?|vans?|buses|bus|suvs?|sedans?|motorcyclists?|cyclists?|pedestrians?)'
    return bool(re.search(rf"\b(?:exact|precise|actual)\s+(?:physical\s+)?speed\s+of\s+(?:[\w'-]+\s+){{0,4}}{target}\b",text,re.I))


def _spatial_vehicle_subject(words):
    from trackrag.retrieval_temporal import current_spatial_relation
    subject = re.search(r"\b(?:is|was)\s+(?:the|that)\s+(truck|lorry|pickup|car|sedan|suv|van|bus|coach|vehicle)"
                        r"\s+(?:near|next to|beside|opposite|behind|under|in front of|to the (?:left|right) of)\b", words, re.I)
    return subject[1].lower() if subject and current_spatial_relation(words) else None


def _which_vehicle(subject, boxes):
    locations = {}
    for track, (left, top, right, bottom), width in boxes:
        centre = (left + right) / (2 * width)
        position = "on the left" if centre < 1/3 else "on the right" if centre > 2/3 else "near the centre"
        locations[track] = (position, left, top, right, bottom)
    same_side = None
    for first, second in combinations(locations.values(), 2):
        where, left, top, right, bottom = first
        other, other_left, other_top, other_right, other_bottom = second
        if not (right < other_left or other_right < left or bottom < other_top or other_bottom < top):
            continue
        if where != other:
            return f"Which {subject} do you mean? There are possible matches {where} and {other} in the paused frame."
        same_side = where
    if same_side:
        return (f"Which {subject} do you mean? There are at least two possible matches {same_side} "
                "in the paused frame. What detail distinguishes yours?")
    return None


def spatial_reference_from_tracks(question, frame_tracks, width):
    """Use the selected frame's verified detector observations to identify an ambiguous reference locally."""
    from trackrag.retrieval_temporal import matches_vehicle_class

    subject = _spatial_vehicle_subject(question['question'])
    if subject is None:
        return None
    boxes = ((row['track_id'], row['bbox_xyxy'], width) for row in frame_tracks
             if row['source'] == 'predicted' and row['recording_id'] == question['recording_id']
             and row['frame_id'] == question['query_frame'] and matches_vehicle_class(subject, row['class_name']))
    return _which_vehicle(subject, boxes)


def spatial_reference_question(question, evidence):
    """Ask which unqualified vehicle the reviewer means when retrieval found distinct current matches."""
    from trackrag.retrieval_temporal import matches_vehicle_class

    subject = _spatial_vehicle_subject(question['question'])
    if subject is None:
        return None
    boxes = []
    for item in evidence:
        track = item['interval']['track_id']
        if track is None:
            continue
        for view in item['images']:
            if view['frame_id'] != question['query_frame'] or not view.get('bbox_xyxy'):
                continue
            if not any(row['track_id'] == track and matches_vehicle_class(subject, row['class_name'])
                       for row in view.get('observations', [])):
                continue
            boxes.append((track, view['bbox_xyxy'], view['privacy']['original_size_wh'][0]))
    return _which_vehicle(subject, boxes)


def _describes_observed_absence(text, track, frame, requested_tracks, required_frames, clock):
    """A cited full scene can support absence; it cannot supply a missing positive track identity."""
    from .playback_time import requested_playback_times
    frame_at = {m['requested_seconds']:m['frame_id'] for m in clock.get('moments', [])}
    for clause in re.split(r'(?<=[.!?])\s+|\b(?:but|whereas|while)\b|;', text, flags=re.I):
        if not re.search(r"\b(?:absent|not (?:seen|visible|present|observed|detected|tracked|in (?:the )?frame)|"
                         r"no (?:matching )?(?:observation|detection)|out of (?:view|frame))\b", clause, re.I):
            continue
        named = set(referenced_track_ids(clause))
        if (named and track not in named) or (not named and len(requested_tracks)>1):
            continue
        anchors = set(requested_frame_ids(clause))
        for relative, seconds in requested_playback_times(clause):
            instant = clock.get('paused_seconds',0) + seconds if relative else seconds
            if round(instant,6) in frame_at:
                anchors.add(frame_at[round(instant,6)])
        if frame in anchors or (not anchors and len(required_frames)==1):
            return True
    return False


def validate_answer(result, evidence, question=None) -> Answer:
    """Keep a supported abstention distinct from provider/validation failure."""
    if not isinstance(result, dict) or set(result) != set(ANSWER_SCHEMA['required']):
        raise ValueError('Model must return exactly status, answer and citations')
    status = result['answer_status']
    text = result['answer']
    citations = result['cited_evidence_ids']
    if status not in {'answered', 'insufficient_evidence'}:
        raise ValueError('Model returned an unsupported answer status')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Model answer requires a nonempty explanation')
    valid_ids = {item['evidence_id'] for item in evidence}
    if (not isinstance(citations, list)
            or any(not isinstance(value, str) or value not in valid_ids for value in citations)
            or len(set(citations)) != len(citations)):
        raise ValueError('Model cited unknown or duplicate evidence IDs')
    if status == 'answered' and not citations:
        raise ValueError('Answered results require supporting citations')
    if status == 'insufficient_evidence' and citations:
        raise ValueError('Abstentions cannot claim supporting citations')
    if status == 'answered' and question is not None:
        unsupported_source = unavailable_observation_source(question['question'], evidence)
        if unsupported_source:
            return make_answer('insufficient_evidence', unsupported_source)
        context = next((item['temporal_scope'] for item in evidence if 'temporal_scope' in item), None)
        if context is not None:
            unsupported = _temporal_support(text, citations, evidence, context)
            if unsupported:
                return make_answer('insufficient_evidence', unsupported)
        available={image['frame_id'] for item in evidence if item['evidence_id'] in citations for image in item['images']}
        required=set(requested_frame_ids(question['question']))|set(requested_frame_ids(text))
        clock = (context or {}).get('playback_scope', {})
        required.update(moment['frame_id'] for moment in clock.get('moments', []))
        if not required.issubset(available):
            raise ValueError('Answered frame claims require their exact cited image frames')
        per_track={}
        scene_frames=set()
        for item in evidence:
            if item['evidence_id'] not in citations:
                continue
            for image in item['images']:
                if item['interval']['track_id'] is None:
                    scene_frames.add(image['frame_id'])
                for row in image.get('observations',[]):
                    per_track.setdefault(row['track_id'],set()).add(image['frame_id'])
        stated=set(referenced_track_ids(text))|set(referenced_track_ids(question['question']))
        if not stated.issubset(per_track):
            raise ValueError('Assigned track IDs require cited source-labeled observations')
        if stated and any(not required.issubset(per_track[track]|scene_frames) for track in stated):
            raise ValueError('Requested temporal endpoints must support the same stated track')
        if clock and any(not _describes_observed_absence(text, track, frame, stated, required, clock)
                         for track in stated for frame in required - per_track[track]):
            return make_answer('insufficient_evidence', 'The cited observations do not establish that track identity at the requested playback time.')
        if (not stated and len(required)>1 and not required.issubset(scene_frames)
                and not any(required.issubset(frames|scene_frames) for frames in per_track.values())):
            raise ValueError('Unrelated identities cannot donate temporal comparison endpoints')
        verify_horizontal_claim(question['question'], text, [item for item in evidence if item['evidence_id'] in citations])
    return make_answer(status, text, list(citations))


def _temporal_context(run_path, question):
    if question is None or not (run_path / 'run.json').is_file():
        return None
    metadata = json.loads((run_path / 'run.json').read_bytes())
    configuration = metadata.get('configuration', {})
    if 'trackrag.retrieval_temporal' not in (configuration.get('retrieval'), configuration.get('retrieval_adapter')):
        return None
    from .retrieval_temporal import question_scope
    from .playback_time import requested_playback_times
    scope = question_scope(question['question'])
    clock = metadata.get('playback_scope', {}).get(question.get('question_id'))
    timed = bool(clock or requested_playback_times(question['question']))
    if scope is None and not timed:
        return None
    if scope is None:
        scope = question_scope(question['question'], current_if_untimed=True)
        scope['latest'] = False
    first = metadata['first_frame_id']
    required = set(scope['exact'])
    moments = clock.get('moments', []) if clock else []
    required.update(m['frame_id'] for m in moments)
    if scope['opening'] and not scope['boundary']:
        required.add(first)
    if (scope['latest'] or (scope['scene'] and not scope['opening'] and not timed)) and not scope['exact'] and not scope['boundary']:
        required.add(question['query_frame'])
    context = {'recording_opening_frame': first, 'latest_allowed_frame': question['query_frame'],
            'required_cited_frames': sorted(required), 'boundary': scope['boundary'], 'tracked_extents': {},
            'edge_directions': sorted(set(re.findall(r'\b(?:left|right|top|bottom)\b', question['question'].casefold())))}
    if timed:
        context['playback_scope'] = clock or {'status': 'unavailable', 'moments': []}
    if (scope['boundary'] and configuration.get('boundary_answer') == 'visual-candidate-v1'
            and re.search(r'\b(?:which|what (?:object|vehicle|car|person|type)|who)\b', question['question'], re.I)):
        context['identification_only'] = True
    return context


def _bracketed_track(item, span, context, scene_frames):
    entering = context['boundary'] == 'enter'
    outer_frame = span[0] if entering else span[1]
    if not any(f < outer_frame if entering else f > outer_frame for f in scene_frames):
        return False
    images = sorted(item['images'], key=lambda im: im['frame_id'], reverse=not entering)
    if len(images) < 2 or images[0]['frame_id'] != outer_frame:
        return False
    def margins(im):
        w,h = im['privacy']['original_size_wh']
        l,t,r,b = im['bbox_xyxy']
        return {'left':l/w, 'right':1-r/w, 'top':t/h, 'bottom':1-b/h}
    outside = margins(images[0])
    directions = context.get('edge_directions') or list(outside)
    return any(outside[d] <= 0.05 and margins(im)[d] > outside[d] + 0.02
               for d in directions for im in images[1:])


def _temporal_support(text, citations, evidence, context):
    from .playback_time import inconsistent_frame_time
    cited = [e for e in evidence if e['evidence_id'] in citations]
    frames = {im['frame_id'] for e in cited for im in e['images']}
    playback = context.get('playback_scope', {})
    if playback.get('status', 'available') != 'available':
        return 'The requested playback time cannot be verified from the available recording.'
    if inconsistent_frame_time(text, playback):
        return 'The stated frame does not match the requested time on the recording clock.'
    if not set(context['required_cited_frames']) <= frames:
        return 'The citations do not show the requested part of the recording.'
    if context['boundary'] is None:
        return None
    scene_frames = {i['frame_id'] for e in cited if e['interval']['track_id'] is None for i in e['images']}
    if context.get('identification_only'):
        # A visible candidate can be identified even if it was already present
        # in an earlier frame. Its attributes must come from the vision model.
        qualified = re.fullmatch(r'The visible candidate is ([^.!?\n]{3,260})\.\s*'
                                 r'The footage does not establish a boundary crossing\.', text.strip(), flags=re.I)
        if qualified is None or len(qualified[1].split()) < 3 or re.search(
                r'\b(?:enter(?:ed|s|ing)?|exit(?:ed|s|ing)?|cross(?:es|ed|ing)?|came|left the|depart(?:ed|s)?|'
                r'arriv(?:es|ed)|first appeared|last seen)\b', qualified[1], re.I):
            return 'The evidence does not establish the claimed crossing or a separately identified visible candidate.'
        qualifying = set()
        for item in cited:
            track = item['interval']['track_id']
            span = context['tracked_extents'].get(str(track))
            if span is None:
                continue
            views = {im['frame_id'] for im in item['images'] if im['bbox_xyxy'] is not None}
            entering = context['boundary'] == 'enter'
            endpoint = span[0] if entering else span[1]
            if len(views) >= 2 and endpoint in views and any(f < endpoint if entering else f > endpoint for f in scene_frames):
                qualifying.add(track)
        if len(qualifying) != 1 or set(referenced_track_ids(qualified[1])) - qualifying:
            return 'A visible candidate needs its own cited track images and a cited comparison scene.'
        return None
    supported = {}
    for item in cited:
        track = item['interval']['track_id']
        span = context['tracked_extents'].get(str(track))
        if span and _bracketed_track(item, span, context, scene_frames):
            supported[track] = span
    stated = set(referenced_track_ids(text))
    if not supported or stated - supported.keys():
        return 'The citations do not bracket this tracked object moving inward or outward at the requested image edge.'
    # Identifying an object is possible even when physical onset is unobserved.
    # An exact crossing time cannot be established from sampled boxes alone.
    for clause in re.split(r'[.;,]|\b(?:because|since|as it)\b', text, flags=re.I):
        if re.search(r'\b(?:cannot|could not|unclear|uncertain|unknown|not (?:shown|established|visible)|may have)\b', clause, re.I):
            continue
        numbers = requested_frame_ids(clause)
        has_time = bool(numbers) or re.search(r'\b(?:second[s]?|minute[s]?|timestamp|exactly when)\b|\d+:\d{2}', clause, re.I)
        crossing = re.search(r'\b(?:enter(?:ed|s|ing)?|exit(?:ed|s|ing)?|first (?:appears?|appeared|visible)|cross(?:ed|es)?|came into view)\b', clause, re.I)
        if has_time and crossing:
            return 'The views support observed positions. They do not establish the exact crossing frame.'
        observed = re.search(r'\b(first|last) (?:observed|detected|recorded|tracked|detection|observation)\b', clause, re.I)
        if observed and numbers:
            relevant = set(referenced_track_ids(clause)) or stated or set(supported)
            endpoint = 0 if observed.group(1).casefold() == 'first' else 1
            if len(relevant) != 1 or set(numbers) != {supported[next(iter(relevant))][endpoint]}:
                return 'The claimed detection time differs from the identified track history.'
    return None


def prepare_evidence(evidence: list[Evidence], run_dir: str | Path, question: Question | None = None) -> list[dict]:
    """Use the actual exported observations; never look up hidden gold labels."""
    run_path = Path(run_dir).resolve()
    recordings = {item['interval']['recording_id'] for item in evidence}
    if len(recordings) != 1:
        raise ValueError('Evidence must belong to one recording')
    wanted = {image['frame_id'] for item in evidence for image in item['images']}
    sources = {item['interval']['source'] for item in evidence}
    if len(sources) != 1:
        raise ValueError('Evidence must use one observation source')
    tracks_path = run_path / 'tracks.jsonl'
    if not tracks_path.is_file():
        raise FileNotFoundError('Privacy processing requires exported tracks.jsonl')
    by_frame = {frame: [] for frame in wanted}
    context = _temporal_context(run_path, question)
    selected_tracks = {item['interval']['track_id'] for item in evidence}
    with tracks_path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if (context and context['boundary'] and row['track_id'] in selected_tracks
                    and row['recording_id'] in recordings and row['source'] in sources
                    and row['frame_id'] <= question['query_frame']):
                frame_id = row['frame_id']
                span = context['tracked_extents'].setdefault(str(row['track_id']), [frame_id, frame_id])
                span[0], span[1] = min(span[0], frame_id), max(span[1], frame_id)
            if row['frame_id'] not in wanted:
                continue
            if row['recording_id'] not in recordings or row['source'] not in sources:
                raise ValueError('Privacy observations mix recordings or sources')
            by_frame[row['frame_id']].append({key: row[key] for key in
                ('recording_id', 'frame_id', 'track_id', 'source', 'class_name', 'bbox_xyxy')})
    prepared = []
    people = {'person', 'pedestrian', 'cyclist', 'motorbike'}
    # A per-question directory prevents one question overwriting another's inputs.
    # The caller supplies this private output directory below; direct preparation
    # callers receive a deterministic namespace based on the evidence inventory.
    key = sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    privacy_dir = run_path / 'privacy_processed' / key
    for position, item in enumerate(evidence, 1):
        images = []
        if not item['images']:
            raise ValueError('Evidence requires at least one image')
        for image in item['images']:
            relative = Path(image['image_path'])
            original_path = (run_path / relative).resolve()
            if relative.is_absolute() or not original_path.is_relative_to(run_path):
                raise ValueError('Evidence path escapes run directory')
            if not original_path.is_file():
                raise FileNotFoundError(f'Evidence image not found: {relative}')
            observations = by_frame[image['frame_id']]
            track = item['interval']['track_id']
            if track is not None and not any(row['track_id'] == track and
                    list(row['bbox_xyxy']) == list(image['bbox_xyxy']) for row in observations):
                raise ValueError('Selected image box does not match its exported observation')
            people_boxes = [row['bbox_xyxy'] for row in observations if row['class_name'].lower() in people]
            vehicle_boxes = [row['bbox_xyxy'] for row in observations if row['class_name'].lower() not in people]
            processed = anonymize_image(original_path,
                privacy_dir / f'evidence-{position}-frame-{image["frame_id"]}.png',
                pedestrian_boxes=people_boxes, vehicle_boxes=vehicle_boxes)
            images.append({**image, 'image_path': str(processed.pop('output_path')),
                           'observations': observations, 'privacy': processed})
        prepared.append({'evidence_id': item['evidence_id'], 'interval': dict(item['interval']), 'images': images})
    if context:
        prepared[0]['temporal_scope'] = context
    return prepared


def answer_question(question: Question, evidence: list[Evidence], run_dir: str | Path,
                    model: AnswerModel | None = None) -> Answer:
    """Task 5's three-argument adapter; optional injection keeps tests offline."""
    receipt = None
    receipt_path = None
    try:
        validate_evidence(question, evidence)
        if not evidence:
            return make_answer('insufficient_evidence')
        run_path = Path(run_dir).resolve()
        hosted = run_path / 'hosted' / sha256(question['question_id'].encode()).hexdigest()
        hosted.mkdir(parents=True, exist_ok=True)
        receipt_path = hosted / 'receipt.json'
        receipt = {'question_id': question['question_id'], 'recording_id': question['recording_id'],
                   'query_frame': question['query_frame'], 'role': 'task3 grounded answer',
                   'request_status': 'not_requested', 'cached': False, 'sent': []}
        unavailable = unavailable_observation_source(question['question'], evidence)
        if not unavailable and asks_for_unmeasured_speed(question['question']):
            receipt['reason'] = 'external_speed_not_measured'
            receipt['public_question'] = dict(question)
            return make_answer('insufficient_evidence', 'The footage and tracker give image positions. They do not measure that road user’s exact physical speed.')
        if unavailable:
            receipt['reason'] = 'requested_observation_source_unavailable'
            receipt['public_question'] = dict(question)
            return make_answer('insufficient_evidence', unavailable)
        prepared = prepare_evidence(evidence, run_path, question)
        requested_tracks = set(referenced_track_ids(question['question']))
        if requested_tracks:
            visible = {row['track_id'] for item in prepared for image in item['images']
                       for row in image.get('observations', ())}
            missing = sorted(requested_tracks - visible)
            if missing:
                receipt['reason'] = 'requested_track_missing_from_evidence'
                receipt['public_question'] = dict(question)
                ids = ', '.join(map(str, missing))
                return make_answer('insufficient_evidence', f'The retrieved observations contain no matching track ID for {ids}.')
        clarification = spatial_reference_question(question, prepared)
        if clarification:
            receipt['reason'] = 'ambiguous_current_vehicle'
            receipt['public_question'] = dict(question)
            return make_answer('insufficient_evidence', clarification)
        # Copy the exact bytes that will be sent into the auditor's hosted namespace.
        # The model receives those same files; neither the audit nor the upload
        # depends on absolute paths surviving Task 5's final directory rename.
        for position, item in enumerate(prepared, 1):
            for image in item['images']:
                payload = Path(image['image_path']).read_bytes()
                name = f'evidence-{position}-frame-{image["frame_id"]}.png'
                destination = hosted / name
                destination.write_bytes(payload)
                image['image_path'] = str(destination)
                receipt['sent'].append({'evidence_id': item['evidence_id'], 'frame_id': image['frame_id'],
                    'image': name, 'sha256': sha256(payload).hexdigest(), 'privacy': image.get('privacy')})
        if model is None:
            from trackrag.model_client import GeminiAnswerModel
            model = GeminiAnswerModel()
        name = getattr(model, 'model_name', None)
        receipt['model'] = name if isinstance(name,str) else type(model).__name__
        receipt['request_status'] = 'pending'
        result = model.generate(question=dict(question), evidence=prepared)
        receipt['request_status'] = 'completed'
        receipt['raw_response_repr'] = repr(result)[:4000]
        json.dumps(result,allow_nan=False)
        metadata = getattr(model, 'last_receipt', {})
        if isinstance(metadata,dict):
            receipt.update(metadata)
        receipt['request_status'] = 'completed'
        receipt['response'] = result
        answer = validate_answer(result, prepared, question)
        return answer
    except Exception as error:
        if receipt is not None:
            metadata = getattr(model, 'last_receipt', {}) if model is not None else {}
            if isinstance(metadata, dict):
                receipt.update(metadata)
            if receipt['request_status'] != 'completed':
                receipt['request_status'] = 'error'
            receipt['error_type'] = type(error).__name__
            receipt['delivery_error'] = str(error)[:1000]
        return make_answer('error', f'{type(error).__name__}: {error}')
    finally:
        if receipt_path is not None and receipt is not None:
            receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf-8')
