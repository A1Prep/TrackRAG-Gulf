"""Answer reviewer questions from bounded, privacy-processed evidence."""

import json
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from trackrag.contracts import Answer, Evidence, Question
from trackrag.privacy import anonymize_image
from trackrag.questions import validate_evidence
from trackrag.evidence_time import requested_frame_ids,referenced_track_ids


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
        available={image['frame_id'] for item in evidence if item['evidence_id'] in citations for image in item['images']}
        required=set(requested_frame_ids(question['question']))|set(requested_frame_ids(text))
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
        if (not stated and len(required)>1 and not required.issubset(scene_frames)
                and not any(required.issubset(frames|scene_frames) for frames in per_track.values())):
            raise ValueError('Unrelated identities cannot donate temporal comparison endpoints')
    return make_answer(status, text, list(citations))


def prepare_evidence(evidence: list[Evidence], run_dir: str | Path) -> list[dict]:
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
    with tracks_path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
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
        prepared = prepare_evidence(evidence, run_path)
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
