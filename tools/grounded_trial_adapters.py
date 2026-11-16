"""Explicit annotated-development adapter with visual selection and fresh answers.

The selector uses observed track galleries. The answerer receives only exported
query-bounded evidence, class/box facts and privacy-processed views. No gold key
is read, and no trained detector or local reranker is supplied here.
"""
import base64
import io
import json
import os
import re
import shutil
import tempfile
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from PIL import Image, ImageDraw
from tools.emt_trial_adapters import (
    _hosted_image, build_index as _build_index, load_annotated_tracks,
    load_questions, open_recording,
)
from tools.temporal_trial_adapters import (
    history_facts, selection_context as _selection_context, selector_views, _attention_image,
)
from tools.trial_gemini import request_json


def build_index(recording, tracks):
    rows=list(tracks)
    if any(row['source'] != 'annotated' for row in rows):
        raise ValueError('This development gallery adapter requires annotated tracks')
    index=_build_index(recording, rows)
    index['selection_traces']={}
    return index


def candidates_at_query(question, index):
    recording=index['recording']
    if question['recording_id']!=recording.recording_id:
        raise ValueError('Question and index belong to different recordings')
    first=next(recording.iter_frames())['frame_id']
    candidates=[]
    for track,observations in sorted(index['tracks'].items()):
        rows=[row for row in observations if row['frame_id']<=question['query_frame']]
        if rows:
            history=history_facts(rows,first)
            for label,row in (('first',rows[0]),('last',rows[-1])):
                frame=recording.get_frame(row['frame_id'])
                left,top,right,bottom=row['bbox_xyxy']
                history[label+'_image_area_fraction']=(right-left)*(bottom-top)/(frame['width']*frame['height'])
                history[label+'_image_center_xy_fraction']=[(left+right)/2/frame['width'],(top+bottom)/2/frame['height']]
            candidates.append({'candidate_id':f'T{track}','track_id':track,'start_frame':rows[0]['frame_id'],
                               'end_frame':rows[-1]['frame_id'],'history':history,'window':history})
    return candidates


def selection_context(candidates):
    context=_selection_context(candidates)
    for track in context['tracks']:
        history=track['history']
        history['first_box_area_pixels2']=history['first_size_wh_pixels'][0]*history['first_size_wh_pixels'][1]
        history['last_box_area_pixels2']=history['last_size_wh_pixels'][0]*history['last_size_wh_pixels'][1]
        track['candidate_id']=track.pop('windows')[0]['candidate_id']
    return context


def identity_views(question, index):
    """Show each observed identity at its largest available, query-bounded view."""
    recording = index["recording"]
    representatives = []
    by_frame = {}
    for track_id, observations in sorted(index["tracks"].items()):
        rows = [row for row in observations if row["frame_id"] <= question["query_frame"]]
        if not rows:
            continue
        for row in rows:
            by_frame.setdefault(row["frame_id"], []).append(row)
        representative = max(rows, key=lambda row: (
            (row["bbox_xyxy"][2] - row["bbox_xyxy"][0]) * (row["bbox_xyxy"][3] - row["bbox_xyxy"][1]),
            row["frame_id"],
        ))
        representatives.append((track_id, representative))
    representatives.sort(key=lambda item: (
        -(item[1]['bbox_xyxy'][2]-item[1]['bbox_xyxy'][0])*(item[1]['bbox_xyxy'][3]-item[1]['bbox_xyxy'][1]),item[0]))
    parts, sent, processed = [], [], {}
    directory = Path(os.environ.get("TRACKRAG_TRIAL_CACHE", "runs/temporal-request-cache")) / "selector-images"
    directory.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(representatives), 16):
        entries = representatives[offset:offset + 16]
        sheet = Image.new("RGB", (896, 176 * ((len(entries) + 3) // 4)), "white")
        draw = ImageDraw.Draw(sheet)
        identities = []
        for i, (track_id, row) in enumerate(entries):
            frame_id = row["frame_id"]
            if frame_id not in processed:
                frame = recording.get_frame(frame_id)
                payload = _hosted_image(frame["image_path"], by_frame[frame_id], max_size=None)
                with Image.open(io.BytesIO(payload)) as image:
                    processed[frame_id] = image.convert("RGB")
            image = processed[frame_id]
            left, top, right, bottom = row["bbox_xyxy"]
            pad = max(3, (bottom - top) * 0.08)
            crop = image.crop((max(0, int(left - pad)), max(0, int(top - pad)),
                               min(image.width, int(right + pad)), min(image.height, int(bottom + pad))))
            crop.thumbnail((208, 124))
            x, y = (i % 4) * 224, (i // 4) * 176
            sheet.paste(crop, (x + 8, y + 29))
            draw.text((x + 6, y + 3), f"T{track_id} @ frame {frame_id}", fill="black")
            draw.text((x + 6, y + 15), f"Label: {row['class_name']}", fill="black")
            fraction=(right-left)*(bottom-top)/(image.width*image.height)
            draw.text((x+6,y+158),f'Original box: {fraction:.1%} of scene',fill='black')
            identities.append({"track_id": track_id, "frame_id": frame_id,'box_area_fraction':fraction})
        output = io.BytesIO()
        sheet.save(output, format="JPEG", quality=92)
        payload = output.getvalue()
        name = sha256(payload).hexdigest() + ".jpg"
        (directory / name).write_bytes(payload)
        parts.extend([
            {"text": "Identity reference gallery. Each tile is a privacy-masked crop at the stated observed frame, not necessarily the query frame. Annotation categories are broader than everyday body-shape descriptions: an SUV can be labeled Car. Preserve annotation labels and distinguish visible subtypes from the pixels. " + json.dumps(identities)},
            {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(payload).decode("ascii")}},
        ])
        sent.append({"image": name, "identities": identities})
    return parts, sent


def _select_evidence(question, index, receipt_path, k=3):
    candidates = candidates_at_query(question, index)
    # Track history is shared by its windows. Send it once, rather than repeating
    # the same history for every overlapping candidate and spending the quota.
    prompt = (
        "Select up to two observed road-user identities for a driving-footage question. Do not answer it. "
        "Use meaning, object class and time constraints, including paraphrases. "
        "The boxes are left/top/right/bottom original-image pixels. Compare observed changes when relevant. "
        "Use the supplied initial/current scene images to identify appearance and distinguish objects sharing a class. "
        "The identity gallery provides a separate labeled crop for every observed track. Verify the requested body's shape against those crops before selecting a track; do not substitute a similarly coloured SUV for a pickup or an open truck for a closed truck. "
        "Annotation classes are broad: Car does not exclude a visually supported SUV subtype. Keep the annotation category separate from visible body shape. Gallery times are explicit; use scene snapshots for current visibility. "
        "Gallery crops are resized independently. Compare original-pixel box areas and dimensions at the requested time when assessing apparent size; crop magnification is not physical size. "
        "T labels map visible objects to track IDs. Other objects remain in the scene; missing labels do not imply visual absence. "
        "Candidate IDs are exactly the T labels in the gallery. Each identity occurs once and includes its full observable history. Choose identities first; do not choose a different vehicle merely because a partial window resembles a requested time. "
        "When similarly coloured or structured vehicles match, use class, body shape, original pixel scale and requested time together. A shared accessory alone does not establish identity. "
        "Use normalized original-image area and center at the first/last observation to interpret apparent size and position; a large gallery crop does not mean that object filled the original scene. "
        "Prefer recall when a description could match similar-looking objects: include both best plausible identities for the answerer to disambiguate with their actual histories, rather than guessing only one. "
        "The caller always supplies the current full scene. For signs, lighting or current visible presence/absence, return an empty selected list if no particular tracked identity is needed. "
        "For temporal questions select the same identity whose history supports the comparison. "
        "Also extract requested_frame_ids: exact observation frames requested by the question, with at most two comparison endpoints. "
        "For a comparison against the beginning, use the first available frame as its opening endpoint. "
        "For general history questions without specific endpoints, return []. Do not confuse track IDs with frame IDs. "
        "Never request a frame after the query. These frame IDs will refine tracked intervals when those observations exist. "
        "A first annotation is not proof of physical entry; gaps may be occlusion or missing labels. "
        "Candidate histories contain only observations available by query time. Never invent candidate IDs. "
        f"Query frame {question['query_frame']}. Question: {question['question']}\n"
        f"Candidates: {json.dumps(selection_context(candidates))}"
    )
    schema = {"type": "OBJECT", "required": ["selected", "requested_frame_ids"], "properties": {
        "selected": {"type": "ARRAY", "maxItems": min(2,k), "items": {"type": "STRING", "enum": [row["candidate_id"] for row in candidates]}},
        "requested_frame_ids": {"type": "ARRAY", "maxItems": 2, "items": {"type": "INTEGER"}},
    }}
    lookup = {row["candidate_id"]: row for row in candidates}
    def validate(response):
        chosen = response.get("selected")
        if (not isinstance(chosen, list) or len(chosen) > min(2,k)
                or any(not isinstance(value, str) or value not in lookup for value in chosen)
                or len(set(chosen)) != len(chosen)):
            raise ValueError("Selector returned invalid or duplicate evidence candidates")
        frames = response.get("requested_frame_ids")
        first = next(index["recording"].iter_frames())["frame_id"]
        if (not isinstance(frames, list) or len(frames) > 2
                or any(type(frame) is not int or not first <= frame <= question["query_frame"] for frame in frames)
                or len(set(frames)) != len(frames)):
            raise ValueError("Selector returned invalid requested frame IDs or future frames")
        for frame in frames:
            index["recording"].get_frame(frame)
    views, sent = selector_views(question, index)
    gallery, gallery_sent = identity_views(question, index)
    selected = request_json([{"text": prompt}, *views, *gallery], schema, "evidence selection", metadata={
        "question_id": question["question_id"], "recording_id": question["recording_id"],
        "query_frame": question["query_frame"], "selection_images": sent,
        "identity_images": gallery_sent,
    }, validate=validate, repair_once=True, receipt_path=receipt_path,
        model_override=os.environ.get('TRACKRAG_SELECTION_MODEL'))
    validate(selected)
    chosen = selected["selected"]
    ranked = []
    seen = set()
    requested = sorted(selected["requested_frame_ids"])
    for candidate_id in chosen:
        candidate = lookup[candidate_id]
        start, end = candidate["start_frame"], candidate["end_frame"]
        if candidate["track_id"] is not None and len(requested)>1:
            observed = {row["frame_id"] for row in index["tracks"][candidate["track_id"]] if row["frame_id"] <= question["query_frame"]}
            if all(frame in observed for frame in requested):
                start, end = requested[0], requested[-1]
        identity = (candidate["track_id"], start, end)
        if identity in seen:
            continue
        seen.add(identity)
        rank = len(ranked) + 1
        ranked.append({
            "recording_id": question["recording_id"], "source": "annotated",
            "track_id": candidate["track_id"], "start_frame": start,
            "end_frame": end, "rank": rank, "score": 1 / rank,
        })
    return ranked


def explicit_frames(text):
    """Extract numeric frame references, keeping track IDs and quantities apart."""
    values=[]
    pattern=r"\bframes?\s*(\d+(?:\s*(?:,\s*(?:and\s*)?|and|to|through|[-–/;])\s*(?:frames?\s*)?\d+)*)"
    for match in re.finditer(pattern,text,re.I):
        numbers=list(re.finditer(r'\d+',match.group(1)))
        for position,number in enumerate(numbers):
            after=text[match.start(1)+number.end():]
            if position and re.match(r'\s*(?:kilomet(?:er|re)s?|km/?h|mph|met(?:er|re)s?|seconds?|minutes?|hours?)\b',after,re.I):
                break
            values.append(int(number.group()))
    return sorted(set(values))


def rank_evidence(question,index,k=3):
    if type(k) is not int or not 0 <= k <= 3:
        raise ValueError('k must be an integer from0 to3')
    if not k:
        return []
    recording=index['recording']
    if question['recording_id'] != recording.recording_id:
        raise ValueError('Question and index belong to different recordings')
    recording.get_frame(question['query_frame'])
    references=list(dict.fromkeys(int(value) for value in re.findall(r'\btrack\s*(\d+)\b',question['question'],re.I)))
    if references:
        selected=[]
        for track in references:
            rows=[row for row in index['tracks'].get(track,[]) if row['frame_id']<=question['query_frame']]
            if rows:
                selected.append({'recording_id':recording.recording_id,'source':'annotated','track_id':track,
                                 'start_frame':rows[0]['frame_id'],'end_frame':rows[-1]['frame_id'],'rank':len(selected)+1,'score':1.0})
        trace={'question_id':question['question_id'],'role':'explicit track reference lookup','request_status':'not_requested',
               'selection_images':[],'identity_images':[],'requested_track_ids':references}
    else:
        if not candidates_at_query(question,index):
            selected=[]
            trace={'question_id':question['question_id'],'role':'empty identity catalog lookup','request_status':'not_requested',
                   'selection_images':[],'identity_images':[]}
        else:
            cache_root=Path(os.environ.get('TRACKRAG_TRIAL_CACHE','runs/temporal-request-cache'))
            cache_root.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='trackrag-selection-',dir=cache_root) as temporary:
                receipt=Path(temporary)/'receipt.json'
                selected=_select_evidence(question,index,receipt)
                trace=json.loads(receipt.read_text(encoding='utf-8'))
    # Merge selected windows of one identity. Different windows are not different
    # vehicles, and should not consume every evidence slot.
    grouped={}
    requested=explicit_frames(question['question'])
    for interval in selected:
        track=interval['track_id']
        if track is None:
            key=('scene',interval['start_frame'])
        else:
            key=('track',track)
            observed={row['frame_id'] for row in index['tracks'][track] if row['frame_id']<=question['query_frame']}
            if len(requested)>1 and all(frame in observed and frame<=question['query_frame'] for frame in requested):
                interval=dict(interval,start_frame=min(requested),end_frame=max(requested))
        if key in grouped:
            grouped[key]['start_frame']=min(grouped[key]['start_frame'],interval['start_frame'])
            grouped[key]['end_frame']=max(grouped[key]['end_frame'],interval['end_frame'])
        else:
            grouped[key]=dict(interval)
    current=('scene',question['query_frame'])
    others=[value for key,value in grouped.items() if key!=current]
    others.sort(key=lambda value:value['track_id'] is None)
    ranked=others[:k-1]+[{'recording_id':recording.recording_id,'source':'annotated','track_id':None,
                         'start_frame':question['query_frame'],'end_frame':question['query_frame'],'rank':1,'score':0.0}]
    for position,interval in enumerate(ranked,1):
        interval['rank']=position
    trace['returned_intervals']=deepcopy(ranked)
    index['selection_traces'][question['question_id']]=deepcopy(trace)
    return ranked


def export_retrieval(question,index,run_dir):
    """Bind the selected catalog inputs to this run before answer generation."""
    trace=deepcopy(index['selection_traces'][question['question_id']])
    hosted=Path(run_dir)/'hosted'/sha256(question['question_id'].encode()).hexdigest()
    directory=hosted/'selection-inputs'
    directory.mkdir(parents=True,exist_ok=True)
    source=Path(os.environ.get('TRACKRAG_TRIAL_CACHE','runs/temporal-request-cache'))/'selector-images'
    for field in ('selection_images','identity_images'):
        for view in trace.get(field,[]):
            name=view['image']
            if Path(name).name!=name:
                raise ValueError('Selector image name must be a local basename')
            shutil.copyfile(source/name,directory/name)
            view['image']='selection-inputs/'+name
    (hosted/'selection-receipt.json').write_text(json.dumps(trace,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def answer_question(question,evidence,run_dir):
    if not evidence:
        return {'answer_status':'insufficient_evidence','answer':'No evidence was selected.','cited_evidence_ids':[]}
    run_dir=Path(run_dir)
    rows=[row for row in load_questions(run_dir/'tracks.jsonl') if row['frame_id']<=question['query_frame']]
    metadata=json.loads((run_dir/'run.json').read_text(encoding='utf-8'))
    facts=[]
    inventory={}
    for item in evidence:
        interval=item['interval']
        inventory[item['evidence_id']]=[image['frame_id'] for image in item['images']]
        matched=[row for row in rows if row['track_id']==interval['track_id'] and interval['start_frame']<=row['frame_id']<=interval['end_frame']]
        history=history_facts(matched,metadata['first_frame_id']) if matched else None
        if history:
            dx,dy=history['center_displacement_xy_pixels']
            history['net_image_motion']={'horizontal':'left' if dx<0 else 'right' if dx>0 else 'unchanged',
                                         'vertical':'up' if dy<0 else 'down' if dy>0 else 'unchanged'}
            last_image=next((image for image in item['images'] if image['frame_id']==matched[-1]['frame_id']),None)
            history['last_box_touches_image_edge']=None
            if last_image:
                with Image.open(run_dir/last_image['image_path']) as image:
                    width,height=image.size
                history['last_box_touches_image_edge']={
                    'left':matched[-1]['bbox_xyxy'][0]<=1,'right':matched[-1]['bbox_xyxy'][2]>=width-1,
                    'top':matched[-1]['bbox_xyxy'][1]<=1,'bottom':matched[-1]['bbox_xyxy'][3]>=height-1}
        facts.append({'evidence_id':item['evidence_id'],'interval':interval,
                      'kind':'full scene' if interval['track_id'] is None else 'tracked identity',
                      'observed_facts':history})
    prompt=(
        'Answer the driving-footage question using only these supplied scene views and observed facts. '
        'Track classes and boxes are human annotations in this development run. They can establish annotation class and image-relative motion even when a small subject is privacy-masked; say when you rely on those facts. '
        'A full-scene evidence ID has no selected vehicle identity. A track evidence ID refers only to its highlighted/cropped identity. Cite the evidence that actually supports the requested object, not a nearby similar vehicle. '
        'The inventory lists the exact image frames supplied. Do not assert a requested endpoint or current presence/absence unless those times are supported by the cited inventory. '
        'A full-scene image is direct visual evidence at its labelled frame, even without object annotations or a highlighted track. It can support visible presence or absence, sign text and lighting. Do not require annotations to inspect visible scene content. '
        'A zero-score scene was added automatically; the score does not invalidate its pixels or indicate that its contents are unknown. Visible absence means the requested object is not visibly present, not that hidden objects cannot exist. '
        'Use box widths/heights and computed center displacement for image growth and direction. Never infer physical speed, lane changes, intentions or passenger counts from image motion. Image growth alone does not establish physical approach, distance change or acceleration. Dashcam telemetry describes the camera vehicle. '
        'First annotation is not first visibility or proof of entry. Missing annotations do not prove absence. Describe visible clipping or image-edge motion without inventing a physical maneuver. '
        'Describe visible colour/appearance and qualify uncertain paint colour under glare, masking or poor resolution. Avoid unsupported precision. '
        'Lower vehicle regions and labelled people were privacy-masked. Do not invent that a particular window or interior was masked unless the image establishes it. '
        'For hidden, masked, future or unmeasured facts return insufficient_evidence with a brief specific explanation and no citations. Explain what the views do not resolve without inventing a cause. Do not treat every visual question as unanswerable just because exact physical measurements are unavailable. '
        'Answer the asked property concisely; avoid unnecessary mechanisms, physical claims or precision. '
        'Return JSON answer_status (answered or insufficient_evidence), nonempty answer text, and cited_evidence_ids. '
        f"Query cutoff: {question['query_frame']}. Question: {question['question']}\n"
        f'Sent image inventory: {json.dumps(inventory)}\nObserved evidence facts: {json.dumps(facts)}'
    )
    parts=[{'text':prompt}]
    sent=[]
    hosted=run_dir/'hosted'/sha256(question['question_id'].encode()).hexdigest()
    hosted.mkdir(parents=True,exist_ok=True)
    for item in evidence:
        for image in item['images']:
            matching=[row for row in rows if row['frame_id']==image['frame_id']]
            payload=_attention_image(run_dir/image['image_path'],matching,image['bbox_xyxy'])
            name=f"{item['evidence_id']}-{image['frame_id']}.jpg"
            (hosted/name).write_bytes(payload)
            parts.extend([{'text':f"Evidence {item['evidence_id']}, track {item['interval']['track_id']}, frame {image['frame_id']}."},
                          {'inlineData':{'mimeType':'image/jpeg','data':base64.b64encode(payload).decode('ascii')}}])
            sent.append({'evidence_id':item['evidence_id'],'frame_id':image['frame_id'],'image':name})
    lookup={item['evidence_id']:item for item in evidence}
    def validate(response):
        status=response.get('answer_status')
        text=response.get('answer')
        citations=response.get('cited_evidence_ids')
        if status not in {'answered','insufficient_evidence'} or not isinstance(text,str) or not text.strip():
            raise ValueError('Answer requires a supported status and nonempty explanation')
        if not isinstance(citations,list) or any(not isinstance(value,str) or value not in lookup for value in citations) or len(set(citations))!=len(citations):
            raise ValueError('Citations must identify distinct supplied evidence')
        if status=='insufficient_evidence':
            if citations:
                raise ValueError('Insufficient evidence requires no supporting citations')
            return
        if not citations:
            raise ValueError('An answer requires a supporting citation')
        available={frame for eid in citations for frame in inventory[eid]}
        required=set(explicit_frames(question['question']))
        asserted=set(explicit_frames(text))
        if not required.issubset(available) or not asserted.issubset(available):
            raise ValueError('An answered temporal claim requires its requested/asserted exact frames in the cited image inventory')
        tracks={lookup[eid]['interval']['track_id'] for eid in citations}
        stated={int(value) for value in re.findall(r'\btrack\s*(\d+)\b',text,re.I)}
        if not stated.issubset(tracks):
            raise ValueError('An assigned track ID must belong to the cited tracked evidence')
        endpoints=required|asserted
        if endpoints:
            scene_frames={frame for eid in citations if lookup[eid]['interval']['track_id'] is None for frame in inventory[eid]}
            per_track={track:{frame for eid in citations if lookup[eid]['interval']['track_id']==track for frame in inventory[eid]}
                       for track in tracks if track is not None}
            if stated:
                if any(not endpoints.issubset(per_track[track]|scene_frames) for track in stated):
                    raise ValueError('Temporal endpoints must cover the same stated track identity')
            elif len(endpoints)>1 and not endpoints.issubset(scene_frames) and not any(endpoints.issubset(frames|scene_frames) for frames in per_track.values()):
                raise ValueError('Unrelated tracked identities cannot jointly support a temporal comparison')
    schema={'type':'OBJECT','required':['answer_status','answer','cited_evidence_ids'],'properties':{
        'answer_status':{'type':'STRING','enum':['answered','insufficient_evidence']},
        'answer':{'type':'STRING'},'cited_evidence_ids':{'type':'ARRAY','items':{'type':'STRING'}}}}
    receipt_path=hosted/'receipt.json'
    request_metadata={
        'question_id':question['question_id'],'recording_id':question['recording_id'],
        'query_frame':question['query_frame'],'sent':sent,'sent_frame_inventory':inventory,
    }
    response=request_json(parts,schema,'grounded answer',receipt_path,request_metadata,
                          validate=validate,repair_once=True,use_cache=False,max_output_tokens=2048)
    if response['answer_status']=='insufficient_evidence':
        # The classification can be correct while its explanation invents why
        # a detail is unavailable. Preserve the original and check that claim
        # against the same pixels/facts, without changing the classification.
        shutil.copyfile(receipt_path,hosted/'draft-receipt.json')
        review_parts=[*parts,{'text':
            'Check the factual support of this abstention explanation using the supplied images and facts. '
            'Keep answer_status insufficient_evidence and cited_evidence_ids empty. '
            'Write one concise sentence stating only which requested fact the supplied views do not establish. '
            'Remove invented causes such as a particular window or seat being masked, invisible or occluded. '
            'The processing masks labelled people/riders and lower vehicle regions; it does not label or mask semantic windows or seats. '
            'Do not add new facts or try to answer an unsupported question. '
            f'Original response: {json.dumps(response)}'}]
        review_schema=deepcopy(schema)
        review_schema['properties']['answer_status']['enum']=['insufficient_evidence']
        def validate_review(value):
            validate(value)
            if value['answer_status']!='insufficient_evidence':
                raise ValueError('This pass checks only the abstention explanation, not its classification')
        response=request_json(review_parts,review_schema,'abstention explanation review',receipt_path,
                              request_metadata,validate=validate_review,repair_once=True,use_cache=False,max_output_tokens=2048)
    return response
