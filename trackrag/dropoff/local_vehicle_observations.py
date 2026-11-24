"""Replay neutral, on-device vehicle crops against their exact photo and current Task 1 tracks."""
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw

from . import bay_landmarks, local_surfaces, vehicle_ground_view
from .parking_evidence import signed_bytes
from .streamed import METADATA_SHA256, STREAMED_MODEL_ID, WEIGHT_SHA256


ORIGIN = 'local_vehicle_observations_v1'
GUIDE = ('Identify only the pictured target vehicle. Give its body PAINT, TYPE and the END visible to the camera. '
         'Paint: white, cream, black, silver, gray, red, blue, green, yellow, orange, brown, unknown. '
         'Type: sedan, suv, pickup, van, truck, bus, car, unknown. End: REAR, FRONT, SIDE, UNCLEAR. '
         'REAR requires visible rear body details such as tail lamps or a tailgate. A front grille or headlights mean FRONT. '
         'If a field is hidden, too small, or ambiguous, use unknown or UNCLEAR for that field. '
         'Return only: paint | type | end.')
VIEWS = {'crop':'Classify the single vehicle at the centre of this close crop.',
         'context':'Classify only the vehicle inside the magenta box. Ignore the surrounding vehicles.'}
MAX_TOKENS = 17
TASK1_ID = 'acf21b53525d5729541aaedbf961a9c634399fa881fc7976909d48575c5e870d'
TASK1_WEIGHTS = '9ea96f1050f1d3375c5cf0f25a086cd5bb9c6a290fa5fdc009804d9a31181d95'
_registered = {}


def digest(value):
    return sha256(signed_bytes(value)).hexdigest()


def tracks(scene):
    f = scene['frame']
    return [t for t in scene['tracks'] if t['recording_id']==f['recording_id'] and t['frame_id']==f['frame_id']]


def anchor(spec):
    return {k:spec[k] for k in ('track_id','class','box')}


def render(original, box, kind):
    if kind not in VIEWS:
        raise ValueError('Unknown vehicle crop.')
    x0,y0,x1,y1=box
    margin=.07 if kind=='crop' else .60
    left=max(0,math.floor(x0-margin*(x1-x0)))
    top=max(0,math.floor(y0-margin*(y1-y0)))
    right=min(original.width,math.ceil(x1+margin*(x1-x0)))
    bottom=min(original.height,math.ceil(y1+margin*(y1-y0)))
    crop=original.crop((left,top,right,bottom)).convert('RGB')
    try:
        crop.thumbnail((640,480),Image.Resampling.LANCZOS)
        if kind=='context':
            sx,sy=crop.width/(right-left),crop.height/(bottom-top)
            ImageDraw.Draw(crop).rectangle([(x0-left)*sx,(y0-top)*sy,(x1-left)*sx,(y1-top)*sy],outline='#ff22cc',width=2)
        target=BytesIO()
        crop.save(target,format='JPEG',quality=92,subsampling=0)
        return target.getvalue()
    finally:
        crop.close()


def parse(raw):
    if not isinstance(raw,str):
        return None
    fields=[s.strip().casefold() for s in re.split(r'\s*\|\s*|\s*,\s*',raw.strip().rstrip('.'))]
    if (len(fields)!=3 or fields[0] not in vehicle_ground_view.COLORS or fields[1] not in vehicle_ground_view.TYPES
            or fields[2].upper() not in vehicle_ground_view.ENDS):
        return None
    return fields[:2]+[fields[2].upper()]


def consensus(calls):
    a,b=(parse(c['reply']) for c in calls)
    empty=['unknown','unknown','UNCLEAR']
    if a is None or b is None:
        return empty+['NONE']
    return [v if v==w else missing for v,w,missing in zip(a,b,empty,strict=True)]+['NONE']


def register(path, expected_sha):
    raw=Path(path).read_bytes().replace(b'\r\n',b'\n')
    if sha256(raw).hexdigest()!=expected_sha.lower():
        raise ValueError('The local vehicle observation archive has changed.')
    package=json.loads(raw)
    if set(package)!={'origin','version','sources'} or package['origin']!=ORIGIN or package['version']!=1:
        raise ValueError('Unsupported local vehicle archive.')
    entries=package['sources']
    if not isinstance(entries,dict) or not entries or entries.keys() & _registered.keys():
        raise ValueError('A local vehicle archive must contain distinct source photos.')
    for key,packet in entries.items():
        _check_packet(key,packet)
    _registered.update(entries)


def get_packet(photo_sha):
    return deepcopy(_registered.get(photo_sha))


def _check_packet(key,packet):
    body=packet['body']
    if (set(packet)!={'origin','body_sha256','body'} or packet['origin']!=ORIGIN or packet['body_sha256']!=digest(body)
            or body['image_sha256']!=key or body['recipe']!=ORIGIN or body['prompt_sha256']!=digest([GUIDE,VIEWS,MAX_TOKENS])):
        raise ValueError('The vehicle archive differs from its recorded source, guide or body.')
    scene={'source':'predicted','frame':{'recording_id':body['recording_id'],'frame_id':body['frame_id'],
           'width':body['size'][0],'height':body['size'][1]},'provenance':{'image_sha256':key,'network_policy':'local_only'}}
    if not local_surfaces.matches_scene(scene):
        raise ValueError('The vehicle observations require the registered original surface.')
    t=body['task1']
    if t['vision_identity_sha256']!=TASK1_ID or t['checkpoint_sha256']!=TASK1_WEIGHTS:
        raise ValueError('The vehicle boxes require the released Task 1 detector and tracker.')
    if any(type(t.get(n)) is not int or t[n]<0 for n in ('native_detection_count','current_track_count')):
        raise ValueError('The raw detection and current-track counts are missing.')
    for name in ('detections_sha256','tracks_sha256','vision_provenance_sha256','current_tracks_sha256','native_frame_sha256'):
        if not re.fullmatch('[a-f0-9]{64}',t[name]):
            raise ValueError('Missing Task 1 source evidence.')
    model=body['model']
    if (model['model_id']!=STREAMED_MODEL_ID or model['revision']!=bay_landmarks.MODELS['base']
            or model['adapter']!=bay_landmarks.MODELS['adapter'] or model['processor']!=bay_landmarks.MODELS['processor']
            or model['weights']!={**WEIGHT_SHA256,**METADATA_SHA256} or model['quantization']!='nf4'
            or model['runtime']!='local_streamed' or (model['min_pixels'],model['max_pixels'])!=(65536,262144)
            or model['inference_resident_layers']!=2):
        raise ValueError('The local vehicle reader differs from the verified model.')
    rows=body['rows']
    if (not isinstance(rows,list) or type(body['count']) is not int or body['count']!=len(rows)
            or len(rows)>min(4,t['current_track_count']) or len(set(row['anchor']['track_id'] for row in rows))!=len(rows)):
        raise ValueError('Repeated or unsupported local vehicle rows.')
    if body['status']!=('complete' if rows else 'no_eligible_current_vehicles'):
        raise ValueError('Local coverage differs from the actual vehicle rows.')
    for row in rows:
        if len(row['calls'])!=2:
            raise ValueError('Missing independent vehicle crop.')
        for kind,call in zip(VIEWS,row['calls'],strict=True):
            if (call['stage']!=kind or not re.fullmatch('[a-f0-9]{64}',call['view_sha256'])
                    or type(call['reply']) is not str or sha256(call['reply'].encode()).hexdigest()!=call['reply_sha256']
                    or type(call['output_tokens']) is not int or not 1<=call['output_tokens']<=MAX_TOKENS
                    or type(call['input_tokens']) is not int or call['input_tokens']<=0
                    or type(call['seconds']) not in (int,float) or not math.isfinite(call['seconds']) or call['seconds']<0):
                raise ValueError('Invalid raw local vehicle response.')
        if row['attributes']!=consensus(row['calls']):
            raise ValueError('Local vehicle attributes differ from the two actual model replies.')


def verify(scene, original_bytes, specs, packet):
    key=sha256(original_bytes).hexdigest()
    official=_registered.get(key)
    if not isinstance(packet,dict) or not official or packet!=official:
        raise ValueError('This source has no matching registered vehicle observation.')
    _check_packet(key,packet)
    body=packet['body']
    f=scene['frame']
    if (scene.get('source')!='predicted' or scene['provenance'].get('network_policy')!='local_only'
            or scene['provenance'].get('image_sha256')!=key or scene['provenance'].get('vision_identity_sha256')!=TASK1_ID
            or (f['recording_id'],f['frame_id'],[f['width'],f['height']])!=(body['recording_id'],body['frame_id'],body['size'])
            or any(t['recording_id']!=f['recording_id'] or t['frame_id']>f['frame_id'] for t in scene['tracks'])
            or not local_surfaces.matches_scene(scene) or digest(tracks(scene))!=body['task1']['current_tracks_sha256']):
        raise ValueError('Local vision evidence belongs to a different scene or current detector observations.')
    surface=local_surfaces.get_packet(key)
    if not surface or surface['signature']!=body['surface_sha256']:
        raise ValueError('The local surface that bounded the vehicle crops has changed.')
    if [anchor(s) for s in specs]!=[row['anchor'] for row in body['rows']]:
        raise ValueError('The measured track order or boxes changed.')
    with Image.open(BytesIO(original_bytes)) as im:
        if list(im.size)!=body['size']:
            raise ValueError('Vehicle-crop dimensions differ from the source.')
        for row in body['rows']:
            for kind,call in zip(VIEWS,row['calls'],strict=True):
                if sha256(render(im,row['anchor']['box'],kind)).hexdigest()!=call['view_sha256']:
                    raise ValueError('A vehicle crop differs from the original photograph.')
    return [list(row['attributes']) for row in body['rows']]
