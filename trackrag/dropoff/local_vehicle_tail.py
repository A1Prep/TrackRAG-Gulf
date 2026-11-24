"""Source-bound original and reflected readings of a pictured vehicle's tail end."""
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageOps

from . import local_surfaces, local_vehicle_observations as vehicle, manual_surfaces


ORIGIN = 'local_vehicle_tail_edge_mirror_v1'
GUIDE = ('Examine only the vehicle outlined in magenta. Which edge of that vehicle in the displayed '
         'image contains its actual boot, tailgate or rear bumper? LEFT means its rear is at the left '
         'end of its body. RIGHT means its rear is at the right end. Use visible body, lamp, boot or '
         'grille details. If the vehicle is head-on, directly rear-on, hidden, or the end cannot be '
         'distinguished, answer UNCLEAR. Return just LEFT, RIGHT or UNCLEAR.')
QUESTION = 'Which image-side end of this outlined vehicle is its rear? Answer with one word.'
STAGES = ('original','mirrored')
TOKEN_LIMIT = 6
PROMPT_SHA = vehicle.digest([GUIDE,QUESTION,TOKEN_LIMIT])
_registered = {}


def eligible(row):
    paint, _, end, _ = row['attributes']
    x, y, right, bottom = row['anchor']['box']
    return paint != 'unknown' and end in ('SIDE','REAR') and right-x >= 1.65*(bottom-y)


def control(row):
    x, y, right, bottom = row['anchor']['box']
    return row['attributes'][2] in ('FRONT','UNCLEAR') and right-x <= 1.45*(bottom-y)


def parse(reply):
    if not isinstance(reply,str):
        return None
    word = reply.strip().rstrip('.').upper()
    return word if word in {'LEFT','RIGHT','UNCLEAR'} else None


def consensus(calls):
    first, reflected = (parse(call['reply']) for call in calls)
    if (first,reflected) in (('LEFT','RIGHT'),('RIGHT','LEFT')):
        return first
    return 'UNCLEAR'


def render(photo, box, stage):
    if stage not in STAGES:
        raise ValueError('Unknown tail-edge view.')
    x, y, r, b = box
    width, height = r-x, b-y
    bounds = (max(0,math.floor(x-.14*width)),max(0,math.floor(y-.20*height)),
              min(photo.width,math.ceil(r+.14*width)),min(photo.height,math.ceil(b+.20*height)))
    view = photo.crop(bounds).convert('RGB')
    view.thumbnail((640,480),Image.Resampling.LANCZOS)
    sx, sy = view.width/(bounds[2]-bounds[0]), view.height/(bounds[3]-bounds[1])
    ImageDraw.Draw(view).rectangle(((x-bounds[0])*sx,(y-bounds[1])*sy,(r-bounds[0])*sx,(b-bounds[1])*sy),outline='#ff22cc',width=2)
    if stage=='mirrored':
        view = ImageOps.mirror(view)
    output=BytesIO()
    view.save(output,format='PNG')
    return output.getvalue()


def _check(key, packet):
    parent=vehicle.get_packet(key,origin=vehicle.ORIGIN_V2)
    if not parent:
        raise ValueError('Install the matching expanded local vehicle observations first.')
    body=packet['body']; previous=parent['body']
    if (set(packet)!={'origin','body','body_sha256'} or packet['origin']!=ORIGIN or vehicle.digest(body)!=packet['body_sha256']
            or body['source_sha256']!=key or body['parent_body_sha256']!=parent['body_sha256'] or body['prompt_sha256']!=PROMPT_SHA
            or body['model']!=previous['model'] or (body['recording_id'],body['frame_id'],body['size'])!=(previous['recording_id'],previous['frame_id'],previous['size'])):
        raise ValueError('Tail measurement differs from its original photo, base crops, guide or model.')
    by_id={r['anchor']['track_id']:r for r in previous['rows']}; seen=set(); measured=set()
    for row in body['rows']:
        ident=row['anchor']['track_id']; initial=by_id.get(ident)
        if (initial is None or row['anchor']!=initial['anchor'] or ident in seen or
                (row['role']=='eligible' and not eligible(initial)) or
                (row['role']=='control' and not control(initial)) or row['role'] not in ('eligible','control')):
            raise ValueError('Wrong underlying vehicle or tail crop role.')
        seen.add(ident)
        if row['role']=='eligible': measured.add(ident)
        calls=row['calls']
        if len(calls)!=2:
            raise ValueError('Tail inspection requires two views.')
        for stage, call in zip(STAGES,calls,strict=True):
            raw=call['reply']
            if (stage!=call['stage'] or not isinstance(raw,str) or sha256(raw.encode()).hexdigest()!=call['reply_sha256']
                    or not re.fullmatch('[a-f0-9]{64}',call['view_sha256']) or type(call['seconds']) not in (float,int)
                    or not math.isfinite(call['seconds']) or call['seconds']<0 or type(call['input_tokens']) is not int
                    or call['input_tokens']<=0 or type(call['output_tokens']) is not int or not 1<=call['output_tokens']<=TOKEN_LIMIT):
                raise ValueError('Invalid original or mirrored decoder receipt.')
        if row['tail_edge']!=consensus(calls):
            raise ValueError('Tail direction differs from the two raw answers.')
    if measured!={i for i,r in by_id.items() if eligible(r)}:
        raise ValueError('The eligible neutral vehicle inventory is incomplete.')
    return parent


def register(path, expected_sha):
    raw=Path(path).read_bytes().replace(b'\r\n',b'\n')
    if sha256(raw).hexdigest()!=expected_sha.lower():
        raise ValueError('Tail-observation archive failed its pinned hash.')
    pack=json.loads(raw)
    if set(pack)!={'origin','version','sources'} or pack['origin']!=ORIGIN or type(pack['version']) is not int or pack['version']!=1:
        raise ValueError('Unknown tail-observation archive.')
    sources=pack['sources']
    if not isinstance(sources,dict) or not sources or sources.keys()&_registered.keys():
        raise ValueError('Tail archive needs distinct original photos.')
    for key,packet in sources.items(): _check(key,packet)
    _registered.update(sources)


def get_packet(source_sha):
    return deepcopy(_registered.get(source_sha))


def verify(scene, photo, packet):
    key=sha256(photo).hexdigest()
    if not packet or packet!=_registered.get(key):
        raise ValueError('No registered tail measurements on this original photo.')
    parent=_check(key,packet)
    classes=manual_surfaces.decode(local_surfaces.get_packet(key),scene['frame'],key)
    vehicle.verify(scene,photo,vehicle.specs_for(parent,scene,classes),parent)
    result={}
    with Image.open(BytesIO(photo)) as original:
        for row in packet['body']['rows']:
            for call in row['calls']:
                if sha256(render(original,row['anchor']['box'],call['stage'])).hexdigest()!=call['view_sha256']:
                    raise ValueError('Original or reflected vehicle pixels differ from the recorded crop.')
            if row['role']=='eligible' and row['tail_edge']!='UNCLEAR':
                result[row['anchor']['track_id']]=row['tail_edge']
    return result
