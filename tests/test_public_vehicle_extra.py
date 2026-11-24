import base64
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import pytest
from PIL import Image

from tools import passenger_vehicle_extra_native as native, passenger_vehicle_extra_views as views


VEHICLES = [{'track_id':i,'class':'car','box':[10+i*30,20,30+i*30,55]} for i in (1,2,3)]


def test_detail_readings_reverse_the_visual_numbers():
    original = Image.new('RGB',(180,100),'white')
    assert sha256(views.render(original,VEHICLES,'context')).digest() != sha256(views.render(original,VEHICLES,'detail')).digest()
    raw='1 | white | van | FRONT | NONE\n2 | red | SUV | SIDE | NONE\n3 | black | pickup | REAR | NONE'
    context=views.readings(raw,VEHICLES,'context')
    assert context==[['white','van','FRONT'],['red','suv','SIDE'],['black','pickup','REAR']]
    assert views.readings(raw,VEHICLES,'detail')==list(reversed(context))


@pytest.mark.parametrize('raw', [
    '1 | white | sedan | REAR | CURBSIDE',
    '1 | white | sedan | REAR | NONE\n1 | blue | SUV | FRONT | NONE',
    'Bad output', '2 | white | sedan | REAR | NONE',
])
def test_malformed_duplicate_or_ground_reading_yields_unknown(raw):
    assert views.readings(raw,VEHICLES[:1],'context')==[['unknown','unknown','UNCLEAR']]


def test_completed_batches_survive_an_interrupted_next_batch(tmp_path):
    images = [f'photo{i}'.encode() for i in range(8)]
    def generate(batch, guide):
        assert guide == 'neutral'
        if len(batch) < 6:
            raise TimeoutError
        return [f'reply {i}' for i in range(len(batch))]
    output = tmp_path / 'completed.jsonl'
    with pytest.raises(TimeoutError):
        native.measure([base64.b64encode(p).decode() for p in images], 'neutral', output, generate)
    with output.open('ab') as stream:
        stream.write(b'[6, "unterminated')
    assert native.completed_replies(output, [sha256(p).hexdigest() for p in images]) == [
        f'reply {i}' for i in range(6)]


@pytest.mark.parametrize('entry', [[1, 'a', 'text'], [0, 'b', 'text'], [0, 'a', None]])
def test_checkpoint_rejects_a_wrong_order_view_or_nontext_reply(tmp_path, entry):
    output = tmp_path / 'completed.jsonl'
    output.write_text(json.dumps(entry)+'\n', encoding='utf-8')
    with pytest.raises(ValueError):
        native.completed_replies(output, ['a'])


def test_local_checkpoint_copy_authenticates_files(tmp_path):
    source = tmp_path / 'volume'
    source.mkdir()
    (source / 'shard').write_bytes(b'abc')
    record = {'name':'shard', 'bytes':3, 'sha256':sha256(b'abc').hexdigest()}
    native.copy_verified_weights(source, tmp_path/'working', [record])
    assert (tmp_path/'working/shard').read_bytes() == b'abc'
    (source / 'shard').write_bytes(b'bad')
    with pytest.raises(ValueError, match='checkpoint file differs'):
        native.copy_verified_weights(source, tmp_path/'changed', [record])


def test_offline_replay_binds_a_signed_record_to_original_track_and_render(monkeypatch):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    from tools import passenger_vehicle_extra_audit as audit
    from trackrag.dropoff import parking_evidence

    gallery = Path(__file__).resolve().parents[1] / 'data/uae-parking-study'
    if not (gallery/'passenger-requests.jsonl').is_file():
        pytest.skip('The optional public photograph bundle is not installed.')
    registry = audit._registry()
    scene = next((s for line in (gallery/'passenger-requests.jsonl').read_bytes().splitlines()
                  if (s := json.loads(line)['scene'])['provenance']['image_sha256'] in registry), None)
    if scene is None:
        pytest.skip('No additional-public-vehicle source installed.')
    key = scene['provenance']['image_sha256']
    source = registry[key]
    photo = (gallery/(key+'.jpg')).read_bytes()
    request = {'request_id':'offline-replay-test', 'scene':scene, 'text':'Inspect the photograph.',
               'input_mode':'typed', 'dialogue':[]}
    signer = Ed25519PrivateKey.generate()
    public = signer.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    monkeypatch.setattr(parking_evidence, 'PUBLIC_KEY', public.hex())
    def signed(body):
        return {'body':body, 'signature':signer.sign(parking_evidence.signed_bytes(body)).hex()}
    body = {'recipe':views.RECIPE, 'source_manifest':views.SOURCE_SHA256, 'source_spec':audit.checksum(source),
            'image_sha256':key, 'model':audit.primary.MODEL, 'revision':audit.primary.REVISION,
            'guide_sha256':audit.checksum(views.GUIDE), 'groups':[]}
    with Image.open(BytesIO(photo)) as original:
        for group in views.groups(source['vehicles']):
            board = {name:{'rendered_sha256':sha256(views.render(original, group, name)).hexdigest(),
                           'reply':'\n'.join(f'{i} | unknown | unknown | UNCLEAR | NONE'
                                             for i in range(1, len(group)+1))}
                     for name in ('context', 'detail')}
            body['groups'].append({'track_ids':[v['track_id'] for v in group], 'views':board})
    assert len(audit.measurements(request, photo, signed(body))) == len(source['vehicles'])
    tampered = deepcopy(body)
    tampered['groups'][0]['views']['detail']['rendered_sha256'] = '0'*64
    with pytest.raises(ValueError, match='rendered vehicle crop'):
        audit.measurements(request, photo, signed(tampered))
    with pytest.raises(ValueError, match='signature'):
        audit.measurements(request, photo, {**signed(body), 'body':tampered})
    changed_scene = deepcopy(scene)
    changed_scene['tracks'][0]['bbox_xyxy'][0] += .1
    with pytest.raises(ValueError, match='current tracks changed'):
        audit.measurements({**request,'scene':changed_scene}, photo, signed(body))
