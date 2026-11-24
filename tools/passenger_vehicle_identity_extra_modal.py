"""One bounded, source-allowlisted measurement of previously unread public-gallery vehicle crops."""
import base64
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import modal
from tools import passenger_qwen_point_protocol as protocol
from tools.passenger_qwen35_download import MODEL_FOLDER, REPO, REVISION, VOLUME
from tools.passenger_roadside35 import image


ROOT = Path(__file__).resolve().parents[1]
INPUTS = 'passenger-vehicle-identity-extra-inputs-v1.json'
app = modal.App('trackrag-public-extra-vehicle-identity')
if modal.is_local():
    image = image.add_local_file(ROOT/'trackrag/dropoff/vehicle_ground_view.py', '/root/trackrag/dropoff/vehicle_ground_view.py')
    for module in ('passenger_vehicle_extra_views', 'passenger_vehicle_extra_native'):
        image = image.add_local_file(ROOT/'tools'/(module+'.py'), '/root/tools/'+module+'.py')
    image = image.add_local_file(ROOT/'experiments/dropoff'/INPUTS, '/root/'+INPUTS)


@app.function(image=image, gpu='H100', cpu=4, memory=55000, max_containers=1, timeout=720,
              retries=0, single_use_containers=True, volumes={'/weights':modal.Volume.from_name(VOLUME).read_only()},
              secrets=[modal.Secret.from_name('trackrag-parking-receipt-v1')])
def inspect(photos):
    import subprocess
    import tempfile
    import time
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from PIL import Image
    from tools import passenger_vehicle_extra_views as views
    from tools.passenger_vehicle_extra_native import completed_replies
    from trackrag.dropoff.vehicle_ground_view import checksum

    deadline = time.monotonic() + 670
    registry = views.sources('/root/'+INPUTS)
    public = set(protocol.read_allowlist('/trial-public').values())
    if (len(photos) != len(registry) or len(registry) != 71 or not set(registry) <= public
            or {sha256(data).hexdigest() for data in photos} != set(registry)):
        raise ValueError('Only the frozen set of 71 previously registered public originals may be measured.')
    weights = Path('/weights')/MODEL_FOLDER
    receipt = json.loads((weights/'trial-download-verification.json').read_bytes())
    if (receipt['repository'], receipt['revision'], receipt['shards']) != (REPO, REVISION, 42):
        raise ValueError('Unexpected public vehicle reader checkpoint.')
    signing = Ed25519PrivateKey.from_private_bytes(base64.b64decode(os.environ.pop('PARKING_SIGNING_KEY'),validate=True))
    bodies, inputs, order = {}, [], []
    for photo in photos:
        key = sha256(photo).hexdigest()
        source = registry[key]
        body = {'recipe':views.RECIPE, 'image_sha256':key, 'source_manifest':views.SOURCE_SHA256,
                'source_spec':checksum(source), 'model':REPO, 'revision':REVISION, 'guide_sha256':checksum(views.GUIDE),
                'groups':[]}
        with Image.open(BytesIO(photo)) as original:
            if list(original.size) != source['size']:
                raise ValueError('Public photograph dimensions changed.')
            for vehicles in views.groups(source['vehicles']):
                group = {'track_ids':[v['track_id'] for v in vehicles], 'views':{}}
                body['groups'].append(group)
                for kind in ('context','detail'):
                    rendered = views.render(original, vehicles, kind)
                    group['views'][kind] = {'rendered_sha256':sha256(rendered).hexdigest()}
                    inputs.append(base64.b64encode(rendered).decode('ascii'))
                    order.append(group['views'][kind])
        bodies[key] = body
    if len(inputs) != 148:
        raise ValueError('The frozen additional crop population changed.')
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        packed, replies, guide = work/'inputs.json', work/'completed.jsonl', work/'guide'
        packed.write_text(json.dumps(inputs))
        guide.write_text(views.GUIDE)
        status = 'complete'
        try:
            subprocess.run(['/usr/bin/python3.12','-u','/root/tools/passenger_vehicle_extra_native.py',
                            str(packed),str(replies),str(weights),str(guide)], check=True,
                           timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            status = 'timed_out'
        except subprocess.CalledProcessError:
            status = 'native_error'
        results = completed_replies(replies, [slot['rendered_sha256'] for slot in order])
    if status == 'complete' and len(results) != len(order):
        raise ValueError('The native reader reported success without all visual inputs.')
    for slot, raw in zip(order[:len(results)], results, strict=True):
        slot['reply'] = raw
    records = {}
    for key, body in bodies.items():
        if any('reply' not in reading for group in body['groups'] for reading in group['views'].values()):
            continue
        payload = json.dumps(body, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(',',':')).encode()
        records[key] = {'body':body, 'signature':signing.sign(payload).hex()}
    return {'status':status, 'packets':records, 'completed_boards':len(results),
            'expected_boards':len(order), 'incomplete_sources':sorted(set(registry) - set(records))}


@app.local_entrypoint()
def main():
    from tools import passenger_vehicle_extra_views as views
    from tools.passenger_vehicle_extra_audit import measurements
    from trackrag.dropoff.parking_evidence import signed_bytes
    from tools.passenger_release import PUBLIC_PHOTO_LISTS

    public = {entry['sha256'] for name in PUBLIC_PHOTO_LISTS
              for entry in json.loads((ROOT/'experiments/dropoff'/name).read_bytes())['photos']}
    registry = views.sources(ROOT/'experiments/dropoff'/INPUTS)
    gallery = ROOT/'data/uae-parking-study'
    measured = {}
    scenes = {}
    for line in (gallery/'passenger-requests.jsonl').read_bytes().splitlines():
        scene = json.loads(line)['scene']
        key = scene['provenance']['image_sha256']
        if key not in registry:
            continue
        source = registry[key]
        if (key not in public or key in measured or scene['source'] != 'predicted' or
                scene['provenance'].get('network_policy') == 'local_only' or
                (scene['frame']['recording_id'], scene['frame']['frame_id']) != (source['recording_id'],source['frame_id']) or
                sha256(signed_bytes(scene['tracks'])).hexdigest() != source['current_tracks_sha256']):
            raise ValueError('Unapproved photo or changed public Task 1 observations.')
        original = (gallery/(key+'.jpg')).read_bytes()
        if sha256(original).hexdigest() != key:
            raise ValueError('Public photo has changed.')
        measured[key] = original
        scenes[key] = scene
    if set(measured) != set(registry):
        raise ValueError('Incomplete public source set.')
    print('Starting one bounded reader for 71 public originals, 236 additional bodies and 148 neutral boards.',flush=True)
    result = inspect.remote([measured[key] for key in sorted(measured)])
    packets = result.pop('packets')
    incomplete = set(result['incomplete_sources'])
    if (set(packets) & incomplete or set(packets) | incomplete != set(registry)
            or (result['status'] == 'complete' and incomplete)):
        raise ValueError('The supplementary identity source inventory changed.')
    trial = ROOT/'runs/passenger-rear-20261124/public-identity-extra'
    trial.mkdir(parents=True,exist_ok=True)
    # Every invocation has its own result, including timeouts. Never replace an earlier evidence archive.
    import tempfile
    attempt = Path(tempfile.mkdtemp(prefix='attempt-', dir=trial))
    result['signed_sources'] = len(packets)
    (attempt/'status.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    if not packets:
        print('No source completed. The attempt status is saved locally.', flush=True)
        return
    target = attempt/'passenger-vehicle-identity-extra-v1.zip'
    staged = target.with_suffix('.pending')
    with ZipFile(staged,'w') as archive:
        for key in sorted(packets):
            packet = packets[key]
            request = {'request_id':'local-source-replay', 'scene':scenes[key], 'input_mode':'typed',
                       'text':'Inspect the source photograph.', 'dialogue':[]}
            readings = measurements(request, measured[key], packet)
            if len(readings) != len(registry[key]['vehicles']):
                raise ValueError('Unexpected number of measured bodies on the reproduced source.')
            name = ZipInfo(key+'.json',(1980,1,1,0,0,0))
            name.create_system = 3
            name.external_attr = 0o100644 << 16
            archive.writestr(name,signed_bytes(packet),compress_type=ZIP_DEFLATED,compresslevel=9)
    os.replace(staged,target)
    print('Signed supplementary receipts:',sha256(target.read_bytes()).hexdigest(),flush=True)
