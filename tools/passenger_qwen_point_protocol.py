"""Paired public-photo trial. The workers receive identical pixels and passenger words, without labels."""
import hashlib
from io import BytesIO
import json
from math import sqrt
from pathlib import Path


GUIDE = (
    "Use the driver-view photograph to identify where the passenger wants to get out. "
    "Choose the centre of one visibly unoccupied off-lane parking or stopping area that matches all "
    "the passenger's stated landmarks, side and spatial relationships. If the image does not show such "
    "an area clearly, answer NONE. Otherwise answer POINT x y using two integers between 0 and 1000, "
    "with x measured from the left and y from the top of the complete photograph. Put the point on "
    "the open ground within the area. Output only the requested point or NONE."
)
MAX_NEW_TOKENS = 32
MAX_PIXELS = 1048576


def read_allowlist(folder):
    result = {}
    for path in Path(folder).glob('*.json'):
        data = json.loads(path.read_bytes())
        for row in data['photos']:
            url = row['photo_url']
            if not url.startswith('https://kartaview.org/details/'):
                raise ValueError('Unrecognized public source')
            if url in result and result[url] != row['sha256']:
                raise ValueError('Public source has conflicting checksums')
            result[url] = row['sha256']
    return result


def prepare(source, words, original, allowed):
    from PIL import Image

    digest = hashlib.sha256(original).hexdigest()
    if allowed.get(source) != digest:
        raise ValueError('Only an allowlisted, original public KartaView photograph may be used')
    with Image.open(BytesIO(original)) as im:
        image = im.convert('RGB')
    w, h = image.size
    scale = min(1, 1280 / max(w, h), sqrt(917504 / (w * h)))
    size = tuple(max(32, round(d * scale / 32) * 32) for d in (w, h))
    scaled = image.resize(size, Image.Resampling.LANCZOS)
    buf = BytesIO()
    scaled.save(buf, format='PNG')
    rgb = buf.getvalue()
    return {'source': source, 'original_sha256': digest, 'native_size': [w,h], 'words': words,
            'input_size': list(size), 'input_sha256': hashlib.sha256(rgb).hexdigest(), 'png': rgb}


def checked_inputs(cases):
    from PIL import Image

    allowed = read_allowlist('/trial-public')
    if not 1 <= len(cases) <= 8:
        raise ValueError('Unexpected trial size')
    for item in cases:
        if allowed.get(item['source']) != item['original_sha256'] or not item['words'] or len(item['words']) > 256:
            raise ValueError('Public-photo identity or passenger instruction differs')
        raw = item['png']
        if len(raw) > 5000000 or hashlib.sha256(raw).hexdigest() != item['input_sha256']:
            raise ValueError('Input checksum mismatch')
        with Image.open(BytesIO(raw)) as im:
            if im.format != 'PNG' or list(im.size) != item['input_size'] or im.width * im.height > MAX_PIXELS:
                raise ValueError('Invalid common view')
            rgb = im.convert('RGB')
        yield {k:v for k,v in item.items() if k!='png'}, rgb


def messages(words, *, image_marker=None):
    marker = {'type':'image'}
    if image_marker is not None:
        marker['image'] = image_marker
    return [{'role':'system','content':GUIDE},
            {'role':'user','content':[marker, {'type':'text','text':words}]}]


def mount_trial(image):
    from tools.passenger_release import PUBLIC_PHOTO_LISTS

    root = Path(__file__).resolve().parents[1]
    image = image.add_local_file(Path(__file__), '/root/tools/passenger_qwen_point_protocol.py')
    for name in PUBLIC_PHOTO_LISTS:
        image = image.add_local_file(root/'experiments/dropoff'/name, '/trial-public/'+name)
    return image


def run_local(worker, manifest, out, **worker_options):
    panel = json.loads(Path(manifest).read_bytes())
    # This manifest has source identity, passenger wording and pixel hashes only.
    rows = []
    for item in panel['cases']:
        png = Path(panel['views_root']) / (item['input_sha256']+'.png')
        rows.append(dict(item, png=png.read_bytes()))
    result = worker.remote(rows, **worker_options)
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='results'}), flush=True)
    print(f'Saved {len(result["results"])} real outputs to {target}', flush=True)
