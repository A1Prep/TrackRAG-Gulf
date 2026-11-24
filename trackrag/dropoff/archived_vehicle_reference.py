"""Use a prior, source-bound visual crop solely to ask which vehicle a passenger meant."""
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import Image

from . import single_vehicle
from .bay_packets import read_photo_packet
from .parking_evidence import _authenticate, _decode_map, source_input
from .passenger import digest


BANK = Path(__file__).resolve().parents[2] / 'experiments/dropoff/passenger-vehicle-archived-crops-v1.zip'
BANK_SHA256 = '175b32d21937154a8c8f9fb086e69f3fa38b9ac180978ddb32cb1adea9d525ec'


@lru_cache(maxsize=1)
def _released_archive(path, modified_ns, size):
    return size < 2_000_000 and sha256(path.read_bytes()).hexdigest() == BANK_SHA256


def get_packet(photo_sha):
    """New requests consume only the frozen public bank. Replay carries the original signed packet."""
    try:
        stat = BANK.stat()
        if _released_archive(BANK, stat.st_mtime_ns, stat.st_size):
            return read_photo_packet(BANK, photo_sha, 120_000)
    except OSError:
        pass
    return None


def eligible(ref):
    return (ref.get('relation') in {'behind', 'beside'} and not ref.get('qualified')
            and not ref.get('unsupported') and not ref.get('ground_side')
            and ref.get('color') in single_vehicle.COLORS)


def _body_matches(wanted, observed):
    # The archived prompt had no pickup category. A truck is only a possible identity.
    return single_vehicle._compatible(wanted, observed) or (wanted == 'pickup' and observed == 'truck')


def possible_vehicle(ref, readings, specs, answers, width):
    """Require one consistent candidate; ignore the historic crop's former ground verdict entirely."""
    if not eligible(ref):
        return None
    colour, wanted = ref['color'], ref['kind']
    current = {spec['track_id']: (spec, row) for spec, row in zip(specs, answers, strict=True)}
    choices = []
    for track_id, box, body, paint in readings:
        if (paint != colour or not _body_matches(wanted, body)
                or not single_vehicle._side(box, width, ref.get('side'))):
            continue
        if track_id in current:
            spec, row = current[track_id]
            if (list(spec['box']) != list(box) or row[0] not in {'unknown', paint}
                    or row[1] != 'unknown' and not (
                        single_vehicle._compatible(body, row[1]) or single_vehicle._compatible(row[1], body))):
                continue
        choices.append((track_id, box, body))
    if len(choices) != 1:
        return None
    track_id, box, body = choices[0]
    for other, (spec, row) in current.items():
        if (other != track_id and row[0] in {colour, 'unknown'}
                and (row[1] == 'unknown' or _body_matches(wanted, row[1]))
                and single_vehicle._side(spec['box'], width, ref.get('side'))):
            return None
    middle = (box[0] + box[2]) / (2 * width)
    position = 'left of centre' if middle < .46 else 'right of centre' if middle > .54 else 'in the centre'
    question = f'I can see a {colour} {body} {position} in the photo.'
    if wanted == 'pickup' and body == 'truck':
        question += ' I couldn’t verify that it is a pickup.'
    return list(box), question + f' Is that the vehicle you meant? I haven’t verified an open space {ref["relation"]} it.'


def clarify(request, photo, ref, specs, answers, packet):
    """Authenticate every recorded response against its original crop, current box and source model."""
    from .local_surfaces import local_only
    if not eligible(ref) or local_only(request['scene']):
        return None
    clean, key, _ = source_input(request, photo)
    context = single_vehicle.context(clean)
    _authenticate(packet)
    body = packet['body']
    if (body.get('recipe') != single_vehicle.RECIPE or body.get('models') != single_vehicle.MODELS
            or body.get('image_sha256') != key or body.get('context_sha256') != digest(context)):
        raise ValueError('Archived vehicle crops do not match this original and its current tracks.')
    frame = clean['scene']['frame']
    classes = _decode_map(body['class_map'], frame)
    options = single_vehicle.opportunities(context, classes)
    expected = options if len(options) <= 8 else []
    readings = []
    with Image.open(BytesIO(photo)) as original:
        for option, call in zip(expected, body['calls'], strict=True):
            rendered = single_vehicle.inspection(original, option)
            identity = single_vehicle.call_identity(
                'single_vehicle_rear', single_vehicle.GUIDE, single_vehicle.WORDS, rendered)
            reply = call['response']
            if (call.get('track_id') != option['track_id'] or any(call.get(k) != v for k, v in identity.items())
                    or reply != 'NONE' and reply not in single_vehicle.RESPONSES):
                raise ValueError('The historical vehicle reply does not belong to this current crop.')
            if reply != 'NONE':
                kind, colour = reply.lower().split()
                readings.append((option['track_id'], option['anchor_box'], kind, colour))
    return possible_vehicle(ref, readings, specs, answers, frame['width'])
