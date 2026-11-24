"""Replay and score the public crop experiment offline. The passenger app does not load it."""
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import Image

from tools import passenger_vehicle_extra_views as views
from trackrag.dropoff import vehicle_ground_reference as primary
from trackrag.dropoff.parking_evidence import _authenticate, signed_bytes, source_input
from trackrag.dropoff.vehicle_ground_view import checksum


RELEASE = Path(__file__).resolve().parents[1] / 'experiments/dropoff'


@lru_cache(maxsize=1)
def _registry():
    return views.sources(RELEASE/views.SOURCE_NAME)


def measurements(request, photo, packet):
    from trackrag.dropoff.local_surfaces import local_only
    if local_only(request['scene']):
        raise ValueError('Public supplemental crops cannot be applied to a local-only source.')
    clean, key, _ = source_input(request, photo)
    source = _registry().get(key)
    if not source:
        raise ValueError('The photograph is not part of the public vehicle supplement.')
    scene = clean['scene']
    frame = scene['frame']
    if (source['recording_id'] != frame['recording_id'] or source['frame_id'] != frame['frame_id'] or
            source['size'] != [frame['width'],frame['height']] or
            source['current_tracks_sha256'] != sha256(signed_bytes(scene['tracks'])).hexdigest()):
        raise ValueError('The public source or its frozen current tracks changed.')
    _authenticate(packet)
    body = packet['body']
    if (body['recipe'] != views.RECIPE or body['source_manifest'] != views.SOURCE_SHA256 or
            body['source_spec'] != checksum(source) or body['image_sha256'] != key or
            (body['model'],body['revision']) != (primary.MODEL,primary.REVISION) or
            body['guide_sha256'] != checksum(views.GUIDE)):
        raise ValueError('Supplementary vehicle readings differ from the approved visual input or model.')
    result = []
    with Image.open(BytesIO(photo)) as original:
        for vehicles, group in zip(views.groups(source['vehicles']), body['groups'], strict=True):
            if (group['track_ids'] != [v['track_id'] for v in vehicles] or set(group['views']) != {'context','detail'}):
                raise ValueError('Supplementary readings do not follow the original vehicle crops.')
            observed = {}
            for kind, reading in group['views'].items():
                if sha256(views.render(original, vehicles, kind)).hexdigest() != reading['rendered_sha256']:
                    raise ValueError('A supplementary reading is attached to another rendered vehicle crop.')
                observed[kind] = views.readings(reading['reply'], vehicles, kind)
            for vehicle, wide, tight in zip(vehicles,observed['context'],observed['detail'],strict=True):
                result.append({'track_id':vehicle['track_id'],'box':vehicle['box'],
                               'context':wide,'detail':tight})
    return result
