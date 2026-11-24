"""Reproduce the two neutral presentations used in the additional-vehicle experiment."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from trackrag.dropoff import vehicle_ground_view


RECIPE = 'public_additional_vehicle_identity_two_views_v1'
SOURCE_SHA256 = '67add8726c288d146ddb57147fd2b20dbd0256ceb4f62d5c45ba33f53769055d'
SOURCE_NAME = 'passenger-vehicle-identity-extra-inputs-v1.json'
GUIDE = (
    'Identify only each numbered vehicle. Report its body paint, body type and visible end. '
    'Use unknown or UNCLEAR when the colour, subtype or orientation is not visible. A pickup has '
    'a visible open cargo bed or other distinctive pickup body evidence; an ambiguous truck is truck. '
    'Colours: white, cream, black, silver, gray, red, blue, green, yellow, orange, brown, unknown. '
    'Body types: sedan, SUV, pickup, van, truck, bus, car, unknown. Visible ends: REAR, FRONT, SIDE, UNCLEAR. '
    'There is no ground target. Always give NONE in the final field. '
    'Return exactly one line per number: number | colour | type | orientation | NONE.'
)


def sources(path):
    data = Path(path).read_bytes().replace(b'\r\n', b'\n')
    if sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('Supplementary source and vehicle inputs do not match the approved public manifest.')
    return json.loads(data)['sources']


def groups(vehicles):
    return [vehicles[i:i+4] for i in range(0, len(vehicles), 4)]


def render(original, vehicles, kind):
    if kind == 'context':
        return vehicle_ground_view.render(original, [{**v, 'ground':None} for v in vehicles])
    if kind != 'detail' or not 1 <= len(vehicles) <= 4:
        raise ValueError('Unexpected neutral crop group or view.')
    width, height = original.size
    columns = min(2, len(vehicles))
    board = Image.new('RGB', (columns*332, ((len(vehicles)+1)//2)*236), '#222222')
    label = ImageDraw.Draw(board)
    font = ImageFont.load_default(size=21)
    for idx, spec in enumerate(reversed(vehicles)):
        left, top, right, bottom = spec['box']
        dx, dy = .06*(right-left), .06*(bottom-top)
        bound = (max(0, int(left-dx)), max(0, int(top-dy)), min(width, int(right+dx)+1), min(height, int(bottom+dy)+1))
        crop = original.crop(bound).convert('RGB')
        crop.thumbnail((326, 205), Image.Resampling.LANCZOS)
        x, y = idx%columns*332, idx//columns*236
        board.paste(crop, (x+(332-crop.width)//2, y+25+(205-crop.height)//2))
        label.text((x+6,y), str(idx+1), font=font, fill='white')
    result = BytesIO()
    board.save(result, format='JPEG', quality=92, subsampling=0)
    return result.getvalue()


def readings(raw, vehicles, kind):
    valid_rows = len(vehicles)
    if not isinstance(raw, str) or kind not in ('context','detail') or not 1 <= valid_rows <= 4:
        raise ValueError('Invalid additional vehicle reading.')
    # A repeated number is ambiguous even when its first occurrence looks plausible.
    indices = {}
    for line in raw.splitlines():
        prefix = line.strip(' |*-.\t').split('|',1)[0].strip()
        if prefix.isdigit():
            index = int(prefix)
            indices[index] = indices.get(index,0)+1
    rows = vehicle_ground_view.parse(raw, valid_rows)
    rows = [row if indices.get(index) == 1 else ['unknown','unknown','UNCLEAR','NONE']
            for index,row in enumerate(rows,1)]
    if kind == 'detail':
        rows.reverse()
    return [row[:3] if row[3] == 'NONE' else ['unknown', 'unknown', 'UNCLEAR'] for row in rows]
