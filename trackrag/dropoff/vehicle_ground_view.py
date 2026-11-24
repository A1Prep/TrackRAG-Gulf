"""Show the actual detected vehicles independently of any candidate ground behind them."""
from hashlib import sha256
from io import BytesIO
import json

from PIL import Image, ImageDraw, ImageFont


RECIPE = "vehicle_identity_and_curb_ground_v1"
GUIDE = (
    "A driver-view image is followed by detail crops of the numbered vehicles. Look at each "
    "numbered rectangle independently. Identify the body paint, body type, and whether its rear "
    "faces the camera. Use cream for the beige body of a taxi and white for white bodywork. "
    "The small cyan ring, when present, is a separate question. Judge only that ring: CURBSIDE "
    "if it is on vacant ground in the same visibly separated curbside parking row immediately "
    "behind this vehicle, LANE if in the moving traffic lane, BLOCKED if on a person, car, "
    "sidewalk, crossing, obstruction or disconnected surface, otherwise UNCLEAR. A ring in a "
    "shadow, crossing a stripe, too close to the bumper, or an obscured parking boundary needs "
    "UNCLEAR if you cannot tell. Do not infer space for a whole car. If there is no ring use NONE. "
    "Return one line per numbered crop: number | colour | type | orientation | ring. Colours: "
    "white, cream, black, silver, gray, red, blue, green, yellow, orange, brown, unknown. Types: "
    "sedan, SUV, pickup, van, truck, bus, car, unknown. Orientations: REAR, FRONT, SIDE, UNCLEAR. "
    "For an absent or poorly visible object write unknown and UNCLEAR. No other text."
)
COLORS = set("white cream black silver gray red blue green yellow orange brown unknown".split())
TYPES = set("sedan suv pickup van truck bus car unknown".split())
ENDS = {"REAR", "FRONT", "SIDE", "UNCLEAR"}
GROUND = {"CURBSIDE", "LANE", "BLOCKED", "UNCLEAR", "NONE"}


def checksum(value):
    encoded = json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256(encoded).hexdigest()


def parse(reply, count):
    """Read validated labels in crop-number order, retaining unknowns for missing rows."""
    answers = {}
    for line in reply.strip().splitlines():
        fields = [field.strip(" -*`\t.").lower() for field in line.strip().strip("|").split("|")]
        if len(fields) != 5 or not fields[0].isdigit():
            continue

        index = int(fields[0])
        if not 1 <= index <= count or index in answers:
            continue
        colour, kind = fields[1:3]
        orientation, ground = fields[3].upper(), fields[4].upper()
        if colour in COLORS and kind in TYPES and orientation in ENDS and ground in GROUND:
            answers[index] = [colour, kind, orientation, ground]

    return [answers.get(i, ["unknown", "unknown", "UNCLEAR", "UNCLEAR"]) for i in range(1, count + 1)]


def _label_vehicle(image, spec, number, font, *, origin, scale):
    """Project one original vehicle box and its independently measured ground point onto a view."""
    origin_x, origin_y = origin
    scale_x, scale_y = scale
    left, top, right, bottom = spec["box"]
    box = [
        (left - origin_x) * scale_x,
        (top - origin_y) * scale_y,
        (right - origin_x) * scale_x,
        (bottom - origin_y) * scale_y,
    ]

    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline="#fa43ec", width=3)
    label_x, label_y = max(0, int(box[0])), max(0, int(box[1]) - 24)
    draw.rectangle((label_x, label_y, label_x + 28, label_y + 25), fill="black")
    draw.text((label_x + 6, label_y), str(number), font=font, fill="white")

    if spec["ground"]:
        x, y = spec["ground"]["point"]
        x, y = (x - origin_x) * scale_x, (y - origin_y) * scale_y
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline="black", width=3)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="cyan", width=3)


def render(original, specs):
    """Render the wide scene above the corresponding numbered vehicle crops."""
    width, height = original.size
    whole = original.convert("RGB")
    whole.thumbnail((1040, 540), Image.Resampling.LANCZOS)
    font = ImageFont.load_default(size=23)
    for number, spec in enumerate(specs, 1):
        _label_vehicle(
            whole, spec, number, font, origin=(0, 0), scale=(whole.width / width, whole.height / height)
        )

    detail_row_width = 256 * len(specs)
    canvas = Image.new("RGB", (max(whole.width, detail_row_width), whole.height + 232), "#222222")
    canvas.paste(whole, ((canvas.width - whole.width) // 2, 0))
    detail_left = (canvas.width - detail_row_width) // 2
    for number, spec in enumerate(specs, 1):
        left, top, right, bottom = spec["box"]
        vehicle_width = right - left
        vehicle_height = bottom - top
        crop_left = max(0, int(left - 0.45 * vehicle_width))
        crop_top = max(0, int(top - 0.16 * vehicle_height))
        crop_right = min(width, int(right + 0.45 * vehicle_width) + 1)
        crop_bottom = min(height, int(bottom + 0.78 * vehicle_height) + 1)

        crop = original.crop((crop_left, crop_top, crop_right, crop_bottom)).convert("RGB")
        crop.thumbnail((254, 229), Image.Resampling.LANCZOS)
        _label_vehicle(
            crop, spec, number, font,
            origin=(crop_left, crop_top),
            scale=(crop.width / (crop_right - crop_left), crop.height / (crop_bottom - crop_top)),
        )
        canvas.paste(crop, (detail_left + (number - 1) * 256, whole.height))

    jpeg = BytesIO()
    canvas.save(jpeg, format="JPEG", quality=90, subsampling=0)
    return jpeg.getvalue()
