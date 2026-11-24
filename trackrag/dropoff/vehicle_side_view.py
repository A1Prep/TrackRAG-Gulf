"""Render repeatable full-scene and source-pixel close views for one lateral ground check."""
from hashlib import sha256
from io import BytesIO
import json

from PIL import Image, ImageDraw, ImageFont


RECIPE = "vehicle_side_ground_two_views_v1"
GUIDE = (
    "Inspect the exact centre of the small cyan ring. The magenta rectangle identifies one vehicle. "
    "The header states an IMAGE side, as seen by the camera. Judge the visible ground at the ring, "
    "not the boxed vehicle or any rings you might expect. If on a vehicle, person, sidewalk, obstruction "
    "or crossing, answer BLOCKED. If in a moving traffic lane, access aisle, driveway or junction, answer LANE. "
    "If separated by a kerb, fence or raised island, answer SEPARATED. If on the opposite side of the boxed "
    "vehicle, or in front or behind instead of alongside, answer WRONG_SIDE. Answer OPEN only for clearly "
    "visible, vacant, distinct off-lane parking or pull-in on the requested image side beside this vehicle "
    "at its bumper depth, accessible from this roadway. Do not guess from the cyan ring, vehicle paint, "
    "or one patch of asphalt. If the relation, vacancy or distinction from an aisle is uncertain answer UNCLEAR. "
    "Respond with exactly OPEN, WRONG_SIDE, LANE, BLOCKED, SEPARATED or UNCLEAR."
)
VERDICTS = {"OPEN", "WRONG_SIDE", "LANE", "BLOCKED", "SEPARATED", "UNCLEAR"}


def digest(value):
    plain = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return sha256(plain.encode()).hexdigest()


def verdict(reply):
    word = str(reply).strip().rstrip(".").upper()
    return word if word in VERDICTS else "UNCLEAR"


def render(original, row, *, close):
    width, height = original.size
    left, top, right, bottom = row["anchor_box"]
    px, py = row["point"]
    side = row["side"]
    if (side not in {"left", "right"} or not 0 <= left < right <= width or not 0 <= top < bottom <= height
            or not 0 <= px < width or not 0 <= py < height):
        raise ValueError("Invalid vehicle side or source coordinates")
    if close:
        padding_x = max((right - left) * .65, width * .055)
        padding_y = max((bottom - top) * .70, height * .08)
        crop = (max(0, int(min(left, px) - padding_x)), max(0, int(min(top, py) - padding_y)),
                min(width, round(max(right, px) + padding_x)), min(height, round(max(bottom, py) + padding_y)))
    else:
        crop = (0, 0, width, height)
    image = original.crop(crop).convert("RGB")
    image.thumbnail((1080, 760), Image.Resampling.LANCZOS)
    sx, sy = image.width / (crop[2] - crop[0]), image.height / (crop[3] - crop[1])
    def project(x, y):
        return (x - crop[0]) * sx, (y - crop[1]) * sy

    draw = ImageDraw.Draw(image)
    draw.rectangle([project(left, top), project(right, bottom)], outline="#ff55e8", width=2)
    cx, cy = project(px, py)
    for radius, color in ((8, "black"), (6, "#00f8ff")):
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=color, width=2)
    canvas = Image.new("RGB", (max(image.width, 355), image.height+33), "#191b1f")
    canvas.paste(image, (0, 33))
    ImageDraw.Draw(canvas).text((10, 4), f"Requested side: PHOTO {side.upper()}",
                                fill="white", font=ImageFont.load_default(size=21))
    output = BytesIO()
    canvas.save(output, format="JPEG", quality=90, subsampling=0)
    return output.getvalue()
