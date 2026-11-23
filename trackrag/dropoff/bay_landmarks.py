"""Image-bound observations of visible public landmarks from the trained Qwen model."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re

from PIL import Image

from .bay_packets import read_photo_packet
from .parking_evidence import _authenticate
from .place_grounding import ADAPTER, BASE, PROCESSOR

RECIPE = "trained_qwen_public_landmarks_v1"
MODELS = {"base":BASE, "adapter":ADAPTER, "processor":PROCESSOR}
BANDS = ((0., .45), (.275, .725), (.55, 1.))
GUIDE = ("Read this real street photograph crop for a passenger. Identify up to two prominent street-facing places "
         "visible in this crop. Give ordinary categories such as restaurant, grocery, pharmacy, shop, garage, "
         "house, parking lot, trees, or building entrance, followed by the name or sign ONLY if legible. "
         "Do not guess a name or list other possibilities. Do not describe people, vehicles, plates, phone numbers "
         "or addresses. Use a short English phrase for each place, separated by |. Reply NONE if unclear.")
WORDS = "Which visible places could a passenger name when describing where to get out?"
BANK = Path(__file__).resolve().parents[2]/"experiments/dropoff/passenger-bay-landmarks-v1.zip"


def band_image(original, band):
    x0, x1 = BANDS[band]
    w, h = original.size
    image = original.crop((round(w*x0), 0, round(w*x1), round(.79*h))).convert("RGB")
    image.thumbnail((1000, 820), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=91, subsampling=0)
    return buffer.getvalue()


def public_phrase(raw):
    phrase = re.sub(r"[\w.+-]+@[\w.-]+|(?:\+?\d[\s()+-]?){5,}", "", raw)
    return " ".join(phrase.split())[:240]


def verify(packet, photo, size):
    _authenticate(packet)
    body = packet["body"]
    if (body.get("recipe") != RECIPE or body.get("models") != MODELS or body.get("size") != list(size)
            or body.get("image_sha256") != sha256(photo).hexdigest()
            or body.get("prompt_sha256") != sha256((GUIDE+"\n"+WORDS).encode()).hexdigest()):
        raise ValueError("Landmark evidence differs from the original image or trained model.")
    rows = body.get("bands")
    if not isinstance(rows, list) or len(rows) != len(BANDS):
        raise ValueError("Landmark evidence is incomplete.")
    with Image.open(BytesIO(photo)) as opened:
        if opened.size != tuple(size):
            raise ValueError("Landmark photo dimensions changed.")
        for index, row in enumerate(rows):
            if (row.get("view_sha256") != sha256(band_image(opened,index)).hexdigest()
                    or not isinstance(row.get("places"),str) or len(row["places"]) > 240):
                raise ValueError("A trained landmark observation differs from its source crop.")
    return [row["places"] for row in rows]


def get_packet(photo_sha):
    return read_photo_packet(BANK, photo_sha, 12000)
