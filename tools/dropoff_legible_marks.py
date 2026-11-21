"""Image-only diagnostic marking variant; changes marking AND pre-resampling.

This is not an accuracy or causal-improvement claim. No instructions, destination
labels, masks, status decisions or model calls are accepted here.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont

from trackrag.dropoff.contracts import validate_box, validate_point


def smart_spatial_size(height, width, min_pixels=65536, max_pixels=262144):
    """Qwen smart-resize rule with patch16 × merge2 =32 alignment."""
    if (not all(isinstance(v, int) and not isinstance(v, bool) and v > 0
                for v in (height, width, min_pixels, max_pixels)) or min_pixels > max_pixels):
        raise ValueError("Positive dimensions and ordered positive integer pixel budgets required")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Aspect ratio exceeds Qwen processor limit")
    factor = 32
    h = round(height/factor)*factor
    w = round(width/factor)*factor
    if h*w > max_pixels:
        beta = math.sqrt(height*width/max_pixels)
        h = max(factor, math.floor(height/beta/factor)*factor)
        w = max(factor, math.floor(width/beta/factor)*factor)
    elif h*w < min_pixels:
        beta = math.sqrt(min_pixels/(height*width))
        h = math.ceil(height*beta/factor)*factor
        w = math.ceil(width*beta/factor)*factor
    return h, w


def draw_legible_candidates(image_path, candidates, frame, output, min_pixels=65536, max_pixels=262144):
    """Resample once with PIL bicubic, then mark at the final model resolution.

    Candidate point/patch positions use independent image-axis scales. Text can
    move to fit the canvas; points and patches never move to accommodate text.
    PIL pre-resampling differs from the original torchvision processor route.
    """
    width, height = frame["width"], frame["height"]
    target_h, target_w = smart_spatial_size(height, width, min_pixels, max_pixels)
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 36:
        raise ValueError("Expected 1..36 candidates")
    seen = set()
    validated = []
    for row in candidates:
        identity = row.get("candidate_id")
        if not isinstance(identity, str) or not re.fullmatch(r"R[0-9]{2}", identity) or identity in seen:
            raise ValueError("Unique R candidate identifiers required")
        seen.add(identity)
        point = validate_point(row.get("point_original_px"), width, height)
        box = validate_box(row.get("bbox_original_px"), width, height)
        if not (box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]):
            raise ValueError("Candidate point must lie within its patch")
        validated.append((identity, point, box))
    source_bytes = Path(image_path).read_bytes()
    with Image.open(BytesIO(source_bytes)) as source:
        if source.size != (width, height):
            raise ValueError("Declared original dimensions differ from image")
        image = source.convert("RGB").resize((target_w, target_h), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(image)
    sx, sy = target_w/width, target_h/height
    rows = []
    for identity, point, box in validated:
        scaled_point = [point[0]*sx, point[1]*sy]
        scaled_box = [box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy]
        draw.rectangle(scaled_box, outline="yellow", width=2)
        x, y = scaled_point
        draw.line((x-3, y, x+3, y), fill="cyan", width=2)
        draw.line((x, y-3, x, y+3), fill="cyan", width=2)
        font_size = 20
        font = ImageFont.load_default(size=font_size)
        bounds = draw.textbbox((0, 0), identity, font=font)
        while (bounds[2]-bounds[0] > target_w-4 or bounds[3]-bounds[1] > target_h-4) and font_size > 8:
            font_size -= 1
            font = ImageFont.load_default(size=font_size)
            bounds = draw.textbbox((0, 0), identity, font=font)
        tx = min(max(x+5-bounds[0], 2-bounds[0]), target_w-2-bounds[2])
        ty = min(max(y+5-bounds[1], 2-bounds[1]), target_h-2-bounds[3])
        text_box = [tx+bounds[0], ty+bounds[1], tx+bounds[2], ty+bounds[3]]
        draw.rectangle([text_box[0]-2, text_box[1]-2, text_box[2]+1, text_box[3]+1], fill="black")
        draw.text((tx, ty), identity, font=font, fill="yellow")
        rows.append({"candidate_id": identity, "point_model_px": scaled_point,
                     "patch_model_px": scaled_box, "label_model_bbox": text_box, "font_size": font_size})
    # Labels near canvas edges may overlap their own point. Restore all crosses
    # last without moving any candidate geometry to make the anchor visible.
    for row in rows:
        x, y = row["point_model_px"]
        draw.line((x-3, y, x+3, y), fill="cyan", width=2)
        draw.line((x, y-3, x, y+3), fill="cyan", width=2)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new diagnostic marked-image path")
    image.save(output)
    image.close()
    return {"kind":"legible_marking_and_PIL_pre_resampling_variant_not_accuracy_claim",
            "source_image_sha256":hashlib.sha256(source_bytes).hexdigest(),
            "output_image_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
            "original_size":[width,height], "model_spatial_size":[target_w,target_h],
            "min_pixels":min_pixels, "max_pixels":max_pixels, "patch_size":16,"merge_size":2,
            "resampling":"PIL_BICUBIC_before_marking", "patch_border_width_px":2,
            "point_cross_radius_px":3, "requested_font_size":20,
            "labels_used":False,"instruction_used":False,"candidates":rows}
