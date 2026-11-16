
"""Privacy processing for images sent to hosted vision models."""

from pathlib import Path

import cv2


def blur_region(image, x1, y1, x2, y2):
    """Apply strong Gaussian blur to a region of an image."""

    height, width = image.shape[:2]

    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))

    if x2 <= x1 or y2 <= y1:
        return image

    region = image[y1:y2, x1:x2]

    # Use a strong blur to obscure identifiable details.
    blurred = cv2.GaussianBlur(region, (51, 51), 30)

    image[y1:y2, x1:x2] = blurred

    return image


def get_pedestrian_boxes(annotation_path, frame_id):
    """Read pedestrian bounding boxes for a particular EMT frame.

    EMT annotation format:
    frame, track_id, class, ..., bbox_left, bbox_top,
    bbox_right, bbox_bottom, ...
    """

    annotation_path = Path(annotation_path)

    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_path}"
        )

    boxes = []

    with annotation_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            values = line.strip().split()

            if not values:
                continue

            if len(values) != 17:
                raise ValueError(
                    f"Expected 17 annotation columns at "
                    f"line {line_number}, found {len(values)}."
                )

            frame = int(float(values[0]))

            if frame != frame_id:
                continue

            object_class = values[2].lower()

            if object_class not in {"pedestrian", "person"}:
                continue

            x1, y1, x2, y2 = map(float, values[6:10])

            boxes.append((x1, y1, x2, y2))

    return boxes


def blur_pedestrians(image, pedestrian_boxes):
    """Blur every pedestrian region supplied by EMT annotations."""

    blurred_count = 0

    for x1, y1, x2, y2 in pedestrian_boxes:
        blur_region(image, x1, y1, x2, y2)
        blurred_count += 1

    return image, blurred_count


def anonymize_image(
    image_path: str | Path,
    output_path: str | Path,
    pedestrian_boxes=None,
) -> dict:
    """Create a privacy-processed image without modifying the original."""

    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    image, pedestrian_count = blur_pedestrians(
        image,
        pedestrian_boxes or [],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), image)

    if not success:
        raise RuntimeError(
            f"Could not save privacy-processed image: {output_path}"
        )

    return {
        "output_path": output_path,
        "pedestrians_blurred": pedestrian_count,
        "faces_blurred": 0,
        "plate_candidates_blurred": 0,
    }
