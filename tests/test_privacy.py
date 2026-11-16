
import cv2
import numpy as np
import pytest

from trackrag.privacy import anonymize_image, blur_region

def test_blur_region_changes_pixels():
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    # Create a pattern with different pixel values.
    image[50:150, 50:150] = np.random.randint(
        0, 256, (100, 100, 3), dtype=np.uint8
    )

    original = image.copy()

    result = blur_region(image, 50, 50, 150, 150)

    assert result.shape == original.shape
    assert not np.array_equal(result, original)


   

def test_blur_region_handles_invalid_coordinates():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = blur_region(image, 80, 80, 20, 20)

    assert result.shape == image.shape


def test_anonymize_image_creates_processed_copy(tmp_path):
    input_path = tmp_path / "original.jpg"
    output_path = tmp_path / "processed.jpg"

    image = np.zeros((300, 400, 3), dtype=np.uint8)

    assert cv2.imwrite(str(input_path), image)

    result = anonymize_image(input_path, output_path)

    assert output_path.exists()
    assert result["output_path"] == output_path
    assert result["faces_blurred"] >= 0
    assert result["plate_candidates_blurred"] >= 0

    # Original image must remain untouched.
    assert input_path.exists()


def test_missing_image_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        anonymize_image(
            tmp_path / "missing.jpg",
            tmp_path / "processed.jpg",
        )
