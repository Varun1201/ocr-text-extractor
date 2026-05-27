import pytest
import numpy as np
from PIL import Image
import io
import cv2


def create_test_image(text="Hello World"):
    """
    Create a simple white image with black text for testing.
    """
    img = np.ones((100, 300), dtype=np.uint8) * 255
    cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img


def test_image_creation():
    """
    Test that we can create a test image successfully.
    """
    img = create_test_image()
    assert img is not None
    assert img.shape == (100, 300)
    assert img.dtype == np.uint8


def test_image_preprocessing():
    """
    Test that preprocessing converts image to grayscale correctly.
    """
    img = create_test_image()

    # Should be grayscale already
    assert len(img.shape) == 2

    # Pixel values should be 0-255
    assert img.min() >= 0
    assert img.max() <= 255


def test_threshold():
    """
    Test that thresholding produces binary image.
    """
    img = create_test_image()
    _, thresh = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Binary image should only have 0 and 255
    unique_values = np.unique(thresh)
    assert all(v in [0, 255] for v in unique_values)


def test_pil_conversion():
    """
    Test converting numpy array to PIL Image.
    """
    img = create_test_image()
    pil_image = Image.fromarray(img)

    assert isinstance(pil_image, Image.Image)
    assert pil_image.size == (300, 100)


def test_image_to_bytes():
    """
    Test converting image to bytes and back.
    """
    img = create_test_image()
    pil_image = Image.fromarray(img)

    # Convert to bytes
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    assert isinstance(img_bytes, bytes)
    assert len(img_bytes) > 0

    # Convert back
    restored = Image.open(io.BytesIO(img_bytes))
    assert restored.size == (300, 100)