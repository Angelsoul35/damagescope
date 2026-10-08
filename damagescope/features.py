"""Image features for the damage classifier.

We use the Histogram of Oriented Gradients (HOG). HOG describes the local shape
of an image by the directions of its edges. It is a good fit here: a scratch or a
dent adds strong, oriented edges that a smooth panel does not have, so HOG makes
damaged and clean panels easy to tell apart without a deep network.
"""

from __future__ import annotations

import numpy as np
from skimage.feature import hog
from skimage.transform import resize

ORIENTATIONS = 8
PIXELS_PER_CELL = (16, 16)
CELLS_PER_BLOCK = (2, 2)


def to_gray_float(image: np.ndarray, size: int) -> np.ndarray:
    """Normalise an image to a grayscale float array of shape (size, size)."""
    arr = np.asarray(image, dtype=float)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.max() > 1.0:
        arr = arr / 255.0
    if arr.shape != (size, size):
        arr = resize(arr, (size, size), anti_aliasing=True)
    return np.clip(arr, 0.0, 1.0)


def extract_features(image: np.ndarray, size: int = 128) -> np.ndarray:
    gray = to_gray_float(image, size)
    return hog(
        gray,
        orientations=ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        feature_vector=True,
    ).astype(np.float32)


def features_matrix(images: list[np.ndarray], size: int = 128) -> np.ndarray:
    return np.array([extract_features(img, size) for img in images], dtype=np.float32)
