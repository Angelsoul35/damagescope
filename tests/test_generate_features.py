from __future__ import annotations

import numpy as np

from damagescope.features import extract_features, to_gray_float
from damagescope.generate import make_dataset, make_sample


def test_damaged_sample_has_boxes_and_mask():
    sample = make_sample(seed=1, size=128, damaged=True)
    assert sample.label == 1
    assert len(sample.boxes) >= 1
    assert sample.mask is not None and sample.mask.any()


def test_clean_sample_has_no_damage():
    sample = make_sample(seed=2, size=128, damaged=False)
    assert sample.label == 0
    assert sample.boxes == []
    assert not sample.mask.any()


def test_dataset_is_balanced():
    _images, labels, _samples = make_dataset(n=100, size=64, seed=5)
    assert labels.sum() == 50  # half damaged


def test_to_gray_float_handles_rgb_and_uint8():
    rgb_uint8 = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    gray = to_gray_float(rgb_uint8, 128)
    assert gray.shape == (128, 128)
    assert 0.0 <= gray.min() and gray.max() <= 1.0


def test_extract_features_is_consistent():
    sample = make_sample(seed=3, size=128, damaged=True)
    f1 = extract_features(sample.image, 128)
    f2 = extract_features(sample.image, 128)
    assert f1.shape == f2.shape
    assert np.allclose(f1, f2)
