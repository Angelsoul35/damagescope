from __future__ import annotations

import pytest

from damagescope.generate import make_sample
from damagescope.localize import iou, localize, mean_best_iou
from damagescope.severity import assess


def test_iou_known_values():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # half overlap on one axis
    assert iou((0, 0, 10, 10), (0, 5, 10, 15)) == pytest.approx(5 / 15, rel=1e-6)


def test_mean_best_iou_edge_cases():
    assert mean_best_iou([], []) == 1.0
    assert mean_best_iou([(0, 0, 5, 5)], []) == 0.0
    assert mean_best_iou([], [(0, 0, 5, 5)]) == 0.0


def test_localize_finds_damage():
    sample = make_sample(seed=11, size=128, damaged=True)
    boxes, binary, area = localize(sample.image)
    assert len(boxes) >= 1
    assert area > 0
    assert mean_best_iou(boxes, sample.boxes) > 0.3


def test_localize_quiet_on_clean():
    sample = make_sample(seed=12, size=128, damaged=False)
    boxes, _binary, area = localize(sample.image)
    assert area < sample.image.size * 0.02  # very little flagged on a clean panel


def test_severity_levels_and_cost():
    none = assess(0, 10000, 0)
    assert none["severity"] == "none" and none["estimated_cost_cop"] == 0

    high = assess(1000, 10000, 3, medium_threshold=0.03, high_threshold=0.08)
    assert high["severity"] == "high"
    assert high["estimated_cost_cop"] > 0

    low = assess(50, 10000, 1, medium_threshold=0.03, high_threshold=0.08)
    assert low["severity"] == "low"
