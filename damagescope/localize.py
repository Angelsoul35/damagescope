"""Locate damage regions and draw bounding boxes (classical image processing).

A clean panel is smooth, so most of the image equals a blurred copy of itself.
Damage is high-frequency: a scratch or a dent differs sharply from the blur. We
take the absolute difference (the "residual"), threshold it, group the bright
pixels into connected regions, and return a bounding box per region. We also score
the result against the true damage boxes with Intersection over Union (IoU).
"""

from __future__ import annotations

import numpy as np
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk


def residual_map(image: np.ndarray, sigma_blur: float = 4.0) -> np.ndarray:
    smooth = gaussian(image, sigma=sigma_blur)
    return np.abs(image - smooth)


def localize(
    image: np.ndarray, residual_sigma: float = 3.0, min_area: int = 20
) -> tuple[list[tuple[int, int, int, int]], np.ndarray, int]:
    """Return (boxes, binary damage map, total damage area in pixels)."""
    residual = residual_map(image)
    threshold = residual.mean() + residual_sigma * residual.std()
    binary = residual > threshold
    binary = closing(binary, disk(2))

    labelled = label(binary)
    boxes: list[tuple[int, int, int, int]] = []
    total_area = 0
    for region in regionprops(labelled):
        if region.area >= min_area:
            r0, c0, r1, c1 = region.bbox
            boxes.append((int(r0), int(c0), int(r1), int(c1)))
            total_area += int(region.area)
    return boxes, binary, total_area


def iou(box_a: tuple, box_b: tuple) -> float:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    inter_r0, inter_c0 = max(ar0, br0), max(ac0, bc0)
    inter_r1, inter_c1 = min(ar1, br1), min(ac1, bc1)
    inter = max(0, inter_r1 - inter_r0) * max(0, inter_c1 - inter_c0)
    if inter == 0:
        return 0.0
    area_a = (ar1 - ar0) * (ac1 - ac0)
    area_b = (br1 - br0) * (bc1 - bc0)
    return inter / (area_a + area_b - inter)


def mean_best_iou(pred_boxes: list[tuple], true_boxes: list[tuple]) -> float:
    """Mean over true boxes of the best IoU against any predicted box."""
    if not true_boxes:
        return 1.0 if not pred_boxes else 0.0
    if not pred_boxes:
        return 0.0
    scores = [max(iou(t, p) for p in pred_boxes) for t in true_boxes]
    return float(np.mean(scores))
