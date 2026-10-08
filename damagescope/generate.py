"""Synthetic vehicle-panel images with and without damage.

Everything is synthetic. A "panel" is a smooth gradient with a little noise and
some broad specular highlights (reflections) — those are clean features the model
must learn to ignore. Damage is added as thin scratches (bright or dark lines) and
dents (shaded elliptical blobs). We keep the true damage mask and bounding boxes so
we can score both the classifier and the localizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Sample:
    image: np.ndarray  # HxW float in [0, 1], grayscale
    label: int  # 1 = damaged, 0 = clean
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)  # (r0, c0, r1, c1)
    mask: np.ndarray | None = None  # bool HxW, True where damage is


def _panel(size: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size] / size
    base = 0.45 + 0.15 * xx + 0.10 * np.sin(3 * yy + rng.uniform(0, 6))
    # Broad specular highlights: smooth gaussian bumps (clean, not damage).
    for _ in range(int(rng.integers(0, 3))):
        cy, cx = rng.uniform(0, size, 2)
        s = rng.uniform(size * 0.18, size * 0.32)
        bump = np.exp(-(((yy * size - cy) ** 2 + (xx * size - cx) ** 2) / (2 * s**2)))
        base += 0.18 * bump
    base += rng.normal(0, 0.01, (size, size))
    return np.clip(base, 0.0, 1.0)


def _add_scratch(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    size = img.shape[0]
    r0, c0 = int(rng.integers(0, size)), int(rng.integers(0, size))
    angle = rng.uniform(0, np.pi)
    length = int(rng.integers(size // 4, size // 2))
    steps = np.arange(length)
    rr = (r0 + steps * np.sin(angle)).astype(int)
    cc = (c0 + steps * np.cos(angle)).astype(int)
    valid = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
    rr, cc = rr[valid], cc[valid]
    if len(rr) == 0:
        return None
    value = float(rng.choice([-0.45, 0.45]))
    for dr in (-1, 0, 1):
        r2 = np.clip(rr + dr, 0, size - 1)
        img[r2, cc] = np.clip(img[r2, cc] + value, 0, 1)
        mask[r2, cc] = True
    return int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1


def _add_dent(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    size = img.shape[0]
    cy, cx = int(rng.integers(size * 0.2, size * 0.8)), int(rng.integers(size * 0.2, size * 0.8))
    ry, rx = int(rng.integers(5, 14)), int(rng.integers(5, 14))
    yy, xx = np.mgrid[0:size, 0:size]
    dist = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    region = dist <= 1.0
    img[region] = np.clip(img[region] - 0.25 * (1 - dist[region]), 0, 1)
    ring = (dist > 0.8) & (dist <= 1.3)
    img[ring] = np.clip(img[ring] + 0.15, 0, 1)
    mask |= region
    rr, cc = np.where(region)
    if len(rr) == 0:
        return None
    return int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1


def make_sample(seed: int, size: int = 128, damaged: bool | None = None) -> Sample:
    rng = np.random.default_rng(seed)
    if damaged is None:
        damaged = bool(rng.integers(0, 2))
    img = _panel(size, rng)
    mask = np.zeros((size, size), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    if damaged:
        n_scratch = int(rng.integers(0, 3))
        n_dent = int(rng.integers(0, 2))
        if n_scratch + n_dent == 0:
            n_scratch = 1
        for _ in range(n_scratch):
            box = _add_scratch(img, mask, rng)
            if box:
                boxes.append(box)
        for _ in range(n_dent):
            box = _add_dent(img, mask, rng)
            if box:
                boxes.append(box)
    return Sample(image=img, label=int(damaged), boxes=boxes, mask=mask)


def make_dataset(n: int = 400, size: int = 128, seed: int = 42):
    """Return (images, labels, samples) with a balanced damaged/clean split."""
    images, labels, samples = [], [], []
    for i in range(n):
        damaged = i % 2 == 0
        sample = make_sample(seed=seed + i, size=size, damaged=damaged)
        images.append(sample.image)
        labels.append(sample.label)
        samples.append(sample)
    return images, np.array(labels), samples
