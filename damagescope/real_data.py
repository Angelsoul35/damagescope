"""Real car-damage photos: a vendored, balanced subset for a reproducible benchmark.

The synthetic generator in :mod:`damagescope.generate` is controlled and fast, and it
proves the detect/localize/score/serve/export flow end to end. This module adds the
companion the flow is built for: **real** car photos. It reuses the project's own HOG +
random-forest pipeline (:func:`damagescope.features.features_matrix`,
:class:`damagescope.model.DamageClassifier`) without changing it, so the number the
benchmark reports is the project's real classifier measured on real images.

Source
------
``ikuldeep1/vehicle-damage-fraud-image-balanced`` on the Hugging Face Hub
(https://huggingface.co/datasets/ikuldeep1/vehicle-damage-fraud-image-balanced):
12,729 real car photos, balanced, with a ``ClassLabel`` where ``0 = no-damage`` and
``1 = damage`` (verified by eye on spread samples of each class). ``fetch_damage`` pulls a
balanced, variety-spread subset through the public datasets-server rows API, downscales
each image, and vendors it under ``datasets/<classname>/*.jpg`` so the benchmark and tests
run offline. ``load_damage`` reads that vendored subset back as grayscale float arrays in
the same ``[0, 1]`` HxW form the rest of the package uses.
"""

from __future__ import annotations

import http.client
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

# label 0 = no-damage, 1 = damage (dataset's ClassLabel order, verified by eye).
CLASS_NAMES: dict[str, int] = {"no_damage": 0, "damage": 1}
LABEL_TO_CLASS: dict[int, str] = {v: k for k, v in CLASS_NAMES.items()}

DATASET_ID = "ikuldeep1/vehicle-damage-fraud-image-balanced"
_ROWS_API = "https://datasets-server.huggingface.co/rows"
_USER_AGENT = "damagescope/1.0 (+https://github.com/Angelsoul35/damagescope)"

# The source is sorted by label; this offset is where label 1 begins (found by a binary
# search over the rows API). It only steers sampling, so the benchmark draws a varied,
# balanced subset from the whole range of each class instead of one contiguous block.
_LABEL1_START = 7614
_TOTAL_ROWS = 12729

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"


def _http_get(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    """GET ``url`` with exponential backoff (2, 4, 8, 16 s). Returns the raw body."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            # urllib.error.URLError and TimeoutError subclass OSError; IncompleteRead and
            # other partial/transport failures subclass HTTPException. Retry them all.
            last = exc
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"failed to GET {url} after {retries} tries") from last


def _fetch_rows(offset: int, length: int, retries: int = 4) -> list[dict]:
    """Return a block of rows from the datasets-server rows API."""
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET_ID,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    body = _http_get(f"{_ROWS_API}?{query}", retries=retries)
    return json.loads(body)["rows"]


def _downscale(raw: bytes, size: int) -> Image.Image:
    """Decode image bytes and resize the longer side to ``size`` (keeps aspect, RGB)."""
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    scale = size / max(image.size)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def _spread_offsets(start: int, stop: int, count: int) -> list[int]:
    """``count`` offsets spread evenly across ``[start, stop)`` for sample variety."""
    if count <= 0 or stop <= start:
        return []
    step = (stop - start) / count
    return [int(start + round(i * step)) for i in range(count)]


def fetch_damage(
    per_class: int = 200,
    size: int = 128,
    data_dir: str | Path | None = None,
    block: int = 100,
    retries: int = 4,
) -> dict[str, int]:
    """Download a balanced, variety-spread subset of real photos and vendor it on disk.

    Images are spread across the whole offset range of each class (not one contiguous
    block) so backgrounds, angles, colors and damage types vary. Each image is downscaled
    to ``size`` px on its longer side and written as ``datasets/<classname>/<name>.jpg``.
    Returns the number of images saved per class.
    """
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    plan = {
        "no_damage": _spread_offsets(0, _LABEL1_START, per_class),
        "damage": _spread_offsets(_LABEL1_START, _TOTAL_ROWS, per_class),
    }
    saved: dict[str, int] = {}
    for class_name, offsets in plan.items():
        label = CLASS_NAMES[class_name]
        out_dir = root / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        # Fetch in blocks so one request serves several target offsets.
        pending = sorted(set(offsets))
        cache: dict[int, dict] = {}
        for off in pending:
            target = out_dir / f"{class_name}_{off:05d}.jpg"
            if target.exists():  # resumable: keep images already vendored
                count += 1
                continue
            if off not in cache:
                base = (off // block) * block
                rows = _fetch_rows(base, block, retries=retries)
                for row in rows:
                    cache[row["row_idx"]] = row["row"]
            row = cache.get(off)
            if row is None or row.get("label") != label:
                continue
            src = row["image"]["src"]
            try:
                image = _downscale(_http_get(src, retries=retries), size)
            except (RuntimeError, OSError):
                continue
            image.save(target, "JPEG", quality=85)
            count += 1
        saved[class_name] = count
    return saved


def load_damage(
    data_dir: str | Path | None = None,
) -> tuple[list[np.ndarray], np.ndarray, list[Path]]:
    """Load the vendored real subset.

    Returns ``(images, labels, paths)`` where each image is a grayscale float array in
    ``[0, 1]`` (the same form :func:`damagescope.generate.make_dataset` returns), ready to
    feed straight into :func:`damagescope.features.features_matrix`.
    """
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    images: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[Path] = []
    for class_name, label in CLASS_NAMES.items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.glob("*.jpg")):
            with Image.open(path) as handle:
                gray = np.asarray(handle.convert("L"), dtype=np.float32) / 255.0
            images.append(gray)
            labels.append(label)
            paths.append(path)
    if not images:
        raise FileNotFoundError(
            f"no vendored real images under {root}; run `fetch_damage()` first "
            "(see datasets/README.md)"
        )
    return images, np.asarray(labels, dtype=int), paths


def class_counts(labels: np.ndarray) -> dict[str, int]:
    """Count images per class name, for reports and tests."""
    labels = np.asarray(labels, dtype=int)
    return {name: int((labels == value).sum()) for name, value in CLASS_NAMES.items()}
