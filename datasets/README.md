# Vendored real car-damage photos

This directory holds a small, balanced subset of **real** car photos used by the
real-data benchmark (`python -m damagescope.real_benchmark`) and its tests
(`tests/test_real_benchmark.py`). It is vendored so the benchmark is reproducible
offline, without a network round-trip at test time.

## Source

* **Dataset:** [`ikuldeep1/vehicle-damage-fraud-image-balanced`](https://huggingface.co/datasets/ikuldeep1/vehicle-damage-fraud-image-balanced)
  on the Hugging Face Hub — 12,729 real car photos, balanced across two classes.
* **Label semantics:** a `ClassLabel` with names `['0', '1']`, where **`0` = no-damage**
  and **`1` = damage**. The source is stored sorted by label (label `1` begins at row
  offset 7,614). The mapping was verified by eye on spread samples of each class before
  building the loader.
* **Access:** images are pulled through the public datasets-server rows API
  (`https://datasets-server.huggingface.co/rows?...`); each row carries a direct
  `image.src` URL to the JPEG/PNG on `datasets-server.huggingface.co/cached-assets/...`.

## What is vendored here

| Class | Directory | Label | Count |
|---|---|---|---|
| No damage | `no_damage/` | 0 | 200 |
| Damage | `damage/` | 1 | 200 |

* **400 images total**, balanced 200 / 200, ~2.6 MB on disk.
* Each image is **downscaled to 128 px on its longer side** and saved as JPEG (quality 85).
  128 px is the size the project's HOG feature extractor resizes to, so nothing the
  detector sees is lost by the downscale.
* Filenames encode the source row offset (`<class>_<offset>.jpg`) so each image is
  traceable to its source row. Offsets are spread evenly across the whole range of each
  class, so backgrounds, angles, colors and damage types vary rather than coming from one
  contiguous block.

## License and attribution

The dataset card declares **no explicit license**. This subset is a small, downscaled
sample vendored solely for reproducible benchmarking and research, with attribution to the
source above. Images remain the property of their respective owners. If you intend to
redistribute or use these images beyond local benchmarking, consult the dataset card on the
Hugging Face Hub and comply with its terms; replace this subset with your own licensed data
if needed.

## How to regenerate

From the repo root, with the package installed:

```python
from damagescope.real_data import fetch_damage
fetch_damage(per_class=200, size=128)   # writes datasets/<class>/*.jpg
```

`fetch_damage` is **resumable** — images already on disk are kept, so a re-run only tops up
what is missing and retries transient network errors with exponential backoff. To rebuild
from scratch, delete the `no_damage/` and `damage/` directories first.
