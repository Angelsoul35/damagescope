# DamageScope

**Vehicle damage detection with computer vision.**

DamageScope looks at a photo of a car panel and answers three questions: is there damage,
where is it, and how bad is it. It flags damage with a classifier, draws bounding boxes
around the damaged areas, estimates severity and a rough repair cost, and routes the
result into an insurance-style report. The trained model exports to ONNX so inference runs
on plain CPU.

> The built-in generator makes **synthetic** panels (scratches and dents) for a fast,
> controlled end-to-end run. The damage detector is **also validated on real car photos** —
> see [Validated on real car-damage photos](#validated-on-real-car-damage-photos).

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Why it exists

A car insurer gets thousands of damage photos. A first pass that says "damaged / not
damaged, here is where, roughly this severity" saves an adjuster a lot of time. This
project does that first pass with **classical computer vision**, which is transparent and
verifiable end to end: it needs no GPU, runs fast, and every step can be explained.

## How it works

```mermaid
flowchart LR
    A[Panel image] --> B[HOG features]
    B --> C[Classifier<br/>damaged?]
    C -->|yes| D[Localize<br/>residual + regions]
    D --> E[Bounding boxes]
    D --> F[Severity + cost]
    C -->|no| G[No damage]
```

* **Detect.** The image is described with **HOG** features (the directions of its edges), and
  a random forest decides damaged or clean. A scratch or dent adds strong oriented edges that
  a smooth panel does not have, so the two are easy to separate.
* **Localize.** A clean panel is smooth, so it almost equals a blurred copy of itself. Damage
  is high-frequency, so it stands out in the difference ("residual"). We threshold the
  residual, group the bright pixels into regions, and draw a box per region.
* **Score.** Severity comes from the damaged-area fraction and the number of regions; a simple
  model turns that into a rough repair cost.

The classifier and the localizer are both measured: ROC-AUC for detection, and **IoU**
(Intersection over Union) against the true damage boxes for localization.

## Results on the synthetic set

```
detection : accuracy 0.85  precision 0.93  recall 0.76  ROC-AUC 0.91
localization: mean IoU ~0.80
ONNX inference matches scikit-learn exactly
```

## Validated on real car-damage photos

The same detector — the project's HOG features into the random-forest classifier, unchanged —
is benchmarked on **real** car photos from the public
[`ikuldeep1/vehicle-damage-fraud-image-balanced`](https://huggingface.co/datasets/ikuldeep1/vehicle-damage-fraud-image-balanced)
dataset. A balanced subset (200 no-damage / 200 damage) is vendored under
[`datasets/`](datasets/README.md) so the benchmark runs offline and reproducibly.

Measured with **5-fold stratified cross-validation plus a held-out test split**
(`python -m damagescope.real_benchmark`):

```
cross-val    : accuracy 0.717  ROC-AUC 0.770  F1 0.708   (per-fold AUC std 0.061)
held-out test: accuracy 0.800  ROC-AUC 0.867  F1 0.792
baseline     : chance ROC-AUC 0.50   (shuffled-label CV AUC 0.461)
verdict      : WELL-FIT
```

* **Clearly above chance.** Both cross-validation (0.77) and the untouched held-out split
  (0.87) beat the 0.50 balanced baseline by a wide margin, and the two agree, so the result
  carries from the folds to fresh data.
* **Well-fit, not memorised.** The random forest reaches ~1.0 ROC-AUC on its own training
  set by construction; generalisation is judged by CV / held-out **agreement**, which holds.
* **Leakage-free signal.** Permuting the labels collapses cross-validated ROC-AUC to 0.461,
  right on the chance line — proof the detector keys on real damage content, not an artefact.

A trained CNN detector or segmenter would **extend** this further on messy real photos
(varied backgrounds, lighting, dirt, many damage types); these numbers are the verifiable
foundation it builds on, measured with the same ROC-AUC the next model is scored by.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
make install
make demo          # train, inspect a damaged and a clean panel, score localization
make test          # full test suite
make train         # train and save the model
make export        # export the model to ONNX
make run           # serve the API on http://localhost:8000 (docs at /docs)
```

### The API

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/model` | Active model version and metrics |
| `POST` | `/inspect` | Upload a panel image; get damage, boxes, severity, cost |

```bash
curl -s -F "file=@panel.png" http://localhost:8000/inspect
```

## ONNX for CPU

The scaler and random forest export to **ONNX** (`make export`), so serving needs only
`onnxruntime`, not scikit-learn. HOG feature extraction stays in Python; the model runs in
ONNX. A test checks that the ONNX output matches scikit-learn exactly.

## Project layout

```
damagescope/
  generate.py   synthetic panels with scratches/dents (+ true boxes and mask)
  features.py   HOG features
  model.py      random forest classifier + ONNX export
  localize.py   residual-based localization, bounding boxes, IoU
  severity.py   severity level and a rough repair cost
  onnx_infer.py CPU inference with onnxruntime
  service.py    the flow (classify -> localize -> severity)
  main.py       FastAPI endpoints
  cli.py        train / export / demo
  real_data.py      load/fetch the vendored real car photos
  real_benchmark.py benchmark the detector on real photos (CV + held-out)
```

## Testing and quality

* **36 tests** cover the generator, features, the classifier, ONNX parity, localization and
  IoU, severity, the service, the API, and the real-photo benchmark (including a
  shuffled-label leakage check and a monkeypatched fetch path). CI requires at least 80%
  coverage.
* `ruff` for lint, `bandit` and `gitleaks` for security.

See [`docs/architecture.md`](docs/architecture.md), the [ADRs](docs/adr/), and the Spanish
study guide in [`docs/estudio/`](docs/estudio/).

## Scope and how to extend it

DamageScope is a complete, verifiable detect → localize → score → serve → export pipeline,
measured on both synthetic panels and real car photos. The natural next step is a trained
deep model (a CNN detector or segmenter) on large real labelled sets, to push accuracy on
messy photos (varied backgrounds, lighting, dirt, many damage types), add per-pixel
segmentation, and fit a data-driven cost model. Because every stage is measured with
standard metrics (ROC-AUC for detection, IoU for localization), any such extension is
compared head-to-head against a rigorous, leakage-free starting point.

## License

[MIT](LICENSE) © Angelo David Padilla Romero
