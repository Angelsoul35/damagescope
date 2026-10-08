# ADR-0002: Use a classical-CV baseline (HOG + random forest)

* Status: accepted
* Date: 2026-02-24

## Context

Damage detection is usually done with a deep model (a CNN detector or segmenter). That needs
a GPU, a large labelled dataset of real photos, and long training. For a first version — and
for a project that must run and be verified anywhere — that is too much.

## Decision

We build a classical-CV baseline: HOG features and a random forest for detection, and
residual-based image processing for localization. It needs no GPU, trains in seconds, and
every step is explainable.

## Consequences

* **Good:** fast, runs on CPU, fully reproducible, and honest about what it does. It is a
  real, measurable baseline (ROC-AUC, IoU) that a deep model must beat.
* **Cost:** on real photos (backgrounds, lighting, many damage types) a deep model would do
  better. The README says so. The architecture — detect, localize, score, serve, export —
  carries over to a deep model later.
