# ADR-0003: Localize damage with a residual map, not a trained detector

* Status: accepted
* Date: 2026-02-25

## Context

We need bounding boxes around the damage. A trained object detector needs box labels and a
lot of data. We have a simpler signal: a clean panel is smooth.

## Decision

We localize with image processing: residual = |image − blurred image|, threshold, group into
connected regions, box each region. Broad reflections are smooth and are ignored by the
residual. We score the boxes with IoU against the true boxes.

## Consequences

* **Good:** no box labels or training needed for localization; it is fast and explainable;
  and it reaches a solid IoU on the synthetic set.
* **Cost:** it assumes damage is higher-frequency than the panel, which holds here but not for
  every real case (e.g. a large smooth dent). A learned detector is the next step; the IoU
  score gives a number to improve against.
