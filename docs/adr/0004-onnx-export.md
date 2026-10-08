# ADR-0004: Export the model to ONNX for CPU inference

* Status: accepted
* Date: 2026-02-25

## Context

At serving time we want light, fast inference without carrying the whole training stack, and
the option to run the model in other runtimes or languages.

## Decision

We export the scaler + random forest to ONNX with `skl2onnx`. Serving loads the ONNX file
with `onnxruntime`; scikit-learn is not needed. HOG stays in Python. A test checks the ONNX
output matches scikit-learn.

## Consequences

* **Good:** a small, portable model file; fast CPU inference; a standard format other systems
  can run. The parity test guards against silent export bugs.
* **Cost:** two code paths (scikit-learn for training, ONNX for serving) and a check to keep
  them in sync — which the parity test provides.
