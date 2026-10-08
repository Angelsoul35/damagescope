"""CPU inference from the exported ONNX model with onnxruntime.

At serving time you do not need scikit-learn: load the ONNX model and run it.
HOG feature extraction stays in Python; the scaler and the random forest run
inside ONNX.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .features import extract_features


class OnnxDamageModel:
    def __init__(self, onnx_path: str | Path, size: int = 128) -> None:
        self.size = size
        self.session = ort.InferenceSession(str(onnx_path))
        self.input_name = self.session.get_inputs()[0].name

    def predict_proba(self, image: np.ndarray) -> float:
        features = extract_features(image, self.size).reshape(1, -1).astype(np.float32)
        outputs = self.session.run(None, {self.input_name: features})
        for out in outputs:
            arr = np.asarray(out)
            if arr.ndim == 2 and arr.shape[1] == 2:
                return float(arr[0, 1])
        return float(np.asarray(outputs[-1]).ravel()[-1])
