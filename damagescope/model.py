"""The damage classifier: HOG features + a random forest, with ONNX export.

A random forest on HOG features is a strong, fast baseline for "is this panel
damaged?", and it exports cleanly to ONNX so inference can run on CPU with
onnxruntime (no scikit-learn needed at serving time).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import extract_features, features_matrix

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": 1,
}


@dataclass
class TrainingReport:
    version: str
    trained_at: str
    n_samples: int
    metrics: dict[str, float]


class DamageClassifier:
    def __init__(self, size: int = 128, params: dict[str, Any] | None = None) -> None:
        self.size = size
        self.params = {**_DEFAULT_PARAMS, **(params or {})}
        self.pipeline: Pipeline | None = None
        self.version: str | None = None
        self.n_features: int | None = None
        self.metadata: dict[str, Any] = {}

    def build_pipeline(self) -> Pipeline:
        """Build the detector pipeline (StandardScaler + random forest), unfitted.

        This is the exact estimator :meth:`train` fits. It is public so the same classifier
        can be reused verbatim elsewhere (for example cross-validated on real photos in
        :mod:`damagescope.real_benchmark`) instead of being re-implemented.
        """
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(**self.params)),
            ]
        )

    def _build(self) -> Pipeline:
        return self.build_pipeline()

    def train(self, images: list[np.ndarray], labels: np.ndarray) -> TrainingReport:
        X = features_matrix(images, self.size)
        y = np.asarray(labels).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )
        self.pipeline = self._build()
        self.pipeline.fit(X_train, y_train)

        proba = self.pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        }
        self.n_features = int(X.shape[1])
        payload = json.dumps({"params": self.params, "n_features": self.n_features}, sort_keys=True)
        self.version = "m_" + hashlib.sha256(payload.encode()).hexdigest()[:12]
        self.metadata = {
            "version": self.version,
            "trained_at": datetime.now(UTC).isoformat(),
            "n_samples": int(len(images)),
            "image_size": self.size,
            "n_features": self.n_features,
            "metrics": metrics,
            "params": self.params,
        }
        return TrainingReport(self.version, self.metadata["trained_at"], len(images), metrics)

    def predict_proba(self, image: np.ndarray) -> float:
        if self.pipeline is None:
            raise RuntimeError("model is not trained or loaded")
        features = extract_features(image, self.size).reshape(1, -1)
        return float(self.pipeline.predict_proba(features)[0, 1])

    def predict(self, image: np.ndarray, threshold: float = 0.5) -> int:
        return int(self.predict_proba(image) >= threshold)

    def save(self, model_dir: str | Path) -> Path:
        if self.pipeline is None or self.version is None:
            raise RuntimeError("nothing to save; train first")
        target = Path(model_dir) / self.version
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, target / "model.joblib")
        (target / "metadata.json").write_text(json.dumps(self.metadata, indent=2))
        return target

    @classmethod
    def load(cls, model_dir: str | Path, version: str | None = None) -> DamageClassifier:
        base = Path(model_dir)
        resolved = version or latest_version(base)
        if resolved is None:
            raise FileNotFoundError(f"no trained model under {base}")
        path = base / resolved
        obj = cls()
        obj.pipeline = joblib.load(path / "model.joblib")
        obj.metadata = json.loads((path / "metadata.json").read_text())
        obj.version = obj.metadata["version"]
        obj.size = obj.metadata.get("image_size", obj.size)
        obj.n_features = obj.metadata.get("n_features")
        obj.params = obj.metadata.get("params", obj.params)
        return obj

    def export_onnx(self, path: str | Path) -> Path:
        """Export the scaler+classifier to ONNX (input: a HOG feature vector)."""
        if self.pipeline is None or self.n_features is None:
            raise RuntimeError("train first")
        from skl2onnx import to_onnx

        sample = np.zeros((1, self.n_features), dtype=np.float32)
        onx = to_onnx(self.pipeline, sample, options={"zipmap": False})
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(onx.SerializeToString())
        return path


def latest_version(model_dir: str | Path) -> str | None:
    base = Path(model_dir)
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and (p / "metadata.json").exists()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: json.loads((p / "metadata.json").read_text())["trained_at"])
    return candidates[-1].name
