"""Inspection flow: classify -> localize -> severity -> report."""

from __future__ import annotations

import numpy as np

from .config import Settings, get_settings
from .features import to_gray_float
from .generate import make_dataset
from .localize import localize
from .model import DamageClassifier, latest_version
from .severity import assess


class InspectionService:
    def __init__(
        self,
        settings: Settings | None = None,
        model: DamageClassifier | None = None,
        train_params: dict | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = model or self._load_or_train(train_params)

    def _load_or_train(self, params: dict | None) -> DamageClassifier:
        if latest_version(self.settings.model_dir) is not None:
            return DamageClassifier.load(self.settings.model_dir)
        model = DamageClassifier(size=self.settings.image_size, params=params or {})
        images, labels, _ = make_dataset(n=300, size=self.settings.image_size, seed=42)
        model.train(images, labels)
        return model

    def model_info(self) -> dict:
        return {"version": self.model.version, "metadata": self.model.metadata}

    def inspect(self, image: np.ndarray) -> dict:
        size = self.settings.image_size
        confidence = self.model.predict_proba(image)
        damaged = confidence >= 0.5
        gray = to_gray_float(image, size)

        if damaged:
            boxes, _binary, area = localize(
                gray, self.settings.residual_sigma, self.settings.min_region_area
            )
        else:
            boxes, area = [], 0

        severity = assess(
            area,
            size * size,
            len(boxes),
            self.settings.severity_medium,
            self.settings.severity_high,
        )
        return {
            "damaged": bool(damaged),
            "confidence": round(float(confidence), 4),
            "model_version": self.model.version,
            "boxes": [{"r0": b[0], "c0": b[1], "r1": b[2], "c1": b[3]} for b in boxes],
            **severity,
        }
