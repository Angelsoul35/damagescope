"""DamageScope HTTP API."""

from __future__ import annotations

import io

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from .service import InspectionService

app = FastAPI(
    title="DamageScope",
    version="1.0.0",
    summary="Detect vehicle damage in a photo, localize it, and estimate severity.",
)

# Lazily built so importing the app does not train a model.
_service: InspectionService | None = None


def get_service() -> InspectionService:
    global _service
    if _service is None:
        _service = InspectionService()
    return _service


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/model", tags=["model"])
def model_info() -> dict:
    return get_service().model_info()


@app.post("/inspect", tags=["inspect"])
async def inspect(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=422, detail="could not read image") from exc
    array = np.asarray(image, dtype=float) / 255.0
    return get_service().inspect(array)
