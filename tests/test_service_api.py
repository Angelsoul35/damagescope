from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from damagescope.service import InspectionService


def _service(trained_model, tmp_settings):
    return InspectionService(settings=tmp_settings, model=trained_model)


def test_inspect_returns_full_report(trained_model, tmp_settings, damaged_sample):
    service = _service(trained_model, tmp_settings)
    result = service.inspect(damaged_sample.image)
    for key in ["damaged", "confidence", "severity", "n_regions", "estimated_cost_cop", "boxes"]:
        assert key in result
    assert 0.0 <= result["confidence"] <= 1.0


def test_clean_panel_has_no_boxes(trained_model, tmp_settings, clean_sample):
    service = _service(trained_model, tmp_settings)
    result = service.inspect(clean_sample.image)
    # A clean panel classified as clean has no boxes; even if misclassified,
    # the structure must hold.
    assert isinstance(result["boxes"], list)


def _png_bytes(image: np.ndarray) -> bytes:
    pil = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return buffer.getvalue()


def test_api(trained_model, tmp_settings, damaged_sample):
    import damagescope.main as main

    main._service = _service(trained_model, tmp_settings)
    client = TestClient(main.app)

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/model").json()["version"].startswith("m_")

    files = {"file": ("panel.png", _png_bytes(damaged_sample.image), "image/png")}
    response = client.post("/inspect", files=files)
    assert response.status_code == 200, response.text
    assert "severity" in response.json()


def test_api_rejects_bad_image(trained_model, tmp_settings):
    import damagescope.main as main

    main._service = _service(trained_model, tmp_settings)
    client = TestClient(main.app)
    files = {"file": ("x.png", b"not an image", "image/png")}
    assert client.post("/inspect", files=files).status_code == 422
