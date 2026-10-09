"""Functional tests for the real-photo benchmark.

These assert behaviour, not wording: the vendored real set is balanced, the project
detector generalises clearly above the 0.5 chance baseline on held-out folds and a held-out
test split, the fit verdict is well-fit, and permuting the labels collapses ROC-AUC toward
chance (which proves the signal is the real damage content, not a leak). The fetch path is
exercised through a monkeypatched ``urlopen`` that serves synthetic rows and a tiny PNG.
"""

from __future__ import annotations

import io
import json
import urllib.parse

import numpy as np
import pytest
from PIL import Image

from damagescope import real_benchmark, real_data
from damagescope.real_benchmark import (
    BASELINE_AUC,
    GENERALISATION_AUC,
    SHUFFLE_AUC_MAX,
    classify_fit,
    format_report,
    run_benchmark,
)
from damagescope.real_data import CLASS_NAMES, class_counts, load_damage


@pytest.fixture(scope="module")
def report():
    return run_benchmark()


# --- the vendored real data ------------------------------------------------------------


def test_load_damage_returns_balanced_real_set():
    images, labels, paths = load_damage()
    counts = class_counts(labels)
    assert counts["no_damage"] >= 150
    assert counts["damage"] >= 150
    # balanced: the two classes differ by no more than a few images
    assert abs(counts["no_damage"] - counts["damage"]) <= 5
    assert len(images) == len(labels) == len(paths) == sum(counts.values())


def test_loaded_images_are_grayscale_floats_in_unit_range():
    images, _labels, _paths = load_damage()
    sample = images[0]
    assert sample.ndim == 2  # grayscale HxW
    assert sample.dtype == np.float32
    assert 0.0 <= float(sample.min()) and float(sample.max()) <= 1.0


# --- generalisation --------------------------------------------------------------------


def test_detector_beats_chance_baseline_on_heldout(report):
    # Cross-validation and the untouched held-out test both clear the 0.5 baseline clearly.
    assert report.cv_roc_auc > GENERALISATION_AUC
    assert report.test_roc_auc > GENERALISATION_AUC
    assert report.cv_accuracy > BASELINE_AUC
    assert report.test_accuracy > BASELINE_AUC


def test_every_fold_beats_chance(report):
    assert len(report.fold_roc_aucs) == report.n_splits
    # each individual held-out fold is above chance, not just the average
    assert min(report.fold_roc_aucs) > 0.55


def test_verdict_is_well_fit(report):
    assert report.verdict == "well-fit"


def test_random_forest_memorises_train_but_still_generalises(report):
    # The train score is near perfect (random forest), yet CV and test agree: well-fit,
    # not overfit. The positive train-vs-CV gap must not be read as overfitting.
    assert report.train_roc_auc > 0.99
    assert report.train_cv_auc_gap > 0.0
    assert report.cv_test_auc_gap <= 0.15


def test_shuffling_labels_collapses_auc(report):
    # With labels permuted, the cross-validated ROC-AUC falls toward the chance line and
    # well below the real-label score: the learned signal is genuine, not leakage.
    assert report.shuffled_roc_auc < SHUFFLE_AUC_MAX
    assert report.shuffled_roc_auc < report.cv_roc_auc - 0.15


# --- the verdict logic in isolation ----------------------------------------------------


def test_classify_fit_well_fit():
    assert classify_fit(train_auc=1.0, cv_auc=0.80, test_auc=0.85, fold_std=0.05) == "well-fit"


def test_classify_fit_overfit():
    # Memorises train, fails to generalise -> overfit.
    assert classify_fit(train_auc=1.0, cv_auc=0.55, test_auc=0.54, fold_std=0.05) == "overfit"


def test_classify_fit_underfit():
    # Cannot even fit the training set -> underfit.
    assert classify_fit(train_auc=0.58, cv_auc=0.55, test_auc=0.54, fold_std=0.05) == "underfit"


def test_classify_fit_disagreeing_folds_not_well_fit():
    # Strong averages but folds that disagree widely are not accepted as well-fit.
    assert classify_fit(train_auc=1.0, cv_auc=0.80, test_auc=0.82, fold_std=0.30) != "well-fit"


# --- the fetch path, with a monkeypatched network --------------------------------------


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc) -> bool:
        return False


def _tiny_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (110, 120, 130)).save(buffer, format="PNG")
    return buffer.getvalue()


def _fake_urlopen_factory(png: bytes):
    def _fake_urlopen(req, timeout=60):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "/rows?" in url:
            params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            offset = int(params["offset"][0])
            length = int(params["length"][0])
            rows = []
            for idx in range(offset, offset + length):
                label = 0 if idx < real_data._LABEL1_START else 1
                src = f"https://datasets-server.huggingface.co/cached-assets/fake/{idx}.png"
                rows.append({"row_idx": idx, "row": {"image": {"src": src}, "label": label}})
            return _FakeResponse(json.dumps({"rows": rows}).encode())
        return _FakeResponse(png)

    return _fake_urlopen


def test_fetch_damage_with_monkeypatched_network(monkeypatch, tmp_path):
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory(_tiny_png()))

    saved = real_data.fetch_damage(per_class=3, size=32, data_dir=tmp_path)
    assert saved == {"no_damage": 3, "damage": 3}

    images, labels, paths = load_damage(tmp_path)
    assert class_counts(labels) == {"no_damage": 3, "damage": 3}
    assert len(images) == 6
    # the saved files land under the expected class folders
    for class_name in CLASS_NAMES:
        assert (tmp_path / class_name).is_dir()
        assert any(p.parent.name == class_name for p in paths)


def test_fetch_damage_is_resumable(monkeypatch, tmp_path):
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory(_tiny_png()))
    first = real_data.fetch_damage(per_class=2, size=32, data_dir=tmp_path)
    assert first == {"no_damage": 2, "damage": 2}

    # On a second run every target already exists; nothing is downloaded again.
    def _boom(*_a, **_k):
        raise AssertionError("urlopen must not be called when images already exist")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    second = real_data.fetch_damage(per_class=2, size=32, data_dir=tmp_path)
    assert second == {"no_damage": 2, "damage": 2}


def test_http_get_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls = {"n": 0}

    def _flaky_urlopen(req, timeout=60):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("transient")
        return _FakeResponse(b"ok")

    monkeypatch.setattr("urllib.request.urlopen", _flaky_urlopen)
    assert real_data._http_get("https://example.test/x", retries=4) == b"ok"
    assert calls["n"] == 3


def test_load_damage_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_damage(tmp_path / "does-not-exist")


# --- reporting -------------------------------------------------------------------------


def test_format_report_contains_key_figures(report):
    text = format_report(report)
    assert "WELL-FIT" in text
    assert "ROC-AUC" in text
    assert f"no_damage={report.counts['no_damage']}" in text
    assert "shuffled-label" in text


def test_main_prints_report(report, monkeypatch, capsys):
    # Reuse the already-computed report so main() does not retrain.
    monkeypatch.setattr(real_benchmark, "run_benchmark", lambda: report)
    real_benchmark.main()
    out = capsys.readouterr().out
    assert "damage detection benchmark" in out
    assert report.verdict.upper() in out
