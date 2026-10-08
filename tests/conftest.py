from __future__ import annotations

import pytest

from damagescope.config import Settings
from damagescope.generate import make_dataset, make_sample
from damagescope.model import DamageClassifier


@pytest.fixture(scope="session")
def dataset():
    return make_dataset(n=200, size=128, seed=42)


@pytest.fixture(scope="session")
def trained_model(dataset):
    images, labels, _ = dataset
    model = DamageClassifier(size=128, params={"n_estimators": 120})
    model.train(images, labels)
    return model


@pytest.fixture
def damaged_sample():
    return make_sample(seed=9001, size=128, damaged=True)


@pytest.fixture
def clean_sample():
    return make_sample(seed=9002, size=128, damaged=False)


@pytest.fixture
def tmp_settings(tmp_path):
    return Settings(model_dir=str(tmp_path / "artifacts"), image_size=128)
