from __future__ import annotations

from damagescope.model import DamageClassifier, latest_version


def test_training_metrics(trained_model):
    assert trained_model.version.startswith("m_")
    assert trained_model.metadata["metrics"]["roc_auc"] > 0.8


def test_predict_proba_is_probability(trained_model, damaged_sample):
    p = trained_model.predict_proba(damaged_sample.image)
    assert 0.0 <= p <= 1.0


def test_detection_accuracy_over_batch(trained_model):
    from damagescope.generate import make_sample

    correct = 0
    for i in range(20):
        damaged = i % 2 == 0
        sample = make_sample(seed=20000 + i, size=128, damaged=damaged)
        correct += int(trained_model.predict(sample.image) == int(damaged))
    assert correct / 20 >= 0.7  # the model is clearly better than chance


def test_save_and_load(trained_model, tmp_path):
    trained_model.save(tmp_path)
    reloaded = DamageClassifier.load(tmp_path, trained_model.version)
    assert reloaded.version == trained_model.version
    assert latest_version(tmp_path) == trained_model.version


def test_onnx_export_matches_sklearn(trained_model, tmp_path, damaged_sample):
    import onnxruntime as ort

    from damagescope.features import extract_features
    from damagescope.onnx_infer import OnnxDamageModel

    onnx_path = trained_model.export_onnx(tmp_path / "model.onnx")
    assert onnx_path.exists()

    onnx_model = OnnxDamageModel(onnx_path, size=128)
    sk_proba = trained_model.predict_proba(damaged_sample.image)
    onnx_proba = onnx_model.predict_proba(damaged_sample.image)
    assert abs(sk_proba - onnx_proba) < 0.02

    # the session input is the HOG feature vector
    session = ort.InferenceSession(str(onnx_path))
    n_features = extract_features(damaged_sample.image, 128).shape[0]
    assert session.get_inputs()[0].shape[1] == n_features
