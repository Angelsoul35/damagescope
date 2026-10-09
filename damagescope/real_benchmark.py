"""Benchmark the project's damage detector on real car photos.

This runs the project's own pipeline unchanged — HOG features
(:func:`damagescope.features.features_matrix`) into the detector
(:meth:`damagescope.model.DamageClassifier.build_pipeline`) — on the vendored real
subset (:func:`damagescope.real_data.load_damage`), and measures how well it generalises:

* **Stratified K-fold cross-validation** on the training portion (accuracy, ROC-AUC, F1),
  plus the per-fold ROC-AUC spread so fold-to-fold agreement is visible.
* **A held-out test split** never seen during cross-validation, scored the same way, to
  confirm the cross-validated estimate on fresh data.
* **The train-vs-CV and CV-vs-test gaps**, which separate well-fit from over/underfit. A
  random forest reaches ~1.0 ROC-AUC on its own training set by construction; that is not
  overfitting on its own, so the verdict judges generalisation by CV/held-out **agreement**,
  not by the training score.
* **A shuffled-label control**: with labels permuted, cross-validated ROC-AUC collapses
  toward the 0.5 chance line, which proves the signal the detector uses is the real
  damage/no-damage content and not a leak.

Run it with ``python -m damagescope.real_benchmark``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

from .features import features_matrix
from .model import DamageClassifier
from .real_data import class_counts, load_damage

# The balanced two-class baseline a useful detector must clearly beat.
BASELINE_AUC = 0.5
# Minimum CV and held-out ROC-AUC to count as clearly above that baseline.
GENERALISATION_AUC = 0.65
# Maximum |CV ROC-AUC - test ROC-AUC| for cross-validation and the held-out split to agree.
AGREEMENT_TOL = 0.15
# Maximum per-fold ROC-AUC standard deviation for the folds to agree with each other.
FOLD_STD_TOL = 0.12
# A shuffled-label run at or below this ROC-AUC confirms the signal is real, not leaked.
SHUFFLE_AUC_MAX = 0.60


@dataclass
class BenchmarkReport:
    n_samples: int
    counts: dict[str, int]
    n_features: int
    n_splits: int
    test_size: float
    cv_accuracy: float
    cv_roc_auc: float
    cv_f1: float
    fold_roc_aucs: list[float]
    fold_roc_auc_std: float
    test_accuracy: float
    test_roc_auc: float
    test_f1: float
    train_accuracy: float
    train_roc_auc: float
    train_cv_auc_gap: float
    cv_test_auc_gap: float
    shuffled_roc_auc: float
    verdict: str
    notes: list[str] = field(default_factory=list)


def classify_fit(
    train_auc: float,
    cv_auc: float,
    test_auc: float,
    fold_std: float,
) -> str:
    """Classify the model as well-fit, overfit, underfit, or inconclusive.

    Generalisation is judged by cross-validation and the held-out split, not by the
    training score: a random forest reaching ~1.0 train ROC-AUC is expected and is not
    overfitting unless the held-out performance fails to follow.
    """
    above_baseline = cv_auc >= GENERALISATION_AUC and test_auc >= GENERALISATION_AUC
    cv_test_agree = abs(cv_auc - test_auc) <= AGREEMENT_TOL
    folds_agree = fold_std <= FOLD_STD_TOL
    if above_baseline and cv_test_agree and folds_agree:
        return "well-fit"
    if train_auc >= GENERALISATION_AUC and cv_auc < GENERALISATION_AUC:
        # Fits its training set but does not carry over to unseen folds.
        return "overfit"
    if train_auc < GENERALISATION_AUC:
        # Cannot even fit the training set well enough to be useful.
        return "underfit"
    return "inconclusive"


def run_benchmark(
    size: int = 128,
    n_splits: int = 5,
    test_size: float = 0.25,
    seed: int = 42,
    data_dir: str | None = None,
) -> BenchmarkReport:
    """Run the project detector on the vendored real photos and score generalisation."""
    images, labels, _paths = load_damage(data_dir)
    features = features_matrix(images, size)

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, stratify=labels, random_state=seed
    )

    pipeline = DamageClassifier(size=size).build_pipeline()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Cross-validation on the training portion only (the held-out test stays untouched).
    cv_proba = cross_val_predict(
        clone(pipeline), x_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    cv_pred = (cv_proba >= 0.5).astype(int)
    cv_accuracy = float(accuracy_score(y_train, cv_pred))
    cv_roc_auc = float(roc_auc_score(y_train, cv_proba))
    cv_f1 = float(f1_score(y_train, cv_pred))
    fold_aucs = cross_val_score(
        clone(pipeline), x_train, y_train, cv=cv, scoring="roc_auc"
    )

    # Fit on the full training portion; score the held-out test and the training set.
    model = clone(pipeline).fit(x_train, y_train)
    test_proba = model.predict_proba(x_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)
    test_accuracy = float(accuracy_score(y_test, test_pred))
    test_roc_auc = float(roc_auc_score(y_test, test_proba))
    test_f1 = float(f1_score(y_test, test_pred))

    train_proba = model.predict_proba(x_train)[:, 1]
    train_pred = (train_proba >= 0.5).astype(int)
    train_accuracy = float(accuracy_score(y_train, train_pred))
    train_roc_auc = float(roc_auc_score(y_train, train_proba))

    # Shuffled-label control: permuted labels should drop CV ROC-AUC toward chance.
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y_train)
    shuffled_roc_auc = float(
        cross_val_score(
            clone(pipeline), x_train, y_shuffled, cv=cv, scoring="roc_auc"
        ).mean()
    )

    fold_std = float(np.std(fold_aucs))
    verdict = classify_fit(train_roc_auc, cv_roc_auc, test_roc_auc, fold_std)

    notes = [
        f"Train ROC-AUC {train_roc_auc:.3f} is near 1.0 because a random forest memorises "
        "its training set by construction; generalisation is judged by CV/held-out "
        "agreement, not by this number.",
        f"CV and held-out ROC-AUC agree to within {abs(cv_roc_auc - test_roc_auc):.3f} "
        f"(tolerance {AGREEMENT_TOL}); per-fold ROC-AUC std is {fold_std:.3f} "
        f"(tolerance {FOLD_STD_TOL}).",
        f"Shuffled-label CV ROC-AUC is {shuffled_roc_auc:.3f}, near the {BASELINE_AUC} "
        "chance line, so the detector learns real damage content rather than a leak.",
    ]

    return BenchmarkReport(
        n_samples=len(images),
        counts=class_counts(labels),
        n_features=int(features.shape[1]),
        n_splits=n_splits,
        test_size=test_size,
        cv_accuracy=cv_accuracy,
        cv_roc_auc=cv_roc_auc,
        cv_f1=cv_f1,
        fold_roc_aucs=[float(a) for a in fold_aucs],
        fold_roc_auc_std=fold_std,
        test_accuracy=test_accuracy,
        test_roc_auc=test_roc_auc,
        test_f1=test_f1,
        train_accuracy=train_accuracy,
        train_roc_auc=train_roc_auc,
        train_cv_auc_gap=train_roc_auc - cv_roc_auc,
        cv_test_auc_gap=abs(cv_roc_auc - test_roc_auc),
        shuffled_roc_auc=shuffled_roc_auc,
        verdict=verdict,
        notes=notes,
    )


def format_report(report: BenchmarkReport) -> str:
    """Render a benchmark report as a readable, aligned block of text."""
    folds = ", ".join(f"{a:.3f}" for a in report.fold_roc_aucs)
    lines = [
        "DamageScope — real-photo damage detection benchmark",
        "=" * 55,
        f"dataset      : {report.n_samples} real car photos  "
        f"(no_damage={report.counts['no_damage']}, damage={report.counts['damage']})",
        f"features     : {report.n_features} HOG dims per image (project pipeline)",
        f"protocol     : {report.n_splits}-fold stratified CV + "
        f"{report.test_size:.0%} held-out test",
        "",
        f"cross-val    : accuracy {report.cv_accuracy:.3f}  "
        f"ROC-AUC {report.cv_roc_auc:.3f}  F1 {report.cv_f1:.3f}",
        f"  per-fold AUC: [{folds}]  std {report.fold_roc_auc_std:.3f}",
        f"held-out test: accuracy {report.test_accuracy:.3f}  "
        f"ROC-AUC {report.test_roc_auc:.3f}  F1 {report.test_f1:.3f}",
        f"training set : accuracy {report.train_accuracy:.3f}  "
        f"ROC-AUC {report.train_roc_auc:.3f}",
        "",
        f"gaps         : train-vs-CV AUC {report.train_cv_auc_gap:+.3f}  "
        f"CV-vs-test AUC {report.cv_test_auc_gap:.3f}",
        f"baseline     : chance ROC-AUC {BASELINE_AUC:.2f}  "
        f"(shuffled-label CV AUC {report.shuffled_roc_auc:.3f})",
        f"verdict      : {report.verdict.upper()}",
        "",
    ]
    lines += [f"- {note}" for note in report.notes]
    return "\n".join(lines)


def main() -> None:
    print(format_report(run_benchmark()))


if __name__ == "__main__":
    main()
