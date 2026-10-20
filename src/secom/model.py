"""Failure prediction models and their evaluation.

Published SECOM results use shuffled 10-fold CV. This data drifts a lot (71% of
kept sensors trend with run order, and the fail rate drops from 12.9% to 4.1%),
so shuffling lets a model pick up on time instead of anything real. The main
evaluation here is a forward time split; shuffled CV is only there to compare.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class EvalResult:
    """Scored performance of one model under one evaluation protocol."""

    name: str
    protocol: str
    n_train: int
    n_test: int
    n_test_fail: int
    ber: float
    sensitivity: float
    specificity: float
    roc_auc: float
    pr_auc: float
    prevalence: float
    threshold: float
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "model": self.name,
            "protocol": self.protocol,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_test_fail": self.n_test_fail,
            "BER": self.ber,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "ROC_AUC": self.roc_auc,
            "PR_AUC": self.pr_auc,
            "prevalence": self.prevalence,
            "lift_over_prevalence": self.pr_auc / self.prevalence if self.prevalence else np.nan,
        }


def make_models(seed: int = 20080719) -> dict[str, Pipeline]:
    """L2 logistic regression plus gradient boosting.

    Boosting is there to check whether a more flexible model does any better.
    Using class_weight='balanced' instead of resampling.
    """
    return {
        "logistic_l2": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=0.01,
                        class_weight="balanced",
                        max_iter=5000,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=3,
                        max_iter=200,
                        learning_rate=0.05,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                )
            ]
        ),
    }


def _score(
    name: str,
    protocol: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    n_train: int,
    threshold: float,
) -> EvalResult:
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[False, True]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    prevalence = float(y_true.mean())
    return EvalResult(
        name=name,
        protocol=protocol,
        n_train=n_train,
        n_test=len(y_true),
        n_test_fail=int(y_true.sum()),
        ber=float(1 - (sens + spec) / 2),
        sensitivity=float(sens),
        specificity=float(spec),
        roc_auc=float(roc_auc_score(y_true, scores)),
        pr_auc=float(average_precision_score(y_true, scores)),
        prevalence=prevalence,
        threshold=float(threshold),
    )


def evaluate_forward(
    X: pd.DataFrame,
    y: np.ndarray,
    models: dict[str, Pipeline],
    *,
    train_frac: float = 0.7,
) -> list[EvalResult]:
    """Train on the first train_frac of runs (by time), test on the rest.

    Threshold is fixed at 0.5, not tuned on the test set.
    """
    n = len(X)
    cut = int(n * train_frac)
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
    ytr, yte = y[:cut], y[cut:]
    out = []
    for name, model in models.items():
        model.fit(Xtr, ytr)
        scores = model.predict_proba(Xte)[:, 1]
        res = _score(name, "forward_chronological", yte, scores, cut, 0.5)
        res.notes.append(
            f"train window ends at run {cut}; test prevalence {yte.mean():.4f} "
            f"vs train prevalence {ytr.mean():.4f}"
        )
        out.append(res)
    return out


def evaluate_shuffled(
    X: pd.DataFrame,
    y: np.ndarray,
    models: dict[str, Pipeline],
    *,
    n_splits: int = 10,
    seed: int = 20080719,
) -> list[EvalResult]:
    """Stratified 10-fold CV like the published baselines, for comparison with
    evaluate_forward."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = []
    for name, model in models.items():
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y):
            model.fit(X.iloc[tr], y[tr])
            oof[te] = model.predict_proba(X.iloc[te])[:, 1]
        res = _score(name, f"stratified_{n_splits}fold_shuffled", y, oof, len(y), 0.5)
        res.notes.append("shuffled folds: training data includes runs after the test run")
        out.append(res)
    return out


def permutation_null(
    X: pd.DataFrame,
    y: np.ndarray,
    model: Pipeline,
    *,
    n_perm: int = 50,
    train_frac: float = 0.7,
    seed: int = 20080719,
) -> tuple[float, np.ndarray]:
    """Forward-split AUC with shuffled labels, i.e. what AUC looks like with no signal.

    Returns (observed_auc, null_aucs).
    """
    rng = np.random.default_rng(seed)
    observed = evaluate_forward(X, y, {"m": model}, train_frac=train_frac)[0].roc_auc
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = evaluate_forward(X, rng.permutation(y), {"m": model}, train_frac=train_frac)[
            0
        ].roc_auc
    return float(observed), null
