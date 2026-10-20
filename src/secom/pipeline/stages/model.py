"""Shuffled vs forward-chronological evaluation, rolling origin, permutation null, triage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...model import evaluate_forward, evaluate_shuffled, make_models, permutation_null
from ..context import PipelineContext

NAME = "model"
REQUIRES = ("clean", "retained", "is_fail")
PRODUCES = (
    "models_df",
    "rolling_df",
    "perm_null",
    "obs_auc",
    "triage_df",
    "forward_cut",
    "forward_scores",
)


def run(ctx: PipelineContext) -> None:
    cfg = ctx.config
    seed, train_frac = cfg.seed, cfg.model.train_frac
    clean = ctx["clean"]
    X, y = clean[ctx["retained"]], ctx["is_fail"]

    shuffled = evaluate_shuffled(
        X, y, make_models(seed=seed), n_splits=cfg.model.cv_folds, seed=seed
    )
    forward = evaluate_forward(X, y, make_models(seed=seed), train_frac=train_frac)
    model_rows = [r.as_row() for r in shuffled + forward]
    models_df = pd.DataFrame(model_rows)

    rolling = []
    for frac in cfg.model.rolling_origin_fracs:
        r = evaluate_forward(
            X, y, {"logistic_l2": make_models(seed)["logistic_l2"]}, train_frac=frac
        )[0]
        rolling.append(
            {
                "train_frac": frac,
                "n_test": r.n_test,
                "n_test_fail": r.n_test_fail,
                "BER": r.ber,
                "ROC_AUC": r.roc_auc,
                "PR_AUC": r.pr_auc,
            }
        )
    rolling_df = pd.DataFrame(rolling)

    obs_auc, null = permutation_null(
        X,
        y,
        make_models(seed)["logistic_l2"],
        n_perm=cfg.model.n_perm,
        train_frac=train_frac,
        seed=seed,
    )

    cut = int(len(clean) * train_frac)
    mdl = make_models(seed)["logistic_l2"]
    mdl.fit(X.iloc[:cut], y[:cut])
    scores_te = mdl.predict_proba(X.iloc[cut:])[:, 1]
    y_te = y[cut:]
    triage = []
    for k in cfg.model.triage_top_k_pct:
        nf = max(1, int(len(y_te) * k / 100))
        thr = np.sort(scores_te)[-nf]
        flagged = scores_te >= thr
        cap = int((flagged & y_te).sum())
        triage.append(
            {
                "top_k_pct": k,
                "n_flagged": int(flagged.sum()),
                "captured": cap,
                "capture_pct": float(100 * cap / y_te.sum()),
                "lift": float((cap / flagged.sum()) / y_te.mean()),
            }
        )
    triage_df = pd.DataFrame(triage)

    def _ber(protocol_mask: pd.Series) -> float:
        rows = models_df.loc[(models_df["model"] == "logistic_l2") & protocol_mask, "BER"]
        return float(rows.iloc[0])

    ctx.results["model"] = {
        "results": model_rows,
        "rolling_origin": rolling,
        "rolling_mean_auc": float(rolling_df["ROC_AUC"].mean()),
        "rolling_sd_auc": float(rolling_df["ROC_AUC"].std()),
        "permutation": {
            "observed_auc": obs_auc,
            "null_mean": float(null.mean()),
            "null_sd": float(null.std()),
            "p_value": float((null >= obs_auc).mean()),
            "n_perm": int(len(null)),
        },
        "triage": triage,
        "best_lift": float(triage_df["lift"].max()),
        "ber_shuffled": _ber(models_df["protocol"].str.contains("shuffled")),
        "ber_forward": _ber(models_df["protocol"] == "forward_chronological"),
    }
    ctx.artifacts.update(
        models_df=models_df,
        rolling_df=rolling_df,
        perm_null=null,
        obs_auc=obs_auc,
        triage_df=triage_df,
        forward_cut=cut,
        forward_scores=scores_te,
    )
    m = ctx.results["model"]
    ctx.log(
        f"BER shuffled {m['ber_shuffled']:.3f} -> forward {m['ber_forward']:.3f}; "
        f"AUC {obs_auc:.3f} vs null {null.mean():.3f}"
    )
