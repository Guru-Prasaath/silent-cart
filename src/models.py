"""Model training, selection, calibration, threshold tuning, evaluation and ablation.

Validation design (forward-chaining, no random split):
    train = snapshots at Jul-2023 and Oct-2023 (labels = Jul-Sep and Oct-Dec 2023)
    valid = snapshot at Jan-2024 (label = Jan-Mar 2024) -> hyper-parameters, calibration, threshold
    test  = snapshot at Apr-2024 with the official CHURNED label  -> reported once, never tuned on
"""
import itertools
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .config import COLORS, OUTPUTS, ROOT, SEED
from .features import ALL_FEATURES, CATEGORICAL, FEATURE_GROUPS
from .viz import apply_style, fig_titled, save

apply_style()
MODEL_COLORS = {"Recency rule": COLORS["neutral"], "Logistic regression": COLORS["aqua"],
                "Random forest": COLORS["violet"], "Gradient boosting": COLORS["blue"]}


# --------------------------------------------------------------------------- estimators
class RecencyRule(BaseEstimator, ClassifierMixin):
    """Rule-based baseline: churn probability = historical churn rate for the member's recency bucket."""

    def fit(self, X, y, **_):
        self.rates_ = pd.Series(y).groupby(X["recency_months"].to_numpy()).mean()
        self.prior_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = X["recency_months"].map(self.rates_).fillna(self.prior_).to_numpy()
        return np.c_[1 - p, p]


def _preprocessor(features, scale: bool):
    cats = [c for c in features if c in CATEGORICAL]
    nums = [c for c in features if c not in CATEGORICAL]
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", nums),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
    ], verbose_feature_names_out=False)


def build(name: str, params: dict, features=ALL_FEATURES):
    if name == "Recency rule":
        return RecencyRule()
    if name == "Logistic regression":
        est = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED, **params)
        return Pipeline([("prep", _preprocessor(features, True)), ("model", est)])
    if name == "Random forest":
        est = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", n_jobs=-1,
                                     random_state=SEED, **params)
        return Pipeline([("prep", _preprocessor(features, False)), ("model", est)])
    if name == "Gradient boosting":
        est = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, subsample=0.8,
                                         random_state=SEED, **params)
        return Pipeline([("prep", _preprocessor(features, False)), ("model", est)])
    raise ValueError(name)


GRIDS = {
    "Recency rule": [{}],
    "Logistic regression": [{"C": c} for c in (0.05, 0.2, 1.0)],
    "Random forest": [{"min_samples_leaf": m, "max_features": f} for m, f in itertools.product((5, 15), ("sqrt", 0.4))],
    "Gradient boosting": [{"max_depth": d, "min_samples_leaf": m} for d, m in itertools.product((2, 3), (10, 30))],
}


def fit(model, X, y, name):
    """Class imbalance: class weights for LR/RF, balanced sample weights for gradient boosting."""
    if name == "Gradient boosting":
        model.fit(X, y, model__sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


class PlattCalibrator:
    """Maps re-weighted scores back to real probabilities (fit on the validation snapshot)."""

    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e6).fit(self._logit(p), y)
        return self

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def transform(self, p):
        return self.lr.predict_proba(self._logit(p))[:, 1]


class ChurnModel:
    """Fitted pipeline + calibrator + decision threshold: the deployable artefact."""

    def __init__(self, name, pipeline, calibrator, threshold, features):
        self.name, self.pipeline, self.calibrator = name, pipeline, calibrator
        self.threshold, self.features = threshold, features

    def raw_proba(self, df):
        return self.pipeline.predict_proba(df[self.features])[:, 1]

    def predict_proba(self, df):
        return self.calibrator.transform(self.raw_proba(df))

    def predict(self, df):
        return (self.predict_proba(df) >= self.threshold).astype(int)


# --------------------------------------------------------------------------- metrics
def metrics(y, p, thr) -> dict:
    yhat = (p >= thr).astype(int)
    n_top = max(1, int(round(0.1 * len(p))))
    top = np.argsort(-p)[:n_top]
    return {
        "n": len(y), "churn_rate": float(np.mean(y)), "threshold": thr,
        "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat),
        "f1": f1_score(y, yhat), "pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
        "accuracy": accuracy_score(y, yhat), "brier": brier_score_loss(y, p),
        "lift_top10pct": float(np.mean(np.asarray(y)[top]) / max(np.mean(y), 1e-9)),
    }


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(thr[np.argmax(f1[:-1])])


def _active(df):
    return (df.segment == "Active").to_numpy()


# --------------------------------------------------------------------------- MLflow (optional)
def _mlflow():
    try:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")   # local file store; on Databricks the tracking URI is "databricks"
        import logging

        import mlflow
        logging.getLogger("mlflow").setLevel(logging.ERROR)   # e.g. git-SHA warnings when git is unavailable
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
        mlflow.set_experiment("freshbasket-churn")
        return mlflow
    except Exception as e:   # tracking must never break the pipeline
        print(f"  [mlflow] tracking disabled: {e}")
        return None


# --------------------------------------------------------------------------- training
def train_and_select(train, valid, test):
    mlf = _mlflow()
    ytr, yva, yte = train.churned.to_numpy(), valid.churned.to_numpy(), test.churned.to_numpy()
    va_act, te_act = _active(valid), _active(test)
    rows, fitted, val_scores = [], {}, {}
    for name, grid in GRIDS.items():
        best = None
        for params in grid:
            m = fit(build(name, params), train, ytr, name)
            p = m.predict_proba(valid)[:, 1]
            score = average_precision_score(yva[va_act], p[va_act])   # selection metric: Active-segment PR-AUC
            if best is None or score > best[0]:
                best = (score, params, m, p)
        score, params, m, p_va = best
        cal = PlattCalibrator().fit(p_va, yva)
        # Threshold maximises F1 on validation ACTIVE members: lapsed members score ~1 at any sensible threshold,
        # so the decision boundary only matters where prediction is genuinely hard.
        thr = best_f1_threshold(yva[va_act], cal.transform(p_va)[va_act])
        cm = ChurnModel(name, m, cal, thr, ALL_FEATURES)
        fitted[name] = cm
        val_scores[name] = score
        for split, df, y, act in [("valid", valid, yva, va_act), ("test", test, yte, te_act)]:
            p = cm.predict_proba(df)
            for pop, mask in [("All eligible", np.ones(len(y), bool)), ("Active only", act)]:
                rows.append({"model": name, "params": str(params), "split": split, "population": pop,
                             **metrics(y[mask], p[mask], thr)})
        if mlf is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with mlf.start_run(run_name=name):
                    mlf.log_params({"model": name, **{k: str(v) for k, v in params.items()}, "threshold": round(thr, 4)})
                    for r in rows[-4:]:
                        tag = f"{r['split']}_{'all' if r['population'].startswith('All') else 'active'}"
                        mlf.log_metrics({f"{tag}_{k}": float(r[k]) for k in ("pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy", "brier")})
    comp = pd.DataFrame(rows)
    comp = comp.merge(bootstrap_ci(fitted, test), on=["model", "split", "population"], how="left")
    best_name = max((n for n in fitted if n != "Recency rule"), key=lambda n: val_scores[n])
    return fitted, best_name, comp


def bootstrap_ci(fitted, test, n_boot=1000) -> pd.DataFrame:
    """95% bootstrap interval of test PR-AUC, so model differences are judged against sampling noise."""
    rng = np.random.default_rng(SEED)
    y = test.churned.to_numpy()
    rows = []
    for pop, mask in [("All eligible", np.ones(len(y), bool)), ("Active only", _active(test))]:
        yy = y[mask]
        idx = [rng.integers(0, len(yy), len(yy)) for _ in range(n_boot)]
        for name, cm in fitted.items():
            pp = cm.predict_proba(test)[mask]
            vals = [average_precision_score(yy[i], pp[i]) for i in idx if yy[i].any()]
            lo, hi = np.percentile(vals, [2.5, 97.5])
            rows.append({"model": name, "split": "test", "population": pop, "pr_auc_ci_low": lo, "pr_auc_ci_high": hi})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- plots
def fig_pr_curves(fitted, test) -> str:
    y = test.churned.to_numpy()
    act = _active(test)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (pop, mask) in zip(axes, [("All eligible members", np.ones(len(y), bool)), ("Active members only", act)]):
        for name, cm in fitted.items():
            p = cm.predict_proba(test)[mask]
            prec, rec, _ = precision_recall_curve(y[mask], p)
            ax.plot(rec, prec, color=MODEL_COLORS[name], lw=2.4 if name == "Gradient boosting" else 1.6,
                    label=f"{name} (PR-AUC {average_precision_score(y[mask], p):.2f})")
        ax.axhline(y[mask].mean(), color=COLORS["muted"], ls=":", lw=1)
        ax.text(0.99, y[mask].mean() + 0.015, f"base rate {y[mask].mean():.1%}", fontsize=8, color=COLORS["muted"], ha="right")
        ax.set_xlabel("Recall (share of churners caught)")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(pop, fontsize=11)
        ax.legend(loc="lower left", fontsize=8.5, bbox_to_anchor=(0, 0.1))
    fig_titled(fig, "Precision-recall on the untouched Apr-Jun 2024 test cohort",
               "On all members every model looks strong (lapsed members are easy); the Active panel is the real test, "
               "where ML clearly beats the recency rule")
    return save(fig, "07_pr_curves.png")


def fig_confusion(cm_model, test) -> str:
    y = test.churned.to_numpy()
    p = cm_model.predict_proba(test)
    yhat = (p >= cm_model.threshold).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"wspace": 0.08})
    for k, (ax, (pop, mask)) in enumerate(zip(axes, [("All eligible", np.ones(len(y), bool)), ("Active only", _active(test))])):
        m = confusion_matrix(y[mask], yhat[mask])
        share = m / m.sum(axis=1, keepdims=True)          # shade by row share so recall is visible
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1.25)
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{v:,}\n({share[i, j]:.0%} of row)", ha="center", va="center", fontsize=10,
                    color="white" if share[i, j] > 0.6 else COLORS["ink"])
        ax.set_xticks([0, 1], ["Predicted stay", "Predicted churn"])
        ax.set_yticks([0, 1], ["Actually stayed", "Actually churned"] if k == 0 else ["", ""])
        ax.grid(False)
        ax.set_title(pop, fontsize=11)
    fig_titled(fig, f"Confusion matrix: {cm_model.name} at threshold {cm_model.threshold:.2f}",
               "Threshold maximises F1 for Active members of the Jan-2024 validation snapshot; applied unchanged to the test cohort")
    return save(fig, "08_confusion_matrix.png")


def fig_calibration(fitted, test) -> str:
    y = test.churned.to_numpy()
    act = _active(test)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (pop, mask) in zip(axes, [("All eligible", np.ones(len(y), bool)), ("Active only", act)]):
        ax.plot([0, 1], [0, 1], color=COLORS["muted"], ls=":", lw=1, label="Perfect calibration")
        for name in ["Logistic regression", "Random forest", "Gradient boosting"]:
            p = fitted[name].predict_proba(test)[mask]
            frac, mean_p = calibration_curve(y[mask], p, n_bins=8, strategy="quantile")
            ax.plot(mean_p, frac, marker="o", ms=4, color=MODEL_COLORS[name], label=name)
        ax.set_xlabel("Predicted churn probability")
        ax.set_ylabel("Observed churn rate")
        ax.set_title(pop, fontsize=11)
        ax.legend(loc="upper left", fontsize=8.5)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig_titled(fig, "Calibration on the test cohort: probabilities can be used as real churn odds",
               "Platt scaling fitted on the validation snapshot undoes the class re-weighting, which matters for value-at-risk maths")
    return save(fig, "09_calibration.png")


# --------------------------------------------------------------------------- ablation
ABLATION_STEPS = [
    ("Demographics only", ["demographic"]),
    ("+ Behavioural", ["demographic", "behavioural"]),
    ("+ Engagement", ["demographic", "behavioural", "engagement"]),
    ("+ Support", ["demographic", "behavioural", "engagement", "support"]),
    ("All features", list(FEATURE_GROUPS)),
]


def ablation(train, test, params) -> pd.DataFrame:
    """Cumulative feature-group ablation with the selected gradient-boosting set-up.

    A second pass removes recency (the 'already lapsed' giveaway) to show what engagement and support add
    when the model cannot lean on the obvious signal.
    """
    rows = []
    y, act = test.churned.to_numpy(), _active(test)
    for variant in ["With recency", "Without recency features"]:
        for step, groups in ABLATION_STEPS:
            feats = [f for g in groups for f in FEATURE_GROUPS[g]]
            if variant != "With recency":
                feats = [f for f in feats if f not in ("recency_months", "txn_last_month", "active_months_l3")]
            m = fit(build("Gradient boosting", params, feats), train[feats], train.churned.to_numpy(), "Gradient boosting")
            p = m.predict_proba(test[feats])[:, 1]
            rows.append({"variant": variant, "step": step, "n_features": len(feats),
                         "pr_auc_all": average_precision_score(y, p), "roc_auc_all": roc_auc_score(y, p),
                         "pr_auc_active": average_precision_score(y[act], p[act]),
                         "roc_auc_active": roc_auc_score(y[act], p[act])})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(OUTPUTS / "ablation_study.csv", index=False)
    return out


def fig_ablation(ab: pd.DataFrame) -> str:
    d = ab[ab.variant == "With recency"]
    x = np.arange(len(d))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, col, pop in [(axes[0], "pr_auc_all", "All eligible members"), (axes[1], "pr_auc_active", "Active members only")]:
        bars = ax.bar(x, d[col], color=COLORS["blue"], width=0.62)
        for b, v in zip(bars, d[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, color=COLORS["ink2"])
        ax.set_xticks(x, d.step, rotation=15)
        ax.set_title(pop, fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("Test PR-AUC (Gradient boosting)")
    lift = d.pr_auc_active.iloc[-1] / d.pr_auc_active.iloc[0]
    fig_titled(fig, "Ablation: behaviour carries the signal; demographics alone are close to random",
               f"Active-member PR-AUC rises {lift:.1f}x from demographics-only to all features; "
               "see ablation_study.csv for the no-recency variant")
    return save(fig, "10_ablation.png")


# --------------------------------------------------------------------------- validation-design checks
def leakage_and_split_checks(test, labels, params) -> pd.DataFrame:
    """Two demonstrations (5-fold CV on the test cohort, clearly NOT the reported evaluation):
    1. adding the leaky label-table columns inflates performance;
    2. a random split looks better than the honest out-of-time result.
    """
    lab = labels.set_index("CUSTOMER_ID")
    d = test.copy()
    d["MONTHS_OBSERVED"] = d.customer_id.map(lab.MONTHS_OBSERVED)
    d["INSUFFICIENT_HISTORY_FLAG"] = d.customer_id.map(lab.INSUFFICIENT_HISTORY_FLAG)
    y = d.churned.to_numpy()
    act = _active(d)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for name, feats in [("Random 5-fold CV, clean features", ALL_FEATURES),
                        ("Random 5-fold CV, + leaky label columns", ALL_FEATURES + ["MONTHS_OBSERVED", "INSUFFICIENT_HISTORY_FLAG"])]:
        m = build("Gradient boosting", params, feats)
        p = cross_val_predict(m, d[feats], y, cv=cv, method="predict_proba",
                              params={"model__sample_weight": compute_sample_weight("balanced", y)})[:, 1]
        rows.append({"setup": name, "pr_auc_all": average_precision_score(y, p),
                     "pr_auc_active": average_precision_score(y[act], p[act])})
    return pd.DataFrame(rows)


def save_model(cm: ChurnModel) -> str:
    path = OUTPUTS / "models" / "churn_model.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cm, path)
    return str(path)


def export_mlflow_runs() -> None:
    """Write a CSV summary of the latest MLflow runs (evidence of experiment tracking)."""
    mlf = _mlflow()
    if mlf is None:
        return
    try:
        runs = mlf.search_runs(experiment_names=["freshbasket-churn"], order_by=["start_time DESC"])
        # Latest run per model, metrics only: run IDs and timestamps are left out so the file is identical across re-runs
        keep = sorted(c for c in runs.columns if c.startswith(("params.", "metrics.test_", "metrics.valid_active")))
        latest = runs.head(len(GRIDS))[keep].sort_values("params.model")
        latest.round(4).to_csv(OUTPUTS / "mlflow_runs_summary.csv", index=False)
    except Exception as e:
        print(f"  [mlflow] export skipped: {e}")
