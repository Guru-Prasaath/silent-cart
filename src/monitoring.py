"""Production monitoring: feature drift (PSI) across snapshots and fairness of errors across groups."""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from . import config as C
from .features import CATEGORICAL, FEATURE_GROUPS
from .viz import apply_style, save, titled

apply_style()
NUMERIC = [f for g in FEATURE_GROUPS.values() for f in g if f not in CATEGORICAL]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with decile bins of the reference sample (<0.1 stable, 0.1-0.25 watch, >0.25 drift)."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:          # near-constant feature: compare the share at the dominant value instead
        v = np.median(expected)
        e, a = np.array([np.mean(expected == v), np.mean(expected != v)]), np.array([np.mean(actual == v), np.mean(actual != v)])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = [{"feature": f, "group": next(g for g, fs in FEATURE_GROUPS.items() if f in fs),
             "psi_valid_vs_train": psi(train[f].to_numpy(float), valid[f].to_numpy(float)),
             "psi_test_vs_train": psi(train[f].to_numpy(float), test[f].to_numpy(float))} for f in NUMERIC]
    out = pd.DataFrame(rows).sort_values("psi_test_vs_train", ascending=False)
    out["status"] = pd.cut(out.psi_test_vs_train, [-1, 0.1, 0.25, 99], labels=["stable", "watch", "drift"])
    out.round(4).to_csv(C.OUTPUTS / "drift_psi.csv", index=False)
    return out


def fig_drift(d: pd.DataFrame, n=15) -> str:
    t = d.head(n).iloc[::-1]
    colors = [C.COLORS["red"] if v > 0.25 else C.COLORS["yellow"] if v > 0.1 else C.COLORS["blue"] for v in t.psi_test_vs_train]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(t.feature, t.psi_test_vs_train, color=colors, height=0.62)
    for x, lab in [(0.1, "watch"), (0.25, "drift")]:
        ax.axvline(x, color=C.COLORS["muted"], ls=":", lw=1)
        ax.text(x, len(t) - 0.4, f" {lab}", fontsize=8.5, color=C.COLORS["muted"])
    ax.set_xlabel("PSI, Apr-2024 test snapshot vs training snapshots")
    ax.grid(axis="y", visible=False)
    n_drift, n_watch = (d.psi_test_vs_train > 0.25).sum(), d.psi_test_vs_train.between(0.1, 0.25).sum()
    titled(ax, "Feature drift monitor: what to watch when the model runs monthly",
           f"{n_drift} of {len(d)} features exceed PSI 0.25; {n_watch} sit in the 'watch' band (tenure drifts by design as the "
           "program ages). Retrain quarterly and alert on PSI > 0.25")
    return save(fig, "17_feature_drift_psi.png")


def fairness_report(test: pd.DataFrame) -> pd.DataFrame:
    """Error rates by group for Active members at the deployed threshold."""
    a = test[test.segment == "Active"].copy()
    a["age_band"] = pd.cut(a.age, [0, 34, 54, 120], labels=["18-34", "35-54", "55+"]).astype(str)
    rows = []
    for dim in ["age_band", "gender", "city", "tier"]:
        for val, g in a.groupby(dim):
            y, yh = g.churned.to_numpy(), g.predicted_label.to_numpy()
            tp, fp = ((y == 1) & (yh == 1)).sum(), ((y == 0) & (yh == 1)).sum()
            rows.append({"dimension": dim, "group": val, "members": len(g), "churners": int(y.sum()),
                         "observed_churn": y.mean(), "mean_predicted": g.churn_probability.mean(),
                         "recall": tp / y.sum() if y.sum() else np.nan,
                         "precision": tp / (tp + fp) if tp + fp else np.nan,
                         "false_positive_rate": fp / (y == 0).sum() if (y == 0).sum() else np.nan})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(C.OUTPUTS / "fairness_by_group.csv", index=False)
    return out
