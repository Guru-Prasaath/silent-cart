"""Decision economics: profit-optimal threshold and the A/B test needed to measure real uplift."""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy import stats

from . import config as C
from .scenario import DEFAULT, Economics
from .viz import apply_style, save, titled

apply_style()


def profit_curve(df: pd.DataFrame, econ: Economics = DEFAULT) -> pd.DataFrame:
    """Net value of sending the coupon to Active members with p >= t, for every threshold t.
    Also returns precision / recall at t on the realised labels, to show the F1-vs-profit trade-off."""
    a = df[df.segment == "Active"]
    p, y, v = a.churn_probability.to_numpy(), a.churned.to_numpy(), a.margin_value.to_numpy()
    rows = []
    for t in np.round(np.arange(0.02, 0.96, 0.01), 2):
        m = p >= t
        tp = (m & (y == 1)).sum()
        rows.append({"threshold": t, "members_targeted": int(m.sum()),
                     "net_value": float((p[m] * econ.coupon_uplift * v[m]).sum() - econ.coupon_cost * m.sum()),
                     "precision": tp / m.sum() if m.sum() else np.nan, "recall": tp / max(y.sum(), 1)})
    out = pd.DataFrame(rows)
    out["f1"] = 2 * out.precision * out.recall / (out.precision + out.recall)
    out.round(4).to_csv(C.OUTPUTS / "profit_threshold_curve.csv", index=False)
    return out


def fig_profit_curve(curve: pd.DataFrame, f1_threshold: float) -> str:
    best = curve.loc[curve.net_value.idxmax()]
    f1_row = curve.iloc[(curve.threshold - f1_threshold).abs().argmin()]
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.plot(curve.threshold, curve.net_value, color=C.COLORS["blue"], lw=2.4, label="Net value of targeted coupon")
    ax.axhline(0, color=C.COLORS["axis"], lw=1)
    ax.set_ylim(top=curve.net_value.max() * 1.15)
    for row, txt, col, dy in [(best, "profit-optimal", C.COLORS["green"], -30), (f1_row, "F1-optimal (used for labels)", C.COLORS["orange"], -52)]:
        ax.scatter([row.threshold], [row.net_value], s=55, color=col, zorder=3, edgecolor="white", lw=1.5)
        ax.annotate(f"{txt}: t={row.threshold:.2f}, {int(row.members_targeted)} members, ${row.net_value:,.0f}",
                    (row.threshold, row.net_value), xytext=(8, dy), textcoords="offset points", fontsize=8.5, color=C.COLORS["ink2"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{'−' if x < 0 else ''}${abs(x):,.0f}"))
    ax.set_xlabel("Decision threshold on churn probability (Active members)")
    ax.set_ylabel("Net value this quarter")
    band = curve[curve.net_value >= 0.9 * best.net_value].threshold
    titled(ax, f"Profit is flat for thresholds {band.min():.2f}-{band.max():.2f}: the targeting decision is robust",
           f"Coupon economics from config.py. Profit-optimal t={best.threshold:.2f} vs F1-optimal t={f1_threshold:.2f}: "
           f"${best.net_value - f1_row.net_value:,.0f} apart")
    return save(fig, "16_profit_threshold.png")


def ab_test_design(df: pd.DataFrame, threshold: float, alpha=0.05, power=0.8, control_share=0.2) -> pd.DataFrame:
    """Members needed to detect a relative churn reduction with a hold-out control group.

    Two-sided two-proportion z-test with unequal allocation (control = 20% of flagged members).
    Baseline = mean calibrated churn probability of members the model flags.
    """
    flagged = df[(df.segment == "Active") & (df.churn_probability >= threshold)]
    p0 = float(flagged.churn_probability.mean())
    per_quarter = len(flagged)
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    k = (1 - control_share) / control_share          # treated per control member
    rows = []
    for rel in (0.10, 0.15, 0.20, 0.25, 0.30):
        p1 = p0 * (1 - rel)
        n_c = (za + zb) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1) / k) / (p0 - p1) ** 2
        n_total = n_c * (1 + k)
        rows.append({"relative_uplift_to_detect": rel, "baseline_churn_flagged": p0, "treated_churn": p1,
                     "control_members": int(np.ceil(n_c)), "total_members": int(np.ceil(n_total)),
                     "quarters_at_current_volume": n_total / per_quarter,
                     "flagged_per_quarter_this_cohort": per_quarter,
                     "eligible_base_for_one_quarter": int(np.ceil(n_total / per_quarter * len(df)))})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(C.OUTPUTS / "ab_test_design.csv", index=False)
    return out
