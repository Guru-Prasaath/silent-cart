"""EDA charts and hypothesis tests on how behaviour, engagement and support relate to churn."""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats

from .config import CHURN_COLOR, COLORS, OUTPUTS, RETAINED_COLOR
from .viz import apply_style, fig_titled, save, titled

apply_style()


# --------------------------------------------------------------------------- DQ visuals
def fig_spend_repair(raw_activity: pd.DataFrame) -> str:
    raw = raw_activity.drop_duplicates()
    exp = raw.TRANSACTIONS * raw.AVG_BASKET_VALUE
    bad = (raw.TRANSACTIONS > 0) & ((raw.TOTAL_SPEND <= 0) | (raw.TOTAL_SPEND > 3 * exp))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(exp[~bad], raw.TOTAL_SPEND[~bad], s=6, alpha=0.25, color=COLORS["neutral"], label="Consistent rows", lw=0)
    ax.scatter(exp[bad], raw.TOTAL_SPEND[bad], s=36, color=CHURN_COLOR, edgecolor="white", lw=1,
               label=f"Corrupted rows ({bad.sum()}): repaired")
    ax.axhline(0, color=COLORS["axis"], lw=1)
    ax.set_xlabel("Transactions x average basket ($)")
    ax.set_ylabel("Reported TOTAL_SPEND ($)")
    ax.legend(loc="upper left")
    titled(ax, "Spend errors are detectable because spend = transactions x basket",
           "99.8% of rows satisfy the identity; the rest are negative, zero or 9-15x inflated and were rebuilt")
    return save(fig, "01_dq_spend_repair.png")


def fig_leaky_bucket(activity: pd.DataFrame) -> str:
    """Inflow (first month seen) vs outflow (final purchase month of members who never buy again)."""
    first = activity.groupby("CUSTOMER_ID").MONTH.min().value_counts()
    last_buy = activity[activity.TRANSACTIONS > 0].groupby("CUSTOMER_ID").MONTH.max()
    months = pd.date_range("2023-03-01", "2024-03-01", freq="MS")  # Feb-23 skipped: inflated by missing Jan rows
    # A final purchase in month m is only confirmed if the member then stays silent through June-2024.
    outflow = last_buy[last_buy < "2024-04-01"].value_counts()
    inflow = first.reindex(months, fill_value=0)
    outflow = outflow.reindex(months, fill_value=0)
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - 0.2, inflow.values, width=0.38, color=RETAINED_COLOR, label="New members (first month active)")
    ax.bar(x + 0.2, outflow.values, width=0.38, color=CHURN_COLOR, label="Members making their final purchase")
    ax.set_xticks(x, [m.strftime("%b\n%y") for m in months])
    ax.set_ylabel("Members per month")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncols=2, bbox_to_anchor=(1, 1.0))
    ax.set_ylim(0, max(inflow.max(), outflow.max()) * 1.25)
    titled(ax, f"A leaky bucket: ~{outflow.mean():.0f} members make their final purchase every month",
           f"Mar-23 to Mar-24: {outflow.sum():,} members bought for the last time vs {inflow.sum():,} new arrivals; "
           "sign-ups hide the leak in headline counts")
    return save(fig, "02_leaky_bucket.png")


# --------------------------------------------------------------------------- churn structure
def fig_recency_cliff(test: pd.DataFrame) -> str:
    g = test.groupby("recency_months").churned.agg(["mean", "size"])
    labels = [f"{int(k)}" if k < 7 else "7+" for k in g.index]
    colors = [RETAINED_COLOR if k <= 3 else CHURN_COLOR for k in g.index]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.bar(labels, g["mean"], color=colors, width=0.7)
    for b, (rate, n) in zip(bars, g.itertuples(index=False)):
        ax.text(b.get_x() + b.get_width() / 2, rate + 0.02, f"{rate:.0%}\nn={n}", ha="center", fontsize=8.5, color=COLORS["ink2"])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("Months since last purchase at 1-Apr-2024")
    ax.set_ylabel("Churn rate (Apr-Jun 2024)")
    act = test[test.segment == "Active"]
    lap = test[test.segment == "Lapsed"]
    ax.text(0.02, 0.97, f"ACTIVE (bought in Jan-Mar): {len(act):,} members, churn {act.churned.mean():.1%}",
            transform=ax.transAxes, color=RETAINED_COLOR, fontsize=9.5, fontweight="bold", va="top")
    ax.text(0.98, 0.97, f"LAPSED: {len(lap):,} members, churn {lap.churned.mean():.1%}",
            transform=ax.transAxes, color=CHURN_COLOR, fontsize=9.5, fontweight="bold", va="top", ha="right")
    ax.grid(axis="x", visible=False)
    titled(ax, "Two different churn problems hide inside one label",
           f"{lap.churned.sum() / test.churned.sum():.0%} of 'churners' had already stopped buying before April; "
           "only the Active group needs prediction")
    return save(fig, "03_recency_cliff_two_churns.png")


def fig_churn_fingerprint(activity: pd.DataFrame, test: pd.DataFrame) -> str:
    """Average monthly behaviour aligned on each member's last pre-April purchase month (t=0)."""
    pre = activity[activity.MONTH < "2024-04-01"]
    last = pre[pre.TRANSACTIONS > 0].groupby("CUSTOMER_ID").MONTH.max().rename("anchor")
    d = pre.merge(last, left_on="CUSTOMER_ID", right_index=True)
    d["t"] = (d.MONTH.dt.year - d.anchor.dt.year) * 12 + d.MONTH.dt.month - d.anchor.dt.month
    d = d[d.t.between(-6, 0)].merge(test[["customer_id", "churned", "segment"]], left_on="CUSTOMER_ID", right_on="customer_id")
    d = d[d.segment == "Active"]      # compare like with like: members who bought in Q1-2024
    panels = [("TRANSACTIONS", "Transactions"), ("APP_SESSIONS", "App sessions"),
              ("EMAILS_OPENED", "Emails opened"), ("SUPPORT_TICKETS", "Support tickets")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True)
    for ax, (col, name) in zip(axes, panels):
        for lab, color, txt in [(0, RETAINED_COLOR, "Retained"), (1, CHURN_COLOR, "Churned")]:
            s = d[d.churned == lab].groupby("t")[col].mean()
            ax.plot(s.index, s.values, color=color, marker="o", ms=4, label=txt)
        ax.set_title(name, fontsize=11)
        ax.set_xticks(range(-6, 1))
        ax.set_xlabel("Months before last purchase")
    axes[0].legend(loc="lower left")
    fig_titled(fig, "Churn fingerprint: purchases, app use and email opens fade while support tickets spike",
               "Active members, monthly averages aligned on each member's last purchase before April 2024 (t = 0). "
               "The decline starts ~2 months before the last purchase: that is the intervention window")
    return save(fig, "04_churn_fingerprint.png")


def _rate_by(df, col, bins=None, labels=None):
    x = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True) if bins is not None else df[col]
    return df.groupby(x, observed=True).churned.agg(["mean", "size"])


def fig_engagement_support(test: pd.DataFrame) -> str:
    a = test[test.segment == "Active"]
    panels = [
        ("tickets_6m", [-0.1, 0, 1, 2, 10], ["0", "1", "2", "3+"], "Support tickets (last 6m)"),
        ("complaints_6m", [-0.1, 0, 1, 10], ["0", "1", "2+"], "Formal complaints (last 6m)"),
        ("app_l3", [-0.1, 0.5, 1.5, 3, 20], ["<0.5", "0.5-1.5", "1.5-3", "3+"], "App sessions / month (last 3m)"),
        ("spend_change_pct", [-1.01, -0.4, -0.1, 0.1, 3.1], ["<-40%", "-40/-10%", "±10%", ">+10%"], "Spend change vs prior 3m"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    base = a.churned.mean()
    for ax, (col, bins, labels, name) in zip(axes, panels):
        g = _rate_by(a, col, bins, labels)
        bars = ax.bar(g.index.astype(str), g["mean"], color=RETAINED_COLOR, width=0.65)
        for b, (rate, n) in zip(bars, g.itertuples(index=False)):
            ax.text(b.get_x() + b.get_width() / 2, rate + 0.005, f"{rate:.0%}\nn={n}", ha="center", fontsize=8, color=COLORS["ink2"])
        ax.axhline(base, color=COLORS["ink2"], lw=1, ls="--")
        ax.set_title(name, fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Churn rate, Active members")
    axes[0].text(0.02, base + 0.004, f"avg {base:.1%}", fontsize=8, color=COLORS["ink2"], transform=axes[0].get_yaxis_transform())
    axes[0].set_ylim(0, 0.55)
    fig_titled(fig, "Support friction and fading engagement both raise churn risk among Active members",
               "Observed churn rate (Apr-Jun 2024) by pre-April signal; dashed line = Active-segment average")
    return save(fig, "05_engagement_support_vs_churn.png")


def fig_segments_observed(test: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, gridspec_kw={"width_ratios": [3, 10, 4]})
    for ax, col, name in zip(axes, ["tier", "city", "signup_cohort"], ["Membership tier", "City", "Signup cohort"]):
        g = test.groupby(col).churned.agg(["mean", "size"])
        if col == "tier":
            g = g.reindex(["Silver", "Gold", "Platinum"])
        elif col == "city":
            g = g.sort_values("mean", ascending=False)
        bars = ax.bar(g.index.astype(str), g["mean"], color=RETAINED_COLOR, width=0.65)
        for b, (rate, n) in zip(bars, g.itertuples(index=False)):
            ax.text(b.get_x() + b.get_width() / 2, rate + 0.008, f"{rate:.0%}", ha="center", fontsize=8.5, color=COLORS["ink2"])
        ax.axhline(test.churned.mean(), color=COLORS["ink2"], lw=1, ls="--")
        ax.set_title(name, fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", rotation=30 if col == "city" else 0)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Observed churn rate (all eligible)")
    fig_titled(fig, "Tier offers no protection: Platinum members churn as often as Silver",
               "Churn rate of all eligible members (Apr-Jun 2024); dashed line = overall rate. Older cohorts carry more "
               "long-lapsed members; city spread is 31-41%")
    return save(fig, "06_observed_churn_by_segment.png")


# --------------------------------------------------------------------------- hypothesis tests
TEST_FEATURES = {
    "Behavioural": ["recency_months", "txn_l3", "txn_change_pct", "spend_change_pct", "active_months_l3", "categories_l3", "promo_pct_6m"],
    "Engagement": ["app_l3", "app_change_pct", "email_l3", "coupons_l3", "marketing_opt_in"],
    "Support": ["tickets_6m", "tickets_l3", "complaints_6m", "complaint_l3"],
    "Profile": ["tier_rank", "tenure_months", "age"],
}


def _bh(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def hypothesis_tests(test: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U (churned vs retained) per feature, Active segment, with BH-FDR correction.

    Effect size = rank-biserial correlation (+ means churners score higher).
    """
    a = test[test.segment == "Active"]
    rows = []
    for group, cols in TEST_FEATURES.items():
        for c in cols:
            # Test observed values only: imputed ages (age_missing=1) are excluded, so results do not depend on imputation
            obs = a[a.age_missing == 0] if c == "age" and "age_missing" in a else a
            x1, x0 = obs.loc[obs.churned == 1, c].dropna(), obs.loc[obs.churned == 0, c].dropna()
            u, p = stats.mannwhitneyu(x1, x0, alternative="two-sided")
            rows.append({"group": group, "feature": c, "mean_churned": x1.mean(), "mean_retained": x0.mean(),
                         "rank_biserial": 2 * u / (len(x1) * len(x0)) - 1, "p_value": p})
    # Chi-square on the most actionable binary signal: any complaint in the last 3 months
    ct = pd.crosstab(a.complaint_l3, a.churned)
    chi2, p_chi, _, _ = stats.chi2_contingency(ct)
    out = pd.DataFrame(rows)
    out["q_value_bh"] = _bh(out.p_value.to_numpy())
    out["significant_5pct"] = out.q_value_bh < 0.05
    out = out.sort_values("rank_biserial", key=np.abs, ascending=False)
    out.attrs["complaint_chi2"] = {"chi2": chi2, "p": p_chi,
                                   "churn_if_complaint": a[a.complaint_l3 == 1].churned.mean(),
                                   "churn_if_none": a[a.complaint_l3 == 0].churned.mean()}
    out.round(4).to_csv(OUTPUTS / "hypothesis_tests_active_segment.csv", index=False)
    return out


def run_eda(raw: dict, activity: pd.DataFrame, test: pd.DataFrame) -> dict:
    return {
        "spend_repair": fig_spend_repair(raw["activity"]),
        "leaky_bucket": fig_leaky_bucket(activity),
        "recency_cliff": fig_recency_cliff(test),
        "fingerprint": fig_churn_fingerprint(activity, test),
        "engagement_support": fig_engagement_support(test),
        "segments_observed": fig_segments_observed(test),
    }
