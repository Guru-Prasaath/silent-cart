"""SHAP driver attribution (global + per customer) and segment risk comparison."""
import warnings

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

from .config import CHURN_COLOR, COLORS, OUTPUTS, RETAINED_COLOR, SEED
from .features import CATEGORICAL, FEATURE_GROUPS
from .viz import apply_style, fig_titled, save, titled

apply_style()
GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
GROUP_COLORS = {"behavioural": COLORS["blue"], "engagement": COLORS["aqua"], "support": COLORS["orange"],
                "demographic": COLORS["violet"], "history": COLORS["neutral"]}

LABELS = {
    "recency_months": "Months since last purchase", "txn_last_month": "Transactions last month",
    "txn_l3": "Transactions / month (last 3m)", "txn_p3": "Transactions / month (prior 3m)",
    "txn_change_pct": "Transaction change vs prior 3m", "spend_l3": "Spend / month (last 3m)",
    "spend_p3": "Spend / month (prior 3m)", "spend_change_pct": "Spend change vs prior 3m",
    "txn_slope_6m": "6-month transaction trend", "spend_cv_6m": "Spend volatility (6m)",
    "active_months_l3": "Months with a purchase (last 3m)", "active_ratio_6m": "Share of months active (6m)",
    "avg_basket_l3": "Average basket (last 3m)", "categories_l3": "Categories bought / month",
    "promo_pct_6m": "Promo share of transactions", "app_l3": "App sessions / month (last 3m)",
    "app_p3": "App sessions / month (prior 3m)", "app_change_pct": "App-use change vs prior 3m",
    "app_slope_6m": "6-month app-use trend", "email_l3": "Emails opened / month (last 3m)",
    "email_p3": "Emails opened / month (prior 3m)", "email_change_pct": "Email-open change vs prior 3m",
    "coupons_l3": "Coupons redeemed / month (last 3m)", "coupons_p3": "Coupons redeemed / month (prior 3m)",
    "tickets_l3": "Support tickets (last 3m)", "tickets_p3": "Support tickets (prior 3m)",
    "tickets_6m": "Support tickets (6m)", "complaints_6m": "Formal complaints (6m)",
    "complaint_l3": "Complaint in last 3m", "months_since_ticket": "Months since last ticket",
    "history_months": "Months of history", "short_history": "Short history (<4 months)",
    "missing_months_6m": "Missing data months", "age": "Age", "age_missing": "Age missing",
    "tier_rank": "Membership tier", "marketing_opt_in": "Marketing opt-in", "tenure_months": "Tenure (months)",
    "gender": "Gender", "city": "City", "preferred_category": "Preferred category",
    "home_store_price_tier": "Home-store price tier",
}


# Related features share a family so the top-3 drivers describe three DIFFERENT reasons.
FAMILY = {
    **dict.fromkeys(["recency_months", "txn_last_month", "txn_l3", "active_months_l3", "active_ratio_6m"], "purchase frequency"),
    **dict.fromkeys(["txn_change_pct", "txn_slope_6m", "txn_p3"], "purchase trend"),
    **dict.fromkeys(["spend_l3", "spend_p3", "spend_change_pct", "spend_cv_6m", "avg_basket_l3"], "spend"),
    **dict.fromkeys(["app_l3", "app_p3", "app_change_pct", "app_slope_6m"], "app"),
    **dict.fromkeys(["email_l3", "email_p3", "email_change_pct"], "email"),
    **dict.fromkeys(["coupons_l3", "coupons_p3", "promo_pct_6m"], "offers"),
    **dict.fromkeys(["tickets_l3", "tickets_p3", "tickets_6m", "complaints_6m", "complaint_l3", "months_since_ticket"], "support"),
    **dict.fromkeys(["history_months", "short_history", "missing_months_6m", "tenure_months"], "history"),
}


def describe(feature: str, v) -> str:
    """Plain-English driver text for the CRM / store team."""
    pct = lambda x: f"{x:+.0%}"
    t = {
        "recency_months": lambda: "No purchase in 6+ months" if v >= 7 else (
            "Bought last month" if v <= 1 else f"No purchase for {int(v) - 1} months"),
        "txn_last_month": lambda: "No transactions last month" if v == 0 else f"Only {v:.0f} transaction(s) last month",
        "txn_l3": lambda: "No purchases in last 3m" if v == 0 else f"{v:.1f} transactions/month (last 3m)",
        "txn_p3": lambda: f"{v:.1f} transactions/month before (prior 3m)",
        "txn_change_pct": lambda: f"Transactions {pct(v)} vs prior 3m",
        "spend_l3": lambda: f"Spend ${v:,.0f}/month (last 3m)",
        "spend_p3": lambda: f"Spend was ${v:,.0f}/month (prior 3m)",
        "spend_change_pct": lambda: f"Spend {pct(v)} vs prior 3m",
        "txn_slope_6m": lambda: f"Transactions trending {'down' if v < 0 else 'up'} ({v:+.1f}/month)",
        "spend_cv_6m": lambda: f"Erratic monthly spend (CV {v:.2f})",
        "active_months_l3": lambda: f"Bought in {int(v)} of last 3 months",
        "active_ratio_6m": lambda: f"Active in {v:.0%} of recent months",
        "avg_basket_l3": lambda: f"Average basket ${v:,.0f}",
        "categories_l3": lambda: f"Buys {v:.1f} categories/month",
        "promo_pct_6m": lambda: f"{v:.0%} of trips use a promotion",
        "app_l3": lambda: "No app use in last 3m" if v == 0 else f"App use {v:.1f} sessions/month",
        "app_p3": lambda: f"App use was {v:.1f} sessions/month",
        "app_change_pct": lambda: f"App use {pct(v)} vs prior 3m",
        "app_slope_6m": lambda: f"App use trending {'down' if v < 0 else 'up'}",
        "email_l3": lambda: "No marketing emails opened in last 3m" if v == 0 else f"Opens {v:.1f} emails/month",
        "email_p3": lambda: f"Opened {v:.1f} emails/month before",
        "email_change_pct": lambda: f"Email opens {pct(v)} vs prior 3m",
        "coupons_l3": lambda: "No coupons redeemed in last 3m" if v == 0 else f"Redeems {v:.1f} coupons/month",
        "coupons_p3": lambda: f"Redeemed {v:.1f} coupons/month before",
        "tickets_l3": lambda: f"{int(v)} support tickets in last 3m",
        "tickets_p3": lambda: f"{int(v)} support tickets in prior 3m",
        "tickets_6m": lambda: f"{int(v)} support tickets in 6m",
        "complaints_6m": lambda: f"{int(v)} formal complaints in 6m",
        "complaint_l3": lambda: "Formal complaint in last 3m" if v else "No recent complaint",
        "months_since_ticket": lambda: "No recent tickets" if v >= 7 else f"Last ticket {int(v)} month(s) ago",
        "history_months": lambda: f"Only {int(v)} months of history",
        "short_history": lambda: "New member (<4 months of history)" if v else "Established member",
        "missing_months_6m": lambda: f"{int(v)} months of missing data",
        "tier_rank": lambda: ["Silver", "Gold", "Platinum"][int(v)] + " tier",
        "marketing_opt_in": lambda: "Opted in to marketing" if v else "Opted out of marketing",
        "tenure_months": lambda: f"Member for {int(v)} months",
        "age": lambda: f"Age {v:.0f}",
        "age_missing": lambda: "Age unknown",
    }
    if feature in t:
        return t[feature]()
    return f"{LABELS.get(feature, feature)}: {v}"


def shap_values(cm, background: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP values (log-odds) per ORIGINAL feature; one-hot columns are summed back to their source feature."""
    prep, model = cm.pipeline.named_steps["prep"], cm.pipeline.named_steps["model"]
    Xt = prep.transform(X[cm.features])
    names = prep.get_feature_names_out()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hasattr(model, "coef_"):
            bg = prep.transform(background[cm.features].sample(min(500, len(background)), random_state=SEED))
            sv = shap.LinearExplainer(model, shap.maskers.Independent(bg, max_samples=len(bg))).shap_values(Xt)
        else:
            sv = shap.TreeExplainer(model).shap_values(Xt)
            if isinstance(sv, list):
                sv = sv[1]
            elif sv.ndim == 3:
                sv = sv[:, :, 1]
    sv = pd.DataFrame(sv, columns=names, index=X.index)
    src = {}
    for n in names:
        src[n] = next((c for c in CATEGORICAL if n.startswith(c + "_")), n)
    return sv.T.groupby(pd.Series(src)).sum().T[cm.features]


def top_drivers(sv: pd.DataFrame, X: pd.DataFrame, k=3) -> pd.Series:
    """Top-k risk-INCREASING drivers per member, as readable text."""
    def pick(order):
        chosen, fams = [], set()
        for f in order:
            fam = FAMILY.get(f, f)
            if fam not in fams:
                chosen.append(f)
                fams.add(fam)
            if len(chosen) == k:
                break
        return chosen

    out = []
    for i in sv.index:
        row = sv.loc[i].sort_values(ascending=False)
        pos = pick(row[row > 0].index)
        if len(pos) < k:   # very low-risk member: fill with the strongest effects either way
            rest = [f for f in row.abs().sort_values(ascending=False).index if f not in pos]
            pos = pick(pos + rest)
        out.append("; ".join(describe(f, X.at[i, f]) if f not in CATEGORICAL else f"{LABELS[f]}: {X.at[i, f]}" for f in pos))
    return pd.Series(out, index=sv.index)


# --------------------------------------------------------------------------- charts
def fig_shap_summary(sv: pd.DataFrame, X: pd.DataFrame, model_name: str, n=15) -> str:
    """Beeswarm of the top features for Active members (colour = feature value, low->high)."""
    order = sv.abs().mean().sort_values(ascending=False).index[:n][::-1]
    fig, ax = plt.subplots(figsize=(9, 6.8))
    rng = np.random.default_rng(SEED)
    cmap = plt.get_cmap("coolwarm")
    for y, f in enumerate(order):
        vals = sv[f].to_numpy()
        if f in CATEGORICAL:
            col = np.full(len(vals), 0.5)
        else:
            x = pd.to_numeric(X[f], errors="coerce").to_numpy(float)
            lo, hi = np.nanpercentile(x, [5, 95])
            col = np.clip((x - lo) / (hi - lo if hi > lo else 1), 0, 1)
        ax.scatter(vals, y + rng.normal(0, 0.12, len(vals)), c=cmap(col), s=7, alpha=0.7, lw=0)
    ax.set_yticks(range(len(order)), [LABELS.get(f, f) for f in order])
    ax.axvline(0, color=COLORS["axis"], lw=1)
    ax.set_xlabel("SHAP value (impact on churn log-odds)   <- lowers risk  |  raises risk ->")
    ax.grid(axis="y", visible=False)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["Low", "High"])
    cb.set_label("Feature value", color=COLORS["ink2"])
    cb.outline.set_visible(False)
    titled(ax, f"What drives churn risk for Active members ({model_name})",
           "Low recent activity and engagement (blue dots on the right) and more support tickets push risk up")
    return save(fig, "11_shap_summary_active.png")


def fig_shap_groups(sv_main: pd.DataFrame, sv_check: pd.DataFrame, names) -> str:
    """Share of total |SHAP| by feature group, selected model vs challenger (robustness check)."""
    def shares(sv):
        g = sv.abs().mean().groupby(lambda f: GROUP_OF[f]).sum()
        return (g / g.sum()).reindex(["behavioural", "engagement", "support", "demographic", "history"])
    a, b = shares(sv_main), shares(sv_check)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for y, (name, s) in enumerate([(names[1], b), (names[0], a)]):
        left = 0
        for g, v in s.items():
            ax.barh(y, v, left=left, color=GROUP_COLORS[g], height=0.55, edgecolor="white", lw=2)
            if v > 0.04:
                ax.text(left + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if g in ("behavioural", "demographic") else COLORS["ink"])
            left += v
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GROUP_COLORS[g], label=g.title()) for g in a.index], loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncols=5, fontsize=9)
    ax.set_yticks([0, 1], [names[1], names[0]])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(False)
    titled(ax, "Where the predictive signal comes from (Active members)",
           "Share of mean |SHAP| by feature group; both models agree that behaviour, engagement and support dominate")
    return save(fig, "12_shap_by_feature_group.png")


def segment_risk(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pop, d in [("All eligible", test), ("Active only", test[test.segment == "Active"])]:
        for dim in ["tier", "city", "signup_cohort"]:
            g = d.groupby(dim).agg(members=("customer_id", "size"), predicted_risk=("churn_probability", "mean"),
                                   observed_churn=("churned", "mean"),
                                   high_risk_share=("predicted_label", "mean"),
                                   revenue_at_risk=("revenue_at_risk", "sum"))
            rows.append(g.reset_index().rename(columns={dim: "segment_value"}).assign(population=pop, dimension=dim))
    out = pd.concat(rows, ignore_index=True)[["population", "dimension", "segment_value", "members", "predicted_risk",
                                             "observed_churn", "high_risk_share", "revenue_at_risk"]]
    out.round(4).to_csv(OUTPUTS / "segment_risk.csv", index=False)
    return out


def fig_segment_risk(seg: pd.DataFrame) -> str:
    d = seg[seg.population == "Active only"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True, gridspec_kw={"width_ratios": [3, 10, 4]})
    for ax, dim, name in zip(axes, ["tier", "city", "signup_cohort"], ["Membership tier", "City", "Signup cohort"]):
        g = d[d.dimension == dim].set_index("segment_value")
        if dim == "tier":
            g = g.reindex(["Silver", "Gold", "Platinum"])
        elif dim == "city":
            g = g.sort_values("predicted_risk", ascending=False)
        x = np.arange(len(g))
        ax.bar(x, g.predicted_risk, color=RETAINED_COLOR, width=0.62, label="Mean predicted risk")
        ax.scatter(x, g.observed_churn, color=CHURN_COLOR, s=40, zorder=3, edgecolor="white", lw=1.5, label="Observed churn")
        sep = " " if dim == "city" else "\n"
        ax.set_xticks(x, [f"{i}{sep}(n={n})" for i, n in zip(g.index, g.members)], rotation=35 if dim == "city" else 0,
                      ha="right" if dim == "city" else "center", fontsize=8.5)
        ax.set_title(name, fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Churn risk, Active members")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig_titled(fig, "Segment risk among Active members: differences are modest; behaviour matters more than who you are",
               "Bars = mean calibrated churn probability; dots = observed Apr-Jun 2024 churn. Use segments for targeting "
               "logistics, not as risk proxies")
    return save(fig, "13_segment_risk_active.png")
