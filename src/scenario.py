"""Retention scenarios, sensitivity analysis and the member-level next-best-action list.

Economics per member i (all assumptions in config.py):
    value_i         = typical monthly spend when shopping x VALUE_HORIZON_MONTHS x GROSS_MARGIN
    churn reduction = p_i x relative_uplift          (expected churners prevented)
    net value       = churn reduction x value_i - cost
Uplifts are assumptions, not measured effects: a sensitivity sweep and break-even uplift are reported,
and the support scenario is cross-checked with a model-based what-if.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

from . import config as C
from .viz import apply_style, fig_titled, save

apply_style()


def _has_support_issue(df):
    return (df.tickets_6m >= 2) | (df.complaint_l3 == 1)


def add_economics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["value_monthly_spend"] = df.value_monthly_spend.fillna(df.value_monthly_spend.median())
    df["margin_value"] = df.value_monthly_spend * C.VALUE_HORIZON_MONTHS * C.GROSS_MARGIN
    df["revenue_at_risk"] = df.churn_probability * df.value_monthly_spend * C.VALUE_HORIZON_MONTHS
    return df


def _result(name, desc, df, target, uplift, cost_each, base_churners):
    """`uplift` applies to Active members; for Lapsed members any offer is effectively a win-back,
    so its effect is capped at the win-back reactivation rate (they have already left)."""
    t = df[target]
    u = np.where(t.segment == "Lapsed", min(uplift, C.WINBACK_REACTIVATION), uplift)
    prevented = (t.churn_probability * u).sum()
    margin = (t.churn_probability * u * t.margin_value).sum()
    cost = cost_each * len(t)
    act = df.segment == "Active"
    prevented_active = (t.churn_probability * u)[t.segment == "Active"].sum()
    return {
        "scenario": name, "description": desc, "members_targeted": len(t),
        "expected_churners_in_target": t.churn_probability.sum(),
        "share_of_all_expected_churners": t.churn_probability.sum() / base_churners,
        "churners_prevented": prevented,
        "churn_rate_before": df.churn_probability.mean(),
        "churn_rate_after": (df.churn_probability.sum() - prevented) / len(df),
        "active_churn_before": df.churn_probability[act].mean(),
        "active_churn_after": (df.churn_probability[act].sum() - prevented_active) / act.sum(),
        "cost": cost, "margin_retained": margin, "net_value": margin - cost,
        "roi": (margin - cost) / cost if cost else np.nan,
        "break_even_uplift": cost / (t.churn_probability * t.margin_value).sum() if len(t) else np.nan,  # if all members responded alike
        "assumed_relative_uplift": uplift, "cost_per_member": cost_each,
    }


def model_whatif_support(cm, df: pd.DataFrame, target) -> dict:
    """Model-implied risk if the targeted members' support issues were fully resolved (associational upper bound)."""
    t = df[target].copy()
    before = cm.predict_proba(t)
    for c in ["tickets_l3", "tickets_p3", "tickets_6m", "complaints_6m", "complaint_l3"]:
        t[c] = 0
    t["months_since_ticket"] = 7
    after = cm.predict_proba(t)
    return {"members": len(t), "mean_risk_before": before.mean(), "mean_risk_after": after.mean(),
            "relative_reduction": 1 - after.mean() / before.mean(), "churners_prevented": (before - after).sum()}


def next_best_action(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the action with the highest expected net value per member; 'Monitor' if none pays back."""
    p, v = df.churn_probability, df.margin_value
    ev = pd.DataFrame({
        "Personalised coupon": p * C.COUPON_RELATIVE_UPLIFT * v - C.COUPON_COST,
        "Proactive service call": np.where(_has_support_issue(df), p * C.OUTREACH_RELATIVE_UPLIFT * v - C.OUTREACH_COST, -np.inf),
        "Win-back offer": np.where(df.segment == "Lapsed", C.WINBACK_REACTIVATION * v - C.WINBACK_COST, -np.inf),
    }, index=df.index)
    ev.loc[df.segment == "Lapsed", ["Personalised coupon", "Proactive service call"]] = -np.inf
    best = ev.idxmax(axis=1)
    best_ev = ev.max(axis=1)
    out = df.copy()
    out["recommended_action"] = np.where(best_ev > 0, best, "Monitor (no paid action)")
    out["expected_net_value"] = np.where(best_ev > 0, best_ev, 0.0)
    return out


def run(cm, df: pd.DataFrame):
    df = add_economics(df)
    base = df.churn_probability.sum()
    act = df.segment == "Active"
    flagged = act & (df.churn_probability >= cm.threshold)
    support = act & _has_support_issue(df)
    rows = [
        _result("A. Blanket coupon (status quo)", "Coupon to every eligible member", df,
                np.ones(len(df), bool), C.COUPON_RELATIVE_UPLIFT, C.COUPON_COST, base),
        _result("B. Targeted coupon", "Coupon only to Active members the model flags", df,
                flagged.to_numpy(), C.COUPON_RELATIVE_UPLIFT, C.COUPON_COST, base),
        _result("C1. Support outreach, rule-based", "Service call to every Active member with 2+ tickets or a recent complaint",
                df, support.to_numpy(), C.OUTREACH_RELATIVE_UPLIFT, C.OUTREACH_COST, base),
        _result("C2. Support outreach, model-targeted", "Service call only to support-issue members the model also flags",
                df, (support & flagged).to_numpy(), C.OUTREACH_RELATIVE_UPLIFT, C.OUTREACH_COST, base),
        _result("D. Win-back for lapsed", "Reactivation offer to members silent 3+ months", df,
                (df.segment == "Lapsed").to_numpy(), C.WINBACK_REACTIVATION, C.WINBACK_COST, base),
    ]
    nba = next_best_action(df)
    # E: per-member best action (each member gets at most one treatment)
    parts = []
    for action, uplift, cost in [("Personalised coupon", C.COUPON_RELATIVE_UPLIFT, C.COUPON_COST),
                                 ("Proactive service call", C.OUTREACH_RELATIVE_UPLIFT, C.OUTREACH_COST),
                                 ("Win-back offer", C.WINBACK_REACTIVATION, C.WINBACK_COST)]:
        parts.append(_result(action, "", df, (nba.recommended_action == action).to_numpy(), uplift, cost, base))
    e = {"scenario": "E. Next-best-action playbook",
         "description": "Each member gets the single action with positive expected value (coupon / service call / win-back)",
         "members_targeted": sum(r["members_targeted"] for r in parts),
         "expected_churners_in_target": sum(r["expected_churners_in_target"] for r in parts),
         "churners_prevented": sum(r["churners_prevented"] for r in parts),
         "cost": sum(r["cost"] for r in parts), "margin_retained": sum(r["margin_retained"] for r in parts)}
    e["share_of_all_expected_churners"] = e["expected_churners_in_target"] / base
    e["net_value"] = e["margin_retained"] - e["cost"]
    e["roi"] = e["net_value"] / e["cost"]
    e["churn_rate_before"] = df.churn_probability.mean()
    e["churn_rate_after"] = (base - e["churners_prevented"]) / len(df)
    prevented_active = sum(r["churners_prevented"] for r in parts[:2])   # coupon + service call go to Active only
    e["active_churn_before"] = df.churn_probability[act].mean()
    e["active_churn_after"] = (df.churn_probability[act].sum() - prevented_active) / act.sum()
    rows.append(e)
    table = pd.DataFrame(rows)
    table.round(4).to_csv(C.OUTPUTS / "retention_scenarios.csv", index=False)
    whatif = model_whatif_support(cm, df, support.to_numpy())
    pd.DataFrame([whatif]).round(4).to_csv(C.OUTPUTS / "scenario_support_model_whatif.csv", index=False)
    sens = sensitivity(df, flagged, support)
    return df, nba, table, whatif, sens


def sensitivity(df, flagged, support) -> pd.DataFrame:
    rows = []
    for u in np.linspace(0, 0.4, 41):
        for name, mask, cost in [("A. Blanket coupon", np.ones(len(df), bool), C.COUPON_COST),
                                 ("B. Targeted coupon", flagged.to_numpy(), C.COUPON_COST),
                                 ("C1. Outreach, rule-based", support.to_numpy(), C.OUTREACH_COST),
                                 ("C2. Outreach, model-targeted", (support & flagged).to_numpy(), C.OUTREACH_COST)]:
            t = df[mask]
            uu = np.where(t.segment == "Lapsed", min(u, C.WINBACK_REACTIVATION), u)
            rows.append({"strategy": name, "relative_uplift": u,
                         "net_value": (t.churn_probability * uu * t.margin_value).sum() - cost * len(t)})
    out = pd.DataFrame(rows)
    out.round(2).to_csv(C.OUTPUTS / "scenario_sensitivity.csv", index=False)
    return out


# --------------------------------------------------------------------------- charts
def fig_scenarios(table: pd.DataFrame) -> str:
    t = table.iloc[::-1]
    def _m(x, _=None):
        sign = "−" if x < 0 else ""
        return f"{sign}${abs(x) / 1000:,.1f}k" if abs(x) >= 1000 else f"{sign}${abs(x):,.0f}"
    money = FuncFormatter(_m)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), sharey=True)
    specs = [("cost", "Cost", C.COLORS["neutral"]), ("churners_prevented", "Expected churners prevented", C.COLORS["blue"]),
             ("net_value", "Net value (margin retained - cost)", None)]
    for ax, (col, name, color) in zip(axes, specs):
        vals = t[col].to_numpy()
        colors = color or [C.COLORS["green"] if v >= 0 else C.COLORS["red"] for v in vals]
        ax.barh(t.scenario, vals, color=colors, height=0.6)
        for y, v in enumerate(vals):
            lab = f"{v:,.0f}" if col == "churners_prevented" else money(v, None)
            ax.text(v, y, f" {lab} ", va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=C.COLORS["ink2"])
        ax.axvline(0, color=C.COLORS["axis"], lw=1)
        ax.set_title(name, fontsize=11)
        ax.grid(axis="y", visible=False)
        if col != "churners_prevented":
            ax.xaxis.set_major_formatter(money)
        lo, hi = min(0, vals.min()), max(0, vals.max())
        ax.set_xlim(lo - 0.25 * (hi - lo) * (lo < 0), hi + 0.3 * (hi - lo))
    a, e = table.iloc[0], table.iloc[-1]
    fig_titled(fig, f"Next-best-action targeting: {1 - e.cost / a.cost:.0%} less spend, {e.churners_prevented / a.churners_prevented:.0%} "
               f"of the saves, and a profit instead of a loss",
               f"Assumptions: coupon ${C.COUPON_COST:.0f} at {C.COUPON_RELATIVE_UPLIFT:.0%} relative uplift; service call "
               f"${C.OUTREACH_COST:.0f} at {C.OUTREACH_RELATIVE_UPLIFT:.0%}; win-back ${C.WINBACK_COST:.0f} at "
               f"{C.WINBACK_REACTIVATION:.0%} reactivation; value = {C.VALUE_HORIZON_MONTHS} months of spend x "
               f"{C.GROSS_MARGIN:.0%} margin")
    return save(fig, "14_retention_scenarios.png")


def fig_sensitivity(sens: pd.DataFrame, table: pd.DataFrame) -> str:
    colors = {"A. Blanket coupon": C.COLORS["neutral"], "B. Targeted coupon": C.COLORS["blue"],
              "C1. Outreach, rule-based": C.COLORS["yellow"], "C2. Outreach, model-targeted": C.COLORS["orange"]}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for name, g in sens.groupby("strategy"):
        ax.plot(g.relative_uplift, g.net_value, color=colors[name], label=name, lw=2.2)
    for i, (name, g) in enumerate(sens.groupby("strategy")):
        pos = g[g.net_value >= 0]
        if len(pos):
            u = pos.relative_uplift.iloc[0]
            ax.scatter([u], [0], color=colors[name], s=45, zorder=3, edgecolor="white", lw=1.5)
            ax.annotate(f"{u:.0%}", (u, 0), xytext=(-4, 7 + 11 * (i % 2)), textcoords="offset points",
                        fontsize=8.5, color=C.COLORS["ink2"], ha="right")
        else:
            ax.text(0.4, g.net_value.iloc[-1], " never breaks even ", fontsize=8.5, color=colors[name], ha="right", va="bottom")
    ax.axhline(0, color=C.COLORS["axis"], lw=1)
    for u, txt in [(C.COUPON_RELATIVE_UPLIFT, "assumed coupon uplift"), (C.OUTREACH_RELATIVE_UPLIFT, "assumed outreach uplift")]:
        ax.axvline(u, color=C.COLORS["muted"], ls=":", lw=1)
        ax.text(u, ax.get_ylim()[1], f" {txt}", fontsize=8, color=C.COLORS["muted"], va="top", rotation=90)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{'−' if x < 0 else ''}${abs(x) / 1000:,.0f}k"))
    ax.set_xlabel("Relative churn reduction achieved by the action (assumption)")
    ax.set_ylabel("Net value")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Sensitivity: targeted actions break even at about half the assumed effect; blanket never does", pad=24)
    ax.text(0, 1.02, "Net value vs assumed uplift (Lapsed members capped at the win-back rate); dots = break-even uplift", transform=ax.transAxes,
            fontsize=9.5, color=C.COLORS["ink2"])
    return save(fig, "15_scenario_sensitivity.png")
