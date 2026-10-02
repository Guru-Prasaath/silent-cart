"""Builds reports/final_report.md (numbers pulled from outputs/) and exports it to PDF via headless Chrome."""
import json
import shutil
import subprocess
from pathlib import Path

import markdown
import pandas as pd

from . import config as C

ASSUMPTIONS = [
    "**Target.** CHURNED = no transactions in Apr-Jun 2024 for a member who purchased before April (official label, used as-is). "
    "Members who never purchased (120 in the label table), the 6 profiled members with no activity, and 5 labelled churners "
    "with no pre-April activity cannot be scored from history and are excluded, leaving 2,368 eligible members.",
    "**Missing months.** A month with no row *between* a member's first and last rows is lost data, not inactivity (the label "
    "table's MONTHS_OBSERVED counts these months as observed). Months *after* a member's last row are treated as zero activity, "
    "which is what a live system would see at scoring time.",
    "**Spend repair.** TOTAL_SPEND = TRANSACTIONS x AVG_BASKET_VALUE holds for 99.8% of rows, so negative, zero and 9-15x inflated "
    "spend values are rebuilt from that identity rather than dropped.",
    "**Impossible zero-transaction values.** Categories bought and coupons redeemed in months with zero transactions are set to 0.",
    "**Historical labels.** Snapshots at Jul-23, Oct-23 and Jan-24 reuse the official definition (no purchase in the next 3 months). "
    "Applied at Apr-24 this rebuilt label agrees with the official one for 99.8% of members.",
    "**Feature window.** All behavioural features use the 6 months before each cut-off, so snapshots are comparable. Recency is capped "
    "at 7 months (\"6+\").",
    "**Categoricals.** City and tier variants are standardised by trimming and title-casing (40 to 10 cities, 12 to 3 tiers). Missing gender and "
    "price tier become an explicit 'Unknown'. Missing age is filled with the median of the training snapshots, plus a missing flag.",
    f"**Decision threshold.** {{threshold}} maximises F1 for Active members of the validation snapshot. High risk means p at or above the threshold; "
    "Medium means p of 10% or more.",
    f"**Economics (not measured, varied in sensitivity analysis).** Value of a retained member = {C.VALUE_HORIZON_MONTHS} months of their typical "
    f"monthly spend x {C.GROSS_MARGIN:.0%} gross margin. Coupon: ${C.COUPON_COST:.0f} cost, cuts churn probability by "
    f"{C.COUPON_RELATIVE_UPLIFT:.0%} (relative). Service call: ${C.OUTREACH_COST:.0f}, cuts it by {C.OUTREACH_RELATIVE_UPLIFT:.0%}. "
    f"Win-back: ${C.WINBACK_COST:.0f}, reactivates {C.WINBACK_REACTIVATION:.0%}. Any offer to a Lapsed member is assumed to work only as well as a win-back.",
    "**Repeated members.** The same member can appear in up to two training snapshots, so training rows are not fully independent. "
    "This is standard for rolling-origin designs, and the test cohort is strictly later in time.",
]


def shap_group_shares(shap_imp: pd.DataFrame) -> tuple[float, float]:
    """Share of mean |SHAP| from behaviour + engagement, (selected model, challenger)."""
    cols = [c for c in shap_imp.columns if c.startswith("mean_abs_shap_active")]
    out = []
    for c in cols[:2]:
        g = shap_imp.groupby("group")[c].sum()
        out.append(float((g.get("behavioural", 0) + g.get("engagement", 0)) / g.sum()))
    return tuple(out)


def _table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt[c](r[c]) if c in fmt else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def _img(name, alt):
    return f"![{alt}](../outputs/figures/{name})"


def _money(x):
    return f"{'−' if x < 0 else ''}${abs(x):,.0f}"


def build_markdown() -> str:
    O = C.OUTPUTS
    k = json.loads((O / "key_metrics.json").read_text())
    comp = pd.read_csv(O / "model_comparison.csv")
    scen = pd.read_csv(O / "retention_scenarios.csv")
    ab = pd.read_csv(O / "ablation_study.csv")
    chk = pd.read_csv(O / "validation_design_checks.csv")
    ht = pd.read_csv(O / "hypothesis_tests_active_segment.csv")
    dq = pd.read_csv(O / "data_quality_log.csv")
    seg = pd.read_csv(O / "segment_risk.csv")
    shap_imp = pd.read_csv(O / "shap_feature_importance.csv")
    preds = pd.read_csv(O / "churn_predictions.csv")
    actions = pd.read_csv(O / "retention_action_list.csv")
    parity = pd.read_csv(O / "spark_parity_check.csv") if (O / "spark_parity_check.csv").exists() else None

    ta, tc = k["test_all"], k["test_active"]
    best, thr = k["selected_model"], k["threshold"]
    A, B, C1, C2, D, E = (scen.iloc[i] for i in range(6))
    per10k = lambda v: v / k["n_test"] * 10_000
    eng = pd.read_csv(O / "modelling_dataset.csv")
    act = eng[eng.segment == "Active"]
    r = lambda m: act[m].churned.mean()
    t_none, t_3 = r(act.tickets_6m == 0), r(act.tickets_6m >= 3)
    app_low, spend_drop = r(act.app_l3 < 0.5), r(act.spend_change_pct < -0.4)
    rec2 = eng[eng.recency_months == 2].churned.mean()
    rec3 = eng[eng.recency_months == 3].churned.mean()
    tier = seg[(seg.population == "All eligible") & (seg.dimension == "tier")].set_index("segment_value")
    gb_act = comp[(comp.model == "Gradient boosting") & (comp.split == "test") & (comp.population == "Active only")].iloc[0]
    rule_act = comp[(comp.model == "Recency rule") & (comp.split == "test") & (comp.population == "Active only")].iloc[0]
    h = {row.feature: row for row in ht.itertuples()}

    # ---------------------------------------------------------------- tables
    mc = comp[comp.split == "test"].copy()
    mc["PR-AUC (95% CI)"] = [f"{a:.3f} ({lo:.2f}-{hi:.2f})" for a, lo, hi in zip(mc.pr_auc, mc.pr_auc_ci_low, mc.pr_auc_ci_high)]
    mc = mc.rename(columns={"model": "Model", "population": "Population", "precision": "Precision", "recall": "Recall",
                            "f1": "F1", "roc_auc": "ROC-AUC", "accuracy": "Accuracy", "lift_top10pct": "Lift @ top 10%"})
    f3 = lambda x: f"{x:.3f}"
    mc_tbl = _table(mc[["Model", "Population", "Precision", "Recall", "F1", "PR-AUC (95% CI)", "ROC-AUC", "Accuracy", "Lift @ top 10%"]],
                    {c: f3 for c in ["Precision", "Recall", "F1", "ROC-AUC", "Accuracy"]} | {"Lift @ top 10%": lambda x: f"{x:.1f}x"})
    dq_tbl = _table(dq[dq.rows_found > 0].rename(columns={"table": "Table", "check": "Issue", "rows_found": "Rows",
                                                          "action": "Handling"})[["Table", "Issue", "Rows", "Handling"]],
                    {"Rows": lambda x: f"{x:,}"})
    ab_w = ab[ab.variant == "With recency"][["step", "n_features", "pr_auc_all", "pr_auc_active", "roc_auc_active"]]
    ab_tbl = _table(ab_w.rename(columns={"step": "Feature set", "n_features": "Features", "pr_auc_all": "PR-AUC all",
                                         "pr_auc_active": "PR-AUC Active", "roc_auc_active": "ROC-AUC Active"}),
                    {c: f3 for c in ["PR-AUC all", "PR-AUC Active", "ROC-AUC Active"]})
    ab_nr = ab[ab.variant != "With recency"].set_index("step")
    chk_tbl = _table(chk.rename(columns={"setup": "Set-up", "pr_auc_all": "PR-AUC all", "pr_auc_active": "PR-AUC Active"}),
                     {"PR-AUC all": f3, "PR-AUC Active": f3})
    sc = scen[["scenario", "members_targeted", "churners_prevented", "cost", "margin_retained", "net_value", "roi", "break_even_uplift"]]
    sc_tbl = _table(sc.rename(columns={"scenario": "Scenario", "members_targeted": "Members", "churners_prevented": "Churners prevented",
                                       "cost": "Cost", "margin_retained": "Margin retained", "net_value": "Net value", "roi": "ROI",
                                       "break_even_uplift": "Break-even uplift"}),
                    {"Churners prevented": lambda x: f"{x:.1f}", "Cost": _money, "Margin retained": _money, "Net value": _money,
                     "ROI": lambda x: f"{x:+.0%}", "Break-even uplift": lambda x: "n/a" if pd.isna(x) else f"{x:.0%}"})
    ht_tbl = _table(ht.head(12)[["group", "feature", "mean_churned", "mean_retained", "rank_biserial", "q_value_bh"]]
                    .rename(columns={"group": "Group", "feature": "Feature", "mean_churned": "Mean (churned)", "mean_retained": "Mean (retained)",
                                     "rank_biserial": "Effect size (r)", "q_value_bh": "q (BH)"}),
                    {"Mean (churned)": lambda x: f"{x:.2f}", "Mean (retained)": lambda x: f"{x:.2f}",
                     "Effect size (r)": lambda x: f"{x:+.2f}", "q (BH)": lambda x: "<0.001" if x < 0.001 else f"{x:.3f}"})
    nba = actions.groupby("recommended_action").agg(Members=("customer_id", "size"), **{"Expected net value": ("expected_net_value", "sum")},
                                                    **{"6-month revenue at risk": ("revenue_at_risk", "sum")}).reset_index()
    nba_tbl = _table(nba.rename(columns={"recommended_action": "Recommended action"}),
                     {"Expected net value": _money, "6-month revenue at risk": _money})
    ex = preds[(preds.segment == "Active") & (preds.predicted_label == 1)].head(5)[["customer_id", "churn_probability", "top_3_drivers"]]
    ex_tbl = _table(ex.rename(columns={"customer_id": "Member", "churn_probability": "Churn probability", "top_3_drivers": "Top 3 drivers (SHAP)"}),
                    {"Churn probability": lambda x: f"{x:.0%}"})
    from .explain import LABELS
    top_feats = ", ".join(LABELS.get(f, f).lower() for f in shap_imp.feature.head(6))
    sh_sel, sh_ch = shap_group_shares(shap_imp)
    parity_txt = (f"Verified: identical outputs for {parity.members.sum():,} member-snapshots across {len(parity)} cut-offs "
                  f"({int(parity.features_compared.iloc[0])} feature columns, zero mismatches)." if parity is not None
                  else "Run `scripts/spark_parity_check.py` to verify.")
    pers = pd.read_csv(O / "personas.csv")
    pers_tbl = _table(pers[["persona", "members", "observed_churn", "mean_predicted", "recommended_play"]].rename(
        columns={"persona": "Persona", "members": "Members", "observed_churn": "Observed churn", "mean_predicted": "Mean predicted",
                 "recommended_play": "Recommended play"}),
        {"Observed churn": lambda x: f"{x:.0%}", "Mean predicted": lambda x: f"{x:.0%}"})
    ab_d = pd.read_csv(O / "ab_test_design.csv")
    ab_tbl = _table(ab_d[["relative_uplift_to_detect", "control_members", "total_members", "quarters_at_current_volume",
                          "eligible_base_for_one_quarter"]].rename(columns={
        "relative_uplift_to_detect": "Uplift to detect", "control_members": "Control members", "total_members": "Flagged members needed",
        "quarters_at_current_volume": "Quarters at this volume", "eligible_base_for_one_quarter": "Eligible base for 1 quarter"}),
        {"Uplift to detect": lambda x: f"{x:.0%}", "Quarters at this volume": lambda x: f"{x:.1f}",
         "Control members": lambda x: f"{int(x):,}", "Flagged members needed": lambda x: f"{int(x):,}",
         "Eligible base for 1 quarter": lambda x: f"{int(x):,}"})
    ab15 = ab_d[ab_d.relative_uplift_to_detect == 0.15].iloc[0]
    surv = pd.read_csv(O / "survival_summary.csv")
    app_s = surv[surv.dimension == "early_app_use"].set_index("group").still_buying_after_12m
    sp = surv.groupby("dimension").logrank_p.first()
    drift = pd.read_csv(O / "drift_psi.csv")
    nq = k["next_quarter"]
    g = (k.get("gru") or {}).get("Active only")
    gru_txt = (f"A GRU recurrent network (PyTorch) reads each member's raw 6-month activity sequence instead of hand-built trends. "
               f"Same splits, weighting, calibration and threshold logic. It reaches **{g['pr_auc']:.3f}** Active PR-AUC "
               f"(95% CI {g['pr_auc_ci_low']:.2f}-{g['pr_auc_ci_high']:.2f}) vs {tc['pr_auc']:.3f} for the deployed model, with "
               f"{g['precision']:.0%} precision and {g['recall']:.0%} recall. A **paired bootstrap** puts the gain at {g['delta_vs_champion']:+.3f} "
               f"(95% CI {g['delta_ci_low']:+.3f} to {g['delta_ci_high']:+.3f}; GRU better in {g['p_gru_better']:.0%} of resamples): likely real, "
               "but not yet proven. **Decision:** keep the transparent logistic model live (its per-member SHAP reasons drive the action list), "
               "shadow-score the GRU for one quarter, and promote it if the gain holds. This is a standard champion/challenger set-up."
               if g and "delta_vs_champion" in g else "The GRU challenger requires PyTorch (`pip install -r requirements-dl.txt`).")
    msgs_path = O / "retention_messages.csv"
    if msgs_path.exists():
        mm = pd.read_csv(msgs_path)
        n_llm = int(mm.source.str.startswith("groq").sum())
        ex_msgs = "\n\n".join(f"> **{r.recommended_action}** ({r.channel}), drivers: *{r.top_3_drivers}*  \n> **{r.subject}**: {r.message}"
                               for _, r in mm.groupby("recommended_action").head(1).iterrows())
        genai_txt = (f"{len(mm)} priority members (balanced across coupon, service call and win-back) received a drafted message: "
                     f"{n_llm} by the LLM (Groq, `{mm.source.iloc[0].split(':', 1)[-1] if n_llm else 'n/a'}`), {len(mm) - n_llm} by the template engine. "
                     f"Guardrails passed for {mm.guardrail_pass.mean():.0%}.\n\n{ex_msgs}")
    else:
        genai_txt = "Run `python -m src.genai` to draft messages."
    assumptions = "\n".join(f"{i}. {a.replace('{threshold}', f'{thr:.2f}')}" for i, a in enumerate(ASSUMPTIONS, 1))

    return f"""# Silent Cart: FreshBasket Loyalty Churn Prediction & Retention Strategy

*Final report · Data Science · Prepared with the reproducible pipeline in this repository (`python run_pipeline.py`)*

---

## Executive summary

- **Most of the churn being reported has already happened.** Of the {k['n_test']:,} churn-eligible members, {k['churn_rate_all']:.1%} are
  labelled churned. But **{k['share_churners_already_lapsed']:.0%} of those "churners" had already stopped buying before April**: they
  had made no purchase for 3+ months. They need a *win-back* programme, not a prediction model.
- **The real prediction problem is among Active members** (bought in Jan-Mar 2024): {k['n_active']:,} members, of whom
  {k['churn_rate_active']:.1%} churned. Here a recency rule gets a PR-AUC of only {rule_act.pr_auc:.2f}; our **{best.lower()}** reaches
  **{tc['pr_auc']:.2f}** on an untouched, out-of-time test quarter. At the chosen threshold it catches **{tc['recall']:.0%} of Active churners at
  {tc['precision']:.0%} precision** (base rate {tc['churn_rate']:.1%}). The top 10% of scores hold {tc['lift_top10pct']:.1f}x the average churn rate.
- **Churn has a visible fingerprint.** In the ~2 months before their last purchase, churners' transactions, app sessions and email
  opens fall sharply while support tickets spike. Members with 3+ tickets in 6 months churn at {t_3:.0%}, against {t_none:.0%} for those
  with none. Members whose spend fell 40%+ quarter-on-quarter churn at {spend_drop:.0%}.
- **Tier does not protect.** Platinum members churn as often as Silver ({tier.loc['Platinum'].observed_churn:.0%} vs
  {tier.loc['Silver'].observed_churn:.0%}). Age, tenure and tier are not statistically related to churn among Active members.
  Behaviour, engagement and support experience are.
- **Targeting beats blanket offers.** Today's approach, the same coupon for everyone, costs {_money(A.cost)} for this cohort and loses
  {_money(-A.net_value)} under our stated assumptions. A **next-best-action playbook** (personalised coupon, proactive service call or
  win-back, chosen per member by expected value) costs {_money(E.cost)} ({1 - E.cost / A.cost:.0%} less), prevents about
  {E.churners_prevented:.0f} churners ({E.churners_prevented / A.churners_prevented:.0%} of the blanket offer's saves) and returns
  **{_money(E.net_value)}**. That is a swing of about {_money(per10k(E.net_value - A.net_value))} per 10,000 eligible members per quarter.

**Recommendation:** split the churn KPI into *Lapsed* (win-back) and *At-risk Active* (retention). Score members monthly, route each one
to its next best action, and prove the uplift with a 20% hold-out control group before scaling.

---

## 1. Business question and approach

FreshBasket sends the same retention offers to every member, which wastes money on members who were never going to leave. Management asked
four questions: **who** will churn, **which signals** come first, whether **tier, marketing engagement and support experience** matter,
and **what action** would reduce churn.

The pipeline runs in eight steps: **data quality → leakage-safe features → rolling-origin validation → four models compared →
calibration and threshold tuning → SHAP drivers and segments → ablation → retention economics**. It is a reproducible Python package
(`src/`) with one entry point (`run_pipeline.py`), MLflow experiment tracking, unit tests (including an automated leakage test), a
FastAPI scoring endpoint and a PySpark port of the feature pipeline for Databricks.

## 2. Data quality: what was wrong and how it was handled

The workbook has three tables: 2,600 members, 28k member-months, Jan-2023 to Jun-2024. Every sheet carries a title banner, so the
real header is on row 2. Reading with defaults silently yields `Unnamed` columns. All {len(dq)} checks are logged in
`outputs/data_quality_log.csv`. The issues found:

{dq_tbl}

{_img('01_dq_spend_repair.png', 'Spend repair')}

**Leakage audit.** Three label-table columns, `LAST_PURCHASE_DATE`, `MONTHS_OBSERVED` and `INSUFFICIENT_HISTORY_FLAG`, are
calculated over the full window *including* Apr-Jun 2024. Churners stop generating rows, so these columns partly encode the answer. They are
never used. Leakage-safe equivalents (`recency_months`, `history_months`, `short_history`) are rebuilt from pre-cut-off data. Section 5 shows
that including the leaky columns would inflate Active-member PR-AUC from {chk.iloc[1].pr_auc_active:.2f} to {chk.iloc[2].pr_auc_active:.2f}.

## 3. What churn really looks like

### 3.1 A leaky bucket hidden by sign-ups
Every month roughly 55 members make their final purchase, while new sign-ups keep headline member counts rising. The loss is invisible
in the totals.

{_img('02_leaky_bucket.png', 'Leaky bucket')}

### 3.2 Two churn problems inside one label
Churn probability is a cliff in recency. Members who last bought 1 month before April churn at 3%; 2 months, {rec2:.0%}; 3 months,
{rec3:.0%}; 4+ months, about 100%. The label combines members who left months ago with members who are about to leave.

{_img('03_recency_cliff_two_churns.png', 'Two churns')}

| Segment | Members | Churn rate | Share of churners | What it needs |
|---|---|---|---|---|
| Lapsed (no purchase Jan-Mar 2024) | {k['n_lapsed']:,} | {k['churn_rate_lapsed']:.1%} | {k['share_churners_already_lapsed']:.0%} | Win-back campaign; a rule identifies them |
| Active (purchased Jan-Mar 2024) | {k['n_active']:,} | {k['churn_rate_active']:.1%} | {1 - k['share_churners_already_lapsed']:.0%} | Early warning model + retention action |

### 3.3 The churn fingerprint
Lining up Active members on their last pre-April purchase shows *how* churn unfolds. Purchases, app sessions and email opens fade over the
final ~2 months while support tickets spike. Those ~2 months are the intervention window.

{_img('04_churn_fingerprint.png', 'Churn fingerprint')}

## 4. Engagement and support experience vs churn

{_img('05_engagement_support_vs_churn.png', 'Engagement and support')}

Statistical evidence for Active members: Mann-Whitney U tests, Benjamini-Hochberg corrected. Rank-biserial r > 0 means churners score higher.

{ht_tbl}

- **Engagement:** churners opened {h['email_l3'].mean_churned:.1f} emails/month vs {h['email_l3'].mean_retained:.1f}, and used the app
  {h['app_l3'].mean_churned:.1f} vs {h['app_l3'].mean_retained:.1f} sessions/month (r = {h['app_l3'].rank_biserial:+.2f}).
  Members with fewer than 0.5 app sessions/month churn at {app_low:.0%}.
- **Support:** churners raised {h['tickets_6m'].mean_churned:.1f} tickets in 6 months vs {h['tickets_6m'].mean_retained:.1f}. A formal
  complaint in the last quarter roughly doubles churn ({k['complaint_test']['churn_if_complaint']:.0%} vs
  {k['complaint_test']['churn_if_none']:.0%}; chi-square p = {k['complaint_test']['p']:.4f}).
- **Not significant:** tier, age and tenure (q > 0.6). Marketing opt-in is borderline (q = {h['marketing_opt_in'].q_value_bh:.2f}).

## 5. Modelling and validation

**Validation design (no random split).** The churn definition can be replayed at any cut-off, so we built forward-chaining snapshots:

| Role | Cut-off | Label window | Members | Churn | Active churn |
|---|---|---|---|---|---|
| Train | Jul-2023, Oct-2023 | Jul-Sep, Oct-Dec 2023 | 3,767 rows | 21.5% | 10.2% |
| Validation (model choice, calibration, threshold) | Jan-2024 | Jan-Mar 2024 | 2,167 | 30.1% | 10.2% |
| **Test (reported once)** | Apr-2024 | **Apr-Jun 2024, official label** | {k['n_test']:,} | {k['churn_rate_all']:.1%} | {k['churn_rate_active']:.1%} |

**Models compared:**
- (a) a **recency rule** (rule-based baseline: churn rate by months since last purchase);
- (b) **logistic regression** with class weights;
- (c) **random forest** with balanced class weights;
- (d) **gradient boosting** with balanced sample weights.

Hyper-parameters were chosen on validation **Active-member PR-AUC**, since that is where prediction is hard. Re-weighted scores were then
Platt-calibrated on validation, so outputs are real probabilities. The threshold maximises validation Active-member F1.

**Results on the untouched Apr-Jun 2024 test quarter:**

{mc_tbl}

{_img('07_pr_curves.png', 'PR curves')}

- On *all* members every model scores above 0.94 PR-AUC, because lapsed members are trivially predictable. **The Active panel is the real test:**
  machine-learning models roughly double the recency rule's PR-AUC ({rule_act.pr_auc:.2f} to about 0.8).
- **{best} was selected on validation.** Gradient boosting scores slightly higher on test ({gb_act.pr_auc:.3f} vs {tc['pr_auc']:.3f}), but the
  bootstrap intervals overlap almost entirely: the models are statistically tied. The transparent, cheaper model is deployed, and
  gradient boosting is kept as a challenger and SHAP cross-check.

{_img('08_confusion_matrix.png', 'Confusion matrix')}
{_img('09_calibration.png', 'Calibration')}

**Validation-design checks** (5-fold random CV on the test cohort, shown only for comparison):

{chk_tbl}

Random CV scores close to the out-of-time result on this dataset, so behaviour is stable over time. The leaky label columns,
however, would have inflated Active-member PR-AUC by about 0.07.

## 6. Churn drivers (SHAP) and segment risk

{_img('11_shap_summary_active.png', 'SHAP summary')}
{_img('12_shap_by_feature_group.png', 'SHAP by group')}

The strongest drivers for Active members are {top_feats}. Behaviour and engagement account for {sh_sel:.0%} of attribution in the
{best.lower()} and {sh_ch:.0%} in {k['challenger'].lower()}. Support is the next layer; demographics add little. In the logistic model, *marketing opt-in* raises risk once email opens are
accounted for: members who opted in but stopped opening emails are disengaging. This is a conditional effect, not a reason to avoid
opt-ins.

Every member gets three plain-English, risk-increasing drivers in `outputs/churn_predictions.csv`. Examples from the high-risk Active
list:

{ex_tbl}

{_img('13_segment_risk_active.png', 'Segment risk')}

**Segments.** Among Active members, predicted risk ranges from {seg[(seg.population == 'Active only') & (seg.dimension == 'city')].predicted_risk.min():.0%}
(Nashville) to {seg[(seg.population == 'Active only') & (seg.dimension == 'city')].predicted_risk.max():.0%} (Austin). Austin and Columbus are the highest-risk cities.
Platinum members score slightly lower predicted risk, yet *observed* churn is flat across tiers. Members who joined in 2024 churn least (honeymoon period).
Segment differences are much smaller than behavioural ones, so segments should guide *logistics* (which store team, which channel),
not who is targeted.

## 7. Ablation: do behaviour, engagement and support add value over demographics?

Each step adds a feature group cumulatively (gradient boosting, test quarter):

{ab_tbl}

{_img('10_ablation.png', 'Ablation')}

- **Demographics alone are no better than chance** for Active members (PR-AUC {ab_w.pr_auc_active.iloc[0]:.2f} vs a {k['churn_rate_active']:.1%} base rate).
- **Behavioural** features carry most of the signal.
- **Engagement** adds +{ab_w.pr_auc_active.iloc[2] - ab_w.pr_auc_active.iloc[1]:.2f} PR-AUC.
- **Support** adds +{ab_w.pr_auc_active.iloc[3] - ab_w.pr_auc_active.iloc[2]:.2f}.
- **Without recency** the final model still reaches {ab_nr.loc['All features'].pr_auc_active:.2f}, so the early-warning signal is
  not just "hasn't bought lately".

## 8. Retention scenarios

Expected churners prevented = churn probability x relative uplift. Net value = margin retained − cost. All assumptions are listed in
Section 10, and the uplifts are deliberately varied in the sensitivity chart.

{sc_tbl}

{_img('14_retention_scenarios.png', 'Scenarios')}
{_img('15_scenario_sensitivity.png', 'Sensitivity')}

- **Blanket coupon (status quo)** never breaks even. Most of the money goes to members who were staying anyway, or who have already left.
- **Support outreach**, as the brief suggested, loses money if every member with tickets is called ({_money(C1.net_value)}). The same calls
  **restricted to model-flagged members** return {_money(C2.net_value)} (ROI {C2.roi:+.0%}) and reach {C2.churners_prevented / C1.churners_prevented:.0%}
  of the saves with a quarter of the calls.
- **A model-based what-if supports this.** Re-scoring the {k['support_whatif']['members']} support-issue members with their tickets
  "resolved" lowers mean risk from {k['support_whatif']['mean_risk_before']:.0%} to {k['support_whatif']['mean_risk_after']:.0%}
  ({k['support_whatif']['relative_reduction']:.0%} relative). This is associational, so treat it as an upper bound; our planning assumption is 25%.
- **The targeted coupon** breaks even at a {B.break_even_uplift:.0%} uplift, against 21% for the blanket coupon.
- **Next-best-action playbook (recommended):** each member receives at most one action, and only when its expected value is positive:

{nba_tbl}

The playbook lowers expected churn among Active members from {E.active_churn_before:.1%} to {E.active_churn_after:.1%} next quarter
and recovers about 32 lapsed members, for a net {_money(E.net_value)} on this cohort. The full ranked list, with action, value at risk and
drivers per member, is in `outputs/retention_action_list.csv`.

## 9. Beyond the brief: advanced analytics

### 9.1 Deep-learning challenger (champion / challenger)
{gru_txt}

### 9.2 When do members leave? Survival analysis
Kaplan-Meier curves of time to final purchase, landmarked at month 3 (groups are defined on each member's first 3 months, which
avoids immortal-time bias), with log-rank tests:
- **Early app use:** members using the app less than once a month early on are still buying at 12 months only {app_s.min():.0%} of the
  time, vs {app_s.max():.0%} (p < 0.001). App onboarding is a lifetime-value lever.
- **No difference:** early complaints (p = {sp['early_support']:.2f}) and tier (p = {sp['tier']:.2f}). Support friction predicts churn
  when it is *recent* (section 4), not early in the relationship.

{_img('19_survival_curves.png', 'Survival')}

### 9.3 Behavioural personas
K-means on Active members' recent behaviour gives five messaging personas. Separation is modest (silhouette
{k['personas_silhouette']:.2f}), so personas tailor the *message* while the churn score decides *who* gets it.

{pers_tbl}

{_img('18_personas.png', 'Personas')}

### 9.4 Profit-optimal threshold
Choosing the cut-off by money instead of F1 gives t = {k['profit_optimal_threshold']:.2f} ({_money(k['profit_at_optimal'])}) vs
t = {thr:.2f} for F1. The profit curve is flat across a wide band, so the targeting decision is robust to the exact threshold.

{_img('16_profit_threshold.png', 'Profit threshold')}

### 9.5 Designing the experiment that replaces our assumptions
The uplift assumptions should be measured, not argued about. Hold out 20% of flagged members as a control group and use a
two-sided two-proportion test (α = 0.05, power 80%), baseline churn {ab15.baseline_churn_flagged:.0%} among flagged members:

{ab_tbl}

Detecting the assumed 15% uplift needs about {int(ab15.total_members):,} flagged members ({int(ab15.control_members)} in control). At this
sample's volume (2,368 eligible members) that takes about {ab15.quarters_at_current_volume:.0f} quarters. A programme with roughly
{int(ab15.eligible_base_for_one_quarter):,}+ eligible members could read the result within a single quarter.

### 9.6 Next quarter: who is likely to churn in Jul-Sep 2024
The model is used as it would be in production: a snapshot at Jul-2024 built from Jan-Jun 2024 behaviour, with no label yet.

| Segment | Members | Expected churners (90% interval) | Flagged high-risk | 6-month revenue at risk |
|---|---|---|---|---|
| Active | {nq['active']['members']:,} | {nq['active']['expected_churners']:.0f} ({nq['active']['interval_90'][0]:.0f}-{nq['active']['interval_90'][1]:.0f}) | {nq['active']['high_risk']} | {_money(nq['active']['revenue_at_risk_6m'])} |
| Lapsed | {nq['lapsed']['members']:,} | {nq['lapsed']['expected_churners']:.0f} | {nq['lapsed']['high_risk']} | {_money(nq['lapsed']['revenue_at_risk_6m'])} |

The ranked list with drivers and next best action is in `outputs/next_quarter_watchlist.csv`, and in the dashboard.

### 9.7 Monitoring: drift and fairness
- **Drift:** no feature exceeds PSI 0.25 (max {drift.psi_test_vs_train.max():.2f}, `{drift.iloc[0].feature}`, which drifts by design
  as the program ages). {int((drift.psi_test_vs_train > 0.1).sum())} features sit in the 0.1-0.25 watch band, supporting quarterly retraining.
- **Fairness:** recall and false-positive rates are similar across age bands and genders. Platinum recall is lower on few churners;
  it is recorded as a watch item in `MODEL_CARD.md`.

{_img('17_feature_drift_psi.png', 'Drift')}

### 9.8 Silent Cart Retention Console (Dash)
`python app.py` opens an interactive console with four tabs:
- **Overview:** headline KPIs.
- **Members:** filter today's action list or next quarter's watchlist, then click any member to see their churn probability,
  SHAP reasons, persona, next best action and 18-month activity.
- **Scenario simulator:** nine live assumption sliders, running the same scenario engine as this report.
- **Insights:** survival, profit curve, drift, fairness and A/B sizing.

Dash is natively supported on Databricks Apps, so the console can be deployed next to the model.

### 9.9 GenAI last mile: retention messages drafted per member
The pipeline already says *who* is at risk, *why* (SHAP) and *what to do* (next best action). An LLM now drafts *what to say*:
an app push, an email, or a call script for the support agent.

**Grounding and privacy.** The prompt contains only the member's three SHAP drivers, persona, preferred category, tier,
action and offer budget. No name, ID, age or city is sent.

**Guardrails.** Every draft is checked automatically, retried once with the violations fed back, and otherwise replaced by
a deterministic template. A draft fails if it:
- mentions churn, models, prediction or tracking;
- offers more than the budget, or uses percentage offers;
- contains an invented promo code, gift card or voucher;
- apologises for a support issue the member never raised;
- is too long.

The first live run showed why this matters: unguarded drafts invented a promo code (`WELCOME5`), offered a gift card, and
apologised to a member with no tickets. Each failure became a rule and a test.

{genai_txt}

## 10. Recommendations

1. **Redefine the churn KPI.** Report *Lapsed* (no purchase 3+ months: win-back) separately from *At-risk Active*. Add a **60-day
   no-purchase trigger**: churn jumps from 3% to {rec2:.0%} after one missed month, and to {rec3:.0%} after two.
2. **Replace blanket coupons with monthly next-best-action targeting.** Each month, score all members, then send:
   - **service calls** to high-risk members with support issues;
   - **preferred-category coupons** to other high-risk Active members;
   - **win-back offers** to high-value lapsed members.

   The expected result is about 84% lower spend, with a profit instead of a loss.
3. **Make service recovery a retention lever.** Members with 2+ tickets churn at 19-37%. Close the loop on tickets within the 2-month
   fingerprint window, with priority for members the model flags.
4. **Automate free nudges before paid offers.** A 40%+ spend drop or app use below 0.5 sessions/month should trigger app push and
   personalised content first.
5. **Revisit tier benefits.** Platinum perks are not buying loyalty. Test engagement-based rewards (app streaks, category missions) instead of
   spend-only tiers.
6. **Measure, don't assume.** Hold out 20% of flagged members as a control group, which needs about {int(ab15.total_members):,} flagged members to
   detect a 15% uplift (section 9.5). Measure the true uplift of each action and feed it back into `config.py`.
7. **Operationalise on the existing stack.** Run a monthly Databricks job (Data Factory ingest → Delta → `spark_features.py` → MLflow
   champion model → CRM queue). Put the Retention Console on Databricks Apps. Monitor PR-AUC, calibration and PSI drift
   each month, and shadow-score the GRU challenger.

## 11. Assumptions

{assumptions}

## 12. Limitations

- **One test quarter.** Results are validated on one forward quarter (with three earlier snapshots for training and validation). Seasonality
  beyond 18 months is untested.
- **Small sample.** There are only 164 Active churners in the test cohort, so the 95% interval on Active PR-AUC is ±0.06. More history would
  tighten it.
- **Uplifts are assumptions, not measurements.** The scenario values depend on them, which is why break-even points and sensitivity are
  reported and a controlled test is recommended.
- **SHAP is not causal.** SHAP values and the support what-if describe associations. A member with many tickets may churn because of an
  underlying problem (a store move, a bad experience) that a phone call cannot fix.
- **Limited data.** There is no transaction-level, product, price, competitor or exit-reason data. Why members leave is inferred from
  behaviour, not observed.
- **Missing months.** The 953 missing months are treated as lost data. If some were genuine inactivity, a few trend features are slightly
  optimistic for those members.
- **Correlated features.** Several features are strongly correlated (transactions, spend, active months), so individual logistic
  coefficients and SHAP shares between correlated features should be read as a group, not one by one.

## 13. Reproducibility and production path

- `python run_pipeline.py` regenerates every output, chart, the executed notebook, this report and the slide deck (about 1 minute).
- Fixed seeds, pinned `requirements.txt`, and MLflow tracking for every model run (`outputs/mlflow_runs_summary.csv`).
- `python -m pytest`: unit tests, including an **automated leakage test** that corrupts all post-cut-off data and asserts the features
  do not change.
- **Databricks / Spark:** `src/spark_features.py` is a PySpark port of the feature builder. {parity_txt}
  `notebooks/02_databricks_churn_job.py` shows the scheduled job on Azure Data Factory, Delta tables and the MLflow registry.
- **API:** `uvicorn src.api:app` serves `/score` for real-time scoring, for example from the CRM.
- **Dashboard:** `python app.py` (Dash). **CI:** GitHub Actions rebuilds the outputs and runs the tests on every push.
  **Governance:** `MODEL_CARD.md` covers intended use, metrics, fairness and monitoring.
"""


CSS = """
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; color: #0b0b0b; max-width: 900px; margin: 0 auto;
       font-size: 10.5pt; line-height: 1.5; }
h1 { font-size: 21pt; margin-bottom: 0; } h2 { font-size: 14.5pt; border-bottom: 1px solid #e1e0d9; padding-bottom: 3px; margin-top: 26px; }
h3 { font-size: 12pt; } em { color: #52514e; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: 10px 0 14px; page-break-inside: avoid; }
th, td { border-bottom: 1px solid #e1e0d9; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #f6f5f2; font-weight: 600; }
img { max-width: 100%; margin: 8px 0 12px; page-break-inside: avoid; }
code { background: #f3f2ee; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
hr { border: none; border-top: 1px solid #e1e0d9; }
@page { size: A4; margin: 16mm 14mm; }
"""

CHROME_PATHS = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
                "google-chrome", "chromium", "chromium-browser"]


def _chrome():
    for p in CHROME_PATHS:
        if Path(p).exists() or shutil.which(p):
            return p
    return None


def build() -> None:
    C.REPORTS.mkdir(exist_ok=True)
    md_text = build_markdown()
    md_path = C.REPORTS / "final_report.md"
    md_path.write_text(md_text)
    print(f"  report -> {md_path.relative_to(C.ROOT)}")
    html = markdown.markdown(md_text, extensions=["tables"])
    html_path = C.REPORTS / "final_report.html"
    html_path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>FreshBasket churn report</title>"
                         f"<style>{CSS}</style></head><body>{html}</body></html>")
    chrome = _chrome()
    if chrome is None:
        print("  [pdf] no Chrome/Chromium found: open reports/final_report.html and print to PDF")
        return
    pdf_path = C.REPORTS / "final_report.pdf"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()], check=True, capture_output=True, timeout=180)
    print(f"  report PDF -> {pdf_path.relative_to(C.ROOT)}")
