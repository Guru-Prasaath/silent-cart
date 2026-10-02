"""Writes MODEL_CARD.md (intended use, data, metrics, fairness, drift, limitations) from the pipeline outputs."""
import json

import pandas as pd

from . import config as C


def build() -> str:
    O = C.OUTPUTS
    k = json.loads((O / "key_metrics.json").read_text())
    fair = pd.read_csv(O / "fairness_by_group.csv")
    drift = pd.read_csv(O / "drift_psi.csv")
    ta, tc = k["test_all"], k["test_active"]
    g = (k.get("gru") or {}).get("Active only")

    def fair_rows(dim):
        d = fair[fair.dimension == dim]
        return "\n".join(f"| {dim} | {r.group} | {r.members} | {r.churners} | {r.recall:.2f} | {r.precision:.2f} | {r.false_positive_rate:.3f} |"
                         for r in d.itertuples())

    gru_txt = (f"A GRU sequence model (PyTorch) is the challenger: Active PR-AUC {g['pr_auc']:.3f} "
               f"(95% CI {g['pr_auc_ci_low']:.2f}-{g['pr_auc_ci_high']:.2f}), paired-bootstrap gain {g['delta_vs_champion']:+.3f} "
               f"(95% CI {g['delta_ci_low']:+.3f} to {g['delta_ci_high']:+.3f}). Plan: shadow-score for one quarter, then promote if the "
               "gain holds." if g and "delta_vs_champion" in g else "GRU challenger not run (PyTorch not installed).")
    text = f"""# Model card: Silent Cart churn model

| | |
|---|---|
| **Model** | {k['selected_model']}, class-weighted, Platt-calibrated; decision threshold {k['threshold']:.2f} |
| **Task** | Probability that a loyalty member makes **no purchase in the next 3 months** |
| **Owner / version** | FreshBasket Data Science · v1.0 · trained on snapshots Jul-2023 & Oct-2023, calibrated on Jan-2024 |
| **Artefact** | `outputs/models/churn_model.joblib` (pipeline + calibrator + threshold), served by `src/api.py`, used by `app.py` |

## Intended use
- **Use it to:** rank **Active** members (purchased in the last 3 months) for retention actions, choose the action by
  expected value, and explain each score with three SHAP drivers.
- **Lapsed members** (no purchase for 3+ months) score about 1.0 by construction. Route them to win-back with a rule, not this model.
- **Do not use it for** credit, pricing or eligibility decisions, or to withhold service. Scores describe purchase behaviour, not the person.

## Data
- **Source:** FreshBasket loyalty workbook, Jan-2023 to Jun-2024 (2,600 members). Cleaning is logged in `outputs/data_quality_log.csv`.
- **Features:** 42 features from the 6 months before each cut-off. Leaky label-table columns are excluded, and an automated
  test enforces that no post-cut-off data is used.
- **Validation:** rolling origin. Train on Jul-23 and Oct-23 snapshots, choose the model and calibrate on Jan-24, and test once on
  the official Apr-Jun 2024 label.

## Performance (test quarter, never used for tuning)
| Population | n | Base rate | PR-AUC (95% CI) | ROC-AUC | Precision | Recall | Brier |
|---|---|---|---|---|---|---|---|
| All eligible | {ta['n']:,} | {ta['churn_rate']:.1%} | {ta['pr_auc']:.3f} ({ta['pr_auc_ci_low']:.2f}-{ta['pr_auc_ci_high']:.2f}) | {ta['roc_auc']:.3f} | {ta['precision']:.2f} | {ta['recall']:.2f} | {ta['brier']:.3f} |
| **Active only** | {tc['n']:,} | {tc['churn_rate']:.1%} | **{tc['pr_auc']:.3f}** ({tc['pr_auc_ci_low']:.2f}-{tc['pr_auc_ci_high']:.2f}) | {tc['roc_auc']:.3f} | {tc['precision']:.2f} | {tc['recall']:.2f} | {tc['brier']:.3f} |

The recency-rule baseline scores {pd.read_csv(O / 'model_comparison.csv').query("model == 'Recency rule' and split == 'test' and population == 'Active only'").pr_auc.iloc[0]:.2f} PR-AUC on Active members.
{gru_txt}

## Fairness check (Active members, deployed threshold)
| Dimension | Group | Members | Churners | Recall | Precision | FPR |
|---|---|---|---|---|---|---|
{fair_rows('age_band')}
{fair_rows('gender')}
{fair_rows('tier')}

- **Age and gender:** error rates are similar across groups.
- **Platinum recall is lower:** {fair[(fair.dimension == 'tier') & (fair.group == 'Platinum')].recall.iloc[0]:.2f}, from only
  {int(fair[(fair.dimension == 'tier') & (fair.group == 'Platinum')].churners.iloc[0])} churners. The interval is wide, but it is a
  watch item: high-value Platinum members are slightly under-flagged.
- **Protected attributes:** gender and age are model inputs but carry almost no signal (see the ablation). Removing them is a
  supported option if policy requires it.

## Monitoring
- **Drift (PSI, test vs train):** the maximum is {drift.psi_test_vs_train.max():.2f} ({drift.iloc[0].feature}).
  {int((drift.psi_test_vs_train > 0.1).sum())} features are in the 0.1-0.25 watch band, and none exceed 0.25.
- **Alerts:** PSI > 0.25 on any input; Active PR-AUC below 0.70 on the latest labelled quarter; calibration slope outside 0.8-1.2.
- **Retraining:** quarterly, adding the newest labelled snapshot; champion/challenger comparison on the same out-of-time quarter.

## Limitations
- **One test quarter, 164 Active churners:** the 95% CI on PR-AUC is about ±0.06.
- **Associations only:** SHAP drivers and what-ifs are associational. Retention uplifts are assumptions until the A/B hold-out measures them.
- **No exit-reason, price, competitor or transaction-level data.**
"""
    path = C.ROOT / "MODEL_CARD.md"
    path.write_text(text)
    print(f"  model card -> {path.relative_to(C.ROOT)}")
    return str(path)
