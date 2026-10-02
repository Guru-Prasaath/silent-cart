# Model card: Silent Cart churn model

| | |
|---|---|
| **Model** | Logistic regression, class-weighted, Platt-calibrated; decision threshold 0.37 |
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
| All eligible | 2,368 | 34.3% | 0.986 (0.98-0.99) | 0.991 | 0.93 | 0.95 | 0.029 |
| **Active only** | 1,715 | 9.6% | **0.794** (0.73-0.85) | 0.961 | 0.70 | 0.76 | 0.038 |

The recency-rule baseline scores 0.46 PR-AUC on Active members.
A GRU sequence model (PyTorch) is the challenger: Active PR-AUC 0.825 (95% CI 0.76-0.87), paired-bootstrap gain +0.031 (95% CI -0.001 to +0.067). Plan: shadow-score for one quarter, then promote if the gain holds.

## Fairness check (Active members, deployed threshold)
| Dimension | Group | Members | Churners | Recall | Precision | FPR |
|---|---|---|---|---|---|---|
| age_band | 18-34 | 524 | 49 | 0.73 | 0.67 | 0.038 |
| age_band | 35-54 | 583 | 51 | 0.73 | 0.65 | 0.038 |
| age_band | 55+ | 608 | 64 | 0.81 | 0.76 | 0.029 |
| gender | F | 897 | 84 | 0.76 | 0.67 | 0.038 |
| gender | M | 723 | 69 | 0.74 | 0.71 | 0.032 |
| gender | Other | 63 | 6 | 0.83 | 0.83 | 0.018 |
| gender | Unknown | 32 | 5 | 1.00 | 0.83 | 0.037 |
| tier | Gold | 554 | 54 | 0.91 | 0.77 | 0.030 |
| tier | Platinum | 202 | 19 | 0.53 | 0.71 | 0.022 |
| tier | Silver | 959 | 91 | 0.73 | 0.65 | 0.040 |

- **Age and gender:** error rates are similar across groups.
- **Platinum recall is lower:** 0.53, from only
  19 churners. The interval is wide, but it is a
  watch item: high-value Platinum members are slightly under-flagged.
- **Protected attributes:** gender and age are model inputs but carry almost no signal (see the ablation). Removing them is a
  supported option if policy requires it.

## Monitoring
- **Drift (PSI, test vs train):** the maximum is 0.23 (tenure_months).
  12 features are in the 0.1-0.25 watch band, and none exceed 0.25.
- **Alerts:** PSI > 0.25 on any input; Active PR-AUC below 0.70 on the latest labelled quarter; calibration slope outside 0.8-1.2.
- **Retraining:** quarterly, adding the newest labelled snapshot; champion/challenger comparison on the same out-of-time quarter.

## Limitations
- **One test quarter, 164 Active churners:** the 95% CI on PR-AUC is about ±0.06.
- **Associations only:** SHAP drivers and what-ifs are associational. Retention uplifts are assumptions until the A/B hold-out measures them.
- **No exit-reason, price, competitor or transaction-level data.**
