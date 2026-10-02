# Silent Cart

[![ci](https://github.com/Guru-Prasaath/silent-cart/actions/workflows/ci.yml/badge.svg)](https://github.com/Guru-Prasaath/silent-cart/actions/workflows/ci.yml)

**Spotting FreshBasket loyalty members who go quiet before they churn: prediction, diagnosis & retention strategy.**

Customer churn prediction for FreshBasket Retail's Silver/Gold/Platinum loyalty program (2,600 members, Jan-2023 to Jun-2024).
One command rebuilds every output: data-quality audit, leakage-safe features, out-of-time validated models, SHAP drivers,
ablation, retention economics, survival analysis, personas, a deep-learning challenger, a next-quarter forecast,
the executed EDA notebook, the final report (MD + PDF), the slide deck and the model card. An interactive Dash console sits on top.

```bash
python run_pipeline.py      # rebuild everything (~1 minute)
python app.py               # Silent Cart Retention Console -> http://127.0.0.1:8050
```

## Key findings

| | |
|---|---|
| **80% of "churners" had already left** | 649 of 813 churners made no purchase in Jan-Mar 2024, before the Apr-Jun label window. They churn at 99.4% and need *win-back*, not prediction. |
| **The real problem: Active members** | 1,715 members who bought in Jan-Mar 2024 churn at 9.6%. On this group a recency rule gets PR-AUC 0.46; the selected model gets **0.79** on an untouched quarter, catching **76% of churners at 70% precision**. |
| **Churn has a fingerprint** | Purchases, app use and email opens fade over ~2 months while support tickets spike. 3+ tickets: 37% churn vs 4% with none. |
| **Tier does not protect** | Platinum churns like Silver (34% vs 35%). Tier, age and tenure are not significant after FDR correction. |
| **Targeting pays** | Blanket coupon (status quo): −$15.4k on this cohort. Next-best-action playbook: **+$3.6k**, at 84% lower cost. |
| **Next quarter (Jul-Sep 2024)** | About **149 Active churners expected** (90% interval 137-161); 144 flagged for action now. |
| **Deep learning helps a little** | A GRU on raw monthly sequences reaches Active PR-AUC 0.825 vs 0.794, winning in 97% of paired bootstrap resamples. It is deployed as a shadow challenger. |
| **Early app use predicts customer life** | Members with low app use in their first 3 months: 46% still buying after 12 months vs 70% (log-rank p < 0.001). |

Full narrative: [reports/final_report.md](reports/final_report.md) (and `.pdf`). Slides: [presentation/churn_presentation.pptx](presentation/churn_presentation.pptx), with speaker notes.

## What makes this approach different

- **Two-segment framing.** Every metric is reported for *all eligible* and *Active-only* members, because all-member metrics are inflated by easy, already-lapsed cases.
- **Rolling-origin validation.** The churn definition is replayed at Jul-23, Oct-23 and Jan-24 cut-offs, giving genuine out-of-time training and validation data. The rebuilt label agrees 99.8% with the official one. The official Apr-Jun 2024 label is used once, as the test set.
- **Leakage audit.** Three label-table columns use Apr-Jun 2024 data. They are excluded, and the inflation they would cause is measured (Active PR-AUC 0.80 → 0.87). A unit test corrupts all post-cut-off data and asserts that no feature changes.
- **Repairs instead of deletions.** `spend = transactions × basket` holds for 99.8% of rows, so 55 corrupted spend values are rebuilt exactly. Missing months are treated as lost data, not zero activity.
- **Calibrated probabilities plus economics.** Platt calibration makes scores usable as real odds. Each member gets a value at risk, a next best action (service call / coupon / win-back / monitor), and three plain-English SHAP drivers.
- **Built for the Tailwyndz stack.** MLflow tracking, a PySpark port of the feature pipeline (verified to match pandas exactly on all 4 snapshots), a Databricks job notebook, a FastAPI scoring endpoint, GitHub Actions CI and pytest tests.
- **Decision-grade extras.** A profit-optimal threshold curve, an A/B test power calculation (about 730 flagged members detect a 15% uplift), Kaplan-Meier survival with log-rank tests, k-means behavioural personas, PSI drift and fairness monitoring, and a [model card](MODEL_CARD.md).
- **A product, not just a notebook.** The Dash **Retention Console** has four tabs: KPIs; a filterable action list with a member drill-down (SHAP reasons, persona, activity history); a live scenario simulator with 9 assumption sliders; and an insights tab. Dash runs natively on Databricks Apps.

## Setup

Requires Python 3.11+ (developed on 3.14).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Gradient boosting uses scikit-learn rather than LightGBM/XGBoost. Those need the OpenMP runtime (`libomp`) installed separately on macOS, and scikit-learn keeps the project installable with pip alone.

Optional deep-learning challenger: `pip install -r requirements-dl.txt` (PyTorch). Without it the pipeline simply skips the GRU.

## Run

```bash
python run_pipeline.py                     # full run, ~1 minute
python run_pipeline.py --skip-notebook     # skip re-executing the EDA notebook
python -m pytest -q                        # leakage, labels, cleaning, API, scenario engine, dashboard
python app.py                              # Retention Console; ?tab=members|sim|insights links straight to a tab
uvicorn src.api:app                        # scoring API: GET /health, POST /score
```

The report PDF is printed with headless Google Chrome (or Brave) if installed; otherwise open `reports/final_report.html` and print it to PDF.

**Optional: Spark / Databricks.** Needs Java 17 or 21 (Spark 4.2 does not run on newer Java).

```bash
pip install -r requirements-spark.txt
export JAVA_HOME=/path/to/jdk-21
python scripts/spark_parity_check.py       # PySpark features == pandas features, all snapshots
```

`notebooks/02_databricks_churn_job.py` shows the production job: Azure Data Factory, then Delta tables, `spark_features.py`, the MLflow registry and the CRM scores table. It is a reference notebook and has not been run on a live workspace.

## Pipeline steps

| Step | Module | Output |
|---|---|---|
| 1. Data-quality checks & cleaning | `src/data_quality.py` | `outputs/data_quality_log.csv`, `data/processed/*_clean.csv` |
| 2. Leakage-safe snapshots (Jul-23, Oct-23, Jan-24, Apr-24) | `src/features.py` | `outputs/modelling_dataset.csv`, `data/processed/snapshot*.csv` |
| 3. EDA & hypothesis tests | `src/analysis.py` | figures 01-06, `hypothesis_tests_active_segment.csv` |
| 4. Models: rule, logistic, RF, GBM; calibration; threshold | `src/models.py` | `model_comparison.csv`, figures 07-09, `models/churn_model.joblib`, MLflow runs |
| 5. SHAP drivers & segment risk | `src/explain.py` | `churn_predictions.csv`, `shap_feature_importance.csv`, `segment_risk.csv`, figures 11-13 |
| 6. Retention scenarios, next best action, personas, profit threshold, A/B design | `src/scenario.py`, `src/personas.py`, `src/business.py` | `retention_scenarios.csv`, `retention_action_list.csv`, `personas.csv`, `profit_threshold_curve.csv`, `ab_test_design.csv`, figures 14-16, 18 |
| 7. Ablation & validation checks | `src/models.py` | `ablation_study.csv`, `validation_design_checks.csv`, figure 10 |
| 8. Survival, drift, fairness, GRU challenger, next-quarter forecast | `src/survival.py`, `src/monitoring.py`, `src/deep.py`, `src/forecast.py` | `survival_*.csv`, `drift_psi.csv`, `fairness_by_group.csv`, `deep_model_results.csv`, `next_quarter_watchlist.csv`, figures 17, 19 |
| 9-10. Key metrics, model card, notebook, report, deck | `src/model_card.py`, `src/notebook.py`, `src/report.py`, `src/presentation.py` | `key_metrics.json`, `MODEL_CARD.md`, `notebooks/01_data_quality_eda.ipynb`, `reports/`, `presentation/` |

## Deliverables checklist (brief → file)

| Required deliverable | Where |
|---|---|
| Reproducible Python project | `run_pipeline.py`, `src/`, `requirements.txt`, `tests/` |
| README with setup, assumptions, execution | this file |
| Data-quality & EDA notebook | `notebooks/01_data_quality_eda.ipynb` (executed, outputs saved) |
| Final customer-level modelling dataset | `outputs/modelling_dataset.csv` (2,368 eligible members × 42 features + label) |
| Model comparison & validation | `outputs/model_comparison.csv`, `outputs/validation_design_checks.csv`, figures 07-09 |
| Predictions with ID, probability, label, drivers | `outputs/churn_predictions.csv` (`customer_id, churn_probability, predicted_label, top_3_drivers`, plus actual and segment) |
| Driver attribution & feature impact | figures 11-13, `outputs/shap_feature_importance.csv`, `outputs/segment_risk.csv` |
| Ablation study | `outputs/ablation_study.csv`, figure 10 |
| Retention scenario analysis | `outputs/retention_scenarios.csv`, `outputs/scenario_sensitivity.csv`, figures 14-15, `outputs/retention_action_list.csv` |
| Final report | `reports/final_report.md`, `reports/final_report.pdf` |
| 10-15 minute presentation | `presentation/churn_presentation.pptx` (12 slides, speaker notes) |
| *Extra:* interactive dashboard | `app.py` + `assets/` (Dash Retention Console) |
| *Extra:* next-quarter forecast | `outputs/next_quarter_watchlist.csv` |
| *Extra:* model governance | `MODEL_CARD.md`, `.github/workflows/ci.yml` |

## Assumptions

1. **Target.** CHURNED = no transactions in Apr-Jun 2024 for a member who purchased before April; the official label is used as-is. Excluded, because they cannot be scored from history:
   - members who never purchased (120 in the label table);
   - 6 profiled members with no activity;
   - 5 labelled churners with no pre-April activity.

   That leaves 2,368 eligible members.
2. **Missing months.** A month absent *between* a member's first and last rows is lost data, not inactivity (MONTHS_OBSERVED counts it as observed). Months *after* the last row are zero activity, as a live system would see them.
3. **Spend repair.** Negative, zero (with transactions > 0) and 9-15x inflated spend values are rebuilt as transactions × average basket.
4. **Impossible zero-transaction values.** Categories bought and coupons redeemed in zero-transaction months are set to 0.
5. **Historical labels.** Earlier snapshots reuse the official definition (no purchase in the next 3 months). This agrees 99.8% with the official label at Apr-24.
6. **Feature window.** Features use the 6 months before each cut-off. Recency is capped at 7 (meaning "6+ months").
7. **Text cleaning.** City and tier variants are trimmed and title-cased. Missing gender and price tier become "Unknown". Missing age is filled with the training median, plus a missing flag.
8. **Model selection and threshold.** Selection uses validation Active-member PR-AUC. The threshold maximises validation Active-member F1 (0.37). Risk bands: High (p ≥ threshold), Medium (p ≥ 10%), Low.
9. **Economics.** These are assumptions, not measurements, and are varied in a sensitivity analysis; all are editable in `src/config.py`:
   - Value of a retained member = 6 months of typical monthly spend × 25% gross margin.
   - Coupon: $10 cost, cuts churn probability by 15% (relative).
   - Service call: $15 cost, cuts churn probability by 25%.
   - Win-back: $5 cost, 5% reactivation.
   - Any offer to a lapsed member works only as well as a win-back.
10. **Repeated members.** A member may appear in up to two training snapshots, so training rows are not fully independent. The test cohort is strictly later in time.

## Limitations

- **Soft personas.** Clusters are modest (silhouette ~0.12), so they are used for messaging, not targeting.
- **One test quarter.** Small Active-churner count (164), so the 95% CI on Active PR-AUC is about ±0.06.
- **Uplifts are assumed.** A hold-out control group is needed to measure them.
- **SHAP and what-ifs are associational, not causal.**
- **Behaviour only.** No transaction-level, price, competitor or exit-reason data.

## Project structure

```
data/raw/            source workbook + brief          data/processed/   cleaned tables, training snapshots
app.py, assets/      Dash Retention Console                MODEL_CARD.md     model governance
notebooks/           01 EDA (executed), 02 Databricks job  .github/workflows CI (pipeline + tests)
src/                 config, data_quality, features, analysis, models, explain, scenario, business,
                     personas, survival, monitoring, deep, forecast, model_card, report, presentation,
                     notebook, spark_features, api, viz
scripts/             spark_parity_check.py            tests/            pytest suite
outputs/             CSVs, figures/, models/           reports/          final_report.md/.pdf
presentation/        churn_presentation.pptx
```

Seeds are fixed (`SEED = 42`). MLflow runs are written to `./mlruns` (git-ignored) and summarised in `outputs/mlflow_runs_summary.csv`.
