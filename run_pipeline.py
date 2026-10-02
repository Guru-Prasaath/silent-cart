"""FreshBasket loyalty churn: end-to-end pipeline.

    python run_pipeline.py                # full run (data -> models -> SHAP -> scenarios -> report -> slides)
    python run_pipeline.py --skip-notebook --skip-deck
"""
import argparse
import json
import logging
import os
import time
import warnings

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

from src import analysis, config as C, data_quality, explain, features, models, scenario

np.random.seed(C.SEED)


def step(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}")


def main(args):
    C.ensure_dirs()

    step("1/9 Data quality checks and cleaning")
    raw = data_quality.load_raw()
    profile, activity, labels, dq = data_quality.run()
    print(f"  {len(dq)} checks logged -> outputs/data_quality_log.csv")

    step("2/9 Leakage-safe feature snapshots (rolling origin)")
    train = pd.concat([features.build_snapshot(profile, activity, c) for c in C.TRAIN_CUTOFFS], ignore_index=True)
    valid = features.build_snapshot(profile, activity, C.VALID_CUTOFF)
    test = features.build_snapshot(profile, activity, C.TEST_CUTOFF, labels)
    features.fill_age([train, valid, test], train)
    train.to_csv(C.PROCESSED / "snapshots_train.csv", index=False)
    valid.to_csv(C.PROCESSED / "snapshot_valid.csv", index=False)
    test.to_csv(C.OUTPUTS / "modelling_dataset.csv", index=False)
    for n, d in [("train", train), ("valid", valid), ("test", test)]:
        print(f"  {n:5s}: {len(d):5,} rows, churn {d.churned.mean():.1%}, Active churn {d[d.segment == 'Active'].churned.mean():.1%}")

    step("3/9 EDA charts and hypothesis tests")
    analysis.run_eda(raw, activity, test)
    tests = analysis.hypothesis_tests(test)

    step("4/9 Train / select / calibrate models (selection = validation Active PR-AUC)")
    fitted, best, comp = models.train_and_select(train, valid, test)
    comp.round(4).to_csv(C.OUTPUTS / "model_comparison.csv", index=False)
    cm = fitted[best]
    models.save_model(cm)
    print(f"  selected: {best} (threshold {cm.threshold:.3f})")
    print(comp[(comp.split == "test")][["model", "population", "precision", "recall", "f1", "pr_auc", "roc_auc", "accuracy"]]
          .round(3).to_string(index=False))
    models.fig_pr_curves(fitted, test)
    models.fig_confusion(cm, test)
    models.fig_calibration(fitted, test)

    step("5/9 SHAP drivers, per-member top-3 drivers, segment risk")
    test["churn_probability"] = cm.predict_proba(test)
    test["predicted_label"] = (test.churn_probability >= cm.threshold).astype(int)
    sv = explain.shap_values(cm, train, test)
    test["top_3_drivers"] = explain.top_drivers(sv, test)
    act = test.segment == "Active"
    explain.fig_shap_summary(sv[act.to_numpy()], test[act], best)
    challenger = "Gradient boosting" if best != "Gradient boosting" else "Logistic regression"
    sv_ch = explain.shap_values(fitted[challenger], train, test[act])
    explain.fig_shap_groups(sv[act.to_numpy()], sv_ch, (best, challenger))
    shap_rank = pd.DataFrame({"feature": sv.columns,
                              "mean_abs_shap_active": sv[act.to_numpy()].abs().mean().values,
                              f"mean_abs_shap_active_{challenger.lower().replace(' ', '_')}": sv_ch.abs().mean().reindex(sv.columns).values,
                              "group": [explain.GROUP_OF[f] for f in sv.columns]}).sort_values("mean_abs_shap_active", ascending=False)
    shap_rank.round(5).to_csv(C.OUTPUTS / "shap_feature_importance.csv", index=False)

    step("6/9 Retention scenarios and next-best-action list")
    test_econ, nba, scen, whatif, sens = scenario.run(cm, test)
    test["revenue_at_risk"] = test_econ.revenue_at_risk
    seg = explain.segment_risk(test)
    explain.fig_segment_risk(seg)
    scenario.fig_scenarios(scen)
    scenario.fig_sensitivity(sens, scen)

    preds = test[["customer_id", "churn_probability", "predicted_label", "top_3_drivers"]].assign(
        actual_churned=test.churned, segment=test.segment)
    preds.round({"churn_probability": 4}).sort_values("churn_probability", ascending=False).to_csv(
        C.OUTPUTS / "churn_predictions.csv", index=False)
    band = np.select([test.churn_probability >= cm.threshold, test.churn_probability >= 0.10], ["High", "Medium"], "Low")
    actions = nba.assign(risk_band=band)[["customer_id", "segment", "tier", "city", "preferred_category", "churn_probability",
                                          "risk_band", "value_monthly_spend", "revenue_at_risk", "recommended_action",
                                          "expected_net_value", "top_3_drivers"]]
    actions = actions.sort_values(["expected_net_value", "revenue_at_risk"], ascending=False)
    actions.insert(0, "priority_rank", np.arange(1, len(actions) + 1))
    actions.round(2).to_csv(C.OUTPUTS / "retention_action_list.csv", index=False)
    print(scen[["scenario", "members_targeted", "churners_prevented", "cost", "margin_retained", "net_value", "roi"]].round(1).to_string(index=False))

    step("7/9 Ablation study and validation-design checks")
    best_gb = eval(comp[comp.model == "Gradient boosting"].params.iloc[0])
    ab = models.ablation(train, test, best_gb)
    models.fig_ablation(ab)
    checks = models.leakage_and_split_checks(test, labels, best_gb)
    oot = comp[(comp.model == "Gradient boosting") & (comp.split == "test")]
    checks = pd.concat([pd.DataFrame([{"setup": "Out-of-time test (reported), clean features",
                                       "pr_auc_all": oot[oot.population == "All eligible"].pr_auc.iloc[0],
                                       "pr_auc_active": oot[oot.population == "Active only"].pr_auc.iloc[0]}]), checks])
    checks.round(4).to_csv(C.OUTPUTS / "validation_design_checks.csv", index=False)
    print(ab.round(3).to_string(index=False))
    print(checks.round(3).to_string(index=False))

    step("8/9 Key metrics snapshot")
    t = comp[(comp.model == best) & (comp.split == "test")].set_index("population")
    key = {
        "selected_model": best, "threshold": cm.threshold, "challenger": challenger,
        "n_test": len(test), "n_active": int(act.sum()), "n_lapsed": int((~act).sum()),
        "churn_rate_all": test.churned.mean(), "churn_rate_active": test[act].churned.mean(),
        "churn_rate_lapsed": test[~act].churned.mean(),
        "share_churners_already_lapsed": test[~act].churned.sum() / test.churned.sum(),
        "test_all": t.loc["All eligible"].drop(["model", "params", "split"]).to_dict(),
        "test_active": t.loc["Active only"].drop(["model", "params", "split"]).to_dict(),
        "complaint_test": tests.attrs["complaint_chi2"],
        "support_whatif": whatif,
        "revenue_at_risk_total": float(test.revenue_at_risk.sum()),
        "revenue_at_risk_active": float(test[act].revenue_at_risk.sum()),
        "high_risk_active": int((act & (test.predicted_label == 1)).sum()),
        "nba_counts": actions.recommended_action.value_counts().to_dict(),
    }
    (C.OUTPUTS / "key_metrics.json").write_text(json.dumps(key, indent=2, default=float))
    models.export_mlflow_runs()

    step("9/9 Notebook, report and presentation")
    if not args.skip_notebook:
        from src import notebook
        notebook.build_and_execute()
    if not args.skip_report:
        from src import report
        report.build()
    if not args.skip_deck:
        from src import presentation
        presentation.build()
    print("\nDone. Outputs in outputs/, reports/, presentation/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-notebook", action="store_true")
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--skip-deck", action="store_true")
    main(ap.parse_args())
