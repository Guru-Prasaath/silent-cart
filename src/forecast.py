"""Forward-looking scoring: who is likely to churn in Jul-Sep 2024, the quarter after the data ends.

This is how the model is used in production: a snapshot at cut-off Jul-2024 uses Jan-Jun 2024 behaviour, and no label
exists yet. Expected churners and a 90% interval come from the calibrated probabilities (Poisson-binomial, normal approx).
"""
import numpy as np
import pandas as pd

from . import config as C, explain, features, scenario

NEXT_CUTOFF = pd.Timestamp("2024-07-01")


def next_quarter(profile, activity, cm, train) -> tuple[pd.DataFrame, dict]:
    snap = features.build_snapshot(profile, activity, NEXT_CUTOFF).drop(columns="churned")  # label unknown: future
    features.fill_age([snap], train)
    snap["churn_probability"] = cm.predict_proba(snap)
    snap["predicted_label"] = (snap.churn_probability >= cm.threshold).astype(int)
    snap["top_3_drivers"] = explain.top_drivers(explain.shap_values(cm, train, snap), snap)
    snap = scenario.next_best_action(scenario.add_economics(snap))
    snap["risk_band"] = np.select([snap.churn_probability >= cm.threshold, snap.churn_probability >= C.MEDIUM_RISK], ["High", "Medium"], "Low")
    out = snap[["customer_id", "segment", "tier", "city", "preferred_category", "churn_probability", "predicted_label", "risk_band",
                "revenue_at_risk", "recommended_action", "expected_net_value", "top_3_drivers",
                "tickets_l3", "tickets_6m", "complaint_l3"]]
    out = out.sort_values(["expected_net_value", "churn_probability"], ascending=False)
    out.round(4).to_csv(C.OUTPUTS / "next_quarter_watchlist.csv", index=False)

    summary = {"cutoff": NEXT_CUTOFF.strftime("%Y-%m"), "window": "Jul-Sep 2024", "members_scored": len(snap)}
    for seg, g in [("all", snap), ("active", snap[snap.segment == "Active"]), ("lapsed", snap[snap.segment == "Lapsed"])]:
        p = g.churn_probability.to_numpy()
        mu, sd = p.sum(), np.sqrt((p * (1 - p)).sum())
        summary[seg] = {"members": len(g), "expected_churners": mu, "interval_90": [mu - 1.645 * sd, mu + 1.645 * sd],
                        "high_risk": int((g.risk_band == "High").sum()),
                        "revenue_at_risk_6m": float(g.revenue_at_risk.sum())}
    return out, summary
