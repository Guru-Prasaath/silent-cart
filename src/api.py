"""Scoring API: the deployable face of the model (e.g. behind Azure ML / Databricks Model Serving).

    uvicorn src.api:app --reload        then POST /score with member feature rows (see /docs)

Returns, per member: calibrated churn probability, label, risk band, the top-3 SHAP drivers in plain English and,
when value/segment/support columns are supplied, the next best action with its expected net value.
"""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from . import config as C, explain, scenario

app = FastAPI(title="Silent Cart churn scoring", version="1.1")
NBA_COLUMNS = {"segment", "value_monthly_spend", "tickets_6m", "complaint_l3"}


class Members(BaseModel):
    members: list[dict]   # one dict per member with the feature columns of outputs/modelling_dataset.csv


@lru_cache
def _model():
    return joblib.load(C.OUTPUTS / "models" / "churn_model.joblib")


@lru_cache
def _background():
    """Training rows used as the SHAP reference distribution (same as the pipeline)."""
    return pd.read_csv(C.PROCESSED / "snapshots_train.csv")


@app.get("/health")
def health():
    m = _model()
    return {"status": "ok", "model": m.name, "threshold": round(m.threshold, 4), "n_features": len(m.features)}


@app.post("/score")
def score(req: Members):
    m = _model()
    df = pd.DataFrame(req.members)
    df["churn_probability"] = m.predict_proba(df)
    df["top_3_drivers"] = explain.top_drivers(explain.shap_values(m, _background(), df), df)
    if NBA_COLUMNS.issubset(df.columns):
        df = scenario.next_best_action(scenario.add_economics(df))
    out = []
    for _, r in df.iterrows():
        p = float(r.churn_probability)
        rec = {"customer_id": r.get("customer_id"), "churn_probability": round(p, 4), "predicted_label": int(p >= m.threshold),
               "risk_band": "High" if p >= m.threshold else "Medium" if p >= C.MEDIUM_RISK else "Low",
               "top_3_drivers": r.top_3_drivers.split("; ")}
        if "recommended_action" in r:
            rec["recommended_action"] = r.recommended_action
            rec["expected_net_value"] = round(float(r.expected_net_value), 2)
        out.append({k: (v.item() if isinstance(v, np.generic) else v) for k, v in rec.items()})
    return out
