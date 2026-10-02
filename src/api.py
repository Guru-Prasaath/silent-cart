"""Minimal scoring API: the deployable face of the model (e.g. behind Azure ML / Databricks Model Serving).

    uvicorn src.api:app --reload        then POST /score with member feature rows
"""
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from .config import OUTPUTS

app = FastAPI(title="FreshBasket churn scoring", version="1.0")


class Members(BaseModel):
    members: list[dict]   # one dict per member with the feature columns of outputs/modelling_dataset.csv


@lru_cache
def _model():
    return joblib.load(OUTPUTS / "models" / "churn_model.joblib")


@app.get("/health")
def health():
    m = _model()
    return {"status": "ok", "model": m.name, "threshold": round(m.threshold, 4), "n_features": len(m.features)}


@app.post("/score")
def score(req: Members):
    m = _model()
    df = pd.DataFrame(req.members)
    p = m.predict_proba(df)
    return [{"customer_id": r.get("customer_id"), "churn_probability": round(float(pi), 4),
             "predicted_label": int(pi >= m.threshold),
             "risk_band": "High" if pi >= m.threshold else "Medium" if pi >= 0.10 else "Low"}
            for r, pi in zip(req.members, p)]
