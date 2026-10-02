"""Run with:  python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import config as C, data_quality, features


@pytest.fixture(scope="module")
def data():
    profile, activity, labels, log = data_quality.run(save=False)
    return profile, activity, labels, log


def test_no_future_leakage(data):
    """Corrupting every row on/after the cutoff must not change a single feature value."""
    profile, activity, _, _ = data
    cutoff = C.TEST_CUTOFF
    base = features.build_snapshot(profile, activity, cutoff).drop(columns="churned")
    noisy = activity.copy()
    future = noisy.MONTH >= cutoff
    for c in ["TRANSACTIONS", "TOTAL_SPEND", "APP_SESSIONS", "SUPPORT_TICKETS", "COMPLAINT_FLAG"]:
        noisy.loc[future, c] = noisy.loc[future, c] * 7 + 3
    after = features.build_snapshot(profile, noisy, cutoff).drop(columns="churned")
    pd.testing.assert_frame_equal(base, after)


def test_label_matches_official(data):
    profile, activity, labels, _ = data
    derived = features.build_snapshot(profile, activity, C.TEST_CUTOFF).set_index("customer_id").churned
    official = features.build_snapshot(profile, activity, C.TEST_CUTOFF, labels).set_index("customer_id").churned
    common = derived.index.intersection(official.index)
    assert (derived[common] == official[common]).mean() > 0.995


def test_cleaning_invariants(data):
    profile, activity, _, log = data
    assert not activity.duplicated(["CUSTOMER_ID", "MONTH"]).any()
    assert (activity.TOTAL_SPEND >= 0).all()
    assert not ((activity.TRANSACTIONS > 0) & (activity.TOTAL_SPEND == 0)).any()
    assert set(profile.MEMBERSHIP_TIER) == {"Silver", "Gold", "Platinum"}
    assert profile.CITY.nunique() == 10
    assert len(log) > 20


def test_api_scores_members():
    from fastapi.testclient import TestClient
    from src.api import app
    rows = pd.read_csv(C.OUTPUTS / "modelling_dataset.csv").head(5)
    rows = rows.replace({np.nan: None}).to_dict(orient="records")
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    out = client.post("/score", json={"members": rows}).json()
    assert len(out) == 5 and all(0 <= r["churn_probability"] <= 1 for r in out)


def test_scenario_engine_matches_pipeline_output():
    """The dashboard's live simulator and the pipeline must produce identical numbers."""
    from src.scenario import simulate
    cohort = pd.read_csv(C.OUTPUTS / "modelling_dataset.csv").merge(
        pd.read_csv(C.OUTPUTS / "churn_predictions.csv")[["customer_id", "churn_probability"]], on="customer_id")
    import json
    thr = json.loads((C.OUTPUTS / "key_metrics.json").read_text())["threshold"]
    table, *_ = simulate(cohort, thr)
    saved = pd.read_csv(C.OUTPUTS / "retention_scenarios.csv")
    assert np.allclose(table.net_value.round(2), saved.net_value.round(2), atol=0.05)


def test_dashboard_callbacks_render():
    import app
    for tab in ["overview", "members", "sim", "insights"]:
        assert app.render(tab) is not None
    rows, cols, summary = app.filter_members("current", ["Active"], ["High"], None, None, None, None)
    assert rows and all(r["risk_band"] == "High" for r in rows)
    kpis, fig, table = app.run_sim(10, 0.15, 15, 0.25, 5, 0.05, 0.25, 6, 0.37)
    assert len(kpis) == 3
