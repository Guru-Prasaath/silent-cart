"""Leakage-safe, customer-level feature snapshots.

A snapshot at `cutoff` uses ONLY activity rows with MONTH < cutoff. Its label is "no purchase
in [cutoff, cutoff + 3 months)", which is exactly how the official CHURNED target is defined
for cutoff = Apr-2024. Building the same snapshot at earlier cutoffs gives us genuine
out-of-time training data (rolling-origin validation) without touching the test window.

Missing-month handling inside the 6-month lookback, per customer and month:
  * row present                          -> observed value
  * no row, after the member's last row  -> 0 (trailing silence: what a live system would see)
  * no row, between two existing rows    -> missing (data gap, not inactivity)
  * no row, before the member's first row-> missing (not yet active)
"""
import warnings

import numpy as np
import pandas as pd

from .config import FEATURE_LOOKBACK_MONTHS as W, LABEL_WINDOW_MONTHS

METRICS = ["TRANSACTIONS", "TOTAL_SPEND", "DISTINCT_CATEGORIES", "PROMO_TXN_PCT", "APP_SESSIONS",
           "EMAILS_OPENED", "COUPONS_REDEEMED", "SUPPORT_TICKETS", "COMPLAINT_FLAG"]
CAP = W + 1   # recency-style features are capped at 7 = "more than 6 months ago / never"

# Feature groups drive the ablation study and the per-customer driver text.
FEATURE_GROUPS = {
    "demographic": ["age", "age_missing", "tier_rank", "marketing_opt_in", "tenure_months",
                    "gender", "city", "preferred_category", "home_store_price_tier"],
    "behavioural": ["recency_months", "txn_last_month", "txn_l3", "txn_p3", "txn_change_pct",
                    "spend_l3", "spend_p3", "spend_change_pct", "txn_slope_6m", "spend_cv_6m",
                    "active_months_l3", "active_ratio_6m", "avg_basket_l3", "categories_l3",
                    "promo_pct_6m"],
    "engagement": ["app_l3", "app_p3", "app_change_pct", "app_slope_6m", "email_l3", "email_p3",
                   "email_change_pct", "coupons_l3", "coupons_p3"],
    "support": ["tickets_l3", "tickets_p3", "tickets_6m", "complaints_6m", "complaint_l3",
                "months_since_ticket"],
    "history": ["history_months", "short_history", "missing_months_6m"],
}
CATEGORICAL = ["gender", "city", "preferred_category", "home_store_price_tier"]
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
TIER_RANK = {"Silver": 0, "Gold": 1, "Platinum": 2}


def _mi(ts) -> np.ndarray:
    ts = pd.DatetimeIndex(ts) if not isinstance(ts, pd.Timestamp) else ts
    return ts.year * 12 + ts.month


def _nanmean(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=1)


def _slope(y: np.ndarray) -> np.ndarray:
    """NaN-aware OLS slope per row; columns are k=1..W (k=1 most recent), x runs forward in time."""
    x = -np.arange(1, y.shape[1] + 1, dtype=float)[None, :].repeat(len(y), 0)
    m = ~np.isnan(y)
    n = m.sum(1)
    xm = np.where(m, x, 0).sum(1) / np.maximum(n, 1)
    ym = np.where(m, y, 0).sum(1) / np.maximum(n, 1)
    num = np.where(m, (x - xm[:, None]) * (y - ym[:, None]), 0).sum(1)
    den = np.where(m, (x - xm[:, None]) ** 2, 0).sum(1)
    return np.where((n >= 3) & (den > 0), num / np.where(den > 0, den, 1), 0.0)


def _pct_change(recent, prior):
    return np.clip((recent - prior) / np.maximum(prior, 0.5), -1, 3)


def build_snapshot(profile: pd.DataFrame, activity: pd.DataFrame, cutoff: pd.Timestamp,
                   official_labels: pd.DataFrame | None = None) -> pd.DataFrame:
    c = int(_mi(cutoff))
    act = activity.assign(k=c - _mi(activity.MONTH))
    hist = act[act.k >= 1]

    last_buy_k = hist[hist.TRANSACTIONS > 0].groupby("CUSTOMER_ID").k.min()
    prof = profile.set_index("CUSTOMER_ID")
    signed_up = prof.index[prof.SIGNUP_DATE < cutoff]
    ids = last_buy_k.index.intersection(signed_up)          # churn-eligible: has purchase history
    if official_labels is not None:
        lab = official_labels.set_index("CUSTOMER_ID")
        ids = ids.intersection(lab.index)

    first_k = hist.groupby("CUSTOMER_ID").k.max().reindex(ids).to_numpy()
    last_k = hist.groupby("CUSTOMER_ID").k.min().reindex(ids).to_numpy()
    K = np.arange(1, W + 1)[None, :]
    win = hist[hist.k <= W]
    wide = {m: win.pivot(index="CUSTOMER_ID", columns="k", values=m).reindex(index=ids, columns=range(1, W + 1)).to_numpy(float, copy=True)
            for m in METRICS}
    present = ~np.isnan(wide["TRANSACTIONS"])
    trailing = (K < last_k[:, None]) & ~present
    before_first = K > first_k[:, None]
    gap = ~present & ~trailing & ~before_first
    for m in METRICS:
        wide[m][trailing] = 0.0

    L, P = slice(0, 3), slice(3, 6)

    def l3p3(m):
        l3, p3 = _nanmean(wide[m][:, L]), _nanmean(wide[m][:, P])
        l3 = np.where(np.isnan(l3), p3, l3)
        p3 = np.where(np.isnan(p3), l3, p3)       # new members: assume flat, flagged by short_history
        return np.nan_to_num(l3), np.nan_to_num(p3)

    f = pd.DataFrame(index=ids)
    f.index.name = "customer_id"
    txn_l3, txn_p3 = l3p3("TRANSACTIONS")
    spend_l3, spend_p3 = l3p3("TOTAL_SPEND")
    app_l3, app_p3 = l3p3("APP_SESSIONS")
    email_l3, email_p3 = l3p3("EMAILS_OPENED")
    cpn_l3, cpn_p3 = l3p3("COUPONS_REDEEMED")
    tkt = np.nan_to_num(wide["SUPPORT_TICKETS"])
    txn = wide["TRANSACTIONS"]

    # Behavioural
    f["recency_months"] = np.minimum(last_buy_k.reindex(ids).to_numpy(), CAP)
    f["txn_last_month"] = np.nan_to_num(np.where(np.isnan(txn[:, 0]), txn_l3, txn[:, 0]))
    f["txn_l3"], f["txn_p3"] = txn_l3, txn_p3
    f["txn_change_pct"] = _pct_change(txn_l3, txn_p3)
    f["spend_l3"], f["spend_p3"] = spend_l3, spend_p3
    f["spend_change_pct"] = _pct_change(spend_l3, spend_p3)
    f["txn_slope_6m"] = _slope(txn)
    sp = wide["TOTAL_SPEND"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f["spend_cv_6m"] = np.nan_to_num(np.nanstd(sp, 1) / np.maximum(np.nanmean(sp, 1), 1))
    f["active_months_l3"] = (np.nan_to_num(txn[:, L]) > 0).sum(1)
    obs = (~np.isnan(txn)).sum(1)
    f["active_ratio_6m"] = (np.nan_to_num(txn) > 0).sum(1) / np.maximum(obs, 1)
    f["avg_basket_l3"] = np.where(txn_l3 > 0, spend_l3 / np.maximum(txn_l3, 1e-9), 0)
    f["categories_l3"] = l3p3("DISTINCT_CATEGORIES")[0]
    promo = np.where(np.nan_to_num(txn) > 0, wide["PROMO_TXN_PCT"], np.nan)
    f["promo_pct_6m"] = np.nan_to_num(_nanmean(promo))

    # Engagement
    f["app_l3"], f["app_p3"], f["app_change_pct"] = app_l3, app_p3, _pct_change(app_l3, app_p3)
    f["app_slope_6m"] = _slope(wide["APP_SESSIONS"])
    f["email_l3"], f["email_p3"], f["email_change_pct"] = email_l3, email_p3, _pct_change(email_l3, email_p3)
    f["coupons_l3"], f["coupons_p3"] = cpn_l3, cpn_p3

    # Support
    f["tickets_l3"] = tkt[:, L].sum(1)
    f["tickets_p3"] = tkt[:, P].sum(1)
    f["tickets_6m"] = tkt.sum(1)
    cf = np.nan_to_num(wide["COMPLAINT_FLAG"])
    f["complaints_6m"] = cf.sum(1)
    f["complaint_l3"] = (cf[:, L].sum(1) > 0).astype(int)
    last_tkt = hist[hist.SUPPORT_TICKETS > 0].groupby("CUSTOMER_ID").k.min().reindex(ids)
    f["months_since_ticket"] = np.minimum(last_tkt.fillna(CAP).to_numpy(), CAP)

    # History / data-quality (leakage-safe replacement for INSUFFICIENT_HISTORY_FLAG)
    f["history_months"] = np.minimum(first_k, W)
    f["short_history"] = (first_k < 4).astype(int)
    f["missing_months_6m"] = gap.sum(1)

    # Profile / demographic
    p = prof.reindex(ids)
    f["age"] = p.AGE.to_numpy()
    f["age_missing"] = p.AGE.isna().astype(int).to_numpy()
    f["tier"] = p.MEMBERSHIP_TIER.to_numpy()
    f["tier_rank"] = p.MEMBERSHIP_TIER.map(TIER_RANK).to_numpy()
    f["marketing_opt_in"] = p.MARKETING_OPT_IN.to_numpy()
    f["tenure_months"] = c - _mi(p.SIGNUP_DATE)
    f["gender"] = p.GENDER.to_numpy()
    f["city"] = p.CITY.to_numpy()
    f["preferred_category"] = p.PREFERRED_CATEGORY.to_numpy()
    f["home_store_price_tier"] = p.HOME_STORE_PRICE_TIER.to_numpy()
    f["signup_cohort"] = p.SIGNUP_DATE.dt.year.astype(str).to_numpy()

    # Context for the business layer (not model inputs)
    f["avg_monthly_spend_6m"] = np.nan_to_num(_nanmean(sp))
    # Typical monthly spend when shopping (full pre-cutoff history): the revenue a retained or
    # re-activated member is expected to bring back. Used for value-at-risk, not as a feature.
    f["value_monthly_spend"] = hist[hist.TRANSACTIONS > 0].groupby("CUSTOMER_ID").TOTAL_SPEND.mean().reindex(ids).to_numpy()
    f["segment"] = np.where(f.recency_months <= 3, "Active", "Lapsed")
    f["cutoff"] = cutoff.strftime("%Y-%m")

    # Target
    if official_labels is not None:
        f["churned"] = lab.CHURNED.reindex(ids).to_numpy()
    else:
        fut = act[(act.k <= 0) & (act.k > -LABEL_WINDOW_MONTHS)].groupby("CUSTOMER_ID").TRANSACTIONS.sum()
        f["churned"] = (fut.reindex(ids).fillna(0) == 0).astype(int).to_numpy()
    return f.reset_index()


def fill_age(frames: list[pd.DataFrame], reference: pd.DataFrame) -> None:
    """Median-impute age using the training data only (in place)."""
    med = reference.age.median()
    for fr in frames:
        fr["age"] = fr["age"].fillna(med)
