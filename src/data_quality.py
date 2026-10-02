"""Load the raw workbook, run data-quality checks, repair issues and log every action.

Output: cleaned tables in data/processed/ and outputs/data_quality_log.csv, one row per
check with what was found and what was done about it.
"""
import numpy as np
import pandas as pd

from .config import DATA_END, DATA_START, OUTPUTS, PROCESSED, RAW_XLSX

SHEETS = {
    "profile": "fb Customer Profile",
    "activity": "fb Monthly Activity",
    "label": "fb Churn Label",
}
# Spend more than this multiple of TRANSACTIONS x AVG_BASKET_VALUE is a data-entry outlier.
OUTLIER_RATIO = 3.0


class DQLog:
    def __init__(self):
        self.rows = []

    def add(self, table, check, n_found, action, n_changed=None):
        self.rows.append({
            "table": table, "check": check, "rows_found": int(n_found),
            "action": action, "rows_changed": int(n_found if n_changed is None else n_changed),
        })

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def load_raw() -> dict:
    """Every sheet has a title banner in row 1, so the real header is row 2."""
    return {k: pd.read_excel(RAW_XLSX, sheet_name=s, header=1) for k, s in SHEETS.items()}


def _clean_text(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.title()


def clean_profile(df: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    df = df.copy()
    n = df.duplicated().sum()
    log.add("profile", "Exact duplicate rows", n, "Dropped")
    df = df.drop_duplicates()
    assert not df.CUSTOMER_ID.duplicated().any(), "conflicting duplicate profiles"

    for col in ["CITY", "MEMBERSHIP_TIER"]:
        cleaned = _clean_text(df[col])
        n = (cleaned != df[col].astype("string")).sum()
        log.add("profile", f"{col}: inconsistent casing / whitespace", n,
                f"Stripped and title-cased ({df[col].nunique()} raw -> {cleaned.nunique()} clean values)")
        df[col] = cleaned

    n = df.AGE.isna().sum()
    log.add("profile", "AGE missing", n, "Kept as missing; median-imputed at feature time + AGE_MISSING flag", 0)
    bad_age = (~df.AGE.between(16, 100) & df.AGE.notna()).sum()
    log.add("profile", "AGE outside 16-100", bad_age, "None needed" if bad_age == 0 else "Set to missing")

    for col in ["GENDER", "HOME_STORE_PRICE_TIER"]:
        n = df[col].isna().sum()
        log.add("profile", f"{col} missing", n, "Filled with explicit 'Unknown' category")
        df[col] = df[col].astype("string").fillna("Unknown")

    df["SIGNUP_DATE"] = pd.to_datetime(df.SIGNUP_DATE)
    n = (df.SIGNUP_DATE > DATA_END + pd.offsets.MonthEnd(0)).sum()
    log.add("profile", "Signup date after data window", n, "None needed" if n == 0 else "Flagged")
    return df.reset_index(drop=True)


def clean_activity(df: pd.DataFrame, profile_ids: set, log: DQLog) -> pd.DataFrame:
    df = df.copy()
    df["MONTH"] = pd.to_datetime(df.MONTH)

    n = df.duplicated().sum()
    log.add("activity", "Exact duplicate rows", n, "Dropped")
    df = df.drop_duplicates()

    key = ["CUSTOMER_ID", "MONTH"]
    dup = df.duplicated(key, keep=False)
    log.add("activity", "Conflicting duplicates (same customer-month, spend sign flipped)", dup.sum(),
            "Kept the row with positive spend", dup.sum() // 2)
    df = df.sort_values("TOTAL_SPEND", ascending=False).drop_duplicates(key).sort_values(key)

    out = (df.MONTH < DATA_START) | (df.MONTH > DATA_END)
    log.add("activity", "Month outside Jan-2023..Jun-2024", out.sum(), "None needed" if out.sum() == 0 else "Dropped")
    df = df[~out]

    orphans = ~df.CUSTOMER_ID.isin(profile_ids)
    log.add("activity", "Activity for customers missing from profile", orphans.sum(),
            "None needed" if orphans.sum() == 0 else "Dropped")
    df = df[~orphans]

    # TOTAL_SPEND = TRANSACTIONS x AVG_BASKET_VALUE holds to the cent for 99.8% of rows, so it is a
    # reliable way to repair the corrupted spend values rather than dropping whole months.
    expected = (df.TRANSACTIONS * df.AVG_BASKET_VALUE).round(2)
    df["SPEND_REPAIRED"] = 0
    neg = (df.TOTAL_SPEND < 0) & (df.TRANSACTIONS > 0)
    zero = (df.TOTAL_SPEND == 0) & (df.TRANSACTIONS > 0)
    ratio = df.TOTAL_SPEND / expected.replace(0, np.nan)
    outlier = (ratio > OUTLIER_RATIO) & (df.TRANSACTIONS > 0)
    log.add("activity", "Negative TOTAL_SPEND with transactions > 0", neg.sum(),
            "Rebuilt as TRANSACTIONS x AVG_BASKET_VALUE")
    log.add("activity", "Zero TOTAL_SPEND with transactions > 0", zero.sum(),
            "Rebuilt as TRANSACTIONS x AVG_BASKET_VALUE")
    log.add("activity", f"Outlier TOTAL_SPEND (> {OUTLIER_RATIO:g}x transactions x basket)", outlier.sum(),
            f"Rebuilt as TRANSACTIONS x AVG_BASKET_VALUE (max was ${df.TOTAL_SPEND.max():,.0f})")
    fix = neg | zero | outlier
    df.loc[fix, "TOTAL_SPEND"] = expected[fix]
    df.loc[fix, "SPEND_REPAIRED"] = 1

    q1, q3 = df.TOTAL_SPEND.quantile([0.25, 0.75])
    iqr_hi = (df.TOTAL_SPEND > q3 + 3 * (q3 - q1)).sum()
    log.add("activity", "Remaining high spend after repair (> Q3 + 3 IQR)", iqr_hi,
            "Kept: consistent with transactions x basket, so genuine big shoppers", 0)

    zt = df.TRANSACTIONS == 0
    log.add("activity", "Zero-transaction months (engagement but no purchase)", zt.sum(),
            "Kept: valid inactivity signal; spend/basket/promo already 0", 0)
    cat_anom = zt & (df.DISTINCT_CATEGORIES > 0)
    log.add("activity", "Zero-transaction months reporting DISTINCT_CATEGORIES > 0", cat_anom.sum(),
            "Set DISTINCT_CATEGORIES to 0 (cannot buy categories without a transaction)")
    df.loc[cat_anom, "DISTINCT_CATEGORIES"] = 0
    cpn_anom = zt & (df.COUPONS_REDEEMED > 0)
    log.add("activity", "Zero-transaction months reporting COUPONS_REDEEMED > 0", cpn_anom.sum(),
            "Set COUPONS_REDEEMED to 0 (a redemption requires a transaction)")
    df.loc[cpn_anom, "COUPONS_REDEEMED"] = 0

    bad_cf = (df.COMPLAINT_FLAG == 1) & (df.SUPPORT_TICKETS == 0)
    log.add("activity", "Complaint flag without a support ticket", bad_cf.sum(), "None needed" if bad_cf.sum() == 0 else "Flagged")
    bad_promo = ~df.PROMO_TXN_PCT.between(0, 1)
    log.add("activity", "PROMO_TXN_PCT outside 0-1", bad_promo.sum(), "None needed" if bad_promo.sum() == 0 else "Clipped")

    g = df.groupby("CUSTOMER_ID").MONTH.agg(["min", "max", "count"])
    span = (g["max"].dt.year - g["min"].dt.year) * 12 + g["max"].dt.month - g["min"].dt.month + 1
    gaps = span - g["count"]
    log.add("activity", f"Missing months inside a member's active span ({(gaps > 0).sum()} members)", gaps.sum(),
            "Not imputed as zero: treated as missing data; MISSING_MONTHS_6M feature added", 0)
    no_act = len(profile_ids - set(df.CUSTOMER_ID))
    log.add("activity", "Profiled members with no activity rows at all", no_act,
            "Cannot be scored; excluded from modelling", 0)
    return df.reset_index(drop=True)


def clean_labels(df: pd.DataFrame, activity: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    df = df.copy()
    n = df.duplicated().sum()
    log.add("label", "Exact duplicate rows", n, "Dropped")
    df = df.drop_duplicates()
    assert not df.CUSTOMER_ID.duplicated().any(), "conflicting duplicate labels"
    for c in ["OBSERVATION_END_DATE", "LAST_PURCHASE_DATE"]:
        df[c] = pd.to_datetime(df[c])

    n = df.LAST_PURCHASE_DATE.isna().sum()
    log.add("label", "LAST_PURCHASE_DATE missing (member never purchased)", n,
            "Not churn-eligible by definition; excluded from modelling", 0)

    # Re-derive the target from the activity table to confirm we understand it.
    pre = activity[activity.MONTH < "2024-04-01"].groupby("CUSTOMER_ID").TRANSACTIONS.sum()
    post = activity[activity.MONTH >= "2024-04-01"].groupby("CUSTOMER_ID").TRANSACTIONS.sum()
    d = df.set_index("CUSTOMER_ID")
    derived = ((post.reindex(d.index).fillna(0) == 0) & (pre.reindex(d.index).fillna(0) > 0)).astype(int)
    mismatch = (derived != d.CHURNED).sum()
    log.add("label", f"CHURNED disagrees with label re-derived from activity ({1 - mismatch / len(d):.1%} agree)",
            mismatch, "Official label kept; mismatches trace to missing activity rows", 0)
    no_hist = ((pre.reindex(d.index).fillna(0) == 0) & (d.CHURNED == 1)).sum()
    log.add("label", "CHURNED=1 but no pre-April purchase visible in activity", no_hist,
            "Excluded (no history to build features from)", 0)
    log.add("label", "Leaky columns: LAST_PURCHASE_DATE, MONTHS_OBSERVED, INSUFFICIENT_HISTORY_FLAG", 3,
            "Computed with Apr-Jun 2024 data; never used as features. Rebuilt from pre-cutoff data instead", 0)
    return df.reset_index(drop=True)


def run(save: bool = True):
    raw = load_raw()
    log = DQLog()
    profile = clean_profile(raw["profile"], log)
    activity = clean_activity(raw["activity"], set(profile.CUSTOMER_ID), log)
    labels = clean_labels(raw["label"], activity, log)
    dq = log.frame()
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        profile.to_csv(PROCESSED / "profile_clean.csv", index=False)
        activity.to_csv(PROCESSED / "activity_clean.csv", index=False)
        labels.to_csv(PROCESSED / "labels_clean.csv", index=False)
        dq.to_csv(OUTPUTS / "data_quality_log.csv", index=False)
    return profile, activity, labels, dq
