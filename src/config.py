"""Central configuration: paths, dates, seeds and business assumptions.

Every number that is an *assumption* (not learned from data) lives here so it can be
reviewed in one place and is echoed in the README / report.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "raw" / "FreshBasket_Loyalty_Churn_Dataset.xlsx"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
REPORTS = ROOT / "reports"
PRESENTATION = ROOT / "presentation"

SEED = 42

# --- Time design -------------------------------------------------------------
DATA_START = pd.Timestamp("2023-01-01")
DATA_END = pd.Timestamp("2024-06-01")        # last activity month in the file
LABEL_WINDOW_MONTHS = 3                       # churn = 0 transactions in the next 3 months
FEATURE_LOOKBACK_MONTHS = 6                   # all behavioural windows use the last 6 months

# Rolling-origin (forward-chaining) snapshots. Each cutoff is the first month of its
# label window; features use only months strictly before the cutoff.
TRAIN_CUTOFFS = [pd.Timestamp("2023-07-01"), pd.Timestamp("2023-10-01")]
VALID_CUTOFF = pd.Timestamp("2024-01-01")     # model selection, calibration, threshold
TEST_CUTOFF = pd.Timestamp("2024-04-01")      # official CHURNED label (Apr-Jun 2024)

# --- Business assumptions for value & scenarios -----------------------------
# Treated as transparent, editable assumptions (sensitivity analysis is reported).
VALUE_HORIZON_MONTHS = 6       # revenue protected if a churner is retained: 6 months of recent spend
GROSS_MARGIN = 0.25            # grocery gross margin applied to retained revenue
COUPON_COST = 10.0             # $ cost per targeted member (coupon face value, assume full redemption)
COUPON_RELATIVE_UPLIFT = 0.15  # coupon reduces a targeted member's churn probability by 15% (relative)
OUTREACH_COST = 15.0           # $ per proactive support call (~15 agent minutes incl. overhead)
OUTREACH_RELATIVE_UPLIFT = 0.25  # service recovery reduces churn probability by 25% (relative)
WINBACK_COST = 5.0             # $ per lapsed member (email + small reactivation coupon)
WINBACK_REACTIVATION = 0.05    # 5% of lapsed members re-activate after a win-back offer

# Colour tokens (validated categorical palette, light mode).
COLORS = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
    "axis": "#c3c2b7", "surface": "#ffffff", "neutral": "#c3c2b7",
}
RETAINED_COLOR = COLORS["blue"]
CHURN_COLOR = COLORS["orange"]


def ensure_dirs() -> None:
    for p in (PROCESSED, OUTPUTS, FIGURES, REPORTS, PRESENTATION):
        p.mkdir(parents=True, exist_ok=True)
