"""Builds and executes notebooks/01_data_quality_eda.ipynb so it always matches the src/ code."""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

from .config import ROOT

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

CELLS = [
    md("""# 01 · Data quality & exploratory analysis — FreshBasket loyalty churn

**Question:** which loyalty members will stop shopping (zero transactions in Apr–Jun 2024), why, and what should we do about it?

This notebook documents every data-quality issue found in the three source tables, how each one was handled, and the
exploratory analysis that shaped the modelling approach. All logic lives in `src/` (the same code `run_pipeline.py` runs);
the notebook calls it and shows the evidence.

**Headline finding:** the CHURNED label mixes two different problems. ~80% of churners had *already stopped buying*
before April 2024 (a reporting problem, solved by a rule), while churn among members still active in Q1 is ~10% (a
genuine prediction problem). Everything downstream is reported for both groups."""),
    code("""import sys, warnings
sys.path.insert(0, '..')
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from IPython.display import Image, display
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 40); pd.set_option('display.max_colwidth', 120)
from src import config as C, data_quality as dq, features as fe, analysis as an"""),
    md("""## 1. Load the raw workbook
Every sheet has a title banner in row 1, so the real header is row 2 (`header=1`). Reading with defaults silently
produces `Unnamed: n` columns and a header row inside the data — the first data-quality trap."""),
    code("""naive = pd.read_excel(C.RAW_XLSX, sheet_name='fb Customer Profile')
print('Default read  ->', list(naive.columns[:4]), '...')
raw = dq.load_raw()
for k, d in raw.items():
    print(f'{k:9s} {d.shape}  columns: {list(d.columns)}')"""),
    md("## 2. Customer profile checks"),
    code("""p = raw['profile']
print('Exact duplicate rows:', p.duplicated().sum(), '| duplicate CUSTOMER_IDs:', p.CUSTOMER_ID.duplicated().sum())
print('Missing values:', p.isna().sum()[p.isna().sum() > 0].to_dict())
print('\\nRaw MEMBERSHIP_TIER values (casing/whitespace issues):')
print(p.MEMBERSHIP_TIER.value_counts().to_dict())
print('\\nRaw CITY distinct values:', p.CITY.nunique(), '->', p.CITY.str.strip().str.title().nunique(), 'after cleaning')"""),
    md("## 3. Monthly activity checks"),
    code("""a = raw['activity']
print('Exact duplicate rows:', a.duplicated().sum())
a1 = a.drop_duplicates()
conflict = a1[a1.duplicated(['CUSTOMER_ID', 'MONTH'], keep=False)].sort_values(['CUSTOMER_ID', 'MONTH'])
print('Conflicting duplicates (same customer-month, different values):')
display(conflict[['CUSTOMER_ID', 'MONTH', 'TRANSACTIONS', 'TOTAL_SPEND', 'AVG_BASKET_VALUE']])"""),
    md("""**Spend integrity.** `TOTAL_SPEND = TRANSACTIONS × AVG_BASKET_VALUE` holds to the cent for 99.8% of rows, which
gives an objective test for corrupted values *and* a way to repair them without throwing away the month."""),
    code("""exp = a1.TRANSACTIONS * a1.AVG_BASKET_VALUE
ratio = (a1.TOTAL_SPEND / exp.replace(0, np.nan)).round(2)
print('Rows where the identity holds (|diff| < $1):', f"{((a1.TOTAL_SPEND - exp).abs() < 1)[a1.TRANSACTIONS > 0].mean():.2%}")
bad = a1[(a1.TRANSACTIONS > 0) & ((a1.TOTAL_SPEND <= 0) | (ratio > 3))]
print('Corrupted rows:', len(bad), '| negative:', (bad.TOTAL_SPEND < 0).sum(), '| zero:', (bad.TOTAL_SPEND == 0).sum(),
      '| inflated 9-15x:', (ratio[bad.index] > 3).sum())
display(bad.assign(ratio_to_expected=ratio[bad.index]).sort_values('ratio_to_expected').head(8)
        [['CUSTOMER_ID', 'MONTH', 'TRANSACTIONS', 'AVG_BASKET_VALUE', 'TOTAL_SPEND', 'ratio_to_expected']])"""),
    code("display(Image(an.fig_spend_repair(raw['activity'])))"),
    md("""**Zero-transaction months and missing months.** Months with 0 transactions are kept (they are a real inactivity
signal), but 1,452 of them report categories purchased and 653 report coupons redeemed, which is impossible without
a transaction, so those fields are zeroed. Separately, 953 months are simply *absent* inside members' active spans.
They are treated as **missing data, not zero activity** (the label table's `MONTHS_OBSERVED` counts them as observed,
confirming rows were lost rather than members being inactive)."""),
    code("""g = a1.drop_duplicates(['CUSTOMER_ID', 'MONTH']).groupby('CUSTOMER_ID').MONTH.agg(['min', 'max', 'count'])
span = (g['max'].dt.year - g['min'].dt.year) * 12 + g['max'].dt.month - g['min'].dt.month + 1
print('Members by number of internal missing months:', (span - g['count']).value_counts().sort_index().to_dict())"""),
    md("""## 4. Label table checks and leakage audit
The target can be re-derived from the activity table (no purchase in Apr–Jun 2024, with purchases before). Agreement
is 99.7%; the 9 disagreements trace to missing activity rows, so the official label is kept.

Three label-table columns are **leaky**: they are computed over the full window *including* Apr–Jun 2024, so they
encode the answer. Churners stop generating rows, so they have fewer `MONTHS_OBSERVED`:"""),
    code("""l = raw['label'].drop_duplicates()
print(l.groupby('CHURNED').MONTHS_OBSERVED.describe()[['mean', '50%']])
print('\\nChurn rate by INSUFFICIENT_HISTORY_FLAG:', l.groupby('INSUFFICIENT_HISTORY_FLAG').CHURNED.mean().round(3).to_dict())
print('Members who never purchased (LAST_PURCHASE_DATE missing):', l.LAST_PURCHASE_DATE.isna().sum(), '-> all CHURNED=0, not churn-eligible')"""),
    md("""These columns are never used as features. Leakage-safe equivalents (`history_months`, `short_history`,
`recency_months`) are rebuilt from pre-cutoff data only. `validation_design_checks.csv` shows that adding the leaky
columns would inflate Active-member PR-AUC from ~0.80 to ~0.87.

## 5. Full data-quality log
Every check, what was found and what was done (also saved as `outputs/data_quality_log.csv`)."""),
    code("""profile, activity, labels, log = dq.run(save=False)
display(log)"""),
    md("## 6. Exploratory analysis\n### 6.1 A leaky bucket hidden by sign-ups"),
    code("""test = fe.build_snapshot(profile, activity, C.TEST_CUTOFF, labels)
display(Image(an.fig_leaky_bucket(activity)))"""),
    md("""### 6.2 Two churn problems inside one label
Churn probability is a cliff in recency. Members who bought in Jan–Mar 2024 (**Active**) churn at ~10%; members already
silent for 3+ months (**Lapsed**) churn at ~99%: their "churn" happened before the label window opened."""),
    code("""display(Image(an.fig_recency_cliff(test)))
seg = test.groupby('segment').churned.agg(members='size', churners='sum', churn_rate='mean')
seg['share_of_churners'] = seg.churners / seg.churners.sum()
display(seg.round(3))"""),
    md("""### 6.3 The churn fingerprint
Aligning Active members on their last pre-April purchase shows *how* churn unfolds: transactions, app sessions and
email opens fall over the final ~2 months while support tickets spike. This is the intervention window."""),
    code("display(Image(an.fig_churn_fingerprint(activity, test)))"),
    md("### 6.4 Engagement and support experience vs churn (Active members)"),
    code("display(Image(an.fig_engagement_support(test)))"),
    md("""### 6.5 Statistical evidence
Mann–Whitney U tests (churned vs retained Active members), Benjamini–Hochberg FDR-corrected, with rank-biserial
effect size (+ = churners score higher)."""),
    code("""tests = an.hypothesis_tests(test)
display(tests.round(4))
c = tests.attrs['complaint_chi2']
print(f"Chi-square, complaint in last 3m vs churn: chi2={c['chi2']:.1f}, p={c['p']:.4f}; churn {c['churn_if_complaint']:.1%} with complaint vs {c['churn_if_none']:.1%} without")"""),
    md("### 6.6 Observed churn by tier, city and signup cohort"),
    code("display(Image(an.fig_segments_observed(test)))"),
    md("""## 7. What this means for modelling
1. **Report everything twice**, for all eligible members and for Active members. The all-member metrics are inflated
   by easy, already-lapsed cases.
2. **Use rolling-origin snapshots.** The label definition can be replayed at earlier cut-offs (Jul-23, Oct-23, Jan-24),
   giving out-of-time training and validation data. The official Apr-24 label is the untouched test set.
3. **Features come from the 6 months before each cut-off only**: recency, 3-vs-3-month trends, engagement, support and
   history. Missing months are treated as missing data, never as zero activity.
4. **Membership tier, age and tenure carry almost no signal** (not significant after FDR correction). Behaviour,
   engagement and support do, so retention should be triggered by behaviour, not by tier."""),
]


def build_and_execute() -> str:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = ROOT / "notebooks" / "01_data_quality_eda.ipynb"
    path.parent.mkdir(exist_ok=True)
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(path.parent)}})
    nbf.write(nb, path)
    print(f"  notebook executed -> {path.relative_to(ROOT)}")
    return str(path)
