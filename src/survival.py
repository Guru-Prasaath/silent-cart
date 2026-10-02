"""Time-to-lapse survival analysis: *when* do members stop buying, and does early experience change it?

Event  = a member's final purchase (no purchase afterwards through Jun-2024), observable when it happens by Mar-2024.
Clock  = months since the member's first observed activity month.
Landmark design: groups are defined on each member's first 3 observed months, and only members still buying after
those 3 months enter the curves, which avoids immortal-time bias. Kaplan-Meier curves + log-rank test (scipy).
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats

from . import config as C
from .viz import apply_style, fig_titled, save

apply_style()
LANDMARK = 3


def _mi(ts):
    return ts.dt.year * 12 + ts.dt.month


def build_survival_table(activity: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    a = activity.assign(mi=_mi(activity.MONTH))
    first = a.groupby("CUSTOMER_ID").mi.min()
    last_buy = a[a.TRANSACTIONS > 0].groupby("CUSTOMER_ID").mi.max()
    censor_mi = 2024 * 12 + 3                      # a final purchase is only confirmed if followed by Apr-Jun silence
    d = pd.DataFrame({"first": first, "last_buy": last_buy}).dropna()
    d["event"] = (d.last_buy <= censor_mi).astype(int)
    d["time"] = np.where(d.event == 1, d.last_buy - d["first"] + 1, censor_mi - d["first"] + 1)
    early = a.merge(first.rename("first"), left_on="CUSTOMER_ID", right_index=True)
    early = early[early.mi < early["first"] + LANDMARK].groupby("CUSTOMER_ID").agg(
        early_tickets=("SUPPORT_TICKETS", "sum"), early_complaint=("COMPLAINT_FLAG", "max"),
        early_app=("APP_SESSIONS", "mean"))
    d = d.join(early).join(profile.set_index("CUSTOMER_ID").MEMBERSHIP_TIER.rename("tier"))
    d = d[d.time > LANDMARK].copy()                # landmark: still buying after the first 3 months
    d["time"] = d.time - LANDMARK
    d["early_support"] = np.where(d.early_complaint > 0, "Complaint in first 3 months", "No early complaint")
    d["early_app_use"] = np.where(d.early_app < 1, "App use < 1/month early", "App use >= 1/month early")
    return d


def kaplan_meier(time, event):
    t = np.sort(np.unique(time[event == 1]))
    s, out = 1.0, [(0, 1.0)]
    for ti in t:
        at_risk = (time >= ti).sum()
        d = ((time == ti) & (event == 1)).sum()
        s *= 1 - d / at_risk
        out.append((ti, s))
    return pd.DataFrame(out, columns=["t", "survival"])


def logrank(time, event, group):
    """Log-rank test for k groups (chi-square, k-1 df)."""
    groups = np.unique(group)
    times = np.sort(np.unique(time[event == 1]))
    O = np.zeros(len(groups)); E = np.zeros(len(groups)); V = np.zeros((len(groups), len(groups)))
    for ti in times:
        at = time >= ti
        n = at.sum(); d = ((time == ti) & (event == 1)).sum()
        ng = np.array([(at & (group == g)).sum() for g in groups])
        dg = np.array([((time == ti) & (event == 1) & (group == g)).sum() for g in groups])
        O += dg; E += d * ng / n
        if n > 1:
            f = d * (n - d) / (n * n * (n - 1))
            V += f * (np.diag(ng * n) - np.outer(ng, ng))
    diff = (O - E)[:-1]
    chi2 = float(diff @ np.linalg.pinv(V[:-1, :-1]) @ diff)
    return chi2, float(stats.chi2.sf(chi2, len(groups) - 1))


def run(activity, profile):
    d = build_survival_table(activity, profile)
    rows, curves = [], {}
    for dim in ["early_support", "early_app_use", "tier"]:
        chi2, p = logrank(d.time.to_numpy(), d.event.to_numpy(), d[dim].to_numpy())
        for g, gd in d.groupby(dim):
            km = kaplan_meier(gd.time.to_numpy(), gd.event.to_numpy())
            curves[(dim, g)] = km
            s12 = km[km.t <= 12].survival.iloc[-1]
            rows.append({"dimension": dim, "group": g, "members": len(gd), "lapses": int(gd.event.sum()),
                         "still_buying_after_12m": s12, "logrank_chi2": chi2, "logrank_p": p})
    summary = pd.DataFrame(rows)
    summary.round(5).to_csv(C.OUTPUTS / "survival_summary.csv", index=False)
    pd.concat([km.assign(dimension=dim, group=g) for (dim, g), km in curves.items()]).round(5).to_csv(
        C.OUTPUTS / "survival_curves.csv", index=False)
    return d, summary, curves


def fig_survival(summary, curves) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    palette = [C.COLORS["orange"], C.COLORS["blue"], C.COLORS["aqua"]]
    for ax, (dim, title) in zip(axes, [("early_support", "Early support experience"), ("early_app_use", "Early app engagement"),
                                       ("tier", "Membership tier")]):
        s = summary[summary.dimension == dim]
        for col, g in zip(palette, s.group):
            km = curves[(dim, g)]
            ax.step(km.t, km.survival, where="post", color=col, lw=2,
                    label=f"{g} ({s[s.group == g].still_buying_after_12m.iloc[0]:.0%} at 12m)")
        p = s.logrank_p.iloc[0]
        ax.set_title(f"{title}  (log-rank p {'< 0.001' if p < 0.001 else f'= {p:.2f}'})", fontsize=11)
        ax.set_xlabel("Months after the first 3 months")
        ax.legend(loc="lower left", fontsize=8.5)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Share still buying (Kaplan-Meier)")
    axes[0].set_ylim(0.4, 1.01)
    sig = summary.groupby("dimension").logrank_p.first()
    app = summary[summary.dimension == "early_app_use"].set_index("group").still_buying_after_12m
    fig_titled(fig, f"Survival: low early app use cuts 12-month retention to {app.min():.0%} (vs {app.max():.0%}); "
               f"early complaints (p={sig['early_support']:.2f}) and tier (p={sig['tier']:.2f}) do not",
               "Time to final purchase, landmarked at month 3. Support friction matters when it is RECENT (see churn model), not early in the relationship")
    return save(fig, "19_survival_curves.png")
