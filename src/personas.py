"""Behavioural personas for Active members (unsupervised), so each group gets the right retention message.

K-means on standardised recent-behaviour features. Cluster separation is modest (silhouette ~0.13), so personas are
used as *messaging* groups layered on top of the churn score, not as a replacement for it. k=5 is fixed for
interpretability; names are assigned by rules on the cluster centroids (robust to label permutation).
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config as C
from .viz import apply_style, fig_titled, save

apply_style()
FEATURES = ["txn_l3", "spend_change_pct", "app_l3", "email_l3", "coupons_l3", "promo_pct_6m", "tickets_6m",
            "categories_l3", "avg_basket_l3"]
LOG = ["txn_l3", "app_l3", "email_l3", "coupons_l3", "tickets_6m", "categories_l3"]
NICE = {"txn_l3": "Transactions", "spend_change_pct": "Spend change", "app_l3": "App sessions", "email_l3": "Email opens",
        "coupons_l3": "Coupons", "promo_pct_6m": "Promo share", "tickets_6m": "Support tickets",
        "categories_l3": "Categories", "avg_basket_l3": "Basket size"}
PLAYS = {
    "Fading & frustrated": "Service-recovery call first, then a preferred-category win-back coupon",
    "Engaged but service-heavy": "Fix friction fast (ticket follow-up); no discount needed while they still shop",
    "Light & low-touch": "Cross-category offers to deepen the basket; low-cost digital nudges",
    "Ramping up": "Reinforce the new habit: app onboarding and a 'next visit' reward",
    "Loyal core": "No discounts; recognition and referral rewards",
}


def _name(z: pd.Series) -> str:
    if z.txn_l3 < -1 and z.tickets_6m > 0.4:
        return "Fading & frustrated"
    if z.spend_change_pct > 1.5:
        return "Ramping up"
    if z.txn_l3 > 0.6 and z.tickets_6m < -0.4:
        return "Loyal core"
    if z.tickets_6m > 0.4:
        return "Engaged but service-heavy"
    return "Light & low-touch"


def fit_personas(test: pd.DataFrame, k: int = 5):
    a = test[test.segment == "Active"].copy()
    X = a[FEATURES].astype(float).copy()
    X[LOG] = np.log1p(X[LOG])
    Z = StandardScaler().fit_transform(X)
    km = KMeans(k, n_init=20, random_state=C.SEED).fit(Z)
    cent = pd.DataFrame(km.cluster_centers_, columns=FEATURES)
    names = cent.apply(_name, axis=1)
    if names.duplicated().any():   # keep names unique if two centroids land on the same rule
        names = names + names.groupby(names).cumcount().map(lambda i: "" if i == 0 else f" ({i + 1})")
    a["persona"] = names.to_numpy()[km.labels_]
    prof = (a.groupby("persona").agg(members=("customer_id", "size"), observed_churn=("churned", "mean"),
                                     mean_predicted=("churn_probability", "mean"),
                                     revenue_at_risk=("revenue_at_risk", "sum"))
            .join(cent.set_index(names.to_numpy()).rename(columns=lambda c: f"z_{c}")))
    prof["recommended_play"] = [PLAYS.get(n.split(" (")[0], "") for n in prof.index]
    prof = prof.sort_values("observed_churn", ascending=False)
    prof.attrs["silhouette"] = float(silhouette_score(Z, km.labels_))
    prof.reset_index(names="persona").round(4).to_csv(C.OUTPUTS / "personas.csv", index=False)
    return a[["customer_id", "persona"]], prof


def fig_personas(prof: pd.DataFrame) -> str:
    z = prof[[f"z_{f}" for f in FEATURES]].to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [2.3, 1]})
    im = ax1.imshow(z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax1.set_xticks(range(len(FEATURES)), [NICE[f] for f in FEATURES], rotation=30, ha="right")
    ax1.set_yticks(range(len(prof)), [f"{p}  (n={n})" for p, n in zip(prof.index, prof.members)])
    for (i, j), v in np.ndenumerate(z):
        ax1.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8.5, color="white" if abs(v) > 1.2 else C.COLORS["ink"])
    ax1.grid(False)
    ax1.set_title("Profile (z-score vs Active average; red = higher)", fontsize=11)
    y = np.arange(len(prof))
    ax2.barh(y, prof.observed_churn, color=C.CHURN_COLOR, height=0.6)
    for yi, v in zip(y, prof.observed_churn):
        ax2.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=9, color=C.COLORS["ink2"])
    ax2.set_yticks(y, [""] * len(y))
    ax2.invert_yaxis()
    ax2.xaxis.set_major_formatter(PercentFormatter(1))
    ax2.set_xlim(0, max(prof.observed_churn) * 1.25)
    ax2.set_title("Observed churn (Apr-Jun 2024)", fontsize=11)
    ax2.grid(axis="y", visible=False)
    top = prof.index[0]
    fig_titled(fig, f"Five behavioural personas: '{top}' members churn at {prof.observed_churn.iloc[0]:.0%}",
               f"K-means on Active members' recent behaviour (silhouette {prof.attrs.get('silhouette', float('nan')):.2f}: soft groups, "
               "used to tailor the message, not to replace the churn score)")
    return save(fig, "18_personas.png")
