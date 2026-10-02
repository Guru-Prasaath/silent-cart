"""Deep-learning challenger: a GRU that reads each member's raw 6-month activity sequence (PyTorch, optional).

Question it answers: does learning directly from the monthly sequence beat hand-engineered trend features?
Same rolling-origin splits, class weighting, Platt calibration and threshold logic as the tabular models.
Runs on CPU with fixed seeds for reproducibility; skipped automatically if torch is not installed.
"""
import numpy as np
import pandas as pd

from . import config as C
from .features import _mi
from .models import PlattCalibrator, best_f1_threshold, metrics

SEQ = ["TRANSACTIONS", "TOTAL_SPEND", "DISTINCT_CATEGORIES", "PROMO_TXN_PCT", "APP_SESSIONS", "EMAILS_OPENED",
       "COUPONS_REDEEMED", "SUPPORT_TICKETS", "COMPLAINT_FLAG"]
STATIC = ["tier_rank", "tenure_months", "marketing_opt_in", "history_months", "recency_months", "age"]
W = 6


def sequences(activity: pd.DataFrame, snap: pd.DataFrame) -> np.ndarray:
    """(members, 6 months oldest->newest, 9 metrics + observed mask). Same missing-month rules as features.py:
    trailing silence = 0 (observed), gaps / before first row = 0 with mask 0."""
    cutoff = pd.Timestamp(snap.cutoff.iloc[0] + "-01")
    c = int(_mi(cutoff))
    act = activity.assign(k=c - _mi(activity.MONTH))
    hist = act[act.k >= 1]
    ids = snap.customer_id.to_numpy()
    last_k = hist.groupby("CUSTOMER_ID").k.min().reindex(ids).to_numpy()
    win = hist[hist.k <= W]
    K = np.arange(1, W + 1)[None, :]
    chans = []
    present = None
    for m in SEQ:
        v = win.pivot(index="CUSTOMER_ID", columns="k", values=m).reindex(index=ids, columns=range(1, W + 1)).to_numpy(float, copy=True)
        if present is None:
            present = ~np.isnan(v)
            trailing = (K < last_k[:, None]) & ~present
        v[trailing] = 0.0
        v = np.nan_to_num(v)
        if m in ("TRANSACTIONS", "TOTAL_SPEND", "APP_SESSIONS", "EMAILS_OPENED", "COUPONS_REDEEMED", "DISTINCT_CATEGORIES"):
            v = np.log1p(v)
        chans.append(v)
    chans.append((present | trailing).astype(float))
    x = np.stack(chans, axis=-1)          # (n, k=1..6, ch)
    return x[:, ::-1, :].copy()           # oldest -> newest


def run(activity, train, valid, test, champion_proba=None, epochs=60, patience=10):
    try:
        import torch
        from torch import nn
    except ImportError:
        print("  [deep] torch not installed: skipping GRU challenger (pip install -r requirements-dl.txt)")
        return None
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    torch.use_deterministic_algorithms(True)

    def prep(snaps):
        xs = np.concatenate([sequences(activity, s) for s in snaps])
        st = pd.concat(snaps)[STATIC].to_numpy(float)
        return xs, st

    xtr, str_ = prep([train[train.cutoff == c] for c in train.cutoff.unique()])
    xva, sva = prep([valid])
    xte, ste = prep([test])
    mu_x, sd_x = xtr.reshape(-1, xtr.shape[-1]).mean(0), xtr.reshape(-1, xtr.shape[-1]).std(0) + 1e-6
    mu_s, sd_s = str_.mean(0), str_.std(0) + 1e-6
    T = lambda x, s: (torch.tensor((x - mu_x) / sd_x, dtype=torch.float32), torch.tensor((s - mu_s) / sd_s, dtype=torch.float32))
    Xtr, Str = T(xtr, str_)
    Xva, Sva = T(xva, sva)
    Xte, Ste = T(xte, ste)
    ytr = torch.tensor(pd.concat([train[train.cutoff == c] for c in train.cutoff.unique()]).churned.to_numpy(), dtype=torch.float32)
    yva, yte = valid.churned.to_numpy(), test.churned.to_numpy()
    va_act = (valid.segment == "Active").to_numpy()

    class GRUNet(nn.Module):
        def __init__(self, n_ch, n_static, h=32):
            super().__init__()
            self.gru = nn.GRU(n_ch, h, batch_first=True)
            self.head = nn.Sequential(nn.Linear(h + n_static, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

        def forward(self, x, s):
            _, hN = self.gru(x)
            return self.head(torch.cat([hN[-1], s], dim=1)).squeeze(1)

    net = GRUNet(Xtr.shape[-1], Str.shape[-1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=((1 - ytr.mean()) / ytr.mean()).detach())
    g = torch.Generator().manual_seed(C.SEED)
    best, best_state, wait = -1, None, 0
    from sklearn.metrics import average_precision_score
    for ep in range(epochs):
        net.train()
        for idx in torch.randperm(len(ytr), generator=g).split(128):
            opt.zero_grad()
            loss_fn(net(Xtr[idx], Str[idx]), ytr[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pva = torch.sigmoid(net(Xva, Sva)).numpy()
        score = average_precision_score(yva[va_act], pva[va_act])
        if score > best:
            best, best_state, wait = score, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pva = torch.sigmoid(net(Xva, Sva)).numpy()
        pte = torch.sigmoid(net(Xte, Ste)).numpy()
    cal = PlattCalibrator().fit(pva, yva)
    thr = best_f1_threshold(yva[va_act], cal.transform(pva)[va_act])
    pte = cal.transform(pte)
    te_act = (test.segment == "Active").to_numpy()
    rows = [{"model": "GRU sequence (PyTorch)", "params": f"hidden=32, epochs={ep + 1}", "split": "test", "population": pop,
             **metrics(yte[m], pte[m], thr)} for pop, m in [("All eligible", np.ones(len(yte), bool)), ("Active only", te_act)]]
    # Same bootstrap as the tabular models (seeded identically), so intervals are comparable
    rng = np.random.default_rng(C.SEED)
    for r, (pop, m) in zip(rows, [("All eligible", np.ones(len(yte), bool)), ("Active only", te_act)]):
        yy, pp = yte[m], pte[m]
        vals = [average_precision_score(yy[i], pp[i]) for i in (rng.integers(0, len(yy), len(yy)) for _ in range(1000)) if yy[i].any()]
        r["pr_auc_ci_low"], r["pr_auc_ci_high"] = np.percentile(vals, [2.5, 97.5])
        r["valid_active_pr_auc"] = best
    if champion_proba is not None:
        # Paired bootstrap on Active members: same resampled members for both models -> CI of the PR-AUC difference
        yy, pg, pc = yte[te_act], pte[te_act], champion_proba[te_act]
        diffs = [average_precision_score(yy[i], pg[i]) - average_precision_score(yy[i], pc[i])
                 for i in (rng.integers(0, len(yy), len(yy)) for _ in range(1000)) if yy[i].any()]
        rows[1]["delta_vs_champion"] = average_precision_score(yy, pg) - average_precision_score(yy, pc)
        rows[1]["delta_ci_low"], rows[1]["delta_ci_high"] = np.percentile(diffs, [2.5, 97.5])
        rows[1]["p_gru_better"] = float(np.mean(np.array(diffs) > 0))
    out = pd.DataFrame(rows)
    out.round(4).to_csv(C.OUTPUTS / "deep_model_results.csv", index=False)
    out.attrs["valid_active_pr_auc"] = best
    return out
