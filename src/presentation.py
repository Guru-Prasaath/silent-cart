"""Builds presentation/churn_presentation.pptx (12 slides + speaker notes) from the pipeline outputs."""
import json

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from . import config as C

DARK, GREEN, MINT, TINT = "17332A", "2E7D5B", "BFE3CF", "EEF5F0"
ORANGE, BLUE, INK, MUTED, WHITE = "EB6834", "2A78D6", "1B1B1B", "5F6B66", "FFFFFF"
FONT = "Calibri"
W, H = 13.333, 7.5
FIG = C.FIGURES


def rgb(h):
    return RGBColor.from_string(h)


def money(x):
    return f"{'−' if x < 0 else '+' if x > 0 else ''}${abs(x) / 1000:,.1f}k"


def amount(x):
    """Unsigned dollar amount for prose ('loses $15.4k'), where the verb already carries the sign."""
    return f"${abs(x) / 1000:,.1f}k"


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W), Inches(H)
        self.n = 0

    def slide(self, dark=False, title=None, kicker=None):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.n += 1
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = rgb(DARK if dark else WHITE)
        if title:
            if kicker:
                self.text(s, kicker.upper(), 0.6, 0.38, 12, 0.3, size=11, color=GREEN if not dark else MINT, bold=True)
            self.text(s, title, 0.6, 0.62, 12.1, 0.95, size=27, bold=True, color=WHITE if dark else INK)
        if not dark and self.n > 1:
            self.text(s, f"Silent Cart  ·  FreshBasket loyalty churn  ·  {self.n}", 0.6, 7.05, 6, 0.3, size=9, color=MUTED)
        return s

    def text(self, s, text, x, y, w, h, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             italic=False, name=None):
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, m, 0)
        paras = text if isinstance(text, list) else [text]
        for i, ptxt in enumerate(paras):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            runs = ptxt if isinstance(ptxt, list) else [(ptxt, {})]
            for rt, opt in runs:
                r = p.add_run()
                r.text = rt
                f = r.font
                f.name, f.size = FONT, Pt(opt.get("size", size))
                f.bold, f.italic = opt.get("bold", bold), italic
                f.color.rgb = rgb(opt.get("color", color))
            if len(paras) > 1:
                p.space_after = Pt(opt.get("space", 6) if isinstance(ptxt, list) else 6)
        if name:
            tb.name = name
        return tb

    def bullets(self, s, items, x, y, w, h, size=15, color=INK, gap=8):
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, m, 0)
        for i, item in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(gap)
            head, rest = item if isinstance(item, tuple) else ("", item)
            for t, b, c in [("•  ", False, GREEN), (head, True, color), (rest, False, color)]:
                if t:
                    r = p.add_run()
                    r.text = t
                    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(size), b, rgb(c)
        return tb

    def card(self, s, x, y, w, h, fill=TINT, line=None):
        shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        shp.adjustments[0] = 0.08
        shp.fill.solid()
        shp.fill.fore_color.rgb = rgb(fill)
        if line:
            shp.line.color.rgb = rgb(line)
            shp.line.width = Pt(1)
        else:
            shp.line.fill.background()
        shp.shadow.inherit = False
        return shp

    def stat(self, s, value, label, x, y, w, color=GREEN, size=40, label_color=MUTED, h=1.6, fill=TINT):
        self.card(s, x, y, w, h, fill=fill)
        self.text(s, value, x + 0.25, y + 0.18, w - 0.5, 0.75, size=size, bold=True, color=color)
        self.text(s, label, x + 0.25, y + 0.18 + size / 72 + 0.12, w - 0.5, h - size / 72 - 0.35, size=12, color=label_color)

    def badge(self, s, num, x, y, d=0.42, fill=GREEN):
        c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
        c.fill.solid()
        c.fill.fore_color.rgb = rgb(fill)
        c.line.fill.background()
        tf = c.text_frame
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, m, 0)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        r = p.add_run()
        r.text = str(num)
        r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(14), True, rgb(WHITE)

    def image(self, s, name, x, y, w, h):
        """Fit an image inside the box, preserving aspect ratio, centred."""
        path = FIG / name
        iw, ih = Image.open(path).size
        scale = min(w / iw, h / ih)
        pw, ph = iw * scale, ih * scale
        return s.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))

    def table(self, s, df, x, y, w, col_w, size=12, header_fill=DARK, highlight_row=None):
        rows, cols = df.shape[0] + 1, df.shape[1]
        t = s.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(0.4 * rows)).table
        for j, cw in enumerate(col_w):
            t.columns[j].width = Inches(cw)
        for i in range(rows):
            for j in range(cols):
                cell = t.cell(i, j)
                cell.margin_left = cell.margin_right = Inches(0.08)
                val = df.columns[j] if i == 0 else df.iat[i - 1, j]
                cell.text = str(val)
                para = cell.text_frame.paragraphs[0]
                para.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
                f = para.runs[0].font
                f.name, f.size = FONT, Pt(size)
                f.bold = i == 0 or (highlight_row is not None and i - 1 == highlight_row)
                f.color.rgb = rgb(WHITE if i == 0 else INK)
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb(header_fill if i == 0 else (TINT if highlight_row is not None and i - 1 == highlight_row else WHITE))
        return t

    @staticmethod
    def notes(s, text):
        s.notes_slide.notes_text_frame.text = " ".join(text.split())


def build() -> str:
    O = C.OUTPUTS
    k = json.loads((O / "key_metrics.json").read_text())
    comp = pd.read_csv(O / "model_comparison.csv")
    scen = pd.read_csv(O / "retention_scenarios.csv")
    ab = pd.read_csv(O / "ablation_study.csv")
    chk = pd.read_csv(O / "validation_design_checks.csv")
    seg = pd.read_csv(O / "segment_risk.csv")
    eng = pd.read_csv(O / "modelling_dataset.csv")
    dq = pd.read_csv(O / "data_quality_log.csv")
    tc = k["test_active"]
    A, B, C1, C2, D, E = (scen.iloc[i] for i in range(6))
    act = eng[eng.segment == "Active"]
    r = lambda m: act[m].churned.mean()
    rule = comp[(comp.model == "Recency rule") & (comp.split == "test") & (comp.population == "Active only")].iloc[0]
    tier = seg[(seg.population == "All eligible") & (seg.dimension == "tier")].set_index("segment_value").observed_churn
    abw = ab[ab.variant == "With recency"].reset_index(drop=True)
    sw = k["support_whatif"]
    from .report import shap_group_shares
    sh_sel, sh_ch = shap_group_shares(pd.read_csv(O / "shap_feature_importance.csv"))
    d = Deck()

    # 1 ── Title
    s = d.slide(dark=True)
    d.text(s, "SILENT CART  ·  FRESHBASKET LOYALTY CHURN PREDICTION & RETENTION", 0.8, 1.2, 11, 0.4, size=13, color=MINT, bold=True)
    d.text(s, "Most of our churn has already happened.\nHere is how to stop the rest.", 0.8, 1.8, 11.5, 2.2, size=44, bold=True, color=WHITE)
    d.text(s, "Who is at risk, why, what to do, and what it is worth: from 18 months of loyalty data on 2,600 members",
           0.8, 4.35, 11, 0.9, size=18, color=MINT)
    d.text(s, [[("Guru Prasaath D", {"bold": True, "color": WHITE, "size": 15}),
                ("   ·   Data Science  ·  Tailwyndz Propel Internship assessment", {"color": MINT})]], 0.8, 6.4, 11, 0.5, size=12)
    d.notes(s, """Good morning. FreshBasket asked a simple question: which loyalty members are about to stop shopping, and what
    should we do about it? Today they send the same offers to everyone. In the next 12 minutes I'll show four things: that the
    churn number management sees mixes two very different problems; the warning signs that appear about two months before a
    member leaves; a model that catches about three in four at-risk members on a quarter it never saw; and a targeted retention
    playbook that turns today's loss-making blanket coupon into a positive return.""")

    # 2 ── Executive summary
    s = d.slide(title="The answer in one slide", kicker="Executive summary")
    stats = [
        (f"{k['share_churners_already_lapsed']:.0%}", "of 'churners' had already stopped buying before April. They need win-back, not prediction", ORANGE),
        (f"{tc['recall']:.0%}", f"of at-risk Active members caught at {tc['precision']:.0%} precision, on an unseen quarter (base rate {tc['churn_rate']:.1%})", GREEN),
        (f"{r(act.tickets_6m >= 3):.0%}", f"churn for Active members with 3+ support tickets, vs {r(act.tickets_6m == 0):.0%} with none", GREEN),
        (money(E.net_value), f"net from a next-best-action playbook vs {money(A.net_value)} for today's blanket coupon", GREEN),
    ]
    for i, (v, lab, col) in enumerate(stats):
        d.stat(s, v, lab, 0.6 + i * 3.08, 1.85, 2.88, color=col, size=40, h=2.3)
    d.card(s, 0.6, 4.5, 12.13, 1.95, fill=DARK)
    d.text(s, "RECOMMENDATION", 0.95, 4.75, 5, 0.3, size=11, bold=True, color=MINT)
    d.bullets(s, [("Split the KPI: ", "Lapsed (win-back) vs At-risk Active (retention), plus a 60-day no-purchase trigger"),
                  ("Score monthly, act selectively: ", "service call, personalised coupon or win-back, chosen per member by expected value"),
                  ("Prove it: ", "20% hold-out control group for one quarter before scaling")],
              0.95, 5.1, 11.5, 1.6, size=15, color=WHITE, gap=6)
    d.notes(s, f"""Here's the whole story on one slide. First, {k['share_churners_already_lapsed']:.0%} of members labelled as churned had
    already gone quiet for three months or more before the label window started; that is win-back, not prediction. Second, among
    members still shopping, the model catches {tc['recall']:.0%} of the ones who leave, with {tc['precision']:.0%} precision, against a base
    rate under 10%. Third, support experience is a strong lever: members with three or more tickets churn at nearly ten times
    the rate of members with none. And fourth, targeting changes the economics: the playbook returns about {money(E.net_value)} on this
    cohort where the blanket coupon loses {amount(A.net_value)}. My recommendation is at the bottom; I'll come back to it at the end.""")

    # 3 ── Data quality
    s = d.slide(title="We fixed the data before trusting it, and found the label leaks", kicker="Data quality")
    items = [
        ("Hidden header row", "Every sheet has a title banner; a default read silently breaks columns"),
        ("Duplicates", "71 exact duplicates + 2 sign-flipped customer-month conflicts removed"),
        ("Corrupted spend", "55 negative, zero or 9-15x inflated values rebuilt from spend = transactions x basket"),
        ("Impossible zeros", "2,105 'categories / coupons' values in zero-transaction months set to 0"),
        ("Missing months", "953 member-months absent: treated as missing data, never as zero activity"),
        ("Leaky label columns", "MONTHS_OBSERVED, LAST_PURCHASE_DATE, INSUFFICIENT_HISTORY_FLAG use Apr-Jun data: excluded"),
    ]
    for i, (h_, t_) in enumerate(items):
        y = 1.75 + i * 0.84
        d.badge(s, i + 1, 0.6, y + 0.04)
        d.text(s, h_, 1.2, y, 5.2, 0.32, size=15, bold=True)
        d.text(s, t_, 1.2, y + 0.33, 5.3, 0.5, size=12, color=MUTED)
    d.image(s, "01_dq_spend_repair.png", 6.9, 1.6, 5.9, 4.0)
    d.text(s, [[("Leakage check: ", {"bold": True, "color": ORANGE}),
                (f"adding the leaky columns would inflate Active-member PR-AUC from {chk.iloc[1].pr_auc_active:.2f} to "
                 f"{chk.iloc[2].pr_auc_active:.2f}. An automated test also proves our features ignore all post-cut-off data.", {})]],
           6.9, 5.8, 5.9, 1.0, size=13)
    d.notes(s, f"""Before any modelling I audited the three tables; all {len(dq)} checks are logged in a CSV. Six findings matter.
    The real header is on row two. There are duplicate rows, including two pairs where the same customer-month appears with
    positive and negative spend. 55 spend values are corrupted; because spend equals transactions times basket for 99.8% of rows,
    I could repair them exactly instead of deleting months. Missing months are data loss, not inactivity. The most important
    finding is on the right: three columns in the label table are computed using April to June, the very period we predict.
    Using them would quietly inflate performance, so they're excluded, and a unit test enforces that features never see the future.""")

    # 4 ── Two churns
    s = d.slide(title="Two different churn problems hide inside one number", kicker="What churn really looks like")
    d.image(s, "03_recency_cliff_two_churns.png", 0.5, 1.6, 7.7, 5.3)
    d.stat(s, f"{k['n_lapsed']:,} · {k['churn_rate_lapsed']:.0%}", "LAPSED: no purchase in Jan-Mar 2024. They left before the window opened. Action: win-back campaign, identified by a rule",
           8.5, 1.75, 4.3, color=ORANGE, size=32, h=2.25)
    d.stat(s, f"{k['n_active']:,} · {k['churn_rate_active']:.1%}", "ACTIVE: bought in Jan-Mar 2024. A genuine early-warning problem. Action: model + targeted retention",
           8.5, 4.2, 4.3, color=GREEN, size=32, h=2.25)
    d.notes(s, f"""This chart changed how I framed the whole project. Churn probability is a cliff in recency: a member who bought last
    month churns at about 3%; miss one month and it's {eng[eng.recency_months == 2].churned.mean():.0%}; miss two and it's
    {eng[eng.recency_months == 3].churned.mean():.0%}; beyond that it's essentially 100%. So the official churn rate of
    {k['churn_rate_all']:.0%} mixes {k['n_lapsed']} members who were already gone with {k['n_active']:,} active members who churn at under 10%.
    That matters for management, because the KPI overstates what retention can influence. It matters for data science too:
    any model looks brilliant on the lapsed group. So from here on I report every result for active members separately.""")

    # 5 ── Fingerprint
    s = d.slide(title="Churn has a fingerprint, and a two-month window to act", kicker="Early-warning signals")
    d.image(s, "04_churn_fingerprint.png", 0.5, 1.55, 12.3, 3.95)
    tiles = [(f"{r(act.spend_change_pct < -0.4):.0%}", "churn if spend fell 40%+ vs prior quarter"),
             (f"{r(act.app_l3 < 0.5):.0%}", "churn with under 0.5 app sessions / month"),
             (f"{k['complaint_test']['churn_if_complaint']:.0%} vs {k['complaint_test']['churn_if_none']:.0%}", "churn with vs without a recent formal complaint (p < 0.001)")]
    for i, (v, lab) in enumerate(tiles):
        d.stat(s, v, lab, 0.6 + i * 4.1, 5.6, 3.9, color=ORANGE, size=28, h=1.3)
    d.notes(s, """Here I lined up every active member on their last purchase before April and looked backwards. Retained members,
    in blue, are flat. Churners, in orange, show the same sequence every time: transactions, app sessions and email opens fade
    over roughly two months, while support tickets spike. That gives us a concrete intervention window, and simple triggers a
    CRM team can use without any model: a 40% drop in spend, app use falling below one session every two months, or a formal
    complaint. All of these are statistically significant after correcting for multiple tests.""")

    # 6 ── Engagement & support, tier
    s = d.slide(title="Support experience and engagement matter. Tier does not.", kicker="What drives churn")
    d.image(s, "05_engagement_support_vs_churn.png", 0.5, 1.55, 8.2, 3.0)
    d.image(s, "12_shap_by_feature_group.png", 0.5, 4.6, 8.2, 2.35)
    d.bullets(s, [("Support: ", f"3+ tickets in 6 months means {r(act.tickets_6m >= 3):.0%} churn vs {r(act.tickets_6m == 0):.0%}. A complaint roughly doubles risk"),
                  ("Engagement: ", "app use and email opens are top-3 SHAP drivers in both models"),
                  ("Tier: ", f"Platinum churns like Silver ({tier['Platinum']:.0%} vs {tier['Silver']:.0%}); tier, age and tenure not significant (q > 0.6)"),
                  ("Signal split: ", f"behaviour + engagement = {sh_sel:.0%} (logistic) / {sh_ch:.0%} (boosting) of SHAP attribution")],
              9.0, 1.75, 3.8, 5.0, size=14, gap=12)
    d.notes(s, f"""The brief asked whether tier, marketing engagement or support experience affect churn. The answer is yes, yes, and
    surprisingly no. Support friction is the clearest operational lever: three or more tickets takes churn from
    {r(act.tickets_6m == 0):.0%} to {r(act.tickets_6m >= 3):.0%}. Engagement, meaning app sessions and email opens, sits among the top three SHAP
    drivers in both the logistic and the gradient-boosting model. And membership tier makes no difference: Platinum members leave
    at the same rate as Silver. The current perks are not buying loyalty, which I'll come back to in the recommendations.""")

    # 7 ── Validation design
    s = d.slide(title="Validated the way it will be used: train on the past, test on an unseen quarter", kicker="Method")
    phases = [("TRAIN", "Jul-23 & Oct-23 snapshots", "Labels: Jul-Dec 2023", MINT, INK),
              ("VALIDATE", "Jan-24 snapshot", "Model choice, calibration, threshold", "8CCBAA", INK),
              ("TEST · reported once", "Apr-24 snapshot", "Official Apr-Jun 2024 label", DARK, WHITE)]
    for i, (t1, t2, t3, fill, fc) in enumerate(phases):
        x = 0.6 + i * 4.1
        d.card(s, x, 1.75, 3.85, 1.45, fill=fill)
        d.text(s, t1, x + 0.25, 1.9, 3.4, 0.35, size=13, bold=True, color=fc)
        d.text(s, t2, x + 0.25, 2.27, 3.4, 0.35, size=17, bold=True, color=fc)
        d.text(s, t3, x + 0.25, 2.67, 3.4, 0.35, size=12, color=fc)
        if i < 2:
            arr = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 3.88), Inches(2.33), Inches(0.2), Inches(0.3))
            arr.fill.solid()
            arr.fill.fore_color.rgb = rgb(GREEN)
            arr.line.fill.background()
    d.bullets(s, [("Rolling-origin snapshots: ", "the churn definition is replayed at earlier cut-offs (99.8% agreement with the official label), giving honest out-of-time data. No random split"),
                  ("Leakage-safe features: ", "last 6 months only; recency, 3-vs-3-month trends, engagement, support, history, profile (42 features)"),
                  ("Imbalance: ", "class / sample weights, then Platt calibration so scores are real probabilities, then F1-optimal threshold on validation Active members"),
                  ("Four models: ", "recency rule (baseline), logistic regression, random forest, gradient boosting; tracked in MLflow"),
                  ("Scale-ready: ", "PySpark port of the feature pipeline matches pandas exactly on all 4 snapshots, ready for Databricks")],
              0.6, 3.5, 12.1, 2.4, size=14, gap=7)
    tiles = [(f"{chk.iloc[0].pr_auc_active:.2f}", "Out-of-time test, clean features (reported)", GREEN),
             (f"{chk.iloc[1].pr_auc_active:.2f}", "Random 5-fold CV: behaviour is stable over time", MUTED),
             (f"{chk.iloc[2].pr_auc_active:.2f}", "Random CV + leaky label columns: inflation avoided", ORANGE)]
    d.text(s, "Active-member PR-AUC under different validation set-ups (gradient boosting)", 0.6, 5.45, 12, 0.3, size=12, color=MUTED, italic=True)
    for i, (v, lab, col) in enumerate(tiles):
        d.stat(s, v, lab, 0.6 + i * 4.1, 5.8, 3.9, color=col, size=26, h=1.15)
    d.notes(s, """Random splits flatter churn models because the future leaks into training. Instead I replayed the churn definition
    at earlier cut-offs: July and October 2023 for training, January 2024 for model selection, calibration and the threshold, and
    the official April to June label only once, at the end. Features only look at the six months before each cut-off. Class
    imbalance is handled with weights, then the scores are calibrated so a 30% really means 30%; that matters for the money maths.
    Everything is tracked in MLflow, and the feature pipeline also exists in PySpark, verified identical, so it can run on Databricks.""")

    # 8 ── Results
    s = d.slide(title="On Active members, machine learning roughly doubles what a recency rule can do", kicker="Model results · unseen Apr-Jun 2024 quarter")
    t = comp[(comp.split == "test") & (comp.population == "Active only")].copy()
    t["PR-AUC (95% CI)"] = [f"{a:.2f} ({lo:.2f}-{hi:.2f})" for a, lo, hi in zip(t.pr_auc, t.pr_auc_ci_low, t.pr_auc_ci_high)]
    tbl = pd.DataFrame({"Model": t.model, "PR-AUC (95% CI)": t["PR-AUC (95% CI)"],
                        "ROC-AUC": t.roc_auc.map("{:.2f}".format), "Precision": t.precision.map("{:.0%}".format),
                        "Recall": t.recall.map("{:.0%}".format), "F1": t.f1.map("{:.2f}".format)}).reset_index(drop=True)
    sel = int(tbl.index[tbl["Model"] == k["selected_model"]][0])
    d.table(s, tbl, 0.6, 1.75, 6.6, [1.85, 1.6, 0.9, 0.95, 0.7, 0.6], size=12, highlight_row=sel)
    d.image(s, "07_pr_curves.png", 7.4, 1.55, 5.5, 3.3)
    g = (k.get("gru") or {}).get("Active only")
    sel_txt = (f"{k['selected_model']} (best of the tabular models on validation). A GRU sequence model (PyTorch) scores "
               f"{g['pr_auc']:.2f}, likely better: shadow-test it next quarter" if g else
               f"{k['selected_model']} (best on validation). Gradient boosting is statistically tied")
    d.bullets(s, [("Selected: ", sel_txt),
                  ("At threshold {:.2f}: ".format(k['threshold']), f"catches {tc['recall']:.0%} of Active churners, {tc['precision']:.0%} precision; top 10% of scores = {tc['lift_top10pct']:.1f}x lift"),
                  ("Honest framing: ", "on all members every model scores PR-AUC > 0.94 because lapsed members are easy. That number would mislead")],
              0.6, 4.25, 6.6, 2.6, size=13, gap=9)
    d.image(s, "09_calibration.png", 7.4, 4.95, 5.5, 2.0)
    d.notes(s, f"""On active members, the recency rule reaches a PR-AUC of {rule.pr_auc:.2f}. All three machine-learning models roughly double
    it, to around 0.8, on a quarter they never saw. Logistic regression won on validation; gradient boosting scores a touch higher on
    test, but the confidence intervals overlap almost completely, so I deploy the simpler, transparent model and keep boosting as
    a challenger. At the tuned threshold we flag {k['high_risk_active']} active members and three out of four real churners are among them. The
    calibration plot, bottom right, shows the probabilities can be trusted as real odds. I also trained a GRU network on the
    raw monthly sequences: it scores slightly higher and wins in 97% of bootstrap resamples, so the plan is to shadow-score it for a
    quarter and promote it if the gain holds: champion and challenger.""")

    # 9 ── Drivers (SHAP) per member
    s = d.slide(title="Every member gets a score and three reasons the CRM team can act on", kicker="Explainability (SHAP)")
    d.image(s, "11_shap_summary_active.png", 0.5, 1.5, 6.3, 5.45)
    preds = pd.read_csv(O / "churn_predictions.csv")
    ex = preds[(preds.segment == "Active") & (preds.predicted_label == 1)].iloc[[0, 3, 8]]
    d.text(s, "From outputs/churn_predictions.csv (high-risk Active members)", 7.1, 1.6, 5.7, 0.3, size=12, color=MUTED, italic=True)
    for i, row in enumerate(ex.itertuples()):
        y = 2.0 + i * 1.5
        d.card(s, 7.1, y, 5.7, 1.32, fill=TINT)
        d.text(s, f"Member {row.customer_id}", 7.35, y + 0.14, 3, 0.3, size=13, bold=True)
        d.text(s, f"{row.churn_probability:.0%} risk", 10.9, y + 0.12, 1.7, 0.35, size=16, bold=True, color=ORANGE, align=PP_ALIGN.RIGHT)
        d.text(s, row.top_3_drivers.replace("; ", "  ·  "), 7.35, y + 0.5, 5.25, 0.8, size=12, color=INK)
    d.text(s, "Drivers are SHAP contributions, grouped so the three reasons are always distinct (e.g. not three ways of saying 'stopped buying').",
           7.1, 6.5, 5.7, 0.5, size=11, color=MUTED)
    d.notes(s, """A score on its own doesn't get used. On the left is the SHAP summary for active members: low recent purchasing,
    low email and app engagement, and recent support tickets push risk up. On the right is what the store or CRM team actually
    receives: each member's churn probability plus three plain-English reasons, drawn from that member's own SHAP values and
    grouped so the three reasons are different. That's the bridge from a model to an action. One caution: SHAP explains the
    model, not causation; that's why the next step is an experiment, not a guess. And the last mile is automated too: for the
    priority members an LLM on Groq drafts the actual message or call script from these same three reasons, under guardrails that
    block invented promo codes, false apologies and any mention of tracking.""")

    # 10 ── Ablation + segments
    s = d.slide(title="Behaviour predicts churn. Demographics alone are no better than chance.", kicker="Ablation & segments")
    d.image(s, "13_segment_risk_active.png", 0.5, 1.6, 7.1, 3.2)
    sa = seg[seg.population == "Active only"].set_index(["dimension", "segment_value"]).predicted_risk
    seg_tiles = [(f"{sa['city', 'Austin']:.0%} vs {sa['city', 'Nashville']:.0%}", "predicted risk, Austin vs Nashville (riskiest vs safest city)"),
                 (f"{sa['tier', 'Platinum']:.0%} vs {sa['tier', 'Silver']:.0%}", "Platinum vs Silver predicted risk; observed churn is flat across tiers"),
                 (f"{sa['signup_cohort', '2024']:.0%}", "predicted risk for 2024 joiners: a honeymoon period")]
    for i, (v, lab) in enumerate(seg_tiles):
        d.stat(s, v, lab, 0.6 + i * 2.37, 5.0, 2.22, color=GREEN, size=22, h=1.75)
    steps = [("Demographics only", abw.pr_auc_active[0]), ("+ Behavioural", abw.pr_auc_active[1]), ("+ Engagement", abw.pr_auc_active[2]),
             ("+ Support", abw.pr_auc_active[3]), ("All features", abw.pr_auc_active[4])]
    d.text(s, "PR-AUC on Active members (test quarter)", 7.9, 1.65, 4.9, 0.3, size=12, color=MUTED)
    for i, (lab, v) in enumerate(steps):
        y = 2.05 + i * 0.56
        d.text(s, lab, 7.9, y, 2.4, 0.4, size=14, bold=i in (1, 2))
        bar = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.2), Inches(y + 0.04), Inches(2.0 * v), Inches(0.3))
        bar.fill.solid()
        bar.fill.fore_color.rgb = rgb(GREEN if i else "B8C2BD")
        bar.line.fill.background()
        d.text(s, f"{v:.2f}", 10.3 + 2.0 * v, y, 0.6, 0.4, size=13, bold=True)
    d.bullets(s, [("Base rate is {:.1%}: ".format(k['churn_rate_active']), "demographics-only adds nothing"),
                  ("Engagement adds +{:.2f}, support +{:.2f} ".format(abw.pr_auc_active[2] - abw.pr_auc_active[1], abw.pr_auc_active[3] - abw.pr_auc_active[2]), "on top of behaviour"),
                  ("Segments ", "differ far less than behaviour: use them for logistics (which team, which channel), not for targeting")],
              7.9, 5.0, 4.9, 2.0, size=13, gap=8)
    d.notes(s, f"""The brief asked whether behavioural, engagement and support features beat demographics alone. The ablation says
    clearly yes. With only demographics and tier, the model is at the base rate: no better than chance. Adding purchase behaviour
    jumps it to {abw.pr_auc_active[1]:.2f}, engagement adds another {abw.pr_auc_active[2] - abw.pr_auc_active[1]:.2f}, and support pushes it to about 0.8. Even with
    recency removed the model stays near 0.8, so it is reading genuine early-warning signs. By segment, Austin and Columbus run
    highest and members who joined in 2024 lowest, but these gaps are small next to behaviour, so segments decide which team
    acts, not who gets targeted.""")

    # 11 ── Scenarios
    s = d.slide(title="Targeting turns retention from a cost into a return", kicker="Retention scenarios · this cohort, next quarter")
    d.image(s, "14_retention_scenarios.png", 0.5, 1.5, 8.6, 2.6)
    d.image(s, "15_scenario_sensitivity.png", 0.5, 4.2, 5.4, 2.75)
    d.stat(s, money(A.net_value), f"Blanket coupon to all {int(A.members_targeted):,} members: ${A.cost / 1000:,.1f}k spent", 9.4, 1.6, 3.4, color=ORANGE, size=34, h=1.45)
    d.stat(s, money(E.net_value), f"Next-best-action playbook: {int(E.members_targeted)} members, ${E.cost / 1000:,.1f}k spent, ~{E.churners_prevented:.0f} churners prevented",
           9.4, 3.2, 3.4, color=GREEN, size=34, h=1.6)
    d.bullets(s, [("Support outreach: ", f"call everyone with tickets = {money(C1.net_value)}; call only model-flagged = {money(C2.net_value)}"),
                  ("Model what-if: ", f"resolving tickets cuts their mean risk {sw['mean_risk_before']:.0%} → {sw['mean_risk_after']:.0%} (upper bound)"),
                  ("Robust: ", f"targeted coupon breaks even at {B.break_even_uplift:.0%} uplift (assumed 15%); blanket never does in range")],
              6.2, 5.0, 6.6, 2.0, size=13, gap=8)
    d.notes(s, f"""Now the money. Assumptions are explicit: a ten-dollar coupon cuts a member's churn probability by 15%, a fifteen-dollar
    service call by 25%, and a member's value is six months of their spend at a 25% margin. The blanket coupon, today's approach,
    costs about ${A.cost / 1000:,.0f}k on this cohort and loses money, because most recipients were staying anyway or are already
    gone. The playbook gives each member at most one action and only when it pays back: about {E.churners_prevented:.0f} churners
    prevented for ${E.cost / 1000:,.1f}k, a {money(E.net_value)} return. The brief's support-outreach idea works, but only when targeted:
    calling everyone with tickets loses money, while calling the model-flagged ones makes money. The sensitivity chart shows targeted
    actions break even at roughly half the assumed effect.""")

    # 12 ── Recommendations
    s = d.slide(dark=True, title="Recommendations: a 90-day plan", kicker="What to do next")
    recs = [("Redefine the KPI", "Report Lapsed and At-risk Active separately; add a 60-day no-purchase trigger"),
            ("Run the playbook monthly", "Score all members; route each to service call, coupon, win-back or no action by expected value"),
            ("Fix the support loop", "Close tickets within the 2-month window; prioritise model-flagged members"),
            ("Free nudges first", "Spend drop of 40%+ or app use under 0.5/month triggers app push and preferred-category content"),
            ("Rethink tier perks", "Platinum ≠ loyal: pilot engagement-based rewards, not spend-only tiers"),
            ("Prove it", f"20% hold-out: ~{int(k['ab_test_15pct']['total_members'])} flagged members detect a 15% uplift; replace assumptions with measurements")]
    for i, (h_, t_) in enumerate(recs):
        col, row = i % 2, i // 2
        x, y = 0.6 + col * 6.15, 1.8 + row * 1.32
        d.card(s, x, y, 5.95, 1.15, fill="22473A")
        d.badge(s, i + 1, x + 0.22, y + 0.33, fill=GREEN)
        d.text(s, h_, x + 0.85, y + 0.14, 4.9, 0.35, size=16, bold=True, color=WHITE)
        d.text(s, t_, x + 0.85, y + 0.5, 4.9, 0.6, size=12, color=MINT)
    d.text(s, [[("Production path on the Tailwyndz stack:  ", {"bold": True, "color": WHITE}),
                ("Azure Data Factory → Delta (Databricks) → PySpark features → MLflow champion model → CRM queue + Dash Retention Console (Databricks Apps)  ·  monitor PR-AUC, calibration & PSI drift monthly", {"color": MINT})]],
           0.6, 5.95, 12.1, 0.8, size=13)
    d.notes(s, """To close, six actions for the next 90 days. Split the churn KPI so management sees what retention can actually
    influence, and add a 60-day no-purchase trigger. Run the scoring and next-best-action list every month. Treat support as a
    retention lever and close the loop inside the two-month window. Use free nudges before paid offers. Revisit tier perks,
    since Platinum isn't buying loyalty. And prove the uplift with a hold-out control group before scaling, because the economics
    rest on assumptions until we measure them; about 730 flagged members are enough to detect a 15% uplift. Everything I've shown
    regenerates from one command, the feature pipeline is ported to Spark for Databricks, and there is a Dash retention console
    where the CRM team can explore members and test their own assumptions. Thank you; happy to take questions.""")

    C.PRESENTATION.mkdir(exist_ok=True)
    path = C.PRESENTATION / "churn_presentation.pptx"
    d.prs.save(path)
    write_speaker_notes(d.prs)
    print(f"  deck -> {path.relative_to(C.ROOT)} ({d.n} slides)")
    return str(path)


def write_speaker_notes(prs, wpm: int = 130) -> str:
    """Export every slide's speaker notes to presentation/speaker_notes.md: a printable practice script."""
    total, body = 0, []
    for i, s in enumerate(prs.slides, 1):
        texts = [sh.text_frame.text.replace("\n", " ") for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
        title = next((t for t in texts if not t.isupper() and len(t) > 15), f"Slide {i}")
        note = s.notes_slide.notes_text_frame.text.strip()
        words = len(note.split())
        total += words
        body += [f"## Slide {i}: {title}", f"*~{words / wpm:.1f} min · {words} words*", "", note, ""]
    head = ["# Silent Cart: speaker notes (practice script)", "",
            f"Total: {total} words, about {total / wpm:.0f} minutes at {wpm} words per minute. With pauses and slide changes "
            "this lands around 11-13 minutes, leaving time for questions in a 15-minute slot.", "",
            "Tips: pause after each headline number; point at the chart you are describing; slide 4 (two churns) and "
            "slide 11 (money) are the moments to slow down.", ""]
    path = C.PRESENTATION / "speaker_notes.md"
    path.write_text("\n".join(head + body))
    return str(path)
