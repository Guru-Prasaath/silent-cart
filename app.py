"""Silent Cart Retention Console: an interactive Dash app over the pipeline outputs.

    python run_pipeline.py      # once, to produce outputs/
    python app.py               # then open http://127.0.0.1:8050

Tabs: Overview · Members (action list + member drill-down) · Scenario simulator (live economics) · Insights.
The simulator calls the same src.scenario.simulate() the pipeline uses, so numbers always agree.
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update
from plotly.subplots import make_subplots

from src import config as C, genai
from src.scenario import Economics, simulate

O = C.OUTPUTS
COL = C.COLORS
GREEN, DARK, ORANGE, BLUE = "#2E7D5B", "#17332A", COL["orange"], COL["blue"]


# --------------------------------------------------------------------------- data
def load():
    k = json.loads((O / "key_metrics.json").read_text())
    cohort = pd.read_csv(O / "modelling_dataset.csv").merge(
        pd.read_csv(O / "churn_predictions.csv")[["customer_id", "churn_probability", "predicted_label", "top_3_drivers"]],
        on="customer_id")
    actions = pd.read_csv(O / "retention_action_list.csv")
    watch = pd.read_csv(O / "next_quarter_watchlist.csv")
    activity = pd.read_csv(C.PROCESSED / "activity_clean.csv", parse_dates=["MONTH"])
    tables = {n: pd.read_csv(O / f"{n}.csv") for n in
              ["model_comparison", "personas", "drift_psi", "fairness_by_group", "ab_test_design", "profit_threshold_curve",
               "survival_curves", "survival_summary", "retention_scenarios"]}
    gru = O / "deep_model_results.csv"
    tables["deep"] = pd.read_csv(gru) if gru.exists() else None
    msgs = O / "retention_messages.csv"
    tables["messages"] = pd.read_csv(msgs) if msgs.exists() else pd.DataFrame(columns=["customer_id"])
    return k, cohort, actions, watch, activity, tables


K, COHORT, ACTIONS, WATCH, ACTIVITY, T = load()


def style_fig(fig, title=None, height=340):
    fig.update_layout(template="simple_white", height=height, margin=dict(l=50, r=20, t=50 if title else 20, b=40),
                      title=dict(text=title, x=0, font=dict(size=15, color="#0b0b0b")) if title else None,
                      font=dict(family="Inter, Helvetica Neue, Arial", size=12, color="#52514e"),
                      legend=dict(orientation="h", y=-0.2), hoverlabel=dict(font_size=12))
    fig.update_xaxes(showgrid=False, linecolor=COL["axis"])
    fig.update_yaxes(gridcolor=COL["grid"], linecolor=COL["axis"])
    return fig


def kpi(value, label, color=GREEN):
    return html.Div([html.Div(value, className="kpi-value", style={"color": color}), html.Div(label, className="kpi-label")],
                    className="kpi")


def money(x):
    return f"{'−' if x < 0 else ''}${abs(x):,.0f}"


# --------------------------------------------------------------------------- overview
def overview():
    nq = K["next_quarter"]["active"]
    rec = COHORT.groupby("recency_months").churned.agg(["mean", "size"]).reset_index()
    rec["label"] = rec.recency_months.map(lambda v: "7+" if v >= 7 else str(int(v)))
    f1 = go.Figure(go.Bar(x=rec.label, y=rec["mean"], marker_color=[BLUE if v <= 3 else ORANGE for v in rec.recency_months],
                          customdata=rec["size"], hovertemplate="Last purchase %{x} month(s) before April<br>"
                                                                "Churn %{y:.0%}<br>%{customdata} members<extra></extra>",
                          text=[f"{v:.0%}" for v in rec["mean"]], textposition="outside"))
    f1.update_yaxes(tickformat=".0%", range=[0, 1.15], title="Churn rate Apr-Jun 2024")
    f1.update_xaxes(title="Months since last purchase (at 1-Apr-2024)")
    style_fig(f1, "Two churns in one label: Active (blue) vs already Lapsed (orange)")

    mc = T["model_comparison"]
    mc = mc[(mc.split == "test") & (mc.population == "Active only")][["model", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high"]]
    if T["deep"] is not None:
        d = T["deep"][T["deep"].population == "Active only"]
        mc = pd.concat([mc, d[["model", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high"]]])
    mc = mc.sort_values("pr_auc")
    f2 = go.Figure(go.Bar(x=mc.pr_auc, y=mc.model, orientation="h",
                          marker_color=[GREEN if m == K["selected_model"] else (ORANGE if "GRU" in m else COL["neutral"]) for m in mc.model],
                          error_x=dict(type="data", symmetric=False, array=mc.pr_auc_ci_high - mc.pr_auc,
                                       arrayminus=mc.pr_auc - mc.pr_auc_ci_low, color=COL["muted"]),
                          text=[f"{v:.2f}" for v in mc.pr_auc], textposition="inside", insidetextanchor="start",
                          hovertemplate="%{y}: PR-AUC %{x:.3f}<extra></extra>"))
    f2.add_vline(x=K["churn_rate_active"], line_dash="dot", line_color=COL["muted"], annotation_text="base rate",
                 annotation_position="bottom right")
    f2.update_xaxes(range=[0, 1], title="Test PR-AUC, Active members (95% CI)")
    style_fig(f2, "Models on the unseen quarter (green = deployed, orange = challenger)")

    p = T["personas"].sort_values("observed_churn")
    f3 = go.Figure(go.Bar(x=p.observed_churn, y=p.persona, orientation="h", marker_color=ORANGE,
                          customdata=np.c_[p.members, p.recommended_play],
                          hovertemplate="%{y}<br>Churn %{x:.0%} · %{customdata[0]} members<br>Play: %{customdata[1]}<extra></extra>",
                          text=[f"{v:.0%}" for v in p.observed_churn], textposition="outside"))
    f3.update_xaxes(tickformat=".0%", range=[0, p.observed_churn.max() * 1.25])
    style_fig(f3, "Behavioural personas (Active members)")

    return html.Div([
        html.Div([
            kpi(f"{K['share_churners_already_lapsed']:.0%}", "of churners had already stopped buying before April", ORANGE),
            kpi(f"{K['churn_rate_active']:.1%}", f"churn among {K['n_active']:,} Active members (the real prediction problem)"),
            kpi(f"{K['test_active']['recall']:.0%} / {K['test_active']['precision']:.0%}", "recall / precision on Active members, unseen quarter"),
            kpi(f"{nq['expected_churners']:.0f}", f"Active churners expected Jul-Sep 2024 (90% interval {nq['interval_90'][0]:.0f}-{nq['interval_90'][1]:.0f})", BLUE),
            kpi(money(K["revenue_at_risk_active"]), "6-month revenue at risk, Active members"),
        ], className="kpi-row"),
        html.Div([dcc.Graph(figure=f1), dcc.Graph(figure=f2)], className="grid-2"),
        html.Div([dcc.Graph(figure=f3), html.Div([
            html.H4("How to read this console"),
            html.Ul([
                html.Li([html.B("Members: "), "today's prioritised action list, or next quarter's watchlist. Click a row to see why a member is at risk."]),
                html.Li([html.B("Scenario simulator: "), "change the business assumptions and watch each strategy's return update."]),
                html.Li([html.B("Insights: "), "survival, profit-optimal threshold, A/B test sizing, drift and fairness monitoring."]),
            ]),
            html.P(f"Deployed model: {K['selected_model']} (threshold {K['threshold']:.2f}), calibrated, out-of-time validated. "
                   "Drivers are SHAP contributions in plain English.", className="muted"),
        ], className="card")], className="grid-2"),
    ])


# --------------------------------------------------------------------------- members
def _opts(series):
    return [{"label": v, "value": v} for v in sorted(series.dropna().unique())]


def members():
    return html.Div([
        html.Div([
            dcc.RadioItems(id="m-source", value="current", inline=True, className="radio",
                           options=[{"label": " Current quarter (Apr-Jun 2024, with outcomes)", "value": "current"},
                                    {"label": " Next quarter watchlist (Jul-Sep 2024)", "value": "next"}]),
        ]),
        html.Div([
            dcc.Dropdown(id="m-segment", options=_opts(ACTIONS.segment), placeholder="Segment", multi=True),
            dcc.Dropdown(id="m-band", options=_opts(ACTIONS.risk_band), placeholder="Risk band", multi=True),
            dcc.Dropdown(id="m-action", options=_opts(ACTIONS.recommended_action), placeholder="Recommended action", multi=True),
            dcc.Dropdown(id="m-persona", options=_opts(ACTIONS.persona), placeholder="Persona", multi=True),
            dcc.Dropdown(id="m-city", options=_opts(ACTIONS.city), placeholder="City", multi=True),
            dcc.Dropdown(id="m-tier", options=_opts(ACTIONS.tier), placeholder="Tier", multi=True),
        ], className="filters"),
        html.Div(id="m-summary", className="muted", style={"margin": "6px 0 10px"}),
        html.Div([
            dash_table.DataTable(
                id="m-table", page_size=14, sort_action="native",
                style_table={"overflowX": "auto"}, style_as_list_view=True,
                style_cell={"fontFamily": "Inter, Helvetica Neue, Arial", "fontSize": 13, "padding": "6px 8px", "textAlign": "left",
                            "maxWidth": 260, "overflow": "hidden", "textOverflow": "ellipsis"},
                style_header={"fontWeight": 600, "backgroundColor": "#f3f6f4", "borderBottom": "1px solid #d9e2dc"},
                style_data_conditional=[
                    {"if": {"filter_query": '{risk_band} = "High"', "column_id": "risk_band"}, "color": ORANGE, "fontWeight": 600},
                    {"if": {"state": "active"}, "backgroundColor": "#eef5f0", "border": "1px solid #2E7D5B"}]),
            html.Div(id="m-detail", className="card detail"),
        ], className="grid-members"),
    ])


TABLE_COLS = ["customer_id", "segment", "persona", "tier", "city", "churn_probability", "risk_band", "revenue_at_risk",
              "recommended_action", "expected_net_value"]


def member_detail(cid, source):
    df = ACTIONS if source == "current" else WATCH
    r = df[df.customer_id == cid]
    if r.empty:
        return html.P("Click a member in the table to see their risk story.", className="muted")
    r = r.iloc[0]
    act = ACTIVITY[ACTIVITY.CUSTOMER_ID == cid].sort_values("MONTH")
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_bar(x=act.MONTH, y=act.TRANSACTIONS, name="Transactions", marker_color=BLUE)
    fig.add_scatter(x=act.MONTH, y=act.APP_SESSIONS, name="App sessions", mode="lines+markers", line=dict(color=COL["aqua"]))
    tk = act[act.SUPPORT_TICKETS > 0]
    fig.add_scatter(x=tk.MONTH, y=tk.SUPPORT_TICKETS, name="Support tickets", mode="markers",
                    marker=dict(color=ORANGE, size=11, symbol="diamond"))
    cut = "2024-04-01" if source == "current" else "2024-07-01"
    fig.add_vline(x=pd.Timestamp(cut).timestamp() * 1000, line_dash="dot", line_color=COL["muted"])
    style_fig(fig, "Monthly activity (dotted line = scoring date)", height=280)
    outcome = None
    if source == "current":
        y = COHORT.loc[COHORT.customer_id == cid, "churned"]
        if len(y):
            outcome = html.Span("Actually churned in Apr-Jun 2024" if y.iloc[0] else "Stayed in Apr-Jun 2024",
                                className="pill " + ("pill-bad" if y.iloc[0] else "pill-good"))
    drivers = [html.Li(d) for d in str(r.top_3_drivers).split("; ")]
    cached = T["messages"][T["messages"].customer_id == cid] if source == "current" else T["messages"].iloc[0:0]
    msg_box = message_card(cached.iloc[0].to_dict()) if len(cached) else html.P(
        "No pre-drafted message for this member yet. Click the button to draft one.", className="muted")
    llm_ready = genai.api_key() is not None
    return html.Div([
        html.Div([html.H3(f"Member {cid}"), outcome], className="detail-head"),
        html.Div([
            kpi(f"{r.churn_probability:.0%}", f"churn probability · {r.risk_band} risk", ORANGE if r.risk_band == "High" else GREEN),
            kpi(money(r.revenue_at_risk), "6-month revenue at risk", BLUE),
        ], className="kpi-row tight"),
        html.P([html.B("Segment: "), f"{r.segment}", "  ·  ", html.B("Tier: "), f"{r.tier}", "  ·  ", html.B("City: "), f"{r.city}"]
               + ([ "  ·  ", html.B("Persona: "), f"{r.persona}"] if "persona" in r and pd.notna(r.get("persona")) else [])),
        html.H4("Why the model flags this member (SHAP)"), html.Ul(drivers),
        html.H4("Next best action"),
        html.P([html.Span(r.recommended_action, className="pill pill-action"),
                f"  expected net value {money(r.expected_net_value)}"]),
        html.Div([
            html.Div([html.H4("Draft retention message"),
                      html.Button("Draft with AI (Groq)" if llm_ready else "Draft (template; add GROQ_API_KEY for AI)",
                                  id="draft-btn", className="btn",
                                  disabled=r.recommended_action == "Monitor (no paid action)")], className="detail-head"),
            dcc.Loading(html.Div(msg_box, id="draft-box"), type="dot"),
            dcc.Store(id="draft-member", data={"cid": int(cid), "source": source}),
        ], className="msg-wrap"),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
    ])


def message_card(m: dict):
    ok = bool(m.get("guardrail_pass", True))
    src = str(m.get("source", ""))
    return html.Div([
        html.Div([html.Span(m.get("channel", ""), className="pill pill-good"),
                  html.Span("AI-drafted" if src.startswith("groq") else "Template", className="pill"),
                  html.Span("guardrails passed" if ok else f"guardrail flags: {m.get('guardrail_notes', '')}",
                            className="pill " + ("pill-good" if ok else "pill-bad"))], className="pills"),
        html.P(html.B(m.get("subject", ""))),
        html.P(m.get("message", ""), className="msg"),
        html.P(f"Source: {src}. Input = drivers, persona, category, tier, action only (no personal data).", className="muted"),
    ], className="msg-card")


# --------------------------------------------------------------------------- simulator
def slider(id_, label, lo, hi, step, value, fmt):
    marks = {v: fmt(v) for v in np.linspace(lo, hi, 5)}
    return html.Div([html.Label(label), dcc.Slider(id=id_, min=lo, max=hi, step=step, value=value, marks=marks,
                                                    tooltip={"placement": "bottom", "always_visible": False})], className="slider")


def simulator():
    pct = lambda v: f"{v:.0%}"
    usd = lambda v: f"${v:.0f}"
    return html.Div([
        html.Div([
            html.Div([
                html.H4("Assumptions (drag to test)"),
                slider("s-coupon-cost", "Coupon cost per member", 2, 30, 1, C.COUPON_COST, usd),
                slider("s-coupon-up", "Coupon: relative churn reduction", 0, 0.5, 0.01, C.COUPON_RELATIVE_UPLIFT, pct),
                slider("s-call-cost", "Service call cost per member", 5, 40, 1, C.OUTREACH_COST, usd),
                slider("s-call-up", "Service call: relative churn reduction", 0, 0.6, 0.01, C.OUTREACH_RELATIVE_UPLIFT, pct),
                slider("s-wb-cost", "Win-back cost per lapsed member", 1, 20, 1, C.WINBACK_COST, usd),
                slider("s-wb-rate", "Win-back reactivation rate", 0, 0.2, 0.01, C.WINBACK_REACTIVATION, pct),
                slider("s-margin", "Gross margin", 0.1, 0.45, 0.01, C.GROSS_MARGIN, pct),
                slider("s-horizon", "Value horizon (months of spend)", 3, 12, 1, C.VALUE_HORIZON_MONTHS, lambda v: f"{v:.0f}m"),
                slider("s-thr", "Targeting threshold (churn probability)", 0.1, 0.9, 0.01, round(K["threshold"], 2), pct),
            ], className="card"),
            html.Div([html.Div(id="s-kpis", className="kpi-row"), dcc.Graph(id="s-chart"), html.Div(id="s-table")]),
        ], className="grid-sim"),
    ])


def scenario_view(econ, thr):
    table, *_ = simulate(COHORT, thr, econ)
    best = table.loc[table.net_value.idxmax()]
    a = table.iloc[0]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Net value (margin retained − cost)", "Expected churners prevented"),
                        shared_yaxes=True, horizontal_spacing=0.04)
    t = table.iloc[::-1]
    fig.add_bar(y=t.scenario, x=t.net_value, orientation="h", row=1, col=1,
                marker_color=[COL["green"] if v >= 0 else COL["red"] for v in t.net_value],
                text=[money(v) for v in t.net_value], textposition="auto", hovertemplate="%{y}: %{x:$,.0f}<extra></extra>")
    fig.add_bar(y=t.scenario, x=t.churners_prevented, orientation="h", row=1, col=2, marker_color=BLUE,
                text=[f"{v:.0f}" for v in t.churners_prevented], textposition="auto",
                hovertemplate="%{y}: %{x:.1f} churners prevented<extra></extra>")
    fig.update_layout(showlegend=False)
    style_fig(fig, None, height=360)
    kpis = [kpi(best.scenario.split(". ", 1)[1], f"best strategy · net {money(best.net_value)}"),
            kpi(money(best.net_value - a.net_value), "better than today's blanket coupon", BLUE),
            kpi(f"{best.active_churn_before:.1%} → {best.active_churn_after:.1%}", "Active churn, before → after (best strategy)", ORANGE)]
    show = table[["scenario", "members_targeted", "churners_prevented", "cost", "margin_retained", "net_value", "roi"]].copy()
    show["churners_prevented"] = show.churners_prevented.round(1)
    for c in ["cost", "margin_retained", "net_value"]:
        show[c] = show[c].map(money)
    show["roi"] = show.roi.map(lambda v: "n/a" if pd.isna(v) else f"{v:+.0%}")
    tbl = dash_table.DataTable(data=show.to_dict("records"), columns=[{"name": c.replace("_", " "), "id": c} for c in show.columns],
                               style_as_list_view=True, style_cell={"fontFamily": "Inter, Helvetica Neue, Arial", "fontSize": 13,
                                                                    "padding": "6px 8px", "textAlign": "left"},
                               style_header={"fontWeight": 600, "backgroundColor": "#f3f6f4"})
    return kpis, fig, tbl


# --------------------------------------------------------------------------- insights
def insights():
    sc = T["survival_curves"]
    fs = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=("Early support experience", "Early app engagement", "Tier"))
    pal = [ORANGE, BLUE, COL["aqua"]]
    for j, dim in enumerate(["early_support", "early_app_use", "tier"], 1):
        for col, (g, d) in zip(pal, sc[sc.dimension == dim].groupby("group")):
            fs.add_scatter(x=d.t, y=d.survival, mode="lines", line=dict(shape="hv", color=col, width=2), name=g,
                           row=1, col=j, hovertemplate=f"{g}<br>month %{{x}}: %{{y:.0%}} still buying<extra></extra>")
    fs.update_yaxes(tickformat=".0%", range=[0.4, 1.01])
    style_fig(fs, "Survival after month 3: early app use matters, early complaints and tier do not", height=360)

    pc = T["profit_threshold_curve"]
    fp = go.Figure(go.Scatter(x=pc.threshold, y=pc.net_value, mode="lines", line=dict(color=BLUE, width=3),
                              customdata=np.c_[pc.members_targeted, pc.precision, pc.recall],
                              hovertemplate="t=%{x:.2f}: %{y:$,.0f} · %{customdata[0]} members<br>precision %{customdata[1]:.0%}, "
                                            "recall %{customdata[2]:.0%}<extra></extra>"))
    fp.add_vline(x=K["threshold"], line_dash="dot", line_color=ORANGE, annotation_text="deployed (F1)")
    fp.add_vline(x=K["profit_optimal_threshold"], line_dash="dot", line_color=COL["green"], annotation_text="profit-optimal",
                 annotation_position="bottom right")
    fp.update_yaxes(title="Net value of targeted coupon")
    fp.update_xaxes(title="Threshold")
    style_fig(fp, "Profit vs threshold: flat optimum = robust decision")

    dr = T["drift_psi"].head(12).iloc[::-1]
    fd = go.Figure(go.Bar(x=dr.psi_test_vs_train, y=dr.feature, orientation="h",
                          marker_color=[COL["red"] if v > 0.25 else COL["yellow"] if v > 0.1 else BLUE for v in dr.psi_test_vs_train]))
    fd.add_vline(x=0.1, line_dash="dot", line_color=COL["muted"])
    fd.add_vline(x=0.25, line_dash="dot", line_color=COL["red"])
    style_fig(fd, "Feature drift (PSI, test vs train)")

    ab = T["ab_test_design"]
    ab_show = pd.DataFrame({"Uplift to detect": ab.relative_uplift_to_detect.map("{:.0%}".format),
                            "Control members": ab.control_members, "Total flagged members": ab.total_members,
                            "Quarters at this volume": ab.quarters_at_current_volume.round(1),
                            "Eligible base for 1 quarter": ab.eligible_base_for_one_quarter.map("{:,}".format)})
    fa = T["fairness_by_group"]
    fa_show = fa[fa.dimension.isin(["age_band", "gender", "tier"])][["dimension", "group", "members", "churners", "recall", "precision",
                                                                     "false_positive_rate"]].round(2)
    tbl = lambda df: dash_table.DataTable(data=df.to_dict("records"), columns=[{"name": c, "id": c} for c in df.columns],
                                          style_as_list_view=True, style_cell={"fontFamily": "Inter, Helvetica Neue, Arial",
                                                                               "fontSize": 12.5, "padding": "5px 8px", "textAlign": "left"},
                                          style_header={"fontWeight": 600, "backgroundColor": "#f3f6f4"})
    gru_note = ""
    if K.get("gru"):
        g = K["gru"]["Active only"]
        gru_note = (f"GRU sequence challenger: Active PR-AUC {g['pr_auc']:.3f} vs {K['test_active']['pr_auc']:.3f} deployed "
                    f"(paired-bootstrap Δ {g.get('delta_vs_champion', float('nan')):+.3f}, 95% CI {g.get('delta_ci_low', float('nan')):+.3f} to "
                    f"{g.get('delta_ci_high', float('nan')):+.3f}). Recommendation: run in shadow mode one quarter, promote if it holds.")
    return html.Div([
        dcc.Graph(figure=fs),
        html.Div([dcc.Graph(figure=fp), dcc.Graph(figure=fd)], className="grid-2"),
        html.Div([
            html.Div([html.H4("A/B test sizing: hold out 20% of flagged members"),
                      html.P("Two-sided test, α = 0.05, power 80%. Baseline = mean churn probability of flagged members.", className="muted"),
                      tbl(ab_show)], className="card"),
            html.Div([html.H4("Fairness check (Active members, deployed threshold)"),
                      html.P("Recall and false-positive rate should be similar across groups; small groups are noisy.", className="muted"),
                      tbl(fa_show)], className="card"),
        ], className="grid-2"),
        html.Div(html.P(gru_note), className="card") if gru_note else None,
    ])


# --------------------------------------------------------------------------- app
app = Dash(__name__, title="Silent Cart · Retention Console", suppress_callback_exceptions=True)
server = app.server   # for gunicorn / Databricks Apps

TABS = ("overview", "members", "sim", "insights")


def serve_layout():
    return html.Div([
    dcc.Location(id="url", refresh=False),
    html.Header([
        html.Div([html.Span("SILENT CART", className="brand"), html.Span("FreshBasket loyalty retention console", className="sub")]),
        html.Div(f"Model: {K['selected_model']} · scored {K['n_test']:,} members · data to Jun-2024", className="sub"),
    ], className="topbar"),
    dcc.Tabs(id="tabs", value="overview", className="tabs", children=[
        dcc.Tab(label="Overview", value="overview"), dcc.Tab(label="Members", value="members"),
        dcc.Tab(label="Scenario simulator", value="sim"), dcc.Tab(label="Insights", value="insights")]),
    html.Main(id="page", className="page"),
    ])


app.layout = serve_layout


@app.callback(Output("tabs", "value"), Input("url", "search"))
def tab_from_url(search):
    """?tab=members|sim|insights opens a tab directly (shareable links)."""
    tab = dict(kv.split("=", 1) for kv in (search or "").lstrip("?").split("&") if "=" in kv).get("tab", "overview")
    return tab if tab in TABS else "overview"


@app.callback(Output("page", "children"), Input("tabs", "value"))
def render(tab):
    return {"overview": overview, "members": members, "sim": simulator, "insights": insights}[tab]()


@app.callback(Output("m-table", "data"), Output("m-table", "columns"), Output("m-summary", "children"),
              Input("m-source", "value"), Input("m-segment", "value"), Input("m-band", "value"), Input("m-action", "value"),
              Input("m-persona", "value"), Input("m-city", "value"), Input("m-tier", "value"))
def filter_members(source, seg, band, action, persona, city, tier):
    df = (ACTIONS if source == "current" else WATCH).copy()
    for col, vals in [("segment", seg), ("risk_band", band), ("recommended_action", action), ("persona", persona),
                      ("city", city), ("tier", tier)]:
        if vals and col in df:
            df = df[df[col].isin(vals)]
    cols = [c for c in TABLE_COLS if c in df.columns]
    show = df[cols].copy()
    show["churn_probability"] = (show.churn_probability * 100).round(1)
    show["revenue_at_risk"] = show.revenue_at_risk.round(0)
    show["expected_net_value"] = show.expected_net_value.round(0)
    show["id"] = show.customer_id
    names = {"churn_probability": "churn %", "revenue_at_risk": "revenue at risk $", "expected_net_value": "action value $",
             "recommended_action": "next best action", "risk_band": "risk", "customer_id": "member"}
    summary = (f"{len(df):,} members · {int((df.risk_band == 'High').sum()):,} high risk · "
               f"{money(df.revenue_at_risk.sum())} 6-month revenue at risk · {money(df.expected_net_value.sum())} expected action value")
    return show.to_dict("records"), [{"name": names.get(c, c), "id": c} for c in cols], summary


@app.callback(Output("m-detail", "children"), Input("m-table", "active_cell"), Input("m-source", "value"))
def show_member(cell, source):
    if not cell or cell.get("row_id") is None:
        first = (ACTIONS if source == "current" else WATCH).customer_id.iloc[0]
        return member_detail(first, source)
    return member_detail(int(cell["row_id"]), source)


@app.callback(Output("draft-box", "children"), Input("draft-btn", "n_clicks"), State("draft-member", "data"),
              prevent_initial_call=True)
def draft_live(n, store):
    if not n or not store:
        return no_update
    df = ACTIONS if store["source"] == "current" else WATCH
    row = df[df.customer_id == store["cid"]].iloc[0]
    return message_card(genai.draft(row))


@app.callback(Output("s-kpis", "children"), Output("s-chart", "figure"), Output("s-table", "children"),
              Input("s-coupon-cost", "value"), Input("s-coupon-up", "value"), Input("s-call-cost", "value"),
              Input("s-call-up", "value"), Input("s-wb-cost", "value"), Input("s-wb-rate", "value"),
              Input("s-margin", "value"), Input("s-horizon", "value"), Input("s-thr", "value"))
def run_sim(cc, cu, oc, ou, wc, wr, m, h, thr):
    econ = Economics(horizon_months=h, gross_margin=m, coupon_cost=cc, coupon_uplift=cu, outreach_cost=oc,
                     outreach_uplift=ou, winback_cost=wc, winback_rate=wr)
    return scenario_view(econ, thr)


if __name__ == "__main__":
    import os
    app.run(debug=False, port=int(os.environ.get("PORT", 8050)))
