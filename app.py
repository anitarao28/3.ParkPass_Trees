"""Decision Trees Lab: Planning Next Season's Campaign (Streamlit).

Run locally:  streamlit run app.py
Data file:    ParkPassSpend.csv in the same folder.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

st.set_page_config(page_title="Decision Trees Lab", page_icon="🌳", layout="wide")

# ----------------------------------------------------------------------------- style
NAVY, TEAL, ORANGE, SLATE, BODY = "#13233B", "#0E7C7B", "#B45309", "#9AA6B2", "#3A4552"
FONT = 16  # chart and tree text size (projector friendly)
st.markdown(
    """
<style>
  .block-container {padding-top: 2.2rem; max-width: 1500px;}
  h1 {font-size: 2.3rem !important;}
  h2 {font-size: 1.8rem !important;}
  h3 {font-size: 1.4rem !important;}
  button[data-baseweb="tab"] p, .stTabs [role="tab"] p, [data-testid="stTab"] p {font-size: 1.2rem !important; font-weight: 600;}
  [data-testid="stWidgetLabel"] p {font-size: 1.05rem !important; font-weight: 600;}
  [data-testid="stMetricValue"] {font-size: 1.7rem !important;}
  [data-testid="stMetricLabel"] p {font-size: 1.0rem !important;}
  .question {background:#E3F1F0; border-radius:8px; padding:14px 18px; color:#0B5F60; font-size:1.1rem; margin-bottom:0.6rem;}
  .groupq {background:#F3F5F7; border-radius:8px; padding:12px 18px; margin-top:0.6rem;}
</style>
""",
    unsafe_allow_html=True,
)


def plotly_style(fig, height=420, title=None):
    legend = len(fig.data) > 1
    fig.update_layout(showlegend=legend,
        height=height, font=dict(size=FONT, color=BODY),
        title=dict(text=title, font=dict(size=FONT + 2, color=NAVY), y=0.985, yanchor="top", x=0, xanchor="left") if title else None,
        margin=dict(l=10, r=10, t=(95 if legend else 50) if title else (40 if legend else 10), b=10), plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=FONT - 1)),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=FONT))
    fig.update_yaxes(gridcolor="#E6EAEE", tickfont=dict(size=FONT - 1), zerolinecolor="#9AA6B2")
    return fig


# ----------------------------------------------------------------------------- data
@st.cache_data
def load_base():
    df = pd.read_csv("ParkPassSpend.csv")
    df["Bought"] = (df.Pass == "YesPass").astype(int)
    for ch in ["Park", "Mail", "Email"]:
        df[ch] = (df.Channel == ch).astype(int)
    df["Bundle"] = (df.Promo == "Bundle").astype(int)
    return df


if "age_seed" not in st.session_state:
    st.session_state.age_seed = 2026  # 2026 reproduces the Age column in the file


@st.cache_data
def load(age_seed):
    df = load_base().copy()
    df["Age"] = np.random.default_rng(age_seed).integers(18, 76, len(df))
    train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df.Bought)
    return df, train, test


df, train, test = load(st.session_state.age_seed)
X_CP = ["Park", "Mail", "Email", "Bundle"]
SEG = pd.DataFrame([(c, p) for c in ["Mail", "Email", "Park"] for p in ["NoBundle", "Bundle"]], columns=["Channel", "Promo"])
for ch in ["Park", "Mail", "Email"]:
    SEG[ch] = (SEG.Channel == ch).astype(int)
SEG["Bundle"] = (SEG.Promo == "Bundle").astype(int)
SEG["Segment"] = SEG.Channel + "<br>" + SEG.Promo.map({"NoBundle": "No bundle", "Bundle": "Bundle"})
ACTUAL = df.groupby(["Channel", "Promo"]).Bought.mean()

QUESTION = {"Park": "Came through the Park?", "Mail": "Came through Mail?", "Email": "Came through Email?",
            "Bundle": "Offered the bundle?", "Bought": "Bought a pass?"}


# ----------------------------------------------------------------------------- tree diagrams (drawn with Plotly for exact font control)
def _mix(hex_a, hex_b, t):
    t = min(max(t, 0), 1)
    a = [int(hex_a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(hex_b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


def tree_figure(model, features, kind="clf", max_depth=None):
    t = model.tree_
    vals = t.value[:, 0, 0] if kind == "reg" else None
    nodes, edges = {}, []          # nodes: id -> dict(depth, y, text, fill, bold)
    order = [0]                    # running leaf counter for vertical placement

    def info(i):
        n = int(t.n_node_samples[i])
        if kind == "clf":
            v = t.value[i][0]
            p = v[1] / v.sum()
            fill = _mix("#FFFFFF", "#8CCBC9", (p - 0.5) * 2) if p >= 0.5 else _mix("#FFFFFF", "#F4BE92", (0.5 - p) * 2)
            stats = f"{n:,} customers<br>{p:.0%} bought"
            verdict = "BUYS" if p > 0.5 else "DOESN'T BUY"
        else:
            m = t.value[i][0][0]
            fill = _mix("#FFFFFF", "#8CCBC9", (m - vals.min()) / max(vals.max() - vals.min(), 1e-9))
            stats = f"{n:,} customers<br>avg ${m:,.2f} per visit"
            verdict = f"${m:,.2f}"
        return stats, fill, verdict

    def n_cust(i):
        return f"{int(t.n_node_samples[i]):,} customers"

    def n_leaves(i):
        return 1 if t.children_left[i] == -1 else n_leaves(t.children_left[i]) + n_leaves(t.children_right[i])

    def walk(i, depth):
        stats, fill, verdict = info(i)
        leaf = t.children_left[i] == -1
        cut = (not leaf) and max_depth is not None and depth >= max_depth
        if leaf or cut:
            y = order[0]; order[0] += 1
            if cut:
                text = f"<b>… {n_leaves(i)} more segments</b><br>{stats}"
            else:
                text = f"<b>{verdict}</b><br>{stats}" if kind == "clf" else f"<b>{verdict} per visit</b><br>{n_cust(i)}"
            nodes[i] = dict(depth=depth, y=y, text=text, fill=fill, leaf=True)
            return y
        f = features[t.feature[i]]
        q = QUESTION.get(f, f"{f} ≤ {t.threshold[i]:.1f}?")
        left, right = t.children_left[i], t.children_right[i]
        yes, no = (left, right) if f not in QUESTION else (right, left)   # 0/1 columns: "<= 0.5" means No
        y_yes = walk(yes, depth + 1)
        y_no = walk(no, depth + 1)
        edges.append((i, yes, "Yes")); edges.append((i, no, "No"))
        y = (y_yes + y_no) / 2
        nodes[i] = dict(depth=depth, y=y, text=f"<b>{q}</b><br>{stats}", fill=fill, leaf=False)
        return y

    walk(0, 0)
    D = max(v["depth"] for v in nodes.values())
    L = order[0]
    bw, bh = 0.78, 0.72          # box size in axis units (x: one column per level, y: one row per leaf)
    fs = FONT if D <= 3 else FONT - 2
    fig = go.Figure()
    for a, b, lab in edges:
        x0, y0 = nodes[a]["depth"] + bw / 2, nodes[a]["y"]
        x1, y1 = nodes[b]["depth"] - bw / 2, nodes[b]["y"]
        fig.add_shape(type="path", path=f"M {x0},{y0} L {x0 + 0.06},{y0} L {x1 - 0.06},{y1} L {x1},{y1}", line=dict(color="#7A8696", width=1.6))
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=f"<b>{lab}</b>", showarrow=False,
                           font=dict(size=fs - 1, color=TEAL if lab == "Yes" else BODY), bgcolor="white")
    for v in nodes.values():
        fig.add_shape(type="rect", x0=v["depth"] - bw / 2, x1=v["depth"] + bw / 2, y0=v["y"] - bh / 2, y1=v["y"] + bh / 2,
                      line=dict(color=NAVY if v["leaf"] else "#9AA6B2", width=2 if v["leaf"] else 1.2), fillcolor=v["fill"], layer="below")
        fig.add_annotation(x=v["depth"], y=v["y"], text=v["text"], showarrow=False, font=dict(size=fs, color=NAVY), align="center")
    fig.update_xaxes(visible=False, range=[-0.45, D + 0.45])
    fig.update_yaxes(visible=False, range=[L - 0.45, -0.55])
    fig.update_layout(height=max(260, 105 * L), margin=dict(l=5, r=5, t=5, b=5), plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
    return fig


def show_tree(model, features, kind="clf", max_draw_leaves=12):
    big = model.get_n_leaves() > max_draw_leaves
    if big:
        st.caption(f"This tree has {model.get_n_leaves()} segments, too many to draw. Showing its first 3 levels of questions.")
    fig = tree_figure(model, features, kind, max_depth=3 if big else None)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "staticPlot": True})


def stat(label, value, sub=None, color=NAVY):
    sub_html = f"<div style='font-size:0.9rem;color:#5B6878'>{sub}</div>" if sub else ""
    st.markdown(f"<div style='margin:4px 0 14px 0'><div style='font-size:1.0rem;font-weight:600;color:{BODY}'>{label}</div>"
                f"<div style='font-size:1.9rem;font-weight:600;color:{color};line-height:1.25'>{value}</div>{sub_html}</div>", unsafe_allow_html=True)


def group_questions(qs):
    items = "".join(f"<li style='margin-bottom:6px'>{q}</li>" for q in qs)
    st.markdown(f"<div class='groupq'><b>Discuss in your group, then record your answers in your team spreadsheet</b><ol style='margin-top:8px'>{items}</ol></div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- sidebar: business assumptions
with st.sidebar:
    st.header("Business assumptions")
    st.caption("Used in Stops 1, 2 and 4. Change them and watch the decisions.")
    PASS_MARGIN = st.slider("Profit per season pass ($)", 20, 200, 100, 10)
    BUNDLE_COST = st.slider("Bundle cost per bundled pass sold ($)", 0, 80, 20, 5)
    OFFER_COST = st.slider("Cost of sending one offer ($)", 0, 40, 5, 1)
    OFFERS = st.select_slider("Offers per channel", options=[1_000, 5_000, 10_000, 20_000, 50_000], value=10_000)
    st.divider()
    st.caption("Stop 4 only")
    VISITS = st.slider("Visits per season (pass holders)", 1, 15, 6)
    SHARE = st.slider("Share of in-park spend the park keeps", 0.0, 0.6, 0.30, 0.05, format="%.2f")

# ----------------------------------------------------------------------------- header + tabs
st.title("🌳 Decision Trees Lab: Planning Next Season's Campaign")
tab0, tab1, tab2, tab3, tab4 = st.tabs(["Start here", "Stop 1 · Where should the bundle go?", "Stop 2 · Who is worth an offer?",
                                        "Stop 3 · Should we pay for age data?", "Stop 4 · Most valuable customers?"])

# ============================================================================= START
with tab0:
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown(f"""
### You are the park's marketing analytics team
Next season's season-pass campaign is being planned, and the VP of Marketing has four questions. Each tab is one **stop**:

1. **Where should the bundle go?**
2. **Who is worth an offer at all?**
3. **Should we pay for age data?**
4. **Who are our most valuable customers?**

At each stop: read the business question, move the controls, look at what changes, and answer the group questions in your team spreadsheet. Then pause for class discussion.

The **business assumptions** in the left sidebar (profit per pass, bundle cost, cost per offer…) apply across stops.
""")
    with c2:
        st.markdown("### The data")
        st.markdown(f"**{len(df):,} customers**, one row each. **{df.Bought.mean():.1%}** bought a season pass.")
        st.markdown("- **Channel**: how the offer reached them (Mail, Park, Email)\n- **Promo**: offer with or without the bundle\n- **Pass**: bought a season pass?\n- **Age**: customer's age\n- **Spend**: average dollars spent per visit")
        st.dataframe(df[["Channel", "Promo", "Pass", "Age", "Spend"]].sample(8, random_state=3), hide_index=True, width="stretch")

# ============================================================================= STOP 1
with tab1:
    st.markdown("<div class='question'><b>The VP asks:</b> the bundle costs money. In which channels should next season's offer include it? "
                "A busy analyst fits the obvious logistic regression (Promo + Channel). Does a decision tree give the same advice?</div>", unsafe_allow_html=True)
    left, right = st.columns([1.25, 2.4], gap="large")
    with left:
        interaction = st.toggle("Add interaction term", value=False, help="Promo × Channel: lets the logit give the bundle a different effect in each channel.")
        leaves1 = st.slider("Most segments (leaves) the tree may create", 2, 8, 6, key="leaves1")
        formula = "Bought ~ Promo * Channel" if interaction else "Bought ~ Promo + Channel"
        st.code(formula, language=None)
    logit = smf.logit(formula, data=df).fit(disp=0)
    tree1 = DecisionTreeClassifier(max_leaf_nodes=leaves1, random_state=0).fit(df[X_CP], df.Bought)
    seg1 = SEG.copy()
    seg1["Actual"] = [ACTUAL[(c, p)] for c, p in zip(seg1.Channel, seg1.Promo)]
    seg1["Logit"] = logit.predict(seg1).values
    seg1["Tree"] = tree1.predict_proba(seg1[X_CP])[:, 1]

    rows = []
    for ch in ["Mail", "Email", "Park"]:
        s = seg1[seg1.Channel == ch].set_index("Promo")
        lp, tp = s.Logit.idxmax(), s.Tree.idxmax()
        rows.append({"Channel": ch, "Logit sends": lp, "Logit passes": round(s.loc[lp, "Actual"] * OFFERS),
                     "Tree sends": tp, "Tree passes": round(s.loc[tp, "Actual"] * OFFERS)})
    plans = pd.DataFrame(rows)
    with left:
        st.markdown("#### Whose advice sells more?")
        st.caption(f"Each model picks the version it predicts sells better in each channel. Plans are scored with what customers actually did, {OFFERS:,} offers per channel.")
        m1, m2 = st.columns(2)
        with m1:
            stat("Logit plan", f"{plans['Logit passes'].sum():,}", "passes sold", color=ORANGE)
        with m2:
            stat("Tree plan", f"{plans['Tree passes'].sum():,}", f"passes sold ({plans['Tree passes'].sum() - plans['Logit passes'].sum():+,})", color=TEAL)
    with right:
        fig = go.Figure()
        for name, color in [("Actual", SLATE), ("Logit", ORANGE), ("Tree", TEAL)]:
            fig.add_bar(name=name if name != "Actual" else "What customers actually did", x=seg1.Segment, y=seg1[name], marker_color=color,
                        text=[f"{v:.0%}" for v in seg1[name]], textposition="outside", textfont=dict(size=FONT - 2))
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
        st.plotly_chart(plotly_style(fig, 440, "Share who buy, by segment: what each model predicts"), width="stretch")
        st.markdown("#### The two plans, channel by channel")
        st.dataframe(plans, hide_index=True, width="stretch")
    st.markdown("#### The tree")
    show_tree(tree1, X_CP)
    group_questions([
        "What does each model tell you to do in the <b>Email</b> channel? Which one matches what Email customers actually did?",
        "How many more passes does the tree's plan sell, and where does the difference come from?",
        "Now switch on the interaction. What changes? What did the analyst need to know in advance, and how often would you know that before looking at data?",
    ])

# ============================================================================= STOP 2
with tab2:
    st.markdown("<div class='question'><b>The VP asks:</b> every offer costs money and every bundle costs money. Which segments are worth an offer, "
                "and which version should each get? Someone objects: <i>\"the tree is only about 72% accurate, why trust it?\"</i></div>", unsafe_allow_html=True)
    tree_all = DecisionTreeClassifier(max_leaf_nodes=6, random_state=0).fit(df[X_CP], df.Bought)
    tree70 = DecisionTreeClassifier(max_leaf_nodes=6, random_state=0).fit(train[X_CP], train.Bought)
    seg2 = SEG.copy()
    seg2["P"] = tree_all.predict_proba(seg2[X_CP])[:, 1]
    seg2["Profit per buyer"] = PASS_MARGIN - BUNDLE_COST * seg2.Bundle
    seg2["Profit per offer"] = seg2.P * seg2["Profit per buyer"] - OFFER_COST

    left, right = st.columns([1.25, 2.4], gap="large")
    with left:
        cutoff = st.slider("Call a customer a \"buyer\" if P(buy) is above…", 0.05, 0.95, 0.50, 0.01, format="%.2f")
        seg2["Label"] = np.where(seg2.P > cutoff, "Labeled \"buys\"", "Labeled \"doesn't buy\"")

        def campaign_profit(cut, use_labels):
            total = 0
            for ch in ["Mail", "Email", "Park"]:
                s = seg2[seg2.Channel == ch]
                if use_labels:
                    best = s.loc[s.P.idxmax()]
                    if best.P > cut:
                        total += best["Profit per offer"] * OFFERS
                else:
                    best = s.loc[s["Profit per offer"].idxmax()]
                    if best["Profit per offer"] > 0:
                        total += best["Profit per offer"] * OFFERS
            return total

        p_test = tree70.predict_proba(test[X_CP])[:, 1]
        acc = ((p_test > cutoff).astype(int) == test.Bought).mean()
        stat("Accuracy on hidden customers", f"{acc:.1%}", f"Guessing \"buys\" for everyone: {test.Bought.mean():.1%}")
        stat("Campaign profit if you contact the segments labeled \"buys\"", f"${campaign_profit(cutoff, True):,.0f}", color=TEAL)
        stat("Campaign profit if you contact every segment that pays", f"${campaign_profit(cutoff, False):,.0f}", color=TEAL)
        st.caption(f"Each channel gets {OFFERS:,} offers of one version (the more profitable one), or none. The tree was grown on 70% of customers and checked on the other 30% for accuracy.")
    with right:
        s = seg2.sort_values("Profit per offer", ascending=False)
        fig = go.Figure()
        for lab, color in [("Labeled \"buys\"", TEAL), ("Labeled \"doesn't buy\"", ORANGE)]:
            d = s[s.Label == lab]
            fig.add_bar(name=lab, x=d.Segment, y=d["Profit per offer"], marker_color=color,
                        text=[f"${v:,.2f}<br>P={p:.0%}" for v, p in zip(d["Profit per offer"], d.P)], textposition="outside", textfont=dict(size=FONT - 2))
        fig.update_xaxes(categoryorder="array", categoryarray=list(s.Segment))
        fig.update_yaxes(tickprefix="$")
        fig.add_hline(y=0, line_color=NAVY, line_width=1.5)
        st.plotly_chart(plotly_style(fig, 420, "Expected profit per offer = P(buy) × profit per buyer − cost of the offer"), width="stretch")

        cuts = np.round(np.arange(0.05, 0.951, 0.01), 2)
        accs = [((p_test > c).astype(int) == test.Bought).mean() for c in cuts]
        profs = [campaign_profit(c, True) / 1000 for c in cuts]
        best = campaign_profit(cutoff, False) / 1000
        fig2 = go.Figure()
        fig2.add_scatter(x=cuts, y=accs, name="Accuracy (hidden customers)", line=dict(color=NAVY, width=3), yaxis="y1")
        fig2.add_scatter(x=cuts, y=profs, name="Profit: contact segments labeled \"buys\"", line=dict(color=TEAL, width=3, shape="hv"), yaxis="y2")
        fig2.add_scatter(x=[cuts[0], cuts[-1]], y=[best, best], name="Profit: contact every segment that pays", line=dict(color=TEAL, width=2, dash="dot"), yaxis="y2")
        fig2.add_vline(x=cutoff, line_dash="dash", line_color=ORANGE)
        fig2.update_layout(yaxis=dict(tickformat=".0%", title="Accuracy"), yaxis2=dict(overlaying="y", side="right", tickprefix="$", ticksuffix="K", tickformat=",.0f", title="Campaign profit", showgrid=False, rangemode="tozero"),
                           xaxis=dict(title="Cutoff for calling someone a buyer", range=[0.03, 0.97]))
        st.plotly_chart(plotly_style(fig2, 430, "Moving the cutoff: accuracy vs. profit"), width="stretch")
    group_questions([
        "The tree labels two segments \"doesn't buy.\" Is either still worth an offer? Why does the yes/no label give the wrong business answer?",
        "Raise the <b>cost of sending one offer</b> to $10, then try a $60 bundle cost. Which segment stops paying? Does the Email decision change?",
        "Move the cutoff. Where is accuracy highest? Where is profit highest? Answer the objection: \"it's only 72% accurate.\"",
    ])

# ============================================================================= STOP 3
with tab3:
    st.markdown("<div class='question'><b>The VP asks:</b> a data vendor offers to add every customer's age to our file, for a fee. "
                "Their analyst says a tree using age is <b>more accurate</b> than ours. Should we buy it? Check their claim yourself.</div>", unsafe_allow_html=True)
    SIZES = [2, 4, 6, 8, 10, 15, 20, 30, 50, 100, 200, "No limit"]
    left, right = st.columns([1.25, 2.4], gap="large")
    with left:
        use_age = st.toggle("Use the vendor's age data", value=True)
        size = st.select_slider("Most segments (leaves) the tree may create", options=SIZES, value=6, key="size3")
    feats = X_CP + (["Age"] if use_age else [])
    mln = None if size == "No limit" else size

    @st.cache_data
    def curve(seed, with_age):
        _, tr, te = load(seed)
        f = X_CP + (["Age"] if with_age else [])
        out = []
        for k in SIZES:
            m = DecisionTreeClassifier(max_leaf_nodes=None if k == "No limit" else k, random_state=0).fit(tr[f], tr.Bought)
            out.append((str(k), m.score(tr[f], tr.Bought), m.score(te[f], te.Bought), m.get_n_leaves()))
        return pd.DataFrame(out, columns=["size", "train", "test", "leaves"])

    big = DecisionTreeClassifier(max_leaf_nodes=mln, random_state=0).fit(train[feats], train.Bought)
    tr_acc, te_acc = big.score(train[feats], train.Bought), big.score(test[feats], test.Bought)
    t = big.tree_
    splits = [i for i in range(t.node_count) if t.children_left[i] != -1]
    age_q = sum(feats[t.feature[i]] == "Age" for i in splits)
    with left:
        st.markdown("**Accuracy**")
        m1, m2 = st.columns(2)
        with m1:
            stat("Customers it learned from", f"{tr_acc:.1%}")
        with m2:
            stat("Hidden customers", f"{te_acc:.1%}", color=ORANGE)
        stat("Segments in this tree", f"{big.get_n_leaves()}")
        st.markdown(f"**{age_q}** of its **{len(splits)}** questions are about Age." if use_age else "Age is not available to this tree.")
        imp = pd.Series(big.feature_importances_, index=feats)
        imp_g = pd.Series({"Channel": imp[["Park", "Mail", "Email"]].sum(), "Promo": imp["Bundle"], "Age": imp.get("Age", 0.0)})
        fig = go.Figure(go.Bar(x=imp_g.values, y=imp_g.index, orientation="h", marker_color=[TEAL, TEAL, ORANGE],
                               text=[f"{v:.0%}" for v in imp_g.values], textposition="outside", textfont=dict(size=FONT)))
        fig.update_xaxes(range=[0, 1.15], tickformat=".0%")
        st.plotly_chart(plotly_style(fig, 230, "Feature importance"), width="stretch")
    with right:
        cv = curve(st.session_state.age_seed, use_age)
        fig = go.Figure()
        fig.add_scatter(x=cv["size"], y=cv.train, name="Customers it learned from", mode="lines+markers", line=dict(color=NAVY, width=3), marker=dict(size=9))
        fig.add_scatter(x=cv["size"], y=cv.test, name="Hidden customers", mode="lines+markers", line=dict(color=ORANGE, width=3), marker=dict(size=9))
        fig.add_vline(x=SIZES.index(size), line_dash="dash", line_color=TEAL)
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(type="category", title="Most segments allowed")
        st.plotly_chart(plotly_style(fig, 380, "Accuracy as the tree is allowed to grow" + (" (with age)" if use_age else " (without age)")), width="stretch")
    st.markdown("#### The tree")
    show_tree(big, feats)

    st.markdown("#### Zoom in: the \"buyer ages\" in the Mail + Bundle segment")
    if use_age:
        mb_tr = train[(train.Channel == "Mail") & (train.Promo == "Bundle")].copy()
        mb_te = test[(test.Channel == "Mail") & (test.Promo == "Bundle")].copy()
        mb_tr["call"] = big.predict(mb_tr[feats])
        by_age = mb_tr.groupby("Age").agg(rate=("Bought", "mean"), call=("call", "max")).reindex(range(18, 76))
        buyer_ages = [int(a) for a in by_age.index[by_age.call == 1]]
        a, b = mb_tr[mb_tr.Age.isin(buyer_ages)], mb_te[mb_te.Age.isin(buyer_ages)]
        c1, c2 = st.columns([2.4, 1], gap="large")
        with c1:
            fig = go.Figure(go.Bar(x=by_age.index, y=by_age.rate.fillna(0), marker_color=[TEAL if c == 1 else "#C5CDD5" for c in by_age.call.fillna(0)]))
            fig.add_hline(y=mb_tr.Bought.mean(), line_dash="dot", line_color=NAVY, annotation_text=f"segment average {mb_tr.Bought.mean():.0%}",
                          annotation_font_size=FONT - 1)
            fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
            fig.update_xaxes(title="Age", dtick=5)
            st.plotly_chart(plotly_style(fig, 320, "Customers it learned from: % who bought at each age (teal = tree calls these ages \"buyers\")"), width="stretch")
        with c2:
            if buyer_ages:
                st.markdown(f"Ages the tree calls **buyers**: {', '.join(map(str, buyer_ages))}")
                stat("Customers it learned from, at those ages", f"{a.Bought.mean():.0%} bought", f"{a.Bought.sum()} of {len(a)} customers")
                stat("Hidden customers, at those ages", f"{b.Bought.mean():.0%} bought" if len(b) else "none", f"{b.Bought.sum()} of {len(b)} customers", color=ORANGE)
            else:
                st.info("At this size the tree doesn't single out any ages in this segment. Let it grow bigger.")
    else:
        st.info("Switch on the vendor's age data to see this.")
    group_questions([
        "Was the vendor's accuracy claim true? Why is it misleading? (Try \"No limit\" with age on.)",
        "Based on the hidden customers, would you pay for the age data? What would you want to see before paying for any new data?",
        "A colleague sees the importance chart and proposes an age-targeted campaign. What do you tell them?",
    ])
    with st.expander("🔒 Reveal: open only when your instructor says so"):
        st.markdown("The **Age** column is **random**: each customer's age was drawn by chance and has no connection to buying. "
                    "Everything the big tree \"learned\" about age was luck among the customers it learned from. Training accuracy couldn't tell you that; only the hidden customers could.")
        st.markdown("Don't believe it? Draw a brand-new set of random ages and watch the \"buyer ages\" change completely.")
        b1, b2 = st.columns(2)
        if b1.button("🎲 Draw new random ages"):
            st.session_state.age_seed += 1
            st.rerun()
        if b2.button("↺ Back to the original ages"):
            st.session_state.age_seed = 2026
            st.rerun()

# ============================================================================= STOP 4
with tab4:
    st.markdown("<div class='question'><b>The VP asks:</b> pass holders keep coming back and spending money in the park. Does the bundle make visitors spend more? "
                "Once spending is counted, where is an offer worth the most? Spend is a dollar amount, so this needs a <b>regression tree</b>: "
                "it splits customers into groups whose spending is as similar as possible and predicts each group's average.</div>", unsafe_allow_html=True)
    left, right = st.columns([1.25, 2.4], gap="large")
    with left:
        leaves4 = st.slider("Most segments (leaves) the spend tree may create", 2, 12, 6, key="leaves4")
        st.caption("The tree may use Channel, Promo, whether they bought a pass, and Age.")
    XS = ["Park", "Mail", "Email", "Bundle", "Bought", "Age"]
    stree = DecisionTreeRegressor(max_leaf_nodes=leaves4, random_state=0).fit(df[XS], df.Spend)
    imp = pd.Series(stree.feature_importances_, index=XS)
    imp_g = pd.Series({"Channel": imp[["Park", "Mail", "Email"]].sum(), "Bought a pass": imp["Bought"], "Promo (bundle)": imp["Bundle"], "Age": imp["Age"]})
    with right:
        fig = go.Figure(go.Bar(x=imp_g.values, y=imp_g.index, orientation="h", marker_color=[TEAL, TEAL, ORANGE, ORANGE],
                               text=[f"{v:.0%}" for v in imp_g.values], textposition="outside", textfont=dict(size=FONT)))
        fig.update_xaxes(range=[0, 1.12], tickformat=".0%")
        st.plotly_chart(plotly_style(fig, 260, "What drives spending? Feature importance in the spend tree"), width="stretch")
    st.markdown("#### The spend tree")
    show_tree(stree, XS, kind="reg")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        g = df.groupby(["Channel", "Pass", "Promo"]).Spend.mean().unstack()
        lab = [f"{c}<br>{'pass holders' if p == 'YesPass' else 'no pass'}" for c, p in g.index]
        fig = go.Figure()
        fig.add_bar(name="No bundle", x=lab, y=g["NoBundle"], marker_color=SLATE, text=[f"${v:.0f}" for v in g["NoBundle"]], textposition="outside")
        fig.add_bar(name="Bundle", x=lab, y=g["Bundle"], marker_color=TEAL, text=[f"${v:.0f}" for v in g["Bundle"]], textposition="outside")
        fig.update_yaxes(tickprefix="$", range=[0, g.values.max() * 1.18])
        st.plotly_chart(plotly_style(fig, 400, "Does the bundle change spending? Average spend per visit"), width="stretch")
    with c2:
        v = SEG.copy()
        v["P"] = tree_all.predict_proba(v[X_CP])[:, 1]
        ph = v[X_CP].assign(Bought=1, Age=40)[XS]
        v["Spend"] = stree.predict(ph)
        v["Value of a pass holder"] = PASS_MARGIN + VISITS * v.Spend * SHARE
        v["With spending"] = v.P * (v["Value of a pass holder"] - BUNDLE_COST * v.Bundle) - OFFER_COST
        v["Pass profit only"] = v.P * (PASS_MARGIN - BUNDLE_COST * v.Bundle) - OFFER_COST
        v = v.sort_values("With spending", ascending=False)
        fig = go.Figure()
        fig.add_bar(name="Pass profit only (Stop 2)", x=v.Segment, y=v["Pass profit only"], marker_color=SLATE, text=[f"${x:.0f}" for x in v["Pass profit only"]], textposition="outside")
        fig.add_bar(name="Counting in-park spending", x=v.Segment, y=v["With spending"], marker_color=TEAL, text=[f"${x:.0f}" for x in v["With spending"]], textposition="outside")
        fig.update_yaxes(tickprefix="$")
        fig.add_hline(y=0, line_color=NAVY, line_width=1.5)
        st.plotly_chart(plotly_style(fig, 400, "Expected value per offer"), width="stretch")
        vals = {c: v[v.Channel == c]['Value of a pass holder'].iloc[0] for c in ['Email', 'Park', 'Mail']}
        st.caption(f"Value of a new pass holder = pass profit + {VISITS} visits × their average spend × {SHARE:.0%} kept by the park "
                   f"(Email ≈ \\${vals['Email']:,.0f}, Park ≈ \\${vals['Park']:,.0f}, Mail ≈ \\${vals['Mail']:,.0f}).")
    group_questions([
        "Does the bundle make visitors spend more per visit? Give two pieces of evidence from this tab.",
        "Whose pass holders are worth the most? Once spending is counted, does the bundle decision change in any channel? Which segment is the best use of an offer?",
        "Write a three-sentence recommendation to the VP: who gets the bundle, where to focus offers, and one caveat.",
    ])
    with st.expander("🔒 How the Spend column was created (instructor)"):
        st.markdown("Spend is **simulated**: true average = \\$40, +\\$25 for pass holders, +\\$20 for Email customers or −\\$10 for Mail customers, plus random noise "
                    "(typically ±\\$18). Promo and Age have no effect. Compare with what the tree found.")
