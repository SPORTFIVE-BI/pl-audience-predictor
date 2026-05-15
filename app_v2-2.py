"""
PL TV Audience Predictor — SPORTFIVE branded
=============================================
Styled to match SPORTFIVE's visual identity:
- Black / white / red palette only
- Heavy, ALL-CAPS sans-serif headlines
- Sharp geometric edges (no rounded corners)
- Right-side red bar with "V" mark (SPORTFIVE signature)
- Black header with red underline

Run with: streamlit run app_v2.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import io
from predict import predict_audience, predict_basket

# ============================================================
# SPORTFIVE brand tokens
# ============================================================
BRAND = {
    "black": "#000000",
    "white": "#FFFFFF",
    "red": "#E2231A",
    "grey_mid": "#7F7F7F",
    "grey_panel": "#EAEAEA",
    "text_muted": "#666666",
    "font_family": '"Helvetica Neue", "Inter", -apple-system, BlinkMacSystemFont, "Arial Black", sans-serif',
    "company_name": "SPORTFIVE",
    "tool_name": "PREMIER LEAGUE AUDIENCE PREDICTOR",
}

CONF_HIGH_MAX_PCT = 85
CONF_MED_MAX_PCT = 145

TEAMS = sorted([
    "AFC Bournemouth","Arsenal","Aston Villa","Brentford","Brighton & Hove Albion",
    "Burnley","Chelsea","Crystal Palace","Everton","Fulham","Ipswich Town",
    "Leeds United","Leicester City","Liverpool","Luton Town","Manchester City",
    "Manchester United","Newcastle United","Norwich City","Nottingham Forest",
    "Sheffield United","Southampton","Sunderland","Tottenham Hotspur","Watford",
    "West Ham United","Wolverhampton Wanderers",
])

KICK_OFF_SLOTS = {
    "12:30 (Sat early)": "12:30",
    "14:00 (Sun afternoon)": "14:00",
    "15:00 (Sat 3pm)": "15:00",
    "16:30 (Sun marquee)": "16:30",
    "17:30": "17:30",
    "18:30": "18:30",
    "19:30 (midweek)": "19:30",
    "20:00 (Mon/Fri evening)": "20:00",
    "(Slot not yet known)": None,
}

REGION_LABELS = {
    "global_audience": "Global", "uk": "UK", "europe_excl_uk": "Europe (excl. UK)",
    "asia_pacific": "Asia Pacific", "latin_america": "Latin America",
    "mena": "MENA", "north_america": "North America",
    "sub_saharan_africa": "Sub-Saharan Africa",
}

# ============================================================
# PAGE SETUP & STYLING
# ============================================================
st.set_page_config(page_title=f"{BRAND['company_name']} — {BRAND['tool_name']}",
                    layout="wide", initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
.stApp {{
    background: {BRAND['white']};
    font-family: {BRAND['font_family']};
    color: {BRAND['black']};
}}
header[data-testid="stHeader"] {{ display: none; }}
.block-container {{
    padding-top: 0;
    padding-bottom: 80px;
    padding-left: 60px;
    padding-right: 96px;
    max-width: 1400px;
}}

/* Black SPORTFIVE header bar with red underline */
.sf-header {{
    background: {BRAND['black']};
    color: {BRAND['white']};
    padding: 28px 60px;
    margin: 0 -60px 0 -96px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 4px solid {BRAND['red']};
}}
.sf-company {{
    font-size: 32px;
    font-weight: 900;
    letter-spacing: 1px;
    line-height: 1;
}}
.sf-tool-name {{
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
    color: {BRAND['white']};
    text-transform: uppercase;
    opacity: 0.95;
}}

/* SPORTFIVE-signature right-side red bar with V */
.sf-right-bar {{
    position: fixed;
    top: 0;
    right: 0;
    width: 36px;
    height: 100vh;
    background: {BRAND['red']};
    z-index: 1000;
    pointer-events: none;
}}
.sf-right-bar::after {{
    content: "V";
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: {BRAND['white']};
    font-weight: 900;
    font-size: 24px;
}}

/* Hero block */
.sf-hero {{
    padding: 70px 0 40px 0;
}}
.sf-hero-title {{
    font-size: 72px;
    font-weight: 900;
    line-height: 0.95;
    letter-spacing: -2px;
    color: {BRAND['black']};
    text-transform: uppercase;
    margin: 0;
}}
.sf-hero-sub {{
    font-size: 16px;
    color: {BRAND['text_muted']};
    margin-top: 24px;
    max-width: 680px;
    line-height: 1.55;
}}

/* Section heading */
.sf-section-h {{
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: {BRAND['black']};
    margin: 36px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid {BRAND['black']};
}}

/* Tab styling — square, high contrast */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    border-bottom: 2px solid {BRAND['black']};
    margin-top: 20px;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 12px;
    color: {BRAND['black']};
    padding: 14px 32px;
    border-radius: 0;
}}
.stTabs [aria-selected="true"] {{
    background: {BRAND['black']} !important;
    color: {BRAND['white']} !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{ display: none; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* Prediction cards */
.sf-card {{
    background: {BRAND['white']};
    border: 2px solid {BRAND['black']};
    padding: 32px;
    margin: 16px 0 24px 0;
}}
.sf-card-headline {{
    background: {BRAND['black']};
    color: {BRAND['white']};
    padding: 40px;
    margin: 16px 0 24px 0;
}}
.sf-pred-label {{
    font-size: 11px;
    color: {BRAND['text_muted']};
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-weight: 700;
    margin-bottom: 8px;
}}
.sf-card-headline .sf-pred-label {{
    color: rgba(255,255,255,0.7);
}}
.sf-pred-headline {{
    font-size: 76px;
    font-weight: 900;
    color: {BRAND['black']};
    line-height: 0.95;
    letter-spacing: -3px;
    margin: 0;
}}
.sf-card-headline .sf-pred-headline {{
    color: {BRAND['white']};
}}
.sf-pred-range {{
    font-size: 13px;
    color: {BRAND['text_muted']};
    margin-top: 20px;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 600;
}}
.sf-pred-range strong {{
    color: {BRAND['black']};
    font-weight: 800;
}}
.sf-card-headline .sf-pred-range {{
    color: rgba(255,255,255,0.85);
}}
.sf-card-headline .sf-pred-range strong {{
    color: {BRAND['white']};
}}

/* Confidence chip — sharp corners, ALL CAPS */
.sf-conf-chip {{
    display: inline-block;
    padding: 8px 16px;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}}
.sf-conf-high {{
    background: {BRAND['black']};
    color: {BRAND['white']};
}}
.sf-conf-medium {{
    background: {BRAND['grey_mid']};
    color: {BRAND['white']};
}}
.sf-conf-low {{
    background: {BRAND['red']};
    color: {BRAND['white']};
}}
.sf-card-headline .sf-conf-high {{
    background: {BRAND['white']};
    color: {BRAND['black']};
}}

/* Region row */
.sf-region-row {{
    padding: 18px 0;
    border-bottom: 1px solid {BRAND['grey_panel']};
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.sf-region-row:last-child {{ border-bottom: none; }}
.sf-region-name {{
    font-weight: 700;
    color: {BRAND['black']};
    font-size: 15px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.sf-region-value {{
    font-weight: 900;
    color: {BRAND['black']};
    font-size: 26px;
    letter-spacing: -0.8px;
    line-height: 1;
}}
.sf-region-range {{
    font-size: 11px;
    color: {BRAND['text_muted']};
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}}
.sf-region-level {{
    font-size: 10px;
    color: #888;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
}}

/* Buttons — black square primary */
.stButton button[kind="primary"] {{
    background: {BRAND['black']};
    color: {BRAND['white']};
    border: 2px solid {BRAND['black']};
    border-radius: 0;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 12px;
    padding: 14px 32px;
}}
.stButton button[kind="primary"]:hover {{
    background: {BRAND['red']};
    border-color: {BRAND['red']};
    color: {BRAND['white']};
}}
.stButton button[kind="primary"]:focus {{
    background: {BRAND['red']};
    border-color: {BRAND['red']};
    color: {BRAND['white']};
}}
.stButton button:not([kind="primary"]) {{
    border-radius: 0;
    border: 2px solid {BRAND['black']};
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 11px;
}}
.stDownloadButton button {{
    border-radius: 0;
    border: 2px solid {BRAND['black']};
    background: {BRAND['white']};
    color: {BRAND['black']};
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 11px;
}}
.stDownloadButton button:hover {{
    background: {BRAND['black']};
    color: {BRAND['white']};
}}

/* Form inputs — sharp corners */
.stSelectbox > div > div, .stDateInput > div > div, .stTextArea textarea {{
    border-radius: 0 !important;
    border-color: {BRAND['black']} !important;
}}
.stTextArea textarea {{
    border: 2px solid {BRAND['black']} !important;
}}
.stSelectbox label, .stDateInput label, .stSlider label, .stTextArea label,
.stFileUploader label, .stRadio label {{
    font-size: 11px !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    color: {BRAND['black']} !important;
}}

/* Data frame */
.stDataFrame {{
    border: 2px solid {BRAND['black']};
}}

/* Alerts */
.stAlert {{
    border-radius: 0 !important;
    border-left-width: 6px !important;
}}

/* Expander */
.streamlit-expanderHeader, .stExpander summary {{
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    font-size: 12px !important;
    border-radius: 0 !important;
}}

/* Footer */
.sf-footer {{
    margin-top: 80px;
    padding-top: 24px;
    border-top: 4px solid {BRAND['black']};
    font-size: 11px;
    color: {BRAND['text_muted']};
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 600;
}}
.sf-footer-brand {{
    color: {BRAND['red']};
    font-weight: 900;
    letter-spacing: 2px;
}}

/* Misc Streamlit overrides */
.stRadio > div {{ flex-direction: row; gap: 24px; }}
.element-container {{ margin-bottom: 4px; }}
</style>
""", unsafe_allow_html=True)

# Right-side red bar (SPORTFIVE signature)
st.markdown('<div class="sf-right-bar"></div>', unsafe_allow_html=True)

# Header
st.markdown(f"""
<div class="sf-header">
    <div class="sf-company">{BRAND['company_name']}</div>
    <div class="sf-tool-name">{BRAND['tool_name']}</div>
</div>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="sf-hero">
    <div class="sf-hero-title">AUDIENCE<br>PREDICTOR</div>
    <div class="sf-hero-sub">
        Estimate Premier League TV audiences globally and by region.
        Built on five seasons of viewership data.
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def assess_confidence(p10, p50, p90):
    if p50 <= 0:
        return ("Low", 0)
    range_pct = (p90 - p10) / p50 * 100
    if range_pct <= CONF_HIGH_MAX_PCT:
        return ("High", range_pct)
    if range_pct <= CONF_MED_MAX_PCT:
        return ("Medium", range_pct)
    return ("Low", range_pct)

def conf_chip_html(level):
    return f'<span class="sf-conf-chip sf-conf-{level.lower()}">{level} Confidence</span>'

def format_audience(v):
    if v >= 1e6: return f"{v/1e6:.2f}M"
    if v >= 1e3: return f"{v/1e3:.0f}K"
    return f"{v:.0f}"

def render_prediction_card(title, p10, p50, p90, prominent=False):
    level, range_pct = assess_confidence(p10, p50, p90)
    card_cls = "sf-card-headline" if prominent else "sf-card"
    html = (
        f'<div class="{card_cls}">'
        f'<div style="display: flex; align-items: flex-start; justify-content: space-between; gap: 24px;">'
        f'<div style="flex: 1;">'
        f'<div class="sf-pred-label">{title}</div>'
        f'<div class="sf-pred-headline">{format_audience(p50)}</div>'
        f'<div class="sf-pred-range">'
        f'Reasonable Range: <strong>{format_audience(p10)} – {format_audience(p90)}</strong>'
        f'<span style="margin-left: 16px; opacity: 0.7;">±{range_pct/2:.0f}%</span>'
        f'</div>'
        f'</div>'
        f'<div style="text-align: right; padding-top: 8px;">{conf_chip_html(level)}</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def show_basket_results(df_pred, n_fixtures):
    df_g = df_pred[df_pred["region"] == "global_audience"].copy()
    is_total = df_g["home_team"] == "PACKAGE TOTAL"
    totals = df_g[is_total].iloc[0] if is_total.any() else None
    fixtures = df_g[~is_total].copy()

    if totals is not None:
        st.markdown('<div class="sf-section-h">Package Total — Global</div>', unsafe_allow_html=True)
        render_prediction_card(f"Package of {n_fixtures} fixtures",
                                totals["p10"], totals["p50"], totals["p90"],
                                prominent=True)

    st.markdown('<div class="sf-section-h">Per-Fixture Predictions</div>', unsafe_allow_html=True)
    fixtures["Fixture"] = fixtures["home_team"] + " vs " + fixtures["away_team"]
    fixtures["P50"] = fixtures["p50"].apply(format_audience)
    fixtures["Range"] = fixtures.apply(
        lambda r: f"{format_audience(r['p10'])} – {format_audience(r['p90'])}", axis=1)
    fixtures["Confidence"] = fixtures.apply(
        lambda r: assess_confidence(r["p10"], r["p50"], r["p90"])[0], axis=1)
    display = fixtures[["match_date","Fixture","P50","Range","Confidence"]].copy()
    display.columns = ["Date","Fixture","Estimate","Range (P10–P90)","Confidence"]
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.download_button("Download Full Predictions",
                       df_pred.to_csv(index=False).encode(),
                       "predictions.csv", "text/csv")

    if totals is not None:
        with st.expander("Regional Breakdown For Package Total"):
            region_df = df_pred[df_pred["home_team"] == "PACKAGE TOTAL"].copy()
            region_df["Region"] = region_df["region"].map(REGION_LABELS)
            region_df["P50"] = region_df["p50"].apply(format_audience)
            region_df["Range"] = region_df.apply(
                lambda r: f"{format_audience(r['p10'])} – {format_audience(r['p90'])}", axis=1)
            region_df["Confidence"] = region_df.apply(
                lambda r: assess_confidence(r["p10"], r["p50"], r["p90"])[0], axis=1)
            st.dataframe(region_df[["Region","P50","Range","Confidence"]],
                          use_container_width=True, hide_index=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["Single Fixture", "Build Basket", "Bulk Upload"])

# --- TAB 1: Single fixture ---
with tab1:
    st.markdown('<div class="sf-section-h">Predict Audience For A Single Fixture</div>',
                  unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox("Home Team", TEAMS, index=TEAMS.index("Liverpool"))
        season = st.selectbox("Season", ["2025/26","2024/25","2023/24","2022/23","2021/22"])
        slot_label = st.selectbox("Kick-Off Slot (UK)", list(KICK_OFF_SLOTS.keys()), index=3)
    with c2:
        away = st.selectbox("Away Team", TEAMS, index=TEAMS.index("Manchester United"))
        match_date = st.date_input("Match Date", value=pd.Timestamp("2025-12-14"))
        n_concurrent = st.slider("Concurrent PL Matches", 0, 9, 0,
                                    help="How many other PL matches kick off in the same ~30 min window?")

    kick_off = KICK_OFF_SLOTS[slot_label]

    if home == away:
        st.error("Home and away teams must differ.")
    else:
        if st.button("Predict", type="primary"):
            with st.spinner("Running model..."):
                kwargs = {"home_team": home, "away_team": away, "season": season,
                            "match_date": str(match_date)}
                if kick_off:
                    kwargs["kick_off_uk"] = kick_off
                    kwargs["n_concurrent"] = n_concurrent
                result = predict_audience(**kwargs)

            g = result["global_audience"]
            st.markdown('<div class="sf-section-h">Global Audience Estimate</div>',
                          unsafe_allow_html=True)
            render_prediction_card("Global Audience",
                                    g["p10"], g["p50"], g["p90"], prominent=True)

            st.markdown('<div class="sf-section-h">Regional Breakdown</div>',
                          unsafe_allow_html=True)
            rows = ""
            for region in ["uk","europe_excl_uk","asia_pacific","latin_america",
                             "mena","north_america","sub_saharan_africa"]:
                v = result[region]
                level, _ = assess_confidence(v["p10"], v["p50"], v["p90"])
                rows += (
                    f'<div class="sf-region-row">'
                    f'<div>'
                    f'<div class="sf-region-name">{REGION_LABELS[region]}</div>'
                    f'<div class="sf-region-range">{format_audience(v["p10"])} – {format_audience(v["p90"])}</div>'
                    f'</div>'
                    f'<div style="text-align: right;">'
                    f'<div class="sf-region-value">{format_audience(v["p50"])}</div>'
                    f'<div class="sf-region-level">{level}</div>'
                    f'</div>'
                    f'</div>'
                )
            st.markdown(f'<div class="sf-card">{rows}</div>', unsafe_allow_html=True)

# --- TAB 2: Manual basket ---
with tab2:
    st.markdown('<div class="sf-section-h">Build A Basket Manually</div>', unsafe_allow_html=True)
    st.caption("For larger lists, use the Bulk Upload tab.")

    if "basket" not in st.session_state:
        st.session_state.basket = []

    with st.expander("Add A Fixture", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            h = st.selectbox("Home Team", TEAMS, key="b_home")
            s = st.selectbox("Season", ["2025/26","2024/25"], key="b_season")
            slot = st.selectbox("Kick-Off Slot", list(KICK_OFF_SLOTS.keys()), key="b_slot")
        with c2:
            a = st.selectbox("Away Team", TEAMS, key="b_away")
            d = st.date_input("Date", value=pd.Timestamp("2025-12-14"), key="b_date")
            nc = st.slider("Concurrent Matches", 0, 9, 0, key="b_conc")

        if st.button("Add To Basket", type="primary"):
            if h == a:
                st.error("Home and away must differ.")
            else:
                fx = {"home_team": h, "away_team": a, "season": s, "match_date": str(d)}
                if KICK_OFF_SLOTS[slot] is not None:
                    fx["kick_off_uk"] = KICK_OFF_SLOTS[slot]
                    fx["n_concurrent"] = nc
                st.session_state.basket.append(fx)
                st.rerun()

    if st.session_state.basket:
        st.markdown(f'<div class="sf-section-h">Current Basket — {len(st.session_state.basket)} Fixture(s)</div>',
                      unsafe_allow_html=True)
        for i, fx in enumerate(st.session_state.basket):
            cc = st.columns([10, 1])
            cc[0].write(f"**{i+1}.** {fx['home_team']} vs {fx['away_team']}  "
                         f"— {fx.get('match_date','')} {fx.get('kick_off_uk','')}")
            if cc[1].button("✕", key=f"rm_{i}"):
                st.session_state.basket.pop(i)
                st.rerun()

        b1, b2 = st.columns([1, 3])
        with b1:
            if st.button("Clear Basket"):
                st.session_state.basket = []
                st.rerun()
        with b2:
            if st.button("Predict Basket", type="primary"):
                with st.spinner("Running model..."):
                    df = predict_basket(st.session_state.basket)
                show_basket_results(df, len(st.session_state.basket))
    else:
        st.info("Basket is empty. Add fixtures above to begin.")

# --- TAB 3: Bulk upload ---
with tab3:
    st.markdown('<div class="sf-section-h">Bulk Upload Fixtures</div>', unsafe_allow_html=True)
    st.caption("Upload CSV/Excel or paste a list. Get predictions for the whole basket at once.")

    with st.expander("Expected Format"):
        st.markdown("""
Required columns (any order, case-insensitive):

| Column | Required? | Example |
|---|---|---|
| `home_team` | yes | Liverpool |
| `away_team` | yes | Manchester United |
| `match_date` | yes | 2025-12-14 |
| `kick_off_uk` | optional | 16:30 |
| `n_concurrent` | optional | 0 |
| `season` | optional (defaults to 2025/26) | 2025/26 |

**Team names must match dropdown spellings exactly** (e.g. "Tottenham Hotspur" not "Spurs").
Without `kick_off_uk` and `n_concurrent`, you'll get long-horizon predictions (wider ranges).
        """)
        template = pd.DataFrame([
            {"home_team":"Manchester City","away_team":"Arsenal","match_date":"2025-11-08",
              "kick_off_uk":"16:30","n_concurrent":0,"season":"2025/26"},
            {"home_team":"Liverpool","away_team":"Chelsea","match_date":"2025-11-09",
              "kick_off_uk":"17:30","n_concurrent":1,"season":"2025/26"},
            {"home_team":"Brighton & Hove Albion","away_team":"Crystal Palace","match_date":"2025-11-08",
              "kick_off_uk":"15:00","n_concurrent":4,"season":"2025/26"},
        ])
        st.download_button("Download Template CSV",
                            template.to_csv(index=False).encode(),
                            "fixture_template.csv", "text/csv")

    upload_mode = st.radio("Input Method", ["Upload file", "Paste CSV"], horizontal=True)

    fixtures_df = None
    if upload_mode == "Upload file":
        uploaded = st.file_uploader("Upload CSV Or Excel", type=["csv","xlsx","xls"])
        if uploaded:
            try:
                fixtures_df = (pd.read_csv(uploaded) if uploaded.name.endswith(".csv")
                                else pd.read_excel(uploaded))
            except Exception as e:
                st.error(f"Could not read file: {e}")
    else:
        pasted = st.text_area("Paste CSV Here", height=200,
                                placeholder="home_team,away_team,match_date,kick_off_uk,n_concurrent\nLiverpool,Manchester United,2025-12-14,16:30,0")
        if pasted.strip():
            try:
                fixtures_df = pd.read_csv(io.StringIO(pasted))
            except Exception as e:
                st.error(f"Could not parse: {e}")

    if fixtures_df is not None and len(fixtures_df) > 0:
        fixtures_df.columns = [c.lower().strip().replace(" ","_") for c in fixtures_df.columns]
        st.markdown(f'<div class="sf-section-h">Found {len(fixtures_df)} Fixtures</div>',
                      unsafe_allow_html=True)

        required = ["home_team","away_team","match_date"]
        missing = [c for c in required if c not in fixtures_df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            invalid = (set(fixtures_df["home_team"]) | set(fixtures_df["away_team"])) - set(TEAMS)
            if invalid:
                st.warning(f"Unknown team names: {invalid}. Predictions may be less accurate. Check spelling against dropdowns.")

            with st.expander("Preview Uploaded Data"):
                st.dataframe(fixtures_df.head(20), use_container_width=True, hide_index=True)

            if st.button("Predict All Fixtures", type="primary"):
                with st.spinner(f"Running model on {len(fixtures_df)} fixtures..."):
                    fixtures_list = []
                    for _, r in fixtures_df.iterrows():
                        fx = {
                            "home_team": str(r["home_team"]),
                            "away_team": str(r["away_team"]),
                            "match_date": pd.to_datetime(r["match_date"]).strftime("%Y-%m-%d"),
                            "season": str(r.get("season", "2025/26"))
                                       if pd.notna(r.get("season", None)) else "2025/26",
                        }
                        if "kick_off_uk" in r and pd.notna(r["kick_off_uk"]):
                            ko = str(r["kick_off_uk"])
                            if len(ko) >= 5: ko = ko[:5]
                            fx["kick_off_uk"] = ko
                        if "n_concurrent" in r and pd.notna(r["n_concurrent"]):
                            fx["n_concurrent"] = int(r["n_concurrent"])
                        fixtures_list.append(fx)

                    df_pred = predict_basket(fixtures_list)
                show_basket_results(df_pred, len(fixtures_list))

# Footer
st.markdown(f"""
<div class="sf-footer">
    <span class="sf-footer-brand">{BRAND['company_name']}</span>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Powered by audience modelling on five Premier League seasons (2021/22 — 2025/26).
    Estimates are P50 with P10–P90 confidence range.
</div>
""", unsafe_allow_html=True)
