# app.py
# ---------------------------------------------
# Project Samarth (Classic) — Streamlit App
# Only uses Crop + Rainfall datasets (no temp/soil).
# Attractive UI • Robust loaders • DuckDB analytics • Plotly charts
# ---------------------------------------------

import os
import io
import duckdb
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ---------------------------
# Page setup & theme
# ---------------------------
st.set_page_config(
    page_title="Project Samarth — Classic (Crop + Rainfall)",
    page_icon="🌾",
    layout="wide"
)

# Minimal custom styling
st.markdown("""
<style>
/* Layout polish */
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
/* Headings */
h1, h2, h3 { font-weight: 700; letter-spacing: 0.2px; }
/* Cards */
.card {
  border-radius: 16px; padding: 16px 18px; background: #ffffff; 
  border: 1px solid rgba(0,0,0,0.06); box-shadow: 0 4px 18px rgba(0,0,0,0.06);
  margin-bottom: 12px;
}
/* Pills */
.pill {display: inline-block; padding: 3px 10px; border-radius: 999px; 
  background: #f0f2f6; margin-right: 6px; font-size: 12px;}
/* Metric */
.metric {font-size: 28px; font-weight: 700; margin-top: 6px;}
/* Footer note */
.footer {color:#667085; font-size: 12px; margin-top: 6px;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Helpers
# ---------------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip().str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
        .str.replace(".", "", regex=False)
    )
    return df

def load_any(file) -> pd.DataFrame:
    """Read CSV or Parquet from upload; normalize columns."""
    name = file.name.lower()
    data = file.read()
    bio = io.BytesIO(data)
    try:
        if name.endswith(".parquet"):
            df = pd.read_parquet(bio)
        else:
            # default to CSV
            df = pd.read_csv(io.BytesIO(data))
    except Exception:
        # robust CSV fallback
        bio.seek(0)
        try:
            df = pd.read_csv(io.BytesIO(data), engine="python", sep=None)
        except Exception as e:
            st.error(f"Could not read file {file.name}: {e}")
            return pd.DataFrame()
    return normalize_cols(df)

def gentle_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def titlecase(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.title()

# ---------------------------
# Sidebar — Data inputs
# ---------------------------
st.sidebar.title("📦 Data Inputs (Classic)")
st.sidebar.caption("Upload **Crop** and **Rainfall** datasets only.\nCSV or Parquet supported.")

crop_file = st.sidebar.file_uploader("Crop dataset (CSV/Parquet)", type=["csv","parquet"])
rain_file = st.sidebar.file_uploader("Rainfall dataset (CSV/Parquet)", type=["csv","parquet"])

with st.sidebar.expander("ℹ️ Expected Schemas", expanded=False):
    st.write("**Crop (long format)**")
    st.code("state, district, crop, production_mt, year", language="text")
    st.write("**Rainfall**")
    st.code("state|subdivision, year, jan..dec, annual, jf, mam, jjas, ond", language="text")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: If your rainfall file uses `subdivision`,\nthis app will auto-rename it to `state`.")

st.title("🌾 Project Samarth — Classic (Crop + Rainfall)")
st.write("A fast, elegant interface over the classic two-dataset model. No temperature or soil features are included here.")

# Guard: require files
if not crop_file or not rain_file:
    st.info("⬅️ Please upload both **Crop** and **Rainfall** files in the sidebar to begin.")
    st.stop()

# ---------------------------
# Load & Clean
# ---------------------------
df_crop = load_any(crop_file)
df_rain = load_any(rain_file)

if df_crop.empty or df_rain.empty:
    st.error("One or both files could not be parsed. Please verify the uploads.")
    st.stop()

# Required columns for crop
needed_crop = {"state","district","crop","production_mt","year"}
if not needed_crop.issubset(df_crop.columns):
    st.error(f"Crop file missing columns: {sorted(list(needed_crop - set(df_crop.columns)))}")
    st.stop()

# Handle rainfall: allow 'subdivision' or 'state'
if "state" not in df_rain.columns and "subdivision" in df_rain.columns:
    df_rain = df_rain.rename(columns={"subdivision":"state"})

if "state" not in df_rain.columns:
    st.error("Rainfall file must have either 'state' or 'subdivision' column.")
    st.stop()

# Normalize key columns
df_crop["state"] = titlecase(df_crop["state"])
df_crop["district"] = titlecase(df_crop["district"])
df_crop["crop"] = df_crop["crop"].astype(str).str.strip().str.title()
df_crop = gentle_numeric(df_crop, ["production_mt","year"])

df_rain["state"] = titlecase(df_rain["state"])
df_rain = gentle_numeric(df_rain, ["year","annual","jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec","jf","mam","jjas","ond"])

# Drop empties
df_crop = df_crop.dropna(subset=["state","crop","production_mt","year"])
df_rain = df_rain.dropna(subset=["state","year"])

# ---------------------------
# DuckDB unified view (classic)
# ---------------------------
con = duckdb.connect(database=":memory:")

con.execute("CREATE TABLE crop AS SELECT * FROM df_crop")
con.execute("CREATE TABLE rain AS SELECT * FROM df_rain")

# rainfall_mm = 'annual' if present, else try to sum months
has_annual = "annual" in df_rain.columns
if not has_annual:
    # create a view that sums monthly cols if annual missing
    month_cols = [c for c in ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"] if c in df_rain.columns]
    if month_cols:
        months_expr = " + ".join([f"COALESCE({m},0)" for m in month_cols])
        con.execute(f"""
            CREATE VIEW rain2 AS
            SELECT state, year, ({months_expr})::DOUBLE AS annual
            FROM rain;
        """)
    else:
        # last fallback: no rainfall available
        con.execute("""
            CREATE VIEW rain2 AS
            SELECT state, year, NULL::DOUBLE AS annual
            FROM rain;
        """)
    rain_source = "rain2"
else:
    rain_source = "rain"

con.execute(f"""
    CREATE VIEW unified AS
    SELECT
        c.state,
        c.district,
        c.crop,
        c.year,
        c.production_mt,
        r.annual AS rainfall_mm
    FROM crop c
    LEFT JOIN {rain_source} r
      ON lower(c.state) = lower(r.state)
     AND c.year = r.year
""")

# ---------- top row: quick context ----------
u_states = con.execute("SELECT DISTINCT state FROM unified ORDER BY 1").df()["state"].tolist()
u_years = con.execute("SELECT MIN(year) AS miny, MAX(year) AS maxy FROM unified").df().iloc[0].to_dict()

col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.markdown('<div class="card"><div class="pill">States</div><div class="metric">{}</div></div>'.format(len(u_states)), unsafe_allow_html=True)
with col2:
    st.markdown('<div class="card"><div class="pill">Years</div><div class="metric">{miny}–{maxy}</div></div>'.format(**u_years), unsafe_allow_html=True)
with col3:
    total_rows = con.execute("SELECT COUNT(*) AS c FROM unified").df()["c"].iloc[0]
    st.markdown('<div class="card"><div class="pill">Records</div><div class="metric">{}</div></div>'.format(int(total_rows)), unsafe_allow_html=True)

# ---------------------------
# Tabs for analyses
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔝 Top Crops by State",
    "📈 Trend & Rainfall",
    "🏆 District High vs Low",
    "⚖️ Compare Two States"
])

# --- Tab 1: Top crops by state ---
with tab1:
    st.subheader("Top Crops in a State")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        state_sel = st.selectbox("Select State", u_states, index=0, key="top_state")
        topN = st.slider("Top N", 3, 15, 5, key="topN")
    with c2:
        year_filter = st.selectbox("Year (optional)", ["All"] + sorted(list(set(df_crop["year"].dropna().astype(int)))), index=0)

    where_year = "" if year_filter == "All" else f"AND year = {int(year_filter)}"
    q = f"""
        SELECT crop, SUM(production_mt) AS total_prod
        FROM unified
        WHERE lower(state) = lower('{state_sel}')
        {where_year}
        GROUP BY crop
        ORDER BY total_prod DESC
        LIMIT {topN}
    """
    df_top = con.execute(q).df()

    if df_top.empty:
        st.warning("No data available for this selection.")
    else:
        fig = px.bar(df_top, x="crop", y="total_prod", title=f"Top {topN} Crops — {state_sel} ({year_filter})", height=420)
        fig.update_layout(xaxis_title="", yaxis_title="Production (MT)", margin=dict(l=20,r=20,t=60,b=20))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Show data"):
            st.dataframe(df_top, use_container_width=True)

# --- Tab 2: Trend & Rainfall ---
with tab2:
    st.subheader("Trend of Crop Production vs Rainfall")
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        state_tr = st.selectbox("State", u_states, key="trend_state")
        crop_tr  = st.text_input("Crop (e.g., Wheat, Rice)", "Wheat")
    with c2:
        n_years = st.slider("Last N years", 3, 15, 8)
    with c3:
        latest_year = con.execute("SELECT MAX(year) AS y FROM unified").df()["y"].iloc[0]
        min_year = int(latest_year) - n_years + 1
        st.markdown(f"**Year Window:** {min_year}–{int(latest_year)}")

    sql_trend = f"""
        SELECT year,
               SUM(production_mt) AS total_production,
               AVG(rainfall_mm) AS avg_rainfall
        FROM unified
        WHERE lower(state) = lower('{state_tr}')
          AND lower(crop) LIKE '%{crop_tr.lower()}%'
          AND year BETWEEN {min_year} AND {int(latest_year)}
        GROUP BY year
        ORDER BY year
    """
    df_trend = con.execute(sql_trend).df()

    if df_trend.empty:
        st.warning("No data for this selection. Try another crop/state or widen the year window.")
    else:
        fig1 = px.line(df_trend, x="year", y="total_production", markers=True,
                       title=f"{crop_tr} Production — {state_tr}")
        fig1.update_layout(yaxis_title="Production (MT)", xaxis_title="Year", height=380, margin=dict(l=20,r=20,t=50,b=20))

        fig2 = px.line(df_trend, x="year", y="avg_rainfall", markers=True,
                       title=f"Average Rainfall — {state_tr}")
        fig2.update_layout(yaxis_title="Rainfall (mm)", xaxis_title="Year", height=380, margin=dict(l=20,r=20,t=50,b=20))

        colA, colB = st.columns(2)
        with colA: st.plotly_chart(fig1, use_container_width=True)
        with colB: st.plotly_chart(fig2, use_container_width=True)

        # simple correlation (only valid if >= 2 pts)
        corr = None
        if len(df_trend) >= 2 and df_trend["avg_rainfall"].notna().sum() >= 2:
            corr = df_trend["total_production"].corr(df_trend["avg_rainfall"])
        st.markdown(f"**Correlation (Production vs Rainfall):** `{round(corr,3) if corr is not None else 'N/A'}`")
        with st.expander("Show data"):
            st.dataframe(df_trend, use_container_width=True)

# --- Tab 3: District High vs Low ---
with tab3:
    st.subheader("District with Highest vs Lowest Production")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        state_d = st.selectbox("State", u_states, key="dist_state")
        crop_d  = st.text_input("Crop (e.g., Rice, Maize)", "Rice")
    with c2:
        # latest year available for that crop-state
        y = con.execute(f"""
            SELECT MAX(year) AS y 
            FROM unified 
            WHERE lower(state)=lower('{state_d}') AND lower(crop) LIKE '%{crop_d.lower()}%'
        """).df()["y"].iloc[0]
        if pd.isna(y):
            st.warning("No year found for that state/crop selection.")
            st.stop()
        st.metric("Latest Year Used", int(y))

    sql_max = f"""
        SELECT district, SUM(production_mt) AS prod
        FROM unified
        WHERE lower(state)=lower('{state_d}') AND lower(crop) LIKE '%{crop_d.lower()}%' AND year={int(y)}
        GROUP BY district
        ORDER BY prod DESC
        LIMIT 1
    """
    sql_min = f"""
        SELECT district, SUM(production_mt) AS prod
        FROM unified
        WHERE lower(state)=lower('{state_d}') AND lower(crop) LIKE '%{crop_d.lower()}%' AND year={int(y)}
        GROUP BY district
        ORDER BY prod ASC
        LIMIT 1
    """
    df_max = con.execute(sql_max).df()
    df_min = con.execute(sql_min).df()

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### 🏆 Highest")
        if df_max.empty:
            st.info("No data.")
        else:
            st.markdown(f"**{df_max.iloc[0]['district']}** — **{int(df_max.iloc[0]['prod'])} MT**")
    with colB:
        st.markdown("#### 🌱 Lowest")
        if df_min.empty:
            st.info("No data.")
        else:
            st.markdown(f"**{df_min.iloc[0]['district']}** — **{int(df_min.iloc[0]['prod'])} MT**")

# --- Tab 4: Compare Two States ---
with tab4:
    st.subheader("Compare Two States — Avg Production & Rainfall (Last N Years)")
    c1, c2, c3 = st.columns([1.1,1.1,0.8])
    with c1:
        s1 = st.selectbox("State A", u_states, key="cmp_a")
    with c2:
        s2 = st.selectbox("State B", [s for s in u_states if s != s1], key="cmp_b")
    with c3:
        N = st.slider("Years (lookback from latest)", 3, 15, 5)

    latest_year = con.execute("SELECT MAX(year) AS y FROM unified").df()["y"].iloc[0]
    start_year = int(latest_year) - N + 1

    sql_cmp = f"""
        SELECT state,
               AVG(production_mt) AS avg_production,
               AVG(rainfall_mm)   AS avg_rainfall
        FROM unified
        WHERE lower(state) IN (lower('{s1}'), lower('{s2}'))
          AND year BETWEEN {start_year} AND {int(latest_year)}
        GROUP BY state
        ORDER BY state
    """
    df_cmp = con.execute(sql_cmp).df()

    if df_cmp.empty:
        st.warning("No data for the selected window.")
    else:
        colA, colB = st.columns(2)
        with colA:
            figp = px.bar(df_cmp, x="state", y="avg_production", title=f"Avg Production (MT) • {start_year}–{int(latest_year)}")
            figp.update_layout(xaxis_title="", yaxis_title="MT", height=380, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(figp, use_container_width=True)
        with colB:
            figr = px.bar(df_cmp, x="state", y="avg_rainfall", title=f"Avg Rainfall (mm) • {start_year}–{int(latest_year)}")
            figr.update_layout(xaxis_title="", yaxis_title="mm", height=380, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(figr, use_container_width=True)

        with st.expander("Show data"):
            st.dataframe(df_cmp, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div class='footer'>Project Samarth — Classic edition (Crop + Rainfall). "
    "This app intentionally excludes temperature/soil features while the advanced pipeline is under development.</div>",
    unsafe_allow_html=True
)
