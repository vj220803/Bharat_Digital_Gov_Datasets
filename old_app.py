# -------------------------------
# Bharat Digital – Crop & Rainfall Intelligence Dashboard
# -------------------------------
# Features:
# - Load rainfall (state-year monthly & seasonal) and crop production (state-district-crop-year)
# - Filter by state, crop, year range
# - KPIs, interactive visualizations (Plotly)
# - Correlation heatmap & climograph
# - ML: RandomForestRegressor to model production vs rainfall (monthly+seasonal+annual)
# - Fast with caching, supports parquet or CSV uploads, download filtered data
# -------------------------------

import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

# -------------------------------
# Page Config & Styling
# -------------------------------
st.set_page_config(
    page_title="Bharat Digital — Crop & Rainfall Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* clean, modern look */
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
.kpi-card {
  background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
  border-radius: 16px; padding: 14px 18px; border: 1px solid #e5e7eb;
  box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}
.section-card {
  background: #ffffff; border-radius: 16px; padding: 14px 18px; border: 1px solid #e5e7eb;
}
h1, h2, h3 { margin-top: 0.4rem; }
footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------------
# Helpers
# -------------------------------
MONTHS = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
SEASONS = ["jf","mam","jjas","ond"]
RAIN_FEATURES = MONTHS + ["annual"] + SEASONS

@st.cache_data(show_spinner=False)
def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        # requires pyarrow (recommended)
        return pd.read_parquet(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

@st.cache_data(show_spinner=False)
def load_data(
    rainfall_path: str | None,
    crop_path: str | None,
    rainfall_file,  # uploaded file or None
    crop_file,      # uploaded file or None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # priority: uploaded files, else local paths
    if rainfall_file is not None:
        rain = pd.read_parquet(rainfall_file) if rainfall_file.name.endswith(".parquet") else pd.read_csv(rainfall_file)
    elif rainfall_path and os.path.exists(rainfall_path):
        rain = read_table(rainfall_path)
    else:
        rain = pd.DataFrame()

    if crop_file is not None:
        crop = pd.read_parquet(crop_file) if crop_file.name.endswith(".parquet") else pd.read_csv(crop_file)
    elif crop_path and os.path.exists(crop_path):
        crop = read_table(crop_path)
    else:
        crop = pd.DataFrame()

    return rain, crop

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def prep_rainfall_df(rain: pd.DataFrame) -> pd.DataFrame:
    if rain.empty:
        return rain
    rain = standardize_columns(rain)

    # Expected columns from your schema: state, year, monthly, annual, jf, mam, jjas, ond
    # Validate presence and coerce numerics
    for col in ["state","year"] + RAIN_FEATURES:
        if col not in rain.columns:
            # some datasets use 'subdivision' -> 'state' mapping already done per your note
            if col == "state" and "subdivision" in rain.columns:
                rain = rain.rename(columns={"subdivision": "state"})
            else:
                # If seasonal columns missing, try to create where possible
                pass

    # coerce month/seasonal numeric
    for col in RAIN_FEATURES:
        if col in rain.columns:
            rain[col] = pd.to_numeric(rain[col], errors="coerce")

    # compute seasonal if missing (optional)
    if "jf" not in rain.columns and all(c in rain.columns for c in ["jan","feb"]):
        rain["jf"] = rain["jan"].fillna(0) + rain["feb"].fillna(0)
    if "mam" not in rain.columns and all(c in rain.columns for c in ["mar","apr","may"]):
        rain["mam"] = rain[["mar","apr","may"]].fillna(0).sum(axis=1)
    if "jjas" not in rain.columns and all(c in rain.columns for c in ["jun","jul","aug","sep"]):
        rain["jjas"] = rain[["jun","jul","aug","sep"]].fillna(0).sum(axis=1)
    if "ond" not in rain.columns and all(c in rain.columns for c in ["oct","nov","dec"]):
        rain["ond"] = rain[["oct","nov","dec"]].fillna(0).sum(axis=1)
    if "annual" not in rain.columns and any(c in rain.columns for c in MONTHS):
        rain["annual"] = rain[[c for c in MONTHS if c in rain.columns]].fillna(0).sum(axis=1)

    # minimal columns
    needed = ["state","year"] + [c for c in RAIN_FEATURES if c in rain.columns]
    rain = rain[[c for c in needed if c in rain.columns]].dropna(subset=["state","year"])
    rain["state"] = rain["state"].astype(str).str.strip()
    rain["year"] = pd.to_numeric(rain["year"], errors="coerce").astype("Int64")
    return rain

@st.cache_data(show_spinner=False)
def prep_crop_df(crop: pd.DataFrame) -> pd.DataFrame:
    if crop.empty:
        return crop
    crop = standardize_columns(crop)
    # expected columns from your schema: state, district, crop, production_mt, year
    rename_map = {
        "production": "production_mt",
        "production_tonnes": "production_mt",
        "production_mt": "production_mt",
    }
    for k,v in list(rename_map.items()):
        if k in crop.columns:
            crop = crop.rename(columns={k:v})
    # keep minimal set
    needed = [c for c in ["state","district","crop","production_mt","year"] if c in crop.columns]
    crop = crop[needed].dropna(subset=["state","year"])
    crop["state"] = crop["state"].astype(str).str.strip()
    crop["year"] = pd.to_numeric(crop["year"], errors="coerce").astype("Int64")
    crop["production_mt"] = pd.to_numeric(crop.get("production_mt", np.nan), errors="coerce")
    # Drop zero/negative if any
    crop = crop[crop["production_mt"].fillna(0) >= 0]
    return crop

@st.cache_data(show_spinner=False)
def merge_state_year(
    crop: pd.DataFrame,
    rain: pd.DataFrame,
    group_by_crop: bool = False
) -> pd.DataFrame:
    """Aggregate crop to state-year (optionally state-year-crop) and left-join rainfall"""
    if crop.empty or rain.empty:
        return pd.DataFrame()

    keys = ["state","year"]
    if group_by_crop and "crop" in crop.columns:
        grp_cols = keys + ["crop"]
    else:
        grp_cols = keys

    agg_cols = {"production_mt":"sum"}
    g = crop.groupby(grp_cols, dropna=False, as_index=False).agg(agg_cols).rename(columns={"production_mt":"production_mt_total"})
    rain_cols = keys + [c for c in RAIN_FEATURES if c in rain.columns]
    merged = g.merge(rain[rain_cols], on=keys, how="left")
    return merged

def kpi(value, label):
    with st.container():
        st.markdown(f"<div class='kpi-card'><h3 style='margin:0'>{label}</h3><h2 style='margin:0'>{value}</h2></div>", unsafe_allow_html=True)

# -------------------------------
# Sidebar – Data Inputs
# -------------------------------
st.sidebar.title("⚙️ Data & Controls")
st.sidebar.caption("Upload to override local files if needed.")

rain_up = st.sidebar.file_uploader("Rainfall file (.parquet or .csv)", type=["parquet","csv"], key="rfu")
crop_up = st.sidebar.file_uploader("Crop file (.parquet or .csv)", type=["parquet","csv"], key="cpu")

DEFAULT_RAIN_PATH = "imd_rainfall.parquet"
DEFAULT_CROP_PATH = "crop_production.parquet"

rain_raw, crop_raw = load_data(DEFAULT_RAIN_PATH, DEFAULT_CROP_PATH, rain_up, crop_up)
rain = prep_rainfall_df(rain_raw)
crop = prep_crop_df(crop_raw)

if rain.empty or crop.empty:
    st.error("Please provide both rainfall and crop datasets (parquet or CSV). Expected columns:\n"
             "- Rainfall: state, year, jan..dec, annual, jf, mam, jjas, ond\n"
             "- Crop: state, district, crop, production_mt, year")
    st.stop()

# -------------------------------
# Filters
# -------------------------------
all_states = sorted(rain["state"].dropna().unique().tolist())
states = st.sidebar.multiselect("Select states", all_states, default=all_states[: min(len(all_states), 8)])

min_year = int(pd.Series([rain["year"].min(), crop["year"].min()]).dropna().min())
max_year = int(pd.Series([rain["year"].max(), crop["year"].max()]).dropna().max())
year_range = st.sidebar.slider("Year range", min_year, max_year, (max(min_year, max_year-10), max_year), step=1)

all_crops = sorted(crop["crop"].dropna().unique().tolist()) if "crop" in crop.columns else []
selected_crops = st.sidebar.multiselect("Select crops (affects group-by)", all_crops, default=all_crops[: min(len(all_crops), 6)])
group_by_crop = st.sidebar.toggle("Group by crop", value=bool(selected_crops))

# -------------------------------
# Data Merge (state-year) and Filter
# -------------------------------
merged = merge_state_year(crop, rain, group_by_crop=group_by_crop)

if states:
    merged = merged[merged["state"].isin(states)]
merged = merged[(merged["year"] >= year_range[0]) & (merged["year"] <= year_range[1])]

if group_by_crop and selected_crops:
    merged = merged[merged["crop"].isin(selected_crops)]

if merged.empty:
    st.warning("No data after applying filters. Adjust your selections.")
    st.stop()

# -------------------------------
# Header
# -------------------------------
st.title("🌾 Bharat Digital — Crop & Rainfall Intelligence")
st.caption("Interactive analytics combining IMD rainfall and crop production across states & years.")

# -------------------------------
# KPIs
# -------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    kpi(f"{merged['production_mt_total'].sum():,.0f} MT", "Total Production")
with col2:
    if "annual" in merged.columns:
        kpi(f"{merged['annual'].mean():,.1f} mm", "Avg Annual Rainfall")
    else:
        kpi("—", "Avg Annual Rainfall")
with col3:
    kpi(f"{merged['year'].nunique()}", "Years Covered")
with col4:
    kpi(f"{merged['state'].nunique()}", "States Selected")

st.divider()

# -------------------------------
# Tabs
# -------------------------------
tab_overview, tab_explore, tab_model = st.tabs(["📊 Overview", "🔎 Explore", "🧠 ML Model"])

# ---- OVERVIEW TAB
with tab_overview:
    st.subheader("Production vs Rainfall (Dual-Axis)")
    if "annual" in merged.columns:
        # aggregate for clarity
        level = ["year"] + (["crop"] if group_by_crop else [])
        show = merged.groupby(level, as_index=False).agg(
            production=("production_mt_total","sum"),
            annual=("annual","mean")
        )
        fig = go.Figure()
        # production (bars)
        if group_by_crop:
            for c in show["crop"].unique():
                dfc = show[show["crop"] == c]
                fig.add_bar(x=dfc["year"], y=dfc["production"], name=f"Production — {c}", yaxis="y1")
        else:
            fig.add_bar(x=show["year"], y=show["production"], name="Production", yaxis="y1")

        # rainfall (line)
        if group_by_crop:
            # same rainfall across crops for a state-year aggregate, just plot once per crop group to show lines
            for c in show["crop"].unique():
                dfc = show[show["crop"] == c]
                fig.add_trace(go.Scatter(x=dfc["year"], y=dfc["annual"], mode="lines+markers",
                                         name=f"Annual Rainfall — {c}", yaxis="y2"))
        else:
            fig.add_trace(go.Scatter(x=show["year"], y=show["annual"], mode="lines+markers",
                                     name="Annual Rainfall", yaxis="y2"))

        fig.update_layout(
            xaxis=dict(title="Year"),
            yaxis=dict(title="Production (MT)", side="left"),
            yaxis2=dict(title="Rainfall (mm)", overlaying="y", side="right"),
            legend_title="Legend",
            margin=dict(l=20,r=20,t=40,b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Annual rainfall not found. Monthly columns will still be available in Explore.")

    st.subheader("Correlation Heatmap (Production & Rainfall)")
    corr_cols = ["production_mt_total"] + [c for c in RAIN_FEATURES if c in merged.columns]
    corr_df = merged[corr_cols].dropna()
    if corr_df.shape[0] >= 3:
        corr = corr_df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto")
        fig.update_layout(margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough rows to compute correlations after filters.")

# ---- EXPLORE TAB
with tab_explore:
    st.subheader("Monthly Climograph")
    if all(m in merged.columns for m in MONTHS):
        # Average across selected states (and crop grouping if on)
        grp_cols = ["year"] + (["crop"] if group_by_crop else [])
        monthly_means = merged.groupby(grp_cols, as_index=False)[MONTHS].mean()
        if group_by_crop:
            crop_to_show = st.selectbox("Select crop for climograph", sorted(merged["crop"].unique().tolist()))
            dfm = monthly_means[monthly_means["crop"] == crop_to_show].drop(columns=["crop"])
        else:
            dfm = monthly_means

        month_long = dfm.melt(id_vars=["year"], value_vars=MONTHS, var_name="month", value_name="rain_mm")
        fig = px.line(month_long, x="month", y="rain_mm", color="year", markers=True)
        fig.update_layout(margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Monthly columns (jan..dec) not found in rainfall data.")

    st.subheader("Production Breakdown")
    if group_by_crop:
        prod = merged.groupby(["crop","year"], as_index=False)["production_mt_total"].sum()
        fig = px.bar(prod, x="year", y="production_mt_total", color="crop", barmode="group")
    else:
        prod = merged.groupby(["state","year"], as_index=False)["production_mt_total"].sum()
        fig = px.bar(prod, x="year", y="production_mt_total", color="state")
    fig.update_layout(yaxis_title="Production (MT)", margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Download filtered & merged data")
    dl = merged.copy()
    st.download_button(
        "⬇️ Download CSV",
        data=dl.to_csv(index=False).encode("utf-8"),
        file_name="merged_filtered.csv",
        mime="text/csv",
    )

# ---- ML MODEL TAB
with tab_model:
    st.subheader("Predict Production from Rainfall")

    # Features & target
    feature_cols = [c for c in RAIN_FEATURES if c in merged.columns]
    base_cols = ["state"] + (["crop"] if group_by_crop else [])
    needed_cols = base_cols + feature_cols + ["production_mt_total"]
    data_ml = merged.dropna(subset=["production_mt_total"] + feature_cols)
    data_ml = data_ml[needed_cols].copy()

    if data_ml.shape[0] < 30 or len(feature_cols) == 0:
        st.info("Not enough data/features to train. Ensure rainfall columns (months/seasonal/annual) are present and filters allow enough rows.")
    else:
        # Train-test split
        X = data_ml[base_cols + feature_cols]
        y = data_ml["production_mt_total"]

        # pipeline: one-hot encode categoricals (state, optional crop) + RF Regressor
        numeric_features = feature_cols
        categorical_features = base_cols

        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
                ("num", "passthrough", numeric_features),
            ]
        )

        model = Pipeline(steps=[
            ("prep", pre),
            ("rf", RandomForestRegressor(
                n_estimators=400,
                max_depth=None,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        c1, c2 = st.columns(2)
        with c1:
            kpi(f"{r2:.3f}", "R² on Test")
        with c2:
            kpi(f"{rmse:,.0f} MT", "RMSE on Test")

        # Save/Load buttons (optional)
        with st.expander("💾 Save/Load model"):
            save_path = st.text_input("Model file path", value="crop_rain_rf_model.joblib")
            cols = st.columns(3)
            with cols[0]:
                if st.button("Save Model"):
                    joblib.dump(model, save_path)
                    st.success(f"Saved to {save_path}")
            with cols[1]:
                if st.button("Load Model"):
                    if os.path.exists(save_path):
                        mdl = joblib.load(save_path)
                        st.success("Model loaded.")
                        model = mdl
                    else:
                        st.error("File not found.")

        st.subheader("What-if Prediction")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            pred_state = st.selectbox("State", sorted(merged["state"].unique().tolist()))
        with pcol2:
            pred_crop = None
            if group_by_crop:
                pred_crop = st.selectbox("Crop", sorted(merged["crop"].unique().tolist()))

        default_annual = float(np.nanmean(merged["annual"])) if "annual" in merged.columns else 1000.0
        user_vals = {}
        if "annual" in feature_cols:
            user_vals["annual"] = st.number_input("Annual Rainfall (mm)", value=round(default_annual,1))
        # allow monthly overrides when available
        if all(m in feature_cols for m in MONTHS):
            with st.expander("Adjust monthly rainfall (mm)"):
                mcols = st.columns(6)
                for i, m in enumerate(MONTHS):
                    with mcols[i%6]:
                        user_vals[m] = st.number_input(m.upper(), value=float(round(np.nanmean(merged[m]),1)))
        # allow seasonal if available (overrides are fine, model will just use columns present)
        for s in [s for s in SEASONS if s in feature_cols]:
            user_vals[s] = user_vals.get(s, float(round(np.nanmean(merged[s]),1)))

        # Build single-row input
        row = {**{ "state": pred_state }, **({ "crop": pred_crop } if group_by_crop else {})}
        for c in feature_cols:
            row[c] = user_vals.get(c, float(round(np.nanmean(merged[c]),1)))
        X_new = pd.DataFrame([row], columns=base_cols + feature_cols)

        if st.button("Predict Production"):
            try:
                pred_val = float(model.predict(X_new)[0])
                st.success(f"Predicted Production: **{pred_val:,.0f} MT**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -------------------------------
# Footer
# -------------------------------
st.caption("© Bharat Digital — Data from IMD rainfall & crop production. Built with Streamlit & Plotly.")
