# --------------------------------------------------------
# ✅ Project Samarth - Streamlit App (Older Stable Version)
#     (Crop + Rainfall only / robust to schema quirks)
# --------------------------------------------------------

import os
import re
import streamlit as st
import pandas as pd
import duckdb

# ===============================
# Page config & header
# ===============================
st.set_page_config(page_title="Bharat Samarth – Agriculture Q&A", page_icon="🌾", layout="wide")
st.title("🌾 Bharat Samarth – Agriculture Q&A Assistant")
st.markdown("""
This app answers analytical questions using **Indian crop production** and **rainfall** data.

**Try:**
- `Top 5 crops in Himachal Pradesh`
- `Trend of wheat over last 5 years in Himachal Pradesh`
- `Compare rainfall between Maharashtra and Karnataka`
""")

DATA_DIR = "data"
CROP_PARQUET = os.path.join(DATA_DIR, "crop_production.parquet")
RAIN_PARQUET = os.path.join(DATA_DIR, "imd_rainfall.parquet")

# ===============================
# Helpers
# ===============================

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """lowercase, strip, replace spaces/dashes/slashes with underscores."""
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_") for c in out.columns]
    return out

def ensure_rainfall_shape(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure rainfall has: state, year, monthly columns (optional) and 'annual'.
    - If 'subdivision' exists, rename to 'state'
    - If 'annual' missing but months exist, compute it
    - Coerce year to int, rainfall to float
    """
    df = normalize_cols(df)

    # Map subdivision -> state if needed
    if "state" not in df.columns:
        if "subdivision" in df.columns:
            df = df.rename(columns={"subdivision": "state"})
        else:
            # nothing we can do: keep as is
            pass

    # Try to compute 'annual' if missing
    month_cols = [c for c in df.columns if c in ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]]
    if "annual" not in df.columns and month_cols:
        # safely to_numeric
        for c in month_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["annual"] = df[month_cols].sum(axis=1)

    # Year as numeric (extract 4-digit if needed)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")

    # Coerce annual to numeric
    if "annual" in df.columns:
        df["annual"] = pd.to_numeric(df["annual"], errors="coerce")

    # Clean state if present
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.strip()

    # Drop rows with missing critical fields (state/year)
    if {"state","year"}.issubset(df.columns):
        df = df.dropna(subset=["state","year"]).reset_index(drop=True)

    return df

def ensure_crop_shape(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expecting: state, district, crop, production_mt, year
    - Normalize headers
    - Coerce year, production_mt
    """
    df = normalize_cols(df)

    # Rename common variants if needed
    # (Your older cleaned file already matches, but keep this defensive)
    rename_map = {
        "production": "production_mt",
        "production_in_mt": "production_mt",
        "prod_mt": "production_mt"
    }
    for k, v in rename_map.items():
        if k in df.columns and "production_mt" not in df.columns:
            df = df.rename(columns={k: v})

    # Ensure expected columns exist
    needed = {"state","district","crop","production_mt","year"}
    missing = needed - set(df.columns)
    if missing:
        st.warning(f"⚠️ Crop file is missing columns: {sorted(list(missing))}. App will still try to work with what's available.")

    # Coerce types
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    if "production_mt" in df.columns:
        df["production_mt"] = pd.to_numeric(df["production_mt"], errors="coerce")

    # Clean text columns
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.strip()
    if "crop" in df.columns:
        df["crop"] = df["crop"].astype(str).str.strip()

    # Drop obviously unusable rows
    crit = [c for c in ["state","year"] if c in df.columns]
    if crit:
        df = df.dropna(subset=crit).reset_index(drop=True)

    return df

def safe_like(s: str) -> str:
    """very small sanitization for LIKE clause"""
    return s.replace("'", "''")

# ===============================
# Load data (with friendly errors)
# ===============================
@st.cache_data(show_spinner=False)
def load_parquet_or_fail(path: str, friendly_name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"❌ Missing file: `{path}`. Please add it to your repo.")
        st.stop()
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.error(f"❌ Could not read {friendly_name} from `{path}`: {e}")
        st.stop()

crop_raw = load_parquet_or_fail(CROP_PARQUET, "Crop")
rain_raw = load_parquet_or_fail(RAIN_PARQUET, "Rainfall")

crop_df = ensure_crop_shape(crop_raw)
rain_df = ensure_rainfall_shape(rain_raw)

# Quick schema report
with st.expander("🧪 Dataset status (click to expand)", expanded=False):
    st.write("**Crop columns:**", list(crop_df.columns))
    st.write("**Rainfall columns:**", list(rain_df.columns))
    st.write("**Crop rows:**", len(crop_df), " | **Rainfall rows:**", len(rain_df))
    if not {"state","year","production_mt","crop"}.issubset(crop_df.columns):
        st.warning("Crop dataset is missing one or more recommended columns.")
    if not {"state","year","annual"}.issubset(rain_df.columns):
        st.warning("Rainfall dataset is missing one or more recommended columns (needs state, year, annual).")

# ===============================
# Register tables in DuckDB
# ===============================
con = duckdb.connect(database=":memory:")
con.register("crop", crop_df)
con.register("rainfall", rain_df)

# ===============================
# NL Query Planner
# ===============================
def samarth_plan(question: str):
    q = question.lower().strip()

    # 1) Top N crops in a state
    m = re.search(r"top\s+(\d+)\s+crops\s+in\s+([\w\s&.-]+)$", q)
    if m and {"state","crop","production_mt"}.issubset(crop_df.columns):
        n, state = m.groups()
        state_l = safe_like(state.strip().lower())
        sql = f"""
            SELECT crop, SUM(production_mt) AS total_prod
            FROM crop
            WHERE lower(state) = '{state_l}'
            GROUP BY crop
            ORDER BY total_prod DESC
            LIMIT {int(n)};
        """
        return {"intent":"top_crops","sql":sql,"state":state.strip().title(),"n":int(n)}

    # 2) Trend of crop over last N years in state
    m = re.search(r"trend.*of\s+([\w\s&.-]+)\s+over\s+last\s+(\d+)\s+years\s+in\s+([\w\s&.-]+)$", q)
    if m and {"state","crop","production_mt","year"}.issubset(crop_df.columns) and {"state","year","annual"}.issubset(rain_df.columns):
        crop_name, n, state = m.groups()
        n = int(n)
        max_year = int(pd.to_numeric(crop_df["year"], errors="coerce").dropna().max())
        min_year = max_year - n + 1
        state_l = safe_like(state.strip().lower())
        crop_l = safe_like(crop_name.strip().lower())
        sql_prod = f"""
            SELECT year, SUM(production_mt) AS total_prod
            FROM crop
            WHERE lower(crop) LIKE '%{crop_l}%'
              AND lower(state) = '{state_l}'
              AND year BETWEEN {min_year} AND {max_year}
            GROUP BY year ORDER BY year;
        """
        sql_rain = f"""
            SELECT year, AVG(annual) AS avg_rain
            FROM rainfall
            WHERE lower(state) = '{state_l}'
              AND year BETWEEN {min_year} AND {max_year}
            GROUP BY year ORDER BY year;
        """
        return {"intent":"trend","crop":crop_name.strip().title(),"state":state.strip().title(),
                "years":(min_year,max_year),"sql_prod":sql_prod,"sql_rain":sql_rain}

    # 3) Compare rainfall between two states (avg across all years)
    m = re.search(r"compare.*rainfall.*between\s+([\w\s&.-]+)\s+and\s+([\w\s&.-]+)$", q)
    if m and {"state","year","annual"}.issubset(rain_df.columns):
        s1, s2 = m.groups()
        s1_l, s2_l = safe_like(s1.strip().lower()), safe_like(s2.strip().lower())
        sql = f"""
            SELECT state, AVG(annual) AS avg_rain
            FROM rainfall
            WHERE lower(state) IN ('{s1_l}', '{s2_l}')
            GROUP BY state;
        """
        return {"intent":"compare_rain","sql":sql,"states":(s1.strip().title(), s2.strip().title())}

    return {"intent":"unknown"}

# ===============================
# Executor & Synthesizer
# ===============================
def execute_and_synthesize(plan):
    if plan["intent"] == "unknown":
        return "⚠️ I couldn't interpret that. Try one of the example phrasings above.", None

    # 1) Top N crops
    if plan["intent"] == "top_crops":
        df = con.execute(plan["sql"]).df()
        if df.empty:
            return f"ℹ️ No crop data found for **{plan['state']}**.", df
        lines = [f"### 🌾 Top {plan['n']} Crops in {plan['state']}"]
        for _, r in df.iterrows():
            lines.append(f"- **{r['crop']}** → {int(r['total_prod'])} tonnes")
        return "\n".join(lines), df

    # 2) Trend
    if plan["intent"] == "trend":
        df_prod = con.execute(plan["sql_prod"]).df()
        df_rain = con.execute(plan["sql_rain"]).df()
        merged = df_prod.merge(df_rain, on="year", how="left")
        corr = merged["total_prod"].corr(merged["avg_rain"]) if not merged.empty else None
        corr_text = "N/A" if (corr is None or pd.isna(corr)) else round(float(corr), 2)
        text = f"""### 📈 Trend for **{plan['crop']}** in **{plan['state']}**
**Years:** {plan['years'][0]} – {plan['years'][1]}
Correlation (production vs rainfall): **{corr_text}**
"""
        return text, merged

    # 3) Compare rainfall
    if plan["intent"] == "compare_rain":
        df = con.execute(plan["sql"]).df()
        if df.empty:
            return "ℹ️ No rainfall data found for those states.", df
        s1, s2 = plan["states"]
        lines = [f"### 🌦 Rainfall Comparison: {s1} vs {s2}"]
        for _, r in df.iterrows():
            lines.append(f"- **{r['state']}** → {round(r['avg_rain'], 1)} mm")
        return "\n".join(lines), df

    return "⚠️ Something went wrong.", None

# ===============================
# UI – Ask & Answer
# ===============================
st.subheader("💬 Ask a Question")
user_q = st.text_input("Type your query here (examples above):")

if user_q:
    with st.spinner("Analyzing your question..."):
        plan = samarth_plan(user_q)
        answer, df_out = execute_and_synthesize(plan)
    st.markdown(answer)
    if isinstance(df_out, pd.DataFrame) and not df_out.empty:
        st.dataframe(df_out, use_container_width=True)

st.markdown("---")
st.caption("Built with ❤️ for Bharat Digital Fellowship (older stable version: Crop + Rainfall).")
