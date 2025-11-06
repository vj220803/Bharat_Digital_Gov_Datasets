import streamlit as st
import pandas as pd
import duckdb
import os
import re

st.set_page_config(page_title="Bharat Samarth – Gov Dataset Q&A", layout="wide")

# ---------------------------------------------------------
# ✅ SMART PARQUET LOADER — auto maps columns + prints debug
# ---------------------------------------------------------
def load_parquet_smart(path, expected_map, label):
    if not os.path.exists(path):
        st.error(f"❌ {label} file missing at: {path}")
        st.stop()

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        st.error(f"❌ Failed reading {label}: {e}")
        st.stop()

    st.write(f"### ✅ Loaded {label} file")
    st.write("Columns detected:", list(df.columns))

    rename_dict = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}

    for expected, aliases in expected_map.items():
        found = None
        for alias in aliases:
            key = alias.lower().strip()
            if key in cols_lower:
                found = cols_lower[key]
                break
        if found:
            rename_dict[found] = expected

    df = df.rename(columns=rename_dict)

    # Check missing
    missing = [k for k in expected_map.keys() if k not in df.columns]
    if missing:
        st.error(f"❌ {label} missing columns: {missing}")
        ststop()

    return df


# ---------------------------------------------------------
# ✅ LOAD BOTH PARQUET FILES
# ---------------------------------------------------------

DATA_DIR = "data"
CROP_PATH = os.path.join(DATA_DIR, "crop_production.parquet")
RAIN_PATH = os.path.join(DATA_DIR, "imd_rainfall.parquet")

CROP_EXPECTED = {
    "state": ["state", "states", "subdivision"],
    "district": ["district", "dist"],
    "crop": ["crop", "crop_name"],
    "production_mt": ["production_mt", "production", "production (mt)"],
    "year": ["year", "yr"]
}

RAIN_EXPECTED = {
    "state": ["state", "subdivision", "state/ut"],
    "year": ["year", "yr"],
    "jan": ["jan"], "feb": ["feb"], "mar": ["mar"], "apr": ["apr"],
    "may": ["may"], "jun": ["jun"], "jul": ["jul"], "aug": ["aug"],
    "sep": ["sep"], "oct": ["oct"], "nov": ["nov"], "dec": ["dec"],
    "annual": ["annual", "annual_rainfall"],
    "jf": ["jf", "jan-feb"],
    "mam": ["mam", "mar-may"],
    "jjas": ["jjas", "jun-sep"],
    "ond": ["ond", "oct-dec"]
}

crop_df = load_parquet_smart(CROP_PATH, CROP_EXPECTED, "CROP")
rain_df = load_parquet_smart(RAIN_PATH, RAIN_EXPECTED, "RAINFALL")

# ---------------------------------------------------------
# ✅ Build DuckDB in-memory view
# ---------------------------------------------------------
con = duckdb.connect(database=":memory:")

con.register("crop", crop_df)
con.register("rain", rain_df)

con.execute("""
CREATE OR REPLACE VIEW unified AS
SELECT
    c.state,
    c.district,
    c.crop,
    c.production_mt,
    c.year,
    r.annual AS rainfall_mm
FROM crop c
LEFT JOIN rain r
    ON lower(c.state) = lower(r.state)
   AND c.year = r.year;
""")

df_unified = con.execute("SELECT * FROM unified LIMIT 5").df()

# ---------------------------------------------------------
# ✅ SIMPLE QUERY PLANNER (only 3 intents)
# ---------------------------------------------------------

def samarth_plan(q):
    text = q.lower().strip()

    # 1) Top N crops in a state
    m = re.search(r"top\s+(\d+)\s+crops\s+in\s+([\w\s]+)", text)
    if m:
        n = int(m.group(1))
        state = m.group(2).strip()
        sql = f"""
            SELECT crop, SUM(production_mt) AS total_prod
            FROM unified
            WHERE lower(state) = lower('{state}')
            GROUP BY crop
            ORDER BY total_prod DESC
            LIMIT {n};
        """
        return {"intent": "top_crops", "sql": sql, "state": state, "n": n}

    # 2) Trend of crop over last N years
    m = re.search(r"trend.*of\s+([\w\s]+).*last\s+(\d+)\s+years.*in\s+([\w\s]+)", text)
    if m:
        crop = m.group(1).strip()
        n = int(m.group(2))
        state = m.group(3).strip()
        max_year = con.execute("SELECT MAX(year) FROM unified").fetchone()[0]
        min_year = max_year - n + 1
        sql = f"""
            SELECT year, SUM(production_mt) AS total_prod
            FROM unified
            WHERE lower(state)=lower('{state}') 
              AND lower(crop)=lower('{crop}')
              AND year BETWEEN {min_year} AND {max_year}
            GROUP BY year ORDER BY year;
        """
        return {"intent": "trend", "sql": sql, "crop": crop, "state": state, "years": (min_year, max_year)}

    # 3) Compare rainfall between two states
    m = re.search(r"compare.*rainfall.*between\s+([\w\s]+)\s+and\s+([\w\s]+)", text)
    if m:
        s1, s2 = m.groups()
        sql = f"""
            SELECT state, AVG(rainfall_mm) AS avg_rain
            FROM unified
            WHERE lower(state) IN (lower('{s1.strip()}'), lower('{s2.strip()}'))
            GROUP BY state;
        """
        return {"intent": "compare_rain", "sql": sql, "states": (s1.strip(), s2.strip())}

    return {"intent": "unknown"}


# ---------------------------------------------------------
# ✅ EXECUTOR + ANSWER BUILDER
# ---------------------------------------------------------

def execute_and_synthesize(plan):
    intent = plan["intent"]

    if intent == "unknown":
        return "⚠️ I couldn't understand that. Try 'Top 5 crops in X' or 'Trend of rice in X last 5 years'.", None

    df = con.execute(plan["sql"]).df()

    if df.empty:
        return "⚠️ No data available for this query.", df

    if intent == "top_crops":
        lines = [f"### 🌾 Top {plan['n']} crops in {plan['state'].title()}"]
        for _, r in df.iterrows():
            lines.append(f"- **{r['crop'].title()}** — {int(r['total_prod'])} tonnes")
        return "\n".join(lines), df

    if intent == "trend":
        crop = plan["crop"].title()
        state = plan["state"].title()
        a, b = plan["years"]
        return (
            f"### 📈 Trend of **{crop}** in **{state}** ({a}–{b})\n"
            f"Showing yearly production from the unified dataset.",
            df
        )

    if intent == "compare_rain":
        s1, s2 = plan["states"]
        lines = [f"### 🌦 Rainfall comparison between {s1.title()} and {s2.title()}"]
        for _, r in df.iterrows():
            lines.append(f"- **{r['state'].title()}** → {round(r['avg_rain'],2)} mm")
        return "\n".join(lines), df

    return "⚠️ Unexpected error. ", None


# ---------------------------------------------------------
# ✅ STREAMLIT UI
# ---------------------------------------------------------

st.title("🌾 Bharat Samarth – Agriculture Q&A Assistant")
st.write("Ask natural questions about crop production or rainfall.")

q = st.text_input("Ask a question:")

if q:
    plan = samarth_plan(q)
    st.write("✅ **Detected Query Plan:**", plan)

    answer, df_out = execute_and_synthesize(plan)
    st.markdown(answer)

    if df_out is not None and not df_out.empty:
        st.write("### 📊 Data Preview:")
        st.dataframe(df_out)
