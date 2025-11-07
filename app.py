import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Project Samarth - Agricultural Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #2ecc71 0%, #27ae60 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2ecc71;
        margin: 1rem 0;
    }
    .citation-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize DuckDB connection
@st.cache_resource
def init_db():
    con = duckdb.connect(database=':memory:', read_only=False)
    return con

@st.cache_data
def load_data():
    """Load the parquet files"""
    try:
        df_crop = pd.read_parquet('crop_production.parquet')
        df_rain = pd.read_parquet('imd_rainfall.parquet')
        return df_crop, df_rain
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure 'crop_production.parquet' and 'imd_rainfall.parquet' are in the same directory as app.py")
        return None, None

def setup_database(con, df_crop, df_rain):
    """Setup DuckDB tables"""
    con.register('crop', df_crop)
    con.register('rainfall', df_rain)

def samarth_plan(question: str, con):
    """Natural language query planner"""
    q = question.strip().lower()
    
    # Pattern 1: Top N crops in state
    m = re.search(r"top\s+(\d+)\s+crops?\s+(?:in|from)\s+([a-z\s]+)", q)
    if m:
        n, state = m.groups()
        n = int(n)
        sql = f"""
        SELECT crop, SUM(production_mt) as total_prod
        FROM crop
        WHERE LOWER(state) LIKE '%{state.strip()}%'
        GROUP BY crop
        ORDER BY total_prod DESC
        LIMIT {n};
        """
        return {"intent": "top_crops_state", "sql": sql, "state": state.strip().title(), "n": n}
    
    # Pattern 2: Crop production in specific state
    m = re.search(r"(?:production|yield)\s+of\s+([a-z\s]+)\s+(?:in|from)\s+([a-z\s]+)", q)
    if m:
        crop, state = m.groups()
        sql = f"""
        SELECT district, SUM(production_mt) as total_prod
        FROM crop
        WHERE LOWER(crop) LIKE '%{crop.strip()}%' AND LOWER(state) LIKE '%{state.strip()}%'
        GROUP BY district
        ORDER BY total_prod DESC;
        """
        return {"intent": "crop_production_state", "sql": sql, "crop": crop.strip().title(), "state": state.strip().title()}
    
    # Pattern 3: Rainfall comparison
    m = re.search(r"rainfall\s+(?:in|of|for)\s+([a-z\s]+)", q)
    if m:
        state = m.group(1)
        sql = f"""
        SELECT year, AVG(annual) as avg_rainfall
        FROM rainfall
        WHERE LOWER(state) LIKE '%{state.strip()}%'
        GROUP BY year
        ORDER BY year DESC
        LIMIT 10;
        """
        return {"intent": "rainfall_trend", "sql": sql, "state": state.strip().title()}
    
    # Pattern 4: Districts with highest/lowest production
    m = re.search(r"(?:district|region).*(?:highest|most|maximum)\s+([a-z\s]+)", q)
    if m:
        crop = m.group(1)
        sql = f"""
        SELECT state, district, SUM(production_mt) as total_prod
        FROM crop
        WHERE LOWER(crop) LIKE '%{crop.strip()}%'
        GROUP BY state, district
        ORDER BY total_prod DESC
        LIMIT 10;
        """
        return {"intent": "top_districts", "sql": sql, "crop": crop.strip().title()}
    
    # Pattern 5: Total production by crop
    m = re.search(r"total\s+production\s+of\s+([a-z\s]+)", q)
    if m:
        crop = m.group(1)
        sql = f"""
        SELECT state, SUM(production_mt) as total_prod
        FROM crop
        WHERE LOWER(crop) LIKE '%{crop.strip()}%'
        GROUP BY state
        ORDER BY total_prod DESC;
        """
        return {"intent": "total_by_crop", "sql": sql, "crop": crop.strip().title()}
    
    return {"intent": "unknown"}

def execute_query(plan, con):
    """Execute the query plan and return results"""
    if plan["intent"] == "unknown":
        return None, "I couldn't understand your question. Try asking about:\n- Top crops in a state\n- Production of a specific crop\n- Rainfall trends\n- Districts with highest production"
    
    try:
        result = con.execute(plan["sql"]).df()
        return result, None
    except Exception as e:
        return None, f"Error executing query: {str(e)}"

def create_visualizations(result, plan):
    """Create appropriate visualizations based on query intent"""
    
    if plan["intent"] == "top_crops_state":
        fig = px.bar(
            result, 
            x='crop', 
            y='total_prod',
            title=f"Top {plan['n']} Crops in {plan['state']}",
            labels={'total_prod': 'Production (MT)', 'crop': 'Crop'},
            color='total_prod',
            color_continuous_scale='Greens'
        )
        fig.update_layout(showlegend=False)
        return fig
    
    elif plan["intent"] == "crop_production_state":
        fig = px.bar(
            result.head(15), 
            x='district', 
            y='total_prod',
            title=f"{plan['crop']} Production by District in {plan['state']}",
            labels={'total_prod': 'Production (MT)', 'district': 'District'},
            color='total_prod',
            color_continuous_scale='Viridis'
        )
        fig.update_xaxis(tickangle=-45)
        return fig
    
    elif plan["intent"] == "rainfall_trend":
        fig = px.line(
            result, 
            x='year', 
            y='avg_rainfall',
            title=f"Rainfall Trend in {plan['state']} (Last 10 Years)",
            labels={'avg_rainfall': 'Average Rainfall (mm)', 'year': 'Year'},
            markers=True
        )
        fig.update_traces(line_color='#3498db', line_width=3)
        return fig
    
    elif plan["intent"] == "top_districts":
        fig = px.bar(
            result, 
            x='district', 
            y='total_prod',
            color='state',
            title=f"Top Districts for {plan['crop']} Production",
            labels={'total_prod': 'Production (MT)', 'district': 'District'},
            barmode='group'
        )
        fig.update_xaxis(tickangle=-45)
        return fig
    
    elif plan["intent"] == "total_by_crop":
        fig = px.pie(
            result.head(10), 
            values='total_prod', 
            names='state',
            title=f"State-wise {plan['crop']} Production Distribution",
            hole=0.4
        )
        return fig
    
    return None

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Project Samarth</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Agricultural Data Intelligence & Q&A System</p>', unsafe_allow_html=True)
    
    # Load data
    df_crop, df_rain = load_data()
    
    if df_crop is None or df_rain is None:
        st.stop()
    
    # Initialize database
    con = init_db()
    setup_database(con, df_crop, df_rain)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/wheat.png", width=100)
        st.title("📊 Data Overview")
        
        st.metric("Total Crop Records", f"{len(df_crop):,}")
        st.metric("Total Rainfall Records", f"{len(df_rain):,}")
        st.metric("States Covered", df_crop['state'].nunique())
        st.metric("Crops Tracked", df_crop['crop'].nunique())
        
        st.markdown("---")
        st.markdown("### 📖 Sample Questions")
        st.markdown("""
        - Top 5 crops in Himachal Pradesh
        - Production of wheat in Himachal Pradesh
        - Rainfall in Maharashtra
        - District with highest wheat production
        - Total production of rice
        """)
        
        st.markdown("---")
        st.markdown("### 📚 Data Sources")
        st.info("**Crop Production**: data.gov.in (2019-20)\n\n**Rainfall Data**: IMD Subdivision Data (1901-2017)")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query System", "📊 Analytics Dashboard", "📈 Visualizations", "📋 Raw Data"])
    
    with tab1:
        st.markdown("### 💬 Ask Questions About Indian Agriculture")
        
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., Top 5 crops in Himachal Pradesh",
            help="Ask natural language questions about crop production and rainfall data"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        with col2:
            if st.button("🔄 Clear", use_container_width=True):
                st.rerun()
        
        if search_button and query:
            with st.spinner("Analyzing your question..."):
                plan = samarth_plan(query, con)
                result, error = execute_query(plan, con)
                
                if error:
                    st.error(error)
                elif result is not None and not result.empty:
                    st.success(f"Found {len(result)} results!")
                    
                    # Display results
                    st.markdown("### 📊 Results")
                    st.dataframe(result, use_container_width=True)
                    
                    # Create visualization
                    fig = create_visualizations(result, plan)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Citation
                    st.markdown("""
                    <div class="citation-box">
                    <strong>📚 Data Sources:</strong><br>
                    • Crop Production: Production Under Different Crops during 2019-20, data.gov.in<br>
                    • Rainfall Data: Sub Division IMD 2017, data.gov.in
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Download option
                    csv = result.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"samarth_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No results found. Try rephrasing your question.")
    
    with tab2:
        st.markdown("### 📊 Analytics Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_production = df_crop['production_mt'].sum()
            st.metric("Total Production", f"{total_production/1e6:.2f}M MT")
        
        with col2:
            avg_rainfall = df_rain['annual'].mean()
            st.metric("Avg Annual Rainfall", f"{avg_rainfall:.1f} mm")
        
        with col3:
            top_crop = df_crop.groupby('crop')['production_mt'].sum().idxmax()
            st.metric("Top Crop", top_crop)
        
        with col4:
            years_covered = df_rain['year'].max() - df_rain['year'].min()
            st.metric("Years of Data", f"{years_covered}")
        
        # Top crops chart
        st.markdown("#### 🌾 Top 10 Crops by Production")
        top_crops = df_crop.groupby('crop')['production_mt'].sum().nlargest(10).reset_index()
        fig_crops = px.bar(
            top_crops,
            x='production_mt',
            y='crop',
            orientation='h',
            color='production_mt',
            color_continuous_scale='Greens',
            labels={'production_mt': 'Production (MT)', 'crop': 'Crop'}
        )
        fig_crops.update_layout(showlegend=False)
        st.plotly_chart(fig_crops, use_container_width=True)
        
        # State-wise analysis
        st.markdown("#### 🗺️ State-wise Production Distribution")
        state_prod = df_crop.groupby('state')['production_mt'].sum().reset_index()
        fig_states = px.pie(
            state_prod,
            values='production_mt',
            names='state',
            title="Production by State",
            hole=0.4
        )
        st.plotly_chart(fig_states, use_container_width=True)
    
    with tab3:
        st.markdown("### 📈 Interactive Visualizations")
        
        # Rainfall trends
        st.markdown("#### 🌧️ Rainfall Trends Over Time")
        selected_states = st.multiselect(
            "Select states to compare:",
            options=sorted(df_rain['state'].unique()),
            default=[df_rain['state'].unique()[0]] if len(df_rain['state'].unique()) > 0 else []
        )
        
        if selected_states:
            rain_data = df_rain[df_rain['state'].isin(selected_states)]
            rain_yearly = rain_data.groupby(['year', 'state'])['annual'].mean().reset_index()
            
            fig_rain = px.line(
                rain_yearly,
                x='year',
                y='annual',
                color='state',
                title="Annual Rainfall Comparison",
                labels={'annual': 'Rainfall (mm)', 'year': 'Year'}
            )
            st.plotly_chart(fig_rain, use_container_width=True)
        
        # Crop comparison
        st.markdown("#### 🌽 Crop Production Comparison")
        selected_crops = st.multiselect(
            "Select crops to compare:",
            options=sorted(df_crop['crop'].unique()),
            default=[df_crop['crop'].unique()[0]] if len(df_crop['crop'].unique()) > 0 else []
        )
        
        if selected_crops:
            crop_data = df_crop[df_crop['crop'].isin(selected_crops)]
            crop_state = crop_data.groupby(['state', 'crop'])['production_mt'].sum().reset_index()
            
            fig_crop_comp = px.bar(
                crop_state,
                x='state',
                y='production_mt',
                color='crop',
                title="State-wise Crop Production",
                labels={'production_mt': 'Production (MT)', 'state': 'State'},
                barmode='group'
            )
            st.plotly_chart(fig_crop_comp, use_container_width=True)
    
    with tab4:
        st.markdown("### 📋 Raw Data Explorer")
        
        data_option = st.selectbox("Select dataset:", ["Crop Production", "Rainfall Data"])
        
        if data_option == "Crop Production":
            st.markdown("#### 🌾 Crop Production Data")
            st.dataframe(df_crop, use_container_width=True)
            
            csv_crop = df_crop.to_csv(index=False)
            st.download_button(
                "📥 Download Crop Data",
                csv_crop,
                "crop_production.csv",
                "text/csv"
            )
        else:
            st.markdown("#### 🌧️ Rainfall Data")
            st.dataframe(df_rain, use_container_width=True)
            
            csv_rain = df_rain.to_csv(index=False)
            st.download_button(
                "📥 Download Rainfall Data",
                csv_rain,
                "rainfall_data.csv",
                "text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
        <p><strong>Project Samarth</strong> - AI-Powered Agricultural Data Integration</p>
        <p>Built with ❤️ using Streamlit | Data from data.gov.in</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()