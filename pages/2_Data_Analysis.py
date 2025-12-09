# pages/2_Data_Analysis.py

import streamlit as st
import pandas as pd
import os
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from google import genai
from io import StringIO
import plotly.express as px
import statsmodels.api as sm # Keep import to satisfy plotly trendline dependency

# ----------------------------------------------------------------------------------
# CRITICAL: Define constants
# ----------------------------------------------------------------------------------
SAMPLE_SIZE_FOR_PROFILE = 5000
MAX_ROWS_FOR_SCATTER = 10000
MAX_ROWS_FOR_CLUSTERING = 10000
MAX_ROWS_FOR_FULL_CHART = 5000
CHART_SAMPLE_SIZE = 1000


# --- Configuration & Initialization ---
st.set_page_config(layout="wide", page_title="Datafy: Analysis")

# =================================================================
# API KEY SETUP - Use st.secrets to read from secrets.toml
# =================================================================
# This is the correct way to read a secret in Streamlit
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") 
GEMINI_AVAILABLE = False
gemini_client = None

if GEMINI_API_KEY is None:
    # This warning will appear if the file is not found or the key is not set
    st.warning("⚠️ **Gemini API Key Not Set:** AI features (Automated Profile, AI Insights) are disabled. Set `GEMINI_API_KEY` in your `.streamlit/secrets.toml` file to enable.")
else:
    # Initialize Gemini Client using the key
    try:
        # Pass the key explicitly to the Client initialization
        gemini_client = genai.Client(api_key=GEMINI_API_KEY) 
        GEMINI_AVAILABLE = True
    except Exception as e:
        # If the key exists but is invalid, this will catch the error
        st.warning(f"⚠️ **Gemini API Error:** {e}. AI features are disabled.")
        gemini_client = None


# Session State Initialization (for analysis page only)
if 'top_drivers' not in st.session_state: st.session_state.top_drivers = None
if 'forecast_data' not in st.session_state: st.session_state.forecast_data = None
if 'clustering_data' not in st.session_state: st.session_state.clustering_data = None
if 'analysis_objective' not in st.session_state: st.session_state.analysis_objective = "Identify the key factors that drive the selected Target Metric."
if 'profile' not in st.session_state: st.session_state.profile = None
if 'target_metric' not in st.session_state: st.session_state.target_metric = None
if 'forecast_ran' not in st.session_state: st.session_state.forecast_ran = False
if 'clustering_ran' not in st.session_state: st.session_state.clustering_ran = False
if 'ai_insight' not in st.session_state: st.session_state.ai_insight = None


# --- Core Analysis Functions ---

@st.cache_data(show_spinner="Generating Automated Profile...")
def generate_automated_profile(df, gemini_available, _gemini_client_obj):
    """Generates an initial data profile using Gemini."""
    if not gemini_available or _gemini_client_obj is None:
        return "⚠️ **Gemini API not available.** Set the `GEMINI_API_KEY` environment variable to use this feature."
    
    # Sample for large datasets to speed up profiling
    df_for_profile = df
    if len(df) > SAMPLE_SIZE_FOR_PROFILE:
        df_for_profile = df.sample(n=SAMPLE_SIZE_FOR_PROFILE, random_state=42)
    
    buffer = StringIO()
    df_for_profile.info(buf=buffer)
    info_str = buffer.getvalue()
    desc_stats = df_for_profile.describe(include='all').to_markdown()

    profile_prompt = f"""
    You are an expert Data Profiler. Analyze the following DataFrame structure and summary statistics.
    1. Identify all data types (Numeric, Categorical, Date).
    2. Note any potential issues (missing values, columns with too many unique values).
    3. Propose a primary Target Metric (numerical) and a primary Date Column.
    4. Provide a 1-sentence summary of the dataset's purpose.

    DataFrame Info: {info_str}
    Descriptive Statistics: {desc_stats}

    Generate the profile as a clean, markdown-formatted report.
    """
    try:
        response = _gemini_client_obj.models.generate_content(
            model='gemini-2.5-flash',
            contents=profile_prompt
        )
        return response.text
    except Exception as e:
        return f"Error generating automated profile: {e}"


@st.cache_data(show_spinner="Running Driver Analysis...")
def run_driver_analysis(df, target_metric):
    """Identifies top correlating factors for the target metric."""
    
    numeric_df = df.select_dtypes(include=np.number).copy()
    
    if numeric_df.empty or target_metric not in numeric_df.columns:
        return None, "Target Metric not found or no other numeric columns available."

    corr_matrix = numeric_df.corr(numeric_only=True)
    target_corr = corr_matrix[target_metric].sort_values(ascending=False).drop(target_metric)
    top_drivers = target_corr[target_corr.abs() > 0.1].head(10).to_markdown()
    
    top_cols = target_corr.abs().sort_values(ascending=False).head(3).index.tolist()
    
    # Sample data for scatter plots if dataset is large
    df_for_plots = df
    if len(df) > MAX_ROWS_FOR_SCATTER:
        df_for_plots = df.sample(n=MAX_ROWS_FOR_SCATTER, random_state=42)
    
    plots = {}
    for col in top_cols:
        if col in df.columns:
            # Requires statsmodels to run trendline="ols"
            fig = px.scatter(df_for_plots, x=col, y=target_metric, trendline="ols",
                             title=f'{col} vs. {target_metric} (Corr: {target_corr.loc[col]:.2f})')
            plots[col] = fig
            
    return {"markdown_table": top_drivers, "plots": plots}, None


@st.cache_data(show_spinner="Running Time Series Forecast...")
def run_prophet_forecast(df, date_col, target_metric, periods):
    """Runs a Prophet time series forecast."""
    df_ts = df[[date_col, target_metric]].copy().rename(columns={date_col: 'ds', target_metric: 'y'})
    
    df_ts.dropna(subset=['ds', 'y'], inplace=True)
    
    df_ts['ds'] = pd.to_datetime(df_ts['ds'])
    df_ts['y'] = pd.to_numeric(df_ts['y'])

    df_ts = df_ts.groupby('ds')['y'].sum().reset_index()

    if len(df_ts) < 5:
        return None, "Not enough historical data (at least 5 unique time points) remaining after cleaning to run a forecast."
    
    try:
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(df_ts)

        future = model.make_future_dataframe(periods=periods, freq='D')
        forecast = model.predict(future)
        
        fig = plot_plotly(model, forecast, xlabel=date_col, ylabel=target_metric)
        fig.update_layout(title=f'Time Series Forecast for {target_metric}')
        
        forecast_summary = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        
        return {"plot": fig, "summary_markdown": forecast_summary.to_markdown(index=False)}, None

    except Exception as e:
        return None, f"Forecasting Error: {e}"


@st.cache_data(show_spinner="Running K-Means Clustering...")
def run_kmeans_clustering(df, features, n_clusters):
    """Runs K-Means clustering on selected features."""
    
    if not features:
        return None, "Please select at least one numeric feature for clustering."

    # Sample for large datasets - K-Means is O(n) and slow on 40k+ rows
    df_for_clustering = df
    sampled = False
    if len(df) > MAX_ROWS_FOR_CLUSTERING:
        df_for_clustering = df.sample(n=MAX_ROWS_FOR_CLUSTERING, random_state=42)
        sampled = True

    # Fill missing values with the mean to prevent data loss.
    df_cluster = df_for_clustering.select_dtypes(include=np.number)[features].copy()
    df_cluster = df_cluster.fillna(df_cluster.mean())
    
    if df_cluster.empty:
        return None, "Data is empty after selection."

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_cluster)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    df_with_cluster = df_for_clustering.copy()
    df_with_cluster['Cluster_ID'] = kmeans.fit_predict(scaled_features)
    
    # FIX: Explicitly round the mean values before calling to_markdown()
    cluster_summary = df_with_cluster.groupby('Cluster_ID')[features].mean().round(2).to_markdown()
    
    plot = None
    if len(features) >= 2:
        fig = px.scatter(df_with_cluster, x=features[0], y=features[1], color='Cluster_ID',
                         title=f'K-Means Clustering ({n_clusters} Clusters)', hover_data=features)
        plot = fig
    
    result = {"summary_markdown": cluster_summary, "plot": plot, "sampled": sampled}
    return result, None


def run_ai_insights(analysis_type, target_metric):
    """Generates final AI insights based on the analysis results."""
    
    report_context = f"## Analysis Summary\n**Target Metric:** {target_metric}\n\n"
    
    if analysis_type == "Driver Analysis" and st.session_state.top_drivers:
        report_context += f"### Driver Analysis Results\n{st.session_state.top_drivers['markdown_table']}\n\n"
    elif analysis_type == "Time Series Forecast" and st.session_state.forecast_data:
        report_context += f"### Forecast Results\n{st.session_state.forecast_data['summary_markdown']}\n\n"
    elif analysis_type == "Clustering" and st.session_state.clustering_data:
        report_context += f"### Clustering Results\n{st.session_state.clustering_data['summary_markdown']}\n\n"
        
    ai_prompt = f"""
    You are a Senior Data Analyst.
    Analysis Type: {analysis_type}
    Objective: {st.session_state.analysis_objective}

    Data:
    {report_context}
    
    Provide:
    1. **Key Insight:** One distinct, non-obvious finding.
    2. **Trend/Pattern:** Describe the direction or grouping found.
    3. **Recommendation:** One concrete business action to take based on the insight.
    
    Keep it concise and professional.
    """

    with st.spinner(f"🤖 Analyzing {analysis_type}..."):
        try:
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=ai_prompt
            )
            return response.text
        except Exception as e:
            return f"Error generating AI insights: {e}"


# ==============================================================================
# MAIN DASHBOARD CONTENT
# ==============================================================================

st.title("💡 2. Data Analysis")
st.markdown("### Step 3: Run advanced models and generate AI insights.")

# --- DEPENDENCY CHECK ---
if 'cleaned_df' not in st.session_state or st.session_state.cleaned_df is None or st.session_state.cleaned_df.empty:
    st.warning("⚠️ **No Filtered Data Found.** Please ensure you have loaded data on the **Launch** page and completed the filtering on the **1 Data Filtering** page.")
    st.stop()

df_final = st.session_state.cleaned_df
categorical_cols = df_final.select_dtypes(include=['object', 'bool']).columns.tolist()
numeric_cols = df_final.select_dtypes(include=['number']).columns.tolist()

if not numeric_cols:
    st.error("The filtered dataset contains no numeric columns for analysis.")
    st.stop()

# 1. VISUAL SUMMARY
# ----------------------------------------------------------------------------------
st.header("1. 📊 Visual Summary")

# Sample data for charts if dataset is large
df_for_charts = df_final
if len(df_final) > MAX_ROWS_FOR_FULL_CHART:
    df_for_charts = df_final.sample(n=CHART_SAMPLE_SIZE, random_state=42)
    st.info(f"ℹ️ Large dataset detected. Sampling {CHART_SAMPLE_SIZE:,} rows for chart visualization.")

chart_type = st.selectbox("Select Chart Type:", ['Bar Chart', 'Line Chart', 'Histogram', 'Pie Chart'])

col1, col2 = st.columns(2)
with col1:
    default_x = categorical_cols[0] if categorical_cols else df_for_charts.columns[0]
    chart_x = st.selectbox("X-Axis (Grouping)", df_for_charts.columns, index=df_for_charts.columns.get_loc(default_x) if default_x in df_for_charts.columns else 0)
with col2:
    chart_y = st.selectbox("Y-Axis (Value)", numeric_cols, index=0)

if chart_type == 'Bar Chart':
    if chart_x in df_for_charts.columns and chart_y in df_for_charts.columns:
        st.bar_chart(df_for_charts.groupby(chart_x)[chart_y].sum(), use_container_width=True)
elif chart_type == 'Line Chart':
    try:
        if chart_x in df_for_charts.columns and chart_y in df_for_charts.columns:
            st.line_chart(df_for_charts.set_index(chart_x)[chart_y], use_container_width=True)
    except:
        st.warning("Cannot plot line chart: X-Axis column must be unique or sortable.")
elif chart_type == 'Histogram':
    if chart_y in df_for_charts.columns:
        st.bar_chart(df_for_charts[chart_y].value_counts().sort_index(), use_container_width=True)
elif chart_type == 'Pie Chart':
    if chart_x in df_for_charts.columns and chart_y in df_for_charts.columns:
        pie_data = df_for_charts.groupby(chart_x)[chart_y].sum().reset_index()
        fig = px.pie(pie_data, values=chart_y, names=chart_x)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------------------------------
# 2. AUTOMATED PROFILE
# ----------------------------------------------------------------------------------
st.header("2. 📋 Automated Profile")

if st.button("Generate Automated Data Profile"):
    st.session_state.profile = generate_automated_profile(
        df=df_final,
        gemini_available=GEMINI_AVAILABLE,
        _gemini_client_obj=gemini_client 
    )

if st.session_state.profile:
    st.markdown(st.session_state.profile)

st.markdown("---")


# ----------------------------------------------------------------------------------
# 3. SET OBJECTIVE & ANALYSIS
# ----------------------------------------------------------------------------------
st.header("3. 🎯 Objective & Analysis")

st.subheader("Set Analysis Objective")
st.session_state.analysis_objective = st.text_area("Analysis Goal:", st.session_state.analysis_objective, height=68)

# Target Metric Selection
if st.session_state.target_metric not in numeric_cols:
    st.session_state.target_metric = numeric_cols[0] if numeric_cols else None

st.session_state.target_metric = st.selectbox(
    "Select Target Metric:", 
    numeric_cols, 
    index=numeric_cols.index(st.session_state.target_metric) if st.session_state.target_metric in numeric_cols else 0
)

st.markdown("#### Select Advanced Analysis")
analysis_options = {
    "Select": "Select Analysis Type...",
    "Driver Analysis": "Driver Analysis (Correlation)",
    "Time Series Forecast": "Time Series Forecast",
    "Clustering": "Clustering (Segmentation)"
}
selected_analysis = st.selectbox("Analysis Type:", options=list(analysis_options.keys()))

# --- Dynamic Analysis Controls ---
if selected_analysis == "Driver Analysis":
    st.info(f"Finding key drivers for **{st.session_state.target_metric}**.")
    if st.button("Run Driver Analysis"):
        st.session_state.top_drivers, error = run_driver_analysis(df_final, st.session_state.target_metric)
        if error: st.error(error)
        else: st.success("Analysis Complete!")

elif selected_analysis == "Time Series Forecast":
    date_cols = df_final.select_dtypes(include=['datetime64']).columns.tolist()
    if date_cols:
        date_col = st.selectbox("Date Column:", date_cols)
        periods = st.slider("Forecast Days:", 7, 365, 30)
        if st.button("Run Forecast"):
            st.session_state.forecast_data, error = run_prophet_forecast(df_final, date_col, st.session_state.target_metric, periods)
            if error: 
                st.error(error)
                st.session_state.forecast_ran = False
            else: 
                st.session_state.forecast_ran = True
                st.success("Analysis Complete!")
    else:
        st.warning("No Date columns found in the filtered data.")

elif selected_analysis == "Clustering":
    cluster_features = st.multiselect("Features:", numeric_cols, default=numeric_cols[:3])
    n_clusters = st.slider("Clusters (K):", 2, 8, 3)
    if st.button("Run Clustering"):
        st.session_state.clustering_data, error = run_kmeans_clustering(df_final, cluster_features, n_clusters)
        if error: 
            st.error(error)
            st.session_state.clustering_ran = False
        else: 
            st.session_state.clustering_ran = True
            st.success("Analysis Complete!")

# --- Results Display ---
if selected_analysis == "Driver Analysis" and st.session_state.top_drivers:
    st.markdown("#### Correlation Drivers")
    st.markdown(st.session_state.top_drivers['markdown_table'])
    if st.session_state.top_drivers['plots']:
        # Only show the first plot for simplicity
        st.plotly_chart(list(st.session_state.top_drivers['plots'].values())[0], use_container_width=True)

elif selected_analysis == "Time Series Forecast" and st.session_state.forecast_ran:
    st.markdown("#### Forecast Plot")
    st.plotly_chart(st.session_state.forecast_data['plot'], use_container_width=True)
    
elif selected_analysis == "Clustering" and st.session_state.clustering_ran:
    st.markdown("#### Cluster Profile")
    
    # Show sampling notification if data was sampled
    if st.session_state.clustering_data.get('sampled', False):
        st.info(f"ℹ️ Dataset sampled to {MAX_ROWS_FOR_CLUSTERING:,} rows for clustering performance.")
    
    st.markdown(st.session_state.clustering_data['summary_markdown'])
    if st.session_state.clustering_data['plot']:
        st.plotly_chart(st.session_state.clustering_data['plot'], use_container_width=True)

st.markdown("---")


# ----------------------------------------------------------------------------------
# 4. AI SUMMARY
# ----------------------------------------------------------------------------------
st.header("4. 💡 AI Agentic Summary")

analysis_ready = (
    (selected_analysis == "Driver Analysis" and st.session_state.top_drivers) or
    (selected_analysis == "Time Series Forecast" and st.session_state.forecast_ran) or
    (selected_analysis == "Clustering" and st.session_state.clustering_ran)
)

if analysis_ready:
    if st.button(f"Generate AI Insights for {selected_analysis}"):
        if GEMINI_AVAILABLE:
            st.session_state.ai_insight = run_ai_insights(selected_analysis, st.session_state.target_metric)
        else:
            st.error("AI features are disabled due to missing Gemini API key.")
else:
    st.info("Run an analysis above to unlock AI insights.")

if st.session_state.ai_insight:
    st.success("AI Analysis Complete")
    st.markdown(st.session_state.ai_insight)