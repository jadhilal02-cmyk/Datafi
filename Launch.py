# Launch.py

import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
from config import MAX_ROWS_DEFAULT, SUGGESTED_ROW_LIMIT, WARN_THRESHOLD_ROWS

# --- Configuration ---
st.set_page_config(
    layout="wide", 
    page_title="Datafy: Launch",
    initial_sidebar_state="expanded" 
)

# Initialize a default random number generator for consistency
rng = np.random.default_rng(42)

# --- GLOBAL SESSION STATE INITIALIZATION ---
# Initialize necessary state variables if they don't exist
if 'df' not in st.session_state: st.session_state.df = None # Raw Data
if 'cleaned_df' not in st.session_state: st.session_state.cleaned_df = None # Filtered Data
if 'last_loaded_sample' not in st.session_state: st.session_state.last_loaded_sample = None
if 'data_mode' not in st.session_state: st.session_state.data_mode = 'Use Built-in Sample Data'
if 'sample_select' not in st.session_state: st.session_state.sample_select = "Sales Data (1000 rows)"


# --- DATA CREATION FUNCTIONS (Ensure 1000 rows are created reliably) ---

def create_sales_sample():
    """Creates a large Sales DataFrame (1000 rows) with date, region, product, revenue."""
    n_rows = 1000
    indices_to_nan = rng.choice(n_rows, size=20, replace=False)
    
    data = {
        'Sale_ID': range(n_rows),
        'Sale_Date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=n_rows, freq='D')),
        'Region': rng.choice(['North', 'South', 'East', 'West'], n_rows),
        'Product_Category': rng.choice(['Electronics', 'Apparel', 'Home Goods', 'Beauty', 'Sports'], n_rows),
        'Price': rng.integers(20, 500, n_rows) + np.arange(n_rows) * 0.1,
        'Units_Sold': rng.integers(1, 100, n_rows),
        'Customer_Rating': rng.choice([1, 2, 3, 4, 5], n_rows, p=[0.05, 0.1, 0.15, 0.4, 0.3])
    }
    df = pd.DataFrame(data)
    df['Revenue'] = df['Price'] * df['Units_Sold']
    
    for col in ['Price', 'Customer_Rating']:
        df.loc[indices_to_nan, col] = np.nan
        
    return df

def create_team_performance_sample():
    """Creates a sample DataFrame for team performance clustering."""
    n_rows = 500
    data = {
        'Employee_ID': range(n_rows),
        'Team': rng.choice(['A-Tech', 'B-Sales', 'C-Support', 'D-Marketing'], n_rows),
        'Tenure_Years': rng.integers(1, 15, n_rows),
        'Projects_Completed': rng.integers(5, 50, n_rows),
        'Error_Rate': rng.uniform(0.01, 0.15, n_rows),
        'Performance_Score': rng.integers(60, 100, n_rows)
    }
    df = pd.DataFrame(data)
    return df

def create_car_maintenance_sample():
    """Creates a sample DataFrame for forecasting or driver analysis on cost."""
    n_rows = 300
    data = {
        'Service_Date': pd.to_datetime(pd.date_range(start='2022-01-01', periods=n_rows, freq='M')),
        'Vehicle_Type': rng.choice(['Sedan', 'SUV', 'Truck', 'Hatchback'], n_rows),
        'Mileage': rng.integers(10000, 150000, n_rows),
        'Service_Category': rng.choice(['Oil Change', 'Tire Repair', 'Engine', 'Brake'], n_rows),
        'Repair_Cost': rng.integers(50, 5000, n_rows)
    }
    df = pd.DataFrame(data)
    return df


@st.cache_data(show_spinner="Cleaning and Preparing Data...")
def load_and_clean_data(data_source, max_rows=None):
    """Loads data, handles types, and performs initial cleaning."""
    if isinstance(data_source, pd.DataFrame):
        df = data_source.copy()
    else: 
        uploaded_file = data_source
        try:
            if uploaded_file.name.endswith('.csv'):
                # Optimize CSV reading for large files
                df = pd.read_csv(
                    uploaded_file, 
                    nrows=max_rows,
                    low_memory=False,  # Prevents dtype warnings on large files
                    encoding_errors='replace'  # Handle encoding issues gracefully
                )
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file, nrows=max_rows)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return None
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None

    # Standardize column names
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '', col.strip().replace(' ', '_')) for col in df.columns]

    # Try to convert date columns
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['date', 'time', 'period']):
            try:
                # Removed deprecated infer_datetime_format parameter
                df[col] = pd.to_datetime(df[col], errors='coerce', format='mixed')
                if df[col].dt.year.max() < 1900 or df[col].isnull().all():
                     df[col] = df[col].astype(str)
            except:
                pass 

    # Drop fully empty columns/rows only
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    
    return df

# Function to create the sample DataFrame based on selection
def get_sample_df(sample_selection):
    if sample_selection == "Sales Data (1000 rows)":
        return create_sales_sample()
    elif sample_selection == "Team Performance (500 rows)":
        return create_team_performance_sample()
    elif sample_selection == "Car Maintenance (300 rows)":
        return create_car_maintenance_sample()
    return None

# --- PAGE LAYOUT ---

st.title("🚀 Datafy: Launch Pad")
st.markdown("### Step 1: Load your data to begin the analysis pipeline.")

# 1. DATA SELECTION
st.subheader("1. Data Source Selection")

data_mode = st.radio(
    "Select a Data Source Option:",
    ('Use Built-in Sample Data', 'Upload Your Own File'),
    key='data_mode_radio'
)

data_loaded = False

if data_mode == 'Use Built-in Sample Data':
    
    sample_selection = st.selectbox(
        "Choose a Sample Dataset:",
        ("Sales Data (1000 rows)", "Team Performance (500 rows)", "Car Maintenance (300 rows)"),
        key='sample_select'
    )
    
    if st.button(f"Load '{sample_selection}'"):
        with st.spinner(f"Generating and loading {sample_selection}..."):
            raw_df = get_sample_df(sample_selection)
            st.session_state.df = load_and_clean_data(raw_df)
            st.session_state.cleaned_df = st.session_state.df.copy() # Initialize cleaned_df
            st.session_state.last_loaded_sample = sample_selection
            st.success(f"Data Loaded: **{sample_selection}**")
            data_loaded = True
    elif st.session_state.df is not None and st.session_state.last_loaded_sample == sample_selection:
        st.info(f"Data already loaded: **{sample_selection}**")
        data_loaded = True

else: # Upload Your Own File
    # Add option to limit rows for large files
    limit_rows = st.checkbox(f"⚡ Limit to first {SUGGESTED_ROW_LIMIT:,} rows (recommended for large files)", value=False)
    max_rows_to_load = SUGGESTED_ROW_LIMIT if limit_rows else MAX_ROWS_DEFAULT
    
    uploaded_file = st.file_uploader("Upload CSV or Excel File", type=['csv', 'xlsx'])
    if uploaded_file:
        with st.spinner(f"Loading and cleaning {uploaded_file.name}..."):
            st.session_state.df = load_and_clean_data(uploaded_file, max_rows=max_rows_to_load)
            st.session_state.cleaned_df = st.session_state.df.copy() # Initialize cleaned_df
            st.session_state.last_loaded_sample = uploaded_file.name
            
            # Show appropriate warnings and success messages
            if limit_rows:
                st.warning(f"⚠️ Loaded first {SUGGESTED_ROW_LIMIT:,} rows only. Uncheck the limit option to load all data.")
            elif st.session_state.df is not None and len(st.session_state.df) > WARN_THRESHOLD_ROWS:
                st.info(f"ℹ️ Large dataset detected ({len(st.session_state.df):,} rows). Operations may take longer. Consider using the row limit option for faster testing.")
            
            st.success(f"File Uploaded: **{uploaded_file.name}**")
            data_loaded = True
        
st.markdown("---")

# 2. RAW DATA PREVIEW
st.subheader(f"2. Raw Data Preview")

if st.session_state.get('df') is None or st.session_state.df.empty:
    st.error("Please load or upload a dataset above to proceed.")
else:
    st.success(f"Raw Data Loaded: **{len(st.session_state.df)}** rows")
    st.markdown(f"**DataFrame Shape Check:** `{st.session_state.df.shape}`") 
    
    with st.expander("Click to expand raw data view", expanded=False):
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Next Step: Filter")
    st.info("👈 **Click on '1 Data Filtering' in the sidebar to clean and filter your data.**")