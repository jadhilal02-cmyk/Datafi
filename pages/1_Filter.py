# 1_Data_Filtering.py (Final Optimized Version)

import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(layout="wide", page_title="Datafy: Filtering")

st.title("🔬 1. Data Filtering")
st.markdown("### Step 2: Prepare the data by applying filters.")

# --- DEPENDENCY CHECK ---
if st.session_state.get('df') is None or st.session_state.df.empty:
    st.warning("⚠️ **No Raw Data Found.** Please go back to the **Launch** page to load a file.")
    st.stop()

df_raw = st.session_state.df

# ----------------------------------------------------------------------
# CRITICAL FIX 1: CACHE THE INITIAL FILTER METADATA EXTRACTION
# This runs only once per file, preventing the script from recalculating 
# mins, maxes, and column types on every filter interaction.
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Analyzing data structure and ranges...")
def get_initial_data_info(df):
    """Calculates column types, unique values, and filter ranges once per DataFrame.
    
    OPTIMIZED for large datasets:
    - Filters out high-cardinality categorical columns (>1000 unique values)
    - Only calculates numeric ranges upfront
    - Defers categorical unique value calculation until needed
    """
    from config import MAX_UNIQUE_VALUES_FOR_FILTER
    
    all_categorical_cols = df.select_dtypes(include=['object', 'string', 'bool']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Calculate numeric ranges only (fast operation)
    numeric_ranges = {}
    for col in numeric_cols:
        # Calculate min/max only on valid data
        valid_data = df[col].dropna().loc[lambda x: np.isfinite(x)]
        if not valid_data.empty:
            numeric_ranges[col] = (float(valid_data.min()), float(valid_data.max()))
    
    # CRITICAL FIX: Filter out high-cardinality categorical columns
    # This prevents calculating unique values for ID columns with 40k+ unique values
    categorical_cols_filtered = []
    categorical_cardinality = {}
    
    for col in all_categorical_cols:
        n_unique = df[col].nunique()
        categorical_cardinality[col] = n_unique
        
        # Only include columns suitable for filtering
        if n_unique <= MAX_UNIQUE_VALUES_FOR_FILTER:
            categorical_cols_filtered.append(col)
    
    return categorical_cols_filtered, numeric_cols, numeric_ranges, categorical_cardinality


@st.cache_data(show_spinner="Loading filter options...")
def get_categorical_unique_values(df, column_name):
    """Lazy-load unique values for a specific categorical column.
    
    This is called only when user selects a column, not upfront for all columns.
    """
    return df[column_name].dropna().unique().tolist()

# Run the cached function to get filter metadata
categorical_cols, numeric_cols, numeric_ranges, categorical_cardinality = get_initial_data_info(df_raw)


st.subheader(f"Data Source: {st.session_state.last_loaded_sample or 'Uploaded File'} ({len(df_raw)} rows)")
st.markdown("---")

# CRITICAL FIX 2: Create the necessary copy only once at the start of filtering
df_filtered = df_raw.copy() 


# 1. FILTERING CONTROLS
st.subheader("1. Apply Filters")

col_filter1, col_filter2 = st.columns(2)

# Category Filter
with col_filter1:
    st.markdown("#### Categorical Filter")
    if categorical_cols:
        cat_col = st.selectbox("Select Category Column:", categorical_cols)
        
        # Show cardinality info
        cardinality = categorical_cardinality.get(cat_col, 0)
        st.caption(f"ℹ️ {cardinality} unique values in this column")
        
        # LAZY LOAD: Only get unique values when column is selected
        unique_vals = get_categorical_unique_values(df_raw, cat_col)
        
        selected_vals = st.multiselect("Select Values to Keep:", unique_vals, default=unique_vals)
        df_filtered = df_filtered[df_filtered[cat_col].isin(selected_vals)]
    else:
        from config import MAX_UNIQUE_VALUES_FOR_FILTER
        st.warning(f"⚠️ No categorical columns suitable for filtering found. Columns with >{MAX_UNIQUE_VALUES_FOR_FILTER} unique values are excluded (likely ID columns).")

# Numeric Filter
with col_filter2:
    st.markdown("#### Numeric Filter")
    if numeric_cols:
        num_col = st.selectbox("Select Numeric Column:", numeric_cols)
        
        # Use the pre-calculated, cached ranges
        if num_col in numeric_ranges:
            min_val, max_val = numeric_ranges[num_col]
            
            if min_val != max_val:
                # The slider uses the cached min/max, which speeds up the rerun
                slider_range = st.slider(f"Filter Range for {num_col}", min_val, max_val, (min_val, max_val))
                df_filtered = df_filtered[
                    (df_filtered[num_col] >= slider_range[0]) & 
                    (df_filtered[num_col] <= slider_range[1])
                ]
            else:
                st.info(f"Column '{num_col}' has a constant value.")
        else:
            st.info(f"Column '{num_col}' contains no valid numeric data.")
    else:
        st.info("No numeric columns found.")

# Save filtered data to global state for use by the analysis page
st.session_state.cleaned_df = df_filtered

st.markdown("---")

# 2. FINAL FILTERED PREVIEW
st.subheader("2. Final Filtered Data Preview")

if len(df_filtered) == 0:
     st.error("Filters have resulted in an empty dataset. Please adjust your filters above.")
     st.session_state.cleaned_df = None
else:
    # CRITICAL FIX 3: Limit the dataframe preview to prevent browser timeout on large renders
    preview_rows = 10 
    
    st.success(f"Ready for Analysis! **{len(df_filtered)}** rows remaining.")
    st.markdown(f"**DataFrame Shape Check:** `{df_filtered.shape}`") 
    
    with st.expander(f"View first {preview_rows} rows of filtered data", expanded=False):
        st.dataframe(df_filtered.head(preview_rows), use_container_width=True)
        
    st.info("👈 **Now click on '2 Data Analysis' in the sidebar to run the advanced analysis.**")