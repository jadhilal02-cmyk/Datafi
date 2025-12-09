# config.py - Performance Configuration for DataFy

"""
Performance thresholds and limits for handling large datasets.
Adjust these values based on your deployment environment and client needs.
"""

# === Data Loading Limits ===
MAX_ROWS_DEFAULT = None  # Set to a number like 50000 to limit by default
SUGGESTED_ROW_LIMIT = 50000  # Recommended limit for UI checkbox

# === Filter Page Limits ===
MAX_UNIQUE_VALUES_FOR_FILTER = 1000  # Skip categorical columns with more unique values than this
MAX_CATEGORICAL_COLUMNS_TO_ANALYZE = 20  # Limit number of categorical columns to prevent slow loading

# === Chart & Visualization Limits ===
MAX_ROWS_FOR_FULL_CHART = 10000  # Sample down if more rows than this for charts
CHART_SAMPLE_SIZE = 10000  # Number of rows to sample for large datasets

# === Analysis Limits ===
MAX_ROWS_FOR_CLUSTERING = 10000  # Warn and suggest sampling for K-Means above this
SAMPLE_SIZE_FOR_PROFILE = 5000  # Sample size for automated profiling on large datasets
MAX_ROWS_FOR_SCATTER = 5000  # Sample size for scatter plots in driver analysis

# === UI Preview Limits ===
PREVIEW_ROWS_DEFAULT = 10  # Default number of rows to show in dataframe previews
MAX_PREVIEW_ROWS = 100  # Maximum rows allowed in preview

# === Performance Warnings ===
WARN_THRESHOLD_ROWS = 20000  # Show performance warnings above this row count
