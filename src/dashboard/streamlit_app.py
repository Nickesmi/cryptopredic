import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add repo root to path so we can import src
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from src.config.settings import TIER1_SYMBOLS, FORECAST_HORIZONS
from src.data.fetch_market_data import run_pipeline

st.set_page_config(page_title="Crypto Alpha Engine", layout="wide")

st.title("Crypto Alpha Engine 🚀")
st.markdown("Forecast closing prices for Tier-1 crypto assets using XGBoost Direct Strategy.")

st.sidebar.header("Forecast Parameters")
symbol = st.sidebar.selectbox("Select Asset", TIER1_SYMBOLS)
horizon = st.sidebar.selectbox("Forecast Horizon (Days)", FORECAST_HORIZONS)

if st.sidebar.button("Run Forecast"):
    with st.spinner(f"Fetching data and running {horizon}d forecast for {symbol.upper()}..."):
        try:
            result = run_pipeline(symbol, horizon=horizon, save_data=False)
            
            st.success("Forecast completed successfully!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric(label=f"Predicted Price ({horizon}d)", value=f"${result.predicted_price:,.2f}")
            col2.metric(label="Validation MAE", value=f"${result.mae:,.2f}")
            col3.metric(label="Validation MAPE", value=f"{result.mape:.2f}%")
            
            st.info("Note: This is a point-in-time forecast based on the latest daily close.")
        except Exception as e:
            st.error(f"Error running pipeline: {str(e)}")
else:
    st.info("Select parameters in the sidebar and click 'Run Forecast' to generate a prediction.")
