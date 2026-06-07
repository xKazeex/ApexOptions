import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Apex Options Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
API_URL = "http://localhost:8000/api"

# Custom CSS for dark theme look
st.markdown("""
<style>
    .report-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #00ff00;
    }
    .metric-label {
        font-size: 14px;
        color: #888;
    }
    .strategy-title {
        color: #ffaa00;
        font-size: 28px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def analyze_ticker(ticker, account_size):
    try:
        response = requests.post(f"{API_URL}/analyze", json={"ticker": ticker, "account_size": account_size})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None

def get_history():
    try:
        response = requests.get(f"{API_URL}/history")
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

# Sidebar
st.sidebar.title("Apex Options")
ticker_input = st.sidebar.text_input("Enter Ticker Symbol (e.g. TSLA, SPY)", "TSLA").upper()
account_size = st.sidebar.number_input("Account Size ($)", value=50000, step=1000)
analyze_btn = st.sidebar.button("Run Apex Analysis")

# Navigation
page = st.sidebar.selectbox("Navigate", ["🏠 Home", "📜 History", "ℹ️ About"])

if page == "🏠 Home":
    st.title("Apex Options Analytics Dashboard")
    
    if analyze_btn:
        with st.spinner(f"Analyzing {ticker_input} market microstructure..."):
            data = analyze_ticker(ticker_input, account_size)
            
            if data:
                report = data['report']
                metadata = data['metadata']
                
                # Metadata Row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Ticker", metadata['ticker'])
                with col2:
                    st.metric("Current Price", f"${metadata['current_price']:.2f}")
                with col3:
                    st.metric("IV Rank", f"{metadata['iv_rank']}%")
                with col4:
                    st.metric("Last Updated", datetime.fromisoformat(metadata['timestamp'].replace('Z', '')).strftime('%H:%M:%S'))
                
                st.markdown("---")
                
                # Strategy Report
                st.markdown(f"<div class='strategy-title'>Recommended Strategy: {report['recommended_strategy']}</div>", unsafe_allow_html=True)
                
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.subheader("Market Thesis")
                    st.write(report['market_thesis'])
                    
                    st.subheader("Execution Parameters")
                    params = report['execution_parameters']
                    
                    # Show strikes as a nice table or list
                    st.write(f"**Strategy:** {params['strategy']}")
                    st.write(f"**Expiration:** {params['expiration_dte']} DTE")
                    st.write(f"**Target Premium:** {params['target_premium']}")
                    
                    st.markdown("**Strikes:**")
                    strike_data = {
                        "Type": ["Sell Strikes", "Buy Strikes"],
                        "Strikes": [", ".join(map(str, params['sell_strikes'])), ", ".join(map(str, params['buy_strikes']))]
                    }
                    st.table(pd.DataFrame(strike_data))

                with col_right:
                    st.subheader("Risk Management")
                    risk = report['risk_management']
                    st.info(f"**Max Risk:** {risk['max_risk']}")
                    st.success(f"**Profit Target:** {risk['profit_target']}")
                    st.warning(f"**Stop Loss:** {risk['stop_loss_trigger']}")
                    
                    st.subheader("Caveats")
                    st.warning(report['caveats'])
    else:
        st.write("Enter a ticker symbol and account size in the sidebar and click 'Run Apex Analysis' to generate a report.")

elif page == "📜 History":
    st.title("Analysis History")
    history = get_history()
    if history:
        history_df = pd.DataFrame([
            {
                "Ticker": h['metadata']['ticker'],
                "Price": h['metadata']['current_price'],
                "IV Rank": h['metadata']['iv_rank'],
                "Strategy": h['report']['recommended_strategy'],
                "Timestamp": h['metadata']['timestamp']
            } for h in history
        ])
        st.dataframe(history_df, use_container_width=True)
    else:
        st.write("No recent analysis history.")

elif page == "ℹ️ About":
    st.title("About Apex Options Analytics")
    st.markdown("""
    ### Service Description
    A quantitative AI agent that analyzes market microstructure, volatility dynamics (IV/RV skew), and historical data to generate institutional-grade, risk-managed options premium-selling recommendations.
    
    ### Strategy Methodology
    Our engine focuses on volatility selling strategies:
    - **Iron Condors**: Neutral outlook with high IV rank.
    - **Credit Spreads**: Directional bias with premium collection.
    - **Short Strangles**: Pure volatility plays (high risk).
    
    We optimize for:
    - **IV Rank/Percentile**: Ensuring we sell rich premium.
    - **Historical Skew**: Selecting strikes with statistical edge.
    - **Risk/Reward**: Maintaining a minimum 1:3 risk/reward ratio on credit.
    """)
