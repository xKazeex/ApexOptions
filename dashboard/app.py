"""Apex Options Analytics — Streamlit Dashboard."""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Apex Options Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8001/api"

# Custom CSS
st.markdown("""
<style>
    .report-card { background-color: #1e1e1e; padding: 20px; border-radius: 10px; border: 1px solid #333; margin-bottom: 20px; }
    .metric-value { font-size: 24px; font-weight: bold; color: #00ff00; }
    .metric-label { font-size: 14px; color: #888; }
    .strategy-title { color: #ffaa00; font-size: 28px; font-weight: bold; }
    .section-label { color: #aaa; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin-top: 10px; }
    div[data-testid="metric-container"] { min-width: 100px; }
    div[data-testid="metric-container"] > label { white-space: nowrap; overflow: visible; }
    div[data-testid="metric-container"] > div { white-space: nowrap; overflow: visible; text-overflow: clip; }
</style>
""", unsafe_allow_html=True)

def analyze_ticker(ticker, account_size):
    try:
        resp = requests.post(f"{API_URL}/analyze", json={"ticker": ticker, "account_size": account_size}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Error: {resp.status_code} — {resp.text}")
        return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None

def get_history():
    try:
        resp = requests.get(f"{API_URL}/history", timeout=10)
        return resp.json() if resp.status_code == 200 else []
    except:
        return []

# Sidebar
st.sidebar.title("⚡ Apex Options")
ticker_input = st.sidebar.text_input("Ticker", "SPY").upper()
account_size = st.sidebar.number_input("Account Size ($)", value=50000, step=1000)
analyze_btn = st.sidebar.button("▶ Run Analysis", type="primary")
page = st.sidebar.selectbox("Navigate", ["🏠 Home", "📜 History", "ℹ️ About"])

if page == "🏠 Home":
    st.title("Apex Options Analytics")

    if analyze_btn:
        with st.spinner(f"Analyzing {ticker_input}..."):
            data = analyze_ticker(ticker_input, account_size)

        if data:
            r = data["report"]
            m = data["metadata"]
            p = r["execution_parameters"]
            rm = r["risk_management"]

            # --- Key metrics row ---
            cols = st.columns(7)
            exp_date = (datetime.now() + timedelta(days=p['expiration_dte'])).strftime("%m/%d")
            cols[0].metric("Ticker", m["ticker"])
            cols[1].metric("Price", f"${m['current_price']:.2f}")
            cols[2].metric("IV Rank", f"{m['iv_rank']}%")
            cols[3].metric("Expires", exp_date)
            cols[4].metric("Premium", p["target_premium"])
            cols[5].metric("Contracts", rm.get("contracts", 1))
            cols[6].metric("Risk %", rm.get("risk_pct", "N/A"))

            st.markdown("---")

            # --- Strategy title ---
            st.markdown(f"<div class='strategy-title'>📌 {r['recommended_strategy']}</div>", unsafe_allow_html=True)

            left, right = st.columns([3, 2])

            with left:
                st.markdown("<div class='section-label'>Market Thesis</div>", unsafe_allow_html=True)
                st.info(r["market_thesis"])

                st.markdown("<div class='section-label'>Strikes</div>", unsafe_allow_html=True)
                sd = {"Side": ["Sell", "Buy"], "Strike": [
                    ", ".join(str(x) for x in p["sell_strikes"]),
                    ", ".join(str(x) for x in p["buy_strikes"]),
                ]}
                st.dataframe(pd.DataFrame(sd), hide_index=True, use_container_width=True)

                # Trade Rationale (formerly Caveats)
                st.markdown("<div class='section-label'>Trade Rationale</div>", unsafe_allow_html=True)
                st.markdown(r["caveats"])

            with right:
                st.markdown("<div class='section-label'>Risk Management</div>", unsafe_allow_html=True)
                st.metric("Max Risk", rm["max_risk"])
                st.metric("Profit Target", rm["profit_target"])
                st.metric("Stop Loss", rm["stop_loss_trigger"])

                st.markdown("<div class='section-label'>Position Sizing</div>", unsafe_allow_html=True)
                st.metric("Account", f"${account_size:,.0f}")
                st.metric("Contracts", rm.get("contracts", 1))
                st.metric("Risk %", rm.get("risk_pct", "N/A"))

                st.markdown("<div class='section-label'>Limit Price</div>", unsafe_allow_html=True)
                if p.get("limit_price"):
                    st.success(f"**Recommended Limit:** {p['limit_price']}")
                if p.get("bid_ask_range"):
                    st.caption(f"🔄 Bid-Ask Range: {p['bid_ask_range']}")
    else:
        st.info("Enter a ticker in the sidebar and click **▶ Run Analysis**.")

elif page == "📜 History":
    st.title("Analysis History")
    history = get_history()
    if history:
        df = pd.DataFrame([
            {
                "Ticker": h["metadata"]["ticker"],
                "Price": f"${h['metadata']['current_price']:.2f}",
                "IV Rank": f"{h['metadata']['iv_rank']:.1f}%",
                "Strategy": h["report"]["recommended_strategy"],
                "Premium": h["report"]["execution_parameters"]["target_premium"],
                "Time": h["metadata"]["timestamp"][:19],
            }
            for h in history
        ])
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No analysis history yet.")

elif page == "ℹ️ About":
    st.title("About Apex Options Analytics")
    st.markdown("""
    **A quantitative AI agent** that analyzes market microstructure, volatility dynamics, and historical data to generate institutional-grade, risk-managed options premium-selling recommendations.

    ### Methodology
    - **Iron Condors** — Neutral outlook with high IV rank
    - **Credit Spreads** — Directional bias with premium collection
    - **Short Strangles** — Pure volatility plays (high risk)

    We optimize for IV Rank/Percentile, historical skew, and minimum 1:3 risk/reward ratios.
    """)

    st.markdown("---")
    st.subheader("📖 Field Legend")
    st.markdown("""
    | Field | Meaning |
    |-------|---------|
    | **IV Rank** | Where current IV sits in its 52-week range (0-100%). Higher = better for selling premium. |
    | **IV Percentile** | Percentage of days IV was lower than current over the past year. |
    | **IV/RV Ratio** | Implied Volatility ÷ Realized Volatility. >1.0 = options are rich vs actual movement. |
    | **Vol Regime** | Classification: crush opportunity / elevated / normal / low. |
    | **Term Structure** | Whether near-term or longer-term vol is higher (contango vs backwardation). |
    | **Skew** | Which side (puts/calls) has elevated implied volatility. |
    | **Opportunity Score** | Composite score 0-100 rating how favorable conditions are for selling premium. |
    | **DTE** | Days until option expiration. 30-60 DTE is the sweet spot for theta decay. |
    | **Premium Collected** | Total credit received per contract from the spread. |
    | **Max Risk** | Maximum potential loss if the trade goes against you (width - premium). |
    | **R/R Ratio** | Risk/Reward ratio. Lower is better (e.g., 3:1 means risk $3 to make $1). |
    | **POP** | Probability of Profit, estimated from short option delta. |
    | **Limit Price** | Recommended price to enter the limit order (mid of bid-ask). |
    | **Bid-Ask Range** | Estimated range the option can be traded at. |
    | **Account Size** | Your total portfolio value. Affects position sizing. |
    | **Risk per Trade** | % of account at risk in this single trade (default cap: 2%). |
    | **Risk Score** | 0-100 composite risk rating. Higher = riskier setup. |
    | **Theta** | Daily time decay — how much premium the position earns per day. |
    | **Vega** | Sensitivity to 1% change in implied volatility. |
    """)
