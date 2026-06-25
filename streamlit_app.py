"""Apex Options Analytics — Standalone Streamlit Demo App.
Self-contained: calls the engine directly, no separate API server needed."""
import streamlit as st
import pandas as pd
import json
import sys
import logging
from datetime import datetime, timedelta
import os
from pathlib import Path

# Add parent dir to path so engine imports work
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)

# Auto-detect data source: prefer Tradier if API key is set, fall back to Yahoo
DATA_SOURCE = "tradier" if os.environ.get("TRADIER_API_KEY") else "yahoo"

from engine import run_analysis

st.set_page_config(
    page_title="Apex Options Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .strategy-title { color: #ffaa00; font-size: 28px; font-weight: bold; }
    .section-label { color: #aaa; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin-top: 10px; }
    div[data-testid="metric-container"] { min-width: 100px; }
    div[data-testid="metric-container"] > label { white-space: nowrap; overflow: visible; }
    div[data-testid="metric-container"] > div { white-space: nowrap; overflow: visible; text-overflow: clip; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("⚡ Apex Options")
ticker = st.sidebar.text_input("Ticker", "SPY").upper()
account_size = st.sidebar.number_input("Account Size ($)", value=50000, step=1000)
run_btn = st.sidebar.button("▶ Run Analysis", type="primary")
page = st.sidebar.selectbox("Navigate", ["🏠 Home", "ℹ️ About"])

if page == "🏠 Home":
    st.title("Apex Options Analytics")

    if run_btn:
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                result_json = run_analysis(
                    ticker,
                    account_size=int(account_size),
                    output_format="json",
                    data_source=DATA_SOURCE,
                )
                data = json.loads(result_json)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                data = None

        if data:
            s = data["strikes"]
            vol = data["volatility_summary"]
            rp = data["risk_parameters"]
            ps = data["position_sizing"]

            # Build strike lists
            sell_s = []
            if s.get("short_strike"): sell_s.append(round(s["short_strike"], 2))
            if s.get("short_strike_2") and float(s["short_strike_2"]) > 0:
                ss2 = round(float(s["short_strike_2"]), 2)
                if ss2 not in sell_s: sell_s.append(ss2)
            buy_s = []
            if s.get("long_strike"): buy_s.append(round(s["long_strike"], 2))
            if s.get("long_strike_2") and float(s["long_strike_2"]) > 0:
                ls2 = round(float(s["long_strike_2"]), 2)
                if ls2 not in buy_s: buy_s.append(ls2)

            dte = int(s.get("dte", 45))
            exp_date = (datetime.now() + timedelta(days=dte)).strftime("%m/%d/%Y")

            cols = st.columns(7)
            cols[0].metric("Ticker", data.get("ticker", ticker))
            cols[1].metric("Price", f"${data.get('underlying_price',0):.2f}")
            cols[2].metric("IV Rank", f"{vol.get('iv_rank',0):.1f}%")
            cols[3].metric("Expires", exp_date)
            cols[4].metric("Contracts", ps.get("contracts", 1))
            cols[5].metric("Risk %", f"{rp.get('max_risk_pct',0)*100:.1f}%")
            cols[6].metric("Premium", f"${s.get('premium_collected',0):.2f}")

            st.markdown("---")
            st.markdown(f"<div class='strategy-title'>📌 {data.get('strategy_name','N/A')}</div>", unsafe_allow_html=True)

            left, right = st.columns([3, 2])

            with left:
                st.markdown("<div class='section-label'>Market Thesis</div>", unsafe_allow_html=True)
                st.info(data.get("market_thesis", ""))

                st.markdown("<div class='section-label'>Strikes</div>", unsafe_allow_html=True)
                sd = {"Side": ["Sell", "Buy"], "Strike": [", ".join(str(x) for x in sell_s), ", ".join(str(x) for x in buy_s)]}
                st.dataframe(pd.DataFrame(sd), hide_index=True, use_container_width=True)

                st.markdown("<div class='section-label'>Volatility Summary</div>", unsafe_allow_html=True)
                vd = {
                    "Metric": ["IV", "IV Rank", "IV Percentile", "RV (20d)", "IV/RV Ratio", "Regime", "Term Structure", "Skew", "Opportunity Score"],
                    "Value": [
                        f"{vol.get('current_iv',0)*100:.1f}%",
                        f"{vol.get('iv_rank',0):.1f}%",
                        f"{vol.get('iv_percentile',0):.1f}%",
                        f"{vol.get('realized_vol_20d',0)*100:.1f}%",
                        f"{vol.get('iv_rv_ratio_20d',0):.2f}x",
                        vol.get('vol_regime',''),
                        vol.get('term_structure',''),
                        vol.get('skew_description',''),
                        f"{vol.get('opportunity_score',0):.0f}/100",
                    ]
                }
                st.dataframe(pd.DataFrame(vd), hide_index=True, use_container_width=True)

            with right:
                st.markdown("<div class='section-label'>Risk Management</div>", unsafe_allow_html=True)
                st.metric("Max Risk", f"${rp.get('max_risk_dollars',0):.2f}")
                st.metric("Profit Target", f"${rp.get('profit_target_dollars',0):.2f}")
                st.metric("Stop Loss", f"${rp.get('stop_loss_dollars',0):.2f}")
                st.metric("Risk Score", f"{rp.get('risk_score',0):.0f}/100")

                st.markdown("<div class='section-label'>Trade Economics</div>", unsafe_allow_html=True)
                premium = s.get("premium_collected", 0)
                st.success(f"**Recommended Limit: ${premium:.2f}**")
                st.caption(f"🔄 Est. Bid-Ask: ${premium*0.85:.2f} - ${premium*1.15:.2f}")

                st.markdown("<div class='section-label'>Position Sizing</div>", unsafe_allow_html=True)
                st.metric("Account", f"${account_size:,.0f}")
                st.metric("Contracts", ps.get("contracts", 1))
                st.metric("Capital Required", f"${ps.get('capital_required',0):.2f}")
                st.metric("Risk per Trade", f"{rp.get('max_risk_pct',0)*100:.1f}%")

                st.markdown("<div class='section-label'>Trade Rationale</div>", unsafe_allow_html=True)
                st.markdown(data.get("trade_rationale", ""))

    else:
        st.info("Enter a ticker in the sidebar and click **▶ Run Analysis**.")

elif page == "ℹ️ About":
    st.title("About Apex Options Analytics")
    st.markdown("""
    **A quantitative AI agent** that analyzes market microstructure, volatility dynamics, and historical data to generate institutional-grade, risk-managed options premium-selling recommendations.

    ### Methodology
    - **Iron Condors** — Neutral outlook with high IV rank
    - **Credit Spreads** — Directional bias with premium collection
    - **Short Strangles** — Pure volatility plays (high risk)

    ### Field Legend
    | Field | Meaning |
    |-------|---------|
    | **IV Rank** | Current IV in its 52-week range (0-100%). Higher = better for selling. |
    | **IV/RV Ratio** | Implied ÷ Realized Volatility. >1.0 = options are rich. |
    | **Opportunity Score** | 0-100 rating of premium-selling conditions. |
    | **Risk %** | % of account at risk in this trade (default cap: 2%). |
    | **Limit Price** | Recommended limit order entry price (mid of bid-ask). |
    | **POP** | Probability of Profit from short option delta. |
    """)