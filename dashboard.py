#!/usr/bin/env python3
"""
ThetaEdge — Streamlit Web Dashboard
====================================

Web interface for the ThetaEdge options selling recommendation engine.

Run with:
    streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
"""

import sys
import os
import io
import contextlib
from datetime import date

import streamlit as st
import pandas as pd
import numpy as np

# Import ThetaEdge engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thetaedge import (
    scan_market, print_recs, DEFAULT_TICKERS, TOP_LIQUID_TICKERS,
    build_ticker_universe, YFinanceProvider, TradierProvider,
    set_data_provider, get_data_provider_instance, OptionRecommendation
)

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ThetaEdge — Options Selling Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
    .sub-header { font-size: 1rem; color: #666; margin-top: 0; }
    .stExpanderHeader { font-weight: 600; }
    .metric-card { background: #f0f2f6; border-radius: 10px; padding: 15px; text-align: center; }
    .flag-warn { color: #e67e22; font-size: 0.9rem; }
    .flag-danger { color: #e74c3c; font-size: 0.9rem; }
    .flag-info { color: #3498db; font-size: 0.9rem; }
    .kelly-good { color: #27ae60; font-weight: 600; }
    .best-pick { border-left: 4px solid #f1c40f; background: #fef9e7; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar Inputs
# ──────────────────────────────────────────────
st.sidebar.markdown("## ⚡ ThetaEdge")
st.sidebar.markdown("### Options Selling Engine")
st.sidebar.markdown("---")

# Capital
capital = st.sidebar.number_input(
    "💰 Capital ($)",
    min_value=1000, max_value=10_000_000, value=25_000, step=5000,
    format="%d"
)

# Duration
duration = st.sidebar.slider(
    "📅 Duration (days)",
    min_value=7, max_value=90, value=30, step=5,
    help="Preferred days to expiration"
)

# Mode
mode = st.sidebar.radio(
    "📊 Strategy",
    options=["csp", "cc"],
    format_func=lambda x: "Cash-Secured Puts" if x == "csp" else "Covered Calls",
    index=0,
    horizontal=True
)

# Delta Range
col1, col2 = st.sidebar.columns(2)
with col1:
    min_delta = st.number_input("Min Delta", min_value=0.0, max_value=0.50, value=0.0, step=0.05, format="%.2f")
with col2:
    max_delta = st.number_input("Max Delta", min_value=0.0, max_value=0.50, value=0.30, step=0.05, format="%.2f")

# DTE Range
col1, col2 = st.sidebar.columns(2)
with col1:
    dte_min = st.number_input("Min DTE", min_value=1, max_value=90, value=7, step=1)
with col2:
    dte_max = st.number_input("Max DTE", min_value=1, max_value=90, value=45, step=1)

# Top N results
top_n = st.sidebar.number_input("Top Results", min_value=1, max_value=50, value=10, step=1)

# Provider toggle
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔌 Data Provider")

provider = st.sidebar.radio(
    "Provider",
    options=["yfinance", "tradier"],
    index=0,
    horizontal=True,
    help="Tradier requires TRADIER_TOKEN env var"
)

# Tradier status
tradier_token = os.environ.get("TRADIER_TOKEN", "")
if provider == "tradier":
    if tradier_token:
        st.sidebar.success("✅ Tradier: Connected")
        st.sidebar.caption(f"Token: {tradier_token[:8]}...")
    else:
        st.sidebar.error("❌ TRADIER_TOKEN not set")
        st.sidebar.caption("Falling back to yfinance")

# Ticker selection
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Ticker Universe")

universe_option = st.sidebar.selectbox(
    "Universe",
    options=["default", "sp500", "nasdaq100", "liquid", "custom"],
    index=0,
    help="Choose ticker universe or custom list"
)

tickers = DEFAULT_TICKERS
if universe_option == "custom":
    ticker_input = st.sidebar.text_input(
        "Tickers (comma-separated)",
        value="SPY,AAPL,MSFT,INTC",
        help="e.g., SPY,AAPL,MSFT,INTC"
    )
    tickers = [t.strip().upper() for t in ticker_input.replace(",", " ").split() if t.strip()]
elif universe_option == "sp500":
    with st.sidebar.status("Fetching S&P 500..."):
        tickers = build_ticker_universe("sp500", max_scanned=100)
    st.sidebar.caption(f"{len(tickers)} tickers loaded")
elif universe_option == "nasdaq100":
    with st.sidebar.status("Fetching NASDAQ-100..."):
        tickers = build_ticker_universe("nasdaq100", max_scanned=100)
    st.sidebar.caption(f"{len(tickers)} tickers loaded")
elif universe_option == "liquid":
    tickers = TOP_LIQUID_TICKERS
    st.sidebar.caption(f"{len(tickers)} liquid tickers")
else:
    if len(tickers) > 10:
        st.sidebar.caption(f"{len(tickers)} default tickers")

# Enable/disable backtesting
enable_backtest = st.sidebar.checkbox("📊 Enable Backtesting", value=False,
                                       help="Slower but shows historical win rates")

# Run button
st.sidebar.markdown("---")
run_scan = st.sidebar.button("🚀 Run Scan", type="primary", use_container_width=True)


# ──────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────
st.markdown('<p class="main-header">⚡ ThetaEdge</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered options selling recommendation engine</p>', unsafe_allow_html=True)
st.markdown("---")

if run_scan:
    # Initialize provider
    if provider == "tradier" and tradier_token:
        set_data_provider(TradierProvider(token=tradier_token))
    
    # Validate
    if min_delta > max_delta:
        st.error("❌ Min delta cannot exceed max delta")
        st.stop()
    
    # Progress display
    progress_bar = st.progress(0, text="Initializing scan...")
    status_text = st.empty()
    
    # Capture stderr for progress
    stderr_output = io.StringIO()
    
    try:
        with st.spinner("Scanning market..."):
            # Run the scan with captured stderr
            old_stderr = sys.stderr
            sys.stderr = stderr_output
            
            result_buffer = io.StringIO()
            with contextlib.redirect_stdout(result_buffer):
                recs = scan_market(
                    tickers=tickers,
                    capital=float(capital),
                    duration_days=int(duration),
                    mode=mode,
                    min_delta=min_delta,
                    max_delta=max_delta,
                    dte_min=int(dte_min),
                    dte_max=int(dte_max),
                    top_n=int(top_n),
                    enable_backtest=enable_backtest,
                )
                print_recs(recs, float(capital))
            
            sys.stderr = old_stderr
        
        progress_bar.progress(100, text="Scan complete!")
        
        # Display progress logs
        with st.expander("📋 Scan Log", expanded=False):
            st.text(stderr_output.getvalue())
        
        if not recs:
            st.warning("⚠️ No candidates found. Try adjusting your parameters:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("💰 **Increase capital** — many options require $10K+ collateral")
            with col2:
                st.info("📅 **Widen DTE range** — expirations outside 7-45 DTE may have better options")
            with col3:
                st.info("📈 **Expand tickers** — try different tickers or use sp500 universe")
            st.stop()
        
        # ── Results Table ──
        st.markdown(f"## 🏆 Top {len(recs)} Recommendations")
        
        # Summary metrics row
        avg_pcr = np.mean([r.pcr for r in recs])
        avg_pop = np.mean([r.probability_of_profit for r in recs])
        avg_ann = np.mean([r.annualized_return for r in recs])
        total_deployed = sum(r.capital_used for r in recs)
        
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        with mcol1:
            st.metric("Avg P/C Ratio", f"{avg_pcr:.2f}%")
        with mcol2:
            st.metric("Avg PoP", f"{avg_pop:.1f}%")
        with mcol3:
            st.metric("Avg Ann. Return", f"{avg_ann:.1f}%")
        with mcol4:
            st.metric("Capital Deployed", f"${total_deployed:,.0f}")
        with mcol5:
            pct_deployed = total_deployed / capital * 100
            color = "🔴" if pct_deployed > 80 else "🟡" if pct_deployed > 50 else "🟢"
            st.metric("Utilization", f"{color} {pct_deployed:.0f}%")
        
        # Best pick highlight
        best = recs[0]
        st.markdown(f"""
        <div class="best-pick">
            <strong>⭐ Best Pick:</strong> 
            {best.ticker} ${best.strike:.2f} {best.option_type.upper()} 
            expiring {best.expiration} (DTE: {best.dte})<br>
            <strong>Premium:</strong> ${best.premium_mid:.2f} (B:${best.bid:.2f} A:${best.ask:.2f}) | 
            <strong>Return:</strong> {best.pcr:.2f}% ({best.annualized_return:.1f}% ann.) |
            <strong>PoP:</strong> {best.probability_of_profit:.1f}% |
            <strong>Score:</strong> {best.composite_score:.1f}/100
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Build display dataframe
        table_data = []
        for i, rec in enumerate(recs, 1):
            moneyness = "OTM" if (rec.option_type == "put" and rec.strike < rec.current_price) or \
                                (rec.option_type == "call" and rec.strike > rec.current_price) else "ITM"
            table_data.append({
                "#": i,
                "Ticker": rec.ticker,
                "Type": rec.option_type.upper(),
                "Strike": f"${rec.strike:.2f}",
                "Expiry": rec.expiration,
                "DTE": rec.dte,
                "Premium": f"${rec.premium_mid:.2f}",
                "Δ": f"{rec.delta:.3f}",
                "Γ": f"{rec.gamma:.4f}",
                "Θ/D": f"${rec.theta_daily:.4f}",
                "PoP": f"{rec.probability_of_profit:.0f}%",
                "P/C%": f"{rec.pcr:.2f}%",
                "Ann.%": f"{rec.annualized_return:.1f}%",
                "Collat": f"${rec.collateral:,.0f}",
                "Score": f"{rec.composite_score:.1f}",
                "Kelly": f"{rec.kelly_fraction:.3f}" if rec.kelly_fraction > 0 else "-",
                "Contracts": rec.recommended_contracts,
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={
                         "Premium": st.column_config.TextColumn(width="small"),
                         "Strike": st.column_config.TextColumn(width="small"),
                     })
        
        # ── Expandable Details per Recommendation ──
        st.markdown("### 📋 Detailed Analysis")
        
        for i, rec in enumerate(recs):
            moneyness = "OTM" if (rec.option_type == "put" and rec.strike < rec.current_price) or \
                                (rec.option_type == "call" and rec.strike > rec.current_price) else "ITM"
            diff = abs(rec.current_price - rec.strike)
            
            with st.expander(f"#{i+1}  {rec.ticker}  ${rec.strike:.2f} {rec.option_type.upper()}  "
                           f"Exp: {rec.expiration}  DTE: {rec.dte}  Score: {rec.composite_score:.1f}/100"):
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**💰 Premium & Pricing**")
                    st.write(f"Mid Price: **${rec.premium_mid:.2f}**")
                    st.write(f"Bid: ${rec.bid:.2f} / Ask: ${rec.ask:.2f}")
                    st.write(f"Total Premium: **${rec.premium_total:.2f}**")
                    st.write(f"Underlying: **${rec.current_price:.2f}**")
                    st.write(f"Strike: **${rec.strike:.2f}** ({moneyness} by ${diff:.2f})")
                    st.write(f"Break Even: **${rec.break_even:.2f}**")
                
                with col2:
                    st.markdown("**📊 Greeks & Returns**")
                    st.write(f"Delta: **{rec.delta:.4f}**  |  Gamma: **{rec.gamma:.4f}**")
                    st.write(f"Theta/D: **${rec.theta_daily:.6f}**  |  Vega: **{rec.vega:.4f}**")
                    st.write(f"Rho: **{rec.rho:.4f}**")
                    st.write(f"PoP: **{rec.probability_of_profit:.1f}%**")
                    st.write(f"P/C Ratio: **{rec.pcr:.2f}%**")
                    st.write(f"Ann. Return: **{rec.annualized_return:.1f}%**")
                
                with col3:
                    st.markdown("**📐 Sizing & Risk**")
                    st.write(f"Collateral: **${rec.collateral:,.0f}**")
                    st.write(f"Kelly: **{rec.kelly_fraction:.4f}**")
                    st.write(f"Conservative Kelly: **{rec.conservative_kelly:.4f}**")
                    st.write(f"Recommended Contracts: **{rec.recommended_contracts}**")
                    st.write(f"Capital Used: **${rec.capital_used:,.0f} ({rec.capital_used_pct:.1f}%)**")
                    st.write(f"Max Loss: **${rec.max_loss:,.0f}**")
                
                # Second row
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📈 IV Analysis**")
                    st.write(f"IV: **{rec.implied_volatility:.2%}**")
                    ivr = f"{rec.iv_rank:.0f}%" if rec.iv_rank else "N/A"
                    ivp = f"{rec.iv_percentile:.0f}%" if rec.iv_percentile else "N/A"
                    st.write(f"IV Rank: **{ivr}**  |  IV Percentile: **{ivp}**")
                
                with col2:
                    st.markdown("**📊 Scoring Breakdown**")
                    c = rec.score_components
                    st.write(f"PCR Eff: **{c.get('pcr',0):.0f}/100**  |  PoP: **{c.get('pop',0):.0f}/100**")
                    st.write(f"Theta: **{c.get('theta',0):.0f}/100**  |  IV: **{c.get('iv',0):.0f}/100**")
                    st.write(f"Liq: **{c.get('liq',0):.0f}/100**  |  RiskAdj: **{c.get('riskadj',0):.0f}/100**")
                    st.write(f"**Composite: {rec.composite_score:.1f}/100**")
                
                with col3:
                    st.markdown("**📉 Technicals**")
                    if rec.technicals:
                        t = rec.technicals
                        st.write(f"RSI: **{t.rsi:.0f}**{' ⚠️ Oversold' if t.oversold_flag else ''}{' ⚠️ Overbought' if t.overbought_flag else ''}")
                        st.write(f"Trend: **{t.trend}**")
                        st.write(f"ATR: **{t.atr_pct:.1f}%**" if t.atr_pct else "ATR: N/A")
                    else:
                        st.write("N/A")
                
                # Flags
                if rec.flags:
                    st.markdown("**🚩 Risk Flags**")
                    for flag in rec.flags:
                        if "over" in flag.lower() or "risk" in flag.lower():
                            st.markdown(f'<span class="flag-danger">🚨 {flag}</span>', unsafe_allow_html=True)
                        elif "low" in flag.lower():
                            st.markdown(f'<span class="flag-warn">⚠️ {flag}</span>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<span class="flag-info">ℹ️ {flag}</span>', unsafe_allow_html=True)
                
                # Backtest
                if rec.backtest and rec.backtest.total_trades > 0:
                    bt = rec.backtest
                    st.markdown("**📊 Backtest Results**")
                    st.write(f"Trades: **{bt.total_trades}**  |  Win Rate: **{bt.win_rate:.0f}%**  |  "
                           f"Avg Return: **{bt.avg_return:+.2f}%**")
        
        # ── Portfolio Summary ──
        st.markdown("---")
        st.markdown("## 📊 Portfolio Summary")
        
        port_col1, port_col2, port_col3 = st.columns(3)
        with port_col1:
            pct_deployed = total_deployed / capital * 100
            st.metric("Total Capital", f"${capital:,.0f}")
            st.metric("Capital Deployed", f"${total_deployed:,.0f} ({pct_deployed:.1f}%)")
        
        with port_col2:
            st.metric("Number of Positions", len(recs))
            st.metric("Weighted Avg PoP", f"{avg_pop:.1f}%")
        
        with port_col3:
            st.metric("Weighted Avg Return", f"{avg_ann:.1f}% ann.")
            if pct_deployed > 80:
                st.error("⚠️ Over 80% deployed — reduce position sizes")
            elif pct_deployed > 50:
                st.warning("⚠️ Over 50% deployed")
        
        # Sector allocation
        from collections import defaultdict
        sectors = defaultdict(float)
        for rec in recs:
            sectors[rec.sector] += rec.capital_used
        
        if sectors:
            st.markdown("**Sector Allocation**")
            sector_data = {"Sector": [], "Amount": [], "Percentage": []}
            for sector, amount in sorted(sectors.items(), key=lambda x: -x[1]):
                sector_data["Sector"].append(sector)
                sector_data["Amount"].append(f"${amount:,.0f}")
                sector_data["Percentage"].append(f"{amount/capital*100:.1f}%")
            
            st.dataframe(
                pd.DataFrame(sector_data),
                use_container_width=True,
                hide_index=True,
            )
        
        st.success("✅ Scan complete! Use the sidebar to run a new scan.")
    
    except Exception as e:
        sys.stderr = old_stderr
        st.error(f"❌ Error during scan: {e}")
        st.exception(e)

else:
    # Welcome state
    st.markdown("""
    ## Welcome to ThetaEdge ⚡
    
    **AI-powered options selling recommendation engine** that scans the market and tells you
    exactly which option to sell, at what strike, and for what premium.
    
    ### Getting Started
    
    1. **Set your capital** in the sidebar
    2. **Choose your strategy** — Cash-Secured Puts or Covered Calls
    3. **Select tickers** or use a universe (S&P 500, NASDAQ-100)
    4. **Toggle providers** — yfinance (free) or Tradier (API key)
    5. **Click "Run Scan"** and get ranked recommendations
    
    ### Features
    
    | Feature | Description |
    |---------|-------------|
    | 🧮 Full Greeks | Delta, Gamma, Theta, Vega, Rho via Black-Scholes |
    | 📊 Multi-Factor Scoring | 6 weighted factors → 0-100 composite |
    | 📉 Technical Analysis | RSI, SMA, ATR, trend detection |
    | 📐 Kelly Sizing | Optimal position sizing with conservative Kelly |
    | 🚩 Risk Flags | Earnings, dividends, gamma spikes, IV expansion |
    | 📈 Backtesting | Historical win rates for similar setups |
    | 🔌 Dual Provider | yfinance (free) or Tradier (live data) |
    
    ---
    *Configure your scan in the sidebar and click **Run Scan** to begin.*
    """)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 📉 Cash-Secured Puts
        Sell OTM puts, collect premium, tie up cash as collateral.
        Best for neutral-to-bullish outlook.
        """)
    with col2:
        st.markdown("""
        ### 📈 Covered Calls
        Sell OTM calls against shares you own. Generate income
        on holdings with neutral-to-bearish outlook.
        """)
    with col3:
        st.markdown("""
        ### 🎯 Smart Scoring
        Multi-factor ranking: PCR efficiency, probability of profit,
        theta capture, IV percentile, liquidity, risk-adjusted return.
        """)