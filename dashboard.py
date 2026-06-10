import sys
import os
import io
import contextlib
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np

# Must be first Streamlit command
st.set_page_config(
    page_title="ThetaEdge — Options Selling Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import ThetaEdge engine
sys.path.insert(0, os.path.dirname(__file__))
from thetaedge import (
    scan_market, print_recs, DEFAULT_TICKERS, TOP_LIQUID_TICKERS,
    build_ticker_universe, YFinanceProvider, TradierProvider,
    set_data_provider, get_data_provider_instance
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; margin-bottom: 0; }
    .sub-header { font-size: 1rem; opacity: 0.7; margin-top: 0; }
    .best-pick {
        background: linear-gradient(135deg, #1a73e822, #0d47a122);
        border: 1px solid #1a73e844;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .flag-danger { color: #ff4444; }
    .flag-warning { color: #ffaa00; }
    .flag-info { color: #4488ff; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
st.sidebar.markdown("## ⚡ ThetaEdge")
st.sidebar.markdown("### Options Selling Engine")
st.sidebar.markdown("---")

capital = st.sidebar.number_input("💰 Capital ($)", min_value=1000, max_value=10_000_000, value=20_000, step=1000, format="%d")
duration = st.sidebar.slider("📅 Duration (DTE)", min_value=7, max_value=90, value=30, step=1)

mode = st.sidebar.radio("📊 Strategy", options=["Cash-Secured Puts", "Covered Calls"], index=0)
mode_key = "csp" if mode == "Cash-Secured Puts" else "cc"

col1, col2 = st.sidebar.columns(2)
with col1:
    min_delta = st.number_input("Min Delta", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
with col2:
    max_delta = st.number_input("Max Delta", min_value=0.0, max_value=1.0, value=0.30, step=0.05)

col1, col2 = st.sidebar.columns(2)
with col1:
    dte_min = st.number_input("Min DTE", min_value=1, max_value=180, value=7, step=1)
with col2:
    dte_max = st.number_input("Max DTE", min_value=1, max_value=180, value=45, step=1)

top_n = st.sidebar.number_input("Top Results", min_value=1, max_value=50, value=10, step=1)
st.sidebar.markdown("---")

# ── Data Provider ──
st.sidebar.markdown("### 🔌 Data Provider")
tradier_token = os.environ.get("TRADIER_TOKEN", "")
tradier_available = bool(tradier_token)

if tradier_available:
    default_provider = "tradier"
    st.sidebar.success("✅ Tradier: Connected (real-time)")
else:
    default_provider = "yfinance"
    st.sidebar.info("ℹ️ yfinance: 15-min delayed")

provider = st.sidebar.radio("Provider", options=["yfinance", "tradier"], index=0 if default_provider == "yfinance" else 1)
if provider == "tradier" and tradier_available:
    st.sidebar.success("✅ Connected")
elif provider == "tradier" and not tradier_available:
    st.sidebar.error("❌ TRADIER_TOKEN not set")

st.sidebar.markdown("---")

# ── Ticker Universe ──
st.sidebar.markdown("### 📈 Ticker Universe")
universe_option = st.sidebar.selectbox(
    "Universe",
    options=["default", "sp500", "nasdaq100", "liquid", "custom"],
    index=0,
    help="Choose ticker universe or custom list"
)

tickers = DEFAULT_TICKERS
if universe_option == "custom":
    ticker_input = st.sidebar.text_input("Tickers (comma-separated)", value="AAPL,INTC,NVDA,AMZN")
    tickers = [t.strip().upper() for t in ticker_input.replace(",", " ").split() if t.strip()]
elif universe_option == "sp500":
    tickers = build_ticker_universe("sp500", max_scanned=100)
    st.sidebar.caption(f"{len(tickers)} S&P 500 tickers")
elif universe_option == "nasdaq100":
    tickers = build_ticker_universe("nasdaq100", max_scanned=100)
    st.sidebar.caption(f"{len(tickers)} NASDAQ-100 tickers")
elif universe_option == "liquid":
    tickers = TOP_LIQUID_TICKERS
    st.sidebar.caption(f"{len(tickers)} liquid tickers")
else:
    st.sidebar.caption(f"{len(tickers)} default tickers")

enable_backtest = st.sidebar.checkbox("📊 Enable Backtesting", value=False)
st.sidebar.markdown("---")
run_scan = st.sidebar.button("🚀 Run Scan", type="primary", use_container_width=True)

# ── Main Content Area ──
st.markdown('<p class="main-header">⚡ ThetaEdge</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered options selling recommendation engine</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Quick Ticker Check ──
st.markdown("### 🔍 Quick Ticker Check")
check_col1, check_col2 = st.columns([3, 1])
with check_col1:
    quick_ticker = st.text_input("Enter a ticker symbol", value="", placeholder="e.g. AAPL, INTC, NVDA", label_visibility="collapsed")
with check_col2:
    check_ticker = st.button("🔍 Check", type="secondary", use_container_width=True)

if check_ticker and quick_ticker:
    quick_ticker = quick_ticker.strip().upper()
    if provider == "tradier" and tradier_token:
        set_data_provider(TradierProvider(token=tradier_token))
    
    with st.spinner(f"Analyzing {quick_ticker}..."):
        recs = scan_market(
            tickers=[quick_ticker], capital=float(capital), duration_days=int(duration),
            mode=mode_key, min_delta=min_delta, max_delta=max_delta,
            dte_min=int(dte_min), dte_max=int(dte_max), top_n=5,
            enable_backtest=enable_backtest,
        )
        st.session_state['recs'] = recs
        st.session_state['capital'] = float(capital)
        st.session_state['quick_ticker'] = quick_ticker
    st.rerun()

st.markdown("---")

# ── Run Scan ──
if run_scan:
    if provider == "tradier" and tradier_token:
        set_data_provider(TradierProvider(token=tradier_token))
    if min_delta > max_delta:
        st.error("❌ Min delta cannot exceed max delta")
        st.stop()

    progress_bar = st.progress(0, text="Initializing scan...")
    stderr_output = io.StringIO()

    try:
        with st.spinner("Scanning market..."):
            old_stderr = sys.stderr
            sys.stderr = stderr_output
            result_buffer = io.StringIO()
            with contextlib.redirect_stdout(result_buffer):
                recs = scan_market(
                    tickers=tickers, capital=float(capital), duration_days=int(duration),
                    mode=mode_key, min_delta=min_delta, max_delta=max_delta,
                    dte_min=int(dte_min), dte_max=int(dte_max), top_n=int(top_n),
                    enable_backtest=enable_backtest,
                )
                print_recs(recs, float(capital))
            sys.stderr = old_stderr

        progress_bar.progress(100, text="Scan complete!")

        # Persist results
        st.session_state['recs'] = recs
        st.session_state['scan_log'] = stderr_output.getvalue()
        st.session_state['result_text'] = result_buffer.getvalue()
        st.session_state['capital'] = float(capital)

        with st.expander("📋 Scan Log", expanded=False):
            st.text(stderr_output.getvalue())

        if not recs:
            st.warning("⚠️ No candidates found. Try adjusting parameters.")
            st.stop()

    except Exception as e:
        sys.stderr = old_stderr
        st.error(f"❌ Error: {e}")
        st.exception(e)
        st.stop()

# ── Display Results (fresh or from session state) ──
if 'recs' in st.session_state and st.session_state['recs']:
    recs = st.session_state['recs']
    capital = st.session_state.get('capital', capital)

    # Show data source
    ticker_context = st.session_state.get('quick_ticker', '')
    if ticker_context:
        context_str = f" for {ticker_context}"
    else:
        context_str = ""
    
    if provider == "tradier" and tradier_available:
        source_label = "🔴 Live (Tradier)"
    else:
        source_label = "🟡 15-min delayed (yfinance)"

    st.markdown(f"## 🏆 Top {len(recs)} Recommendations{context_str} — {source_label}")

    # Summary metrics
    avg_pcr = np.mean([r.pcr for r in recs])
    avg_pop = np.mean([r.probability_of_profit for r in recs])
    avg_ann = np.mean([r.annualized_return for r in recs])
    total_deployed = sum(r.capital_used for r in recs)

    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1: st.metric("Avg P/C Ratio", f"{avg_pcr:.2f}%")
    with mcol2: st.metric("Avg PoP", f"{avg_pop:.1f}%")
    with mcol3: st.metric("Avg Ann. Return", f"{avg_ann:.1f}%")
    with mcol4: st.metric("Capital Deployed", f"${total_deployed:,.0f}")
    with mcol5:
        pct_deployed = total_deployed / capital * 100
        color = "🔴" if pct_deployed > 80 else "🟡" if pct_deployed > 50 else "🟢"
        st.metric("Utilization", f"{color} {pct_deployed:.0f}%")

    # Best pick
    best = recs[0]
    st.markdown(f"""
    <div class="best-pick">
        <strong>⭐ Best Pick:</strong> {best.ticker} @ <strong>${best.current_price:.2f}</strong> 
        → Sell ${best.strike:.2f} {best.option_type.upper()} 
        expiring {best.expiration} (DTE: {best.dte})<br>
        <strong>Premium:</strong> ${best.premium_mid:.2f} (B:${best.bid:.2f} A:${best.ask:.2f}) | 
        <strong>Return:</strong> {best.pcr:.2f}% ({best.annualized_return:.1f}% ann.) |
        <strong>PoP:</strong> {best.probability_of_profit:.1f}% |
        <strong>Score:</strong> {best.composite_score:.1f}/100
    </div>
    """, unsafe_allow_html=True)

    # Table
    table_data = []
    for i, rec in enumerate(recs, 1):
        distance = abs(rec.current_price - rec.strike)
        moneyness = "OTM" if (rec.option_type == "put" and rec.strike < rec.current_price) or \
                            (rec.option_type == "call" and rec.strike > rec.current_price) else "ITM"
        table_data.append({
            "#": i, "Ticker": rec.ticker, "Type": rec.option_type.upper(),
            "Price": f"${rec.current_price:.2f}",
            "Strike": f"${rec.strike:.2f}", "Expiry": rec.expiration,
            "DTE": rec.dte, "Premium": f"${rec.premium_mid:.2f}",
            "Delta": f"{rec.delta:.3f}", "PoP": f"{rec.probability_of_profit:.0f}%",
            "P/C%": f"{rec.pcr:.2f}%", "Ann%": f"{rec.annualized_return:.1f}%",
            "IV": f"{rec.implied_volatility:.0%}", "Score": f"{rec.composite_score:.0f}/100",
            "Contracts": rec.recommended_contracts,
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True, column_config={
        "Premium": st.column_config.TextColumn(width="small"),
        "Score": st.column_config.TextColumn(width="small"),
    })

    # Expandable details per recommendation
    for i, rec in enumerate(recs, 1):
        with st.expander(f"📊 #{i} {rec.ticker} {rec.strike:.0f} {rec.option_type.upper()} — Score: {rec.composite_score:.0f}/100"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Premium & Pricing**")
                st.write(f"Bid: ${rec.bid:.2f} | Ask: ${rec.ask:.2f}")
                st.write(f"Mid: ${rec.premium_mid:.2f} | Total: ${rec.premium_total:.2f}")
                st.write(f"Current: ${rec.current_price:.2f} | Strike: ${rec.strike:.2f}")
                distance = abs(rec.current_price - rec.strike)
                moneyness = "OTM" if (rec.option_type == "put" and rec.strike < rec.current_price) else "ITM"
                st.write(f"Moneyness: {moneyness} by ${distance:.2f}")
                st.write(f"Break-even: ${rec.break_even:.2f}")

            with col2:
                st.markdown("**Greeks & Risk**")
                st.write(f"Delta: {rec.delta:.4f} | Gamma: {rec.gamma:.4f}")
                st.write(f"Theta/D: ${rec.theta_daily:.6f} | Vega: {rec.vega:.4f}")
                st.write(f"Rho: {rec.rho:.4f}")
                st.write(f"PoP: {rec.probability_of_profit:.1f}%")
                st.write(f"Collateral: ${rec.collateral:,.0f} | Max Loss: ${rec.max_loss:,.0f}")
                st.write(f"Kelly: {rec.kelly_fraction:.4f} | Conservative: {rec.conservative_kelly:.4f}")

            with col3:
                st.markdown("**IV & Scoring**")
                ivr = f"{rec.iv_rank:.0f}%" if rec.iv_rank else "N/A"
                ivp = f"{rec.iv_percentile:.0f}%" if rec.iv_percentile else "N/A"
                st.write(f"IV: {rec.implied_volatility:.2%} | Rank: {ivr} | Perc: {ivp}")
                st.write(f"Theta Capture: {rec.theta_capture_rate:.4f}")
                st.write(f"Tail Sensitivity: {rec.tail_sensitivity:.4f}")
                st.write(f"Liquidity: Vol={rec.volume:,} | OI={rec.open_interest:,}")
                c = rec.score_components
                st.write(f"PCR={c.get('pcr',0):.0f} PoP={c.get('pop',0):.0f} Theta={c.get('theta',0):.0f}")
                st.write(f"IV={c.get('iv',0):.0f} Liq={c.get('liq',0):.0f} RiskAdj={c.get('riskadj',0):.0f} Tech={c.get('tech',50):.0f}")

            # Technicals & Backtest
            tcol1, tcol2 = st.columns(2)
            with tcol1:
                if rec.technicals:
                    t = rec.technicals
                    st.markdown("**Technical Analysis**")
                    rsi_str = f"{t.rsi:.0f}" if t.rsi else "N/A"
                    st.write(f"RSI(14): {rsi_str} | Trend: {t.trend}")
                    atr_str = f"{t.atr_pct:.1f}%" if t.atr_pct else "N/A"
                    st.write(f"ATR: {atr_str}")
                    support_str = f"${t.support_level:.2f}" if t.support_level else "N/A"
                    st.write(f"Support: {support_str}")

            with tcol2:
                if hasattr(rec, 'backtest') and rec.backtest:
                    bt = rec.backtest
                    st.markdown("**Backtest Results**")
                    st.write(f"Win Rate: {bt.win_rate*100:.1f}%")
                    st.write(f"Trades: {bt.total_trades}")
                    st.write(f"Avg Return: {bt.avg_return:.2f}%")

            # Risk flags
            if rec.flags:
                st.markdown("**⚠️ Risk Flags**")
                for flag in rec.flags:
                    flag_type = "🔴" if any(w in flag.lower() for w in ["spike", "loss", "risk", "warning", "danger"]) else \
                                "🟠" if any(w in flag.lower() for w in ["low", "oversold", "overbought"]) else "🔵"
                    st.write(f"{flag_type} {flag}")

    # Portfolio summary
    st.markdown("---")
    st.markdown("## 📊 Portfolio Summary")
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1: st.metric("Total Deployed", f"${total_deployed:,.0f}")
    with pcol2: st.metric("Utilization", f"{total_deployed/capital*100:.1f}%")
    with pcol3:
        avg_ann = np.mean([r.annualized_return for r in recs if r.annualized_return > 0])
        st.metric("Avg Ann. Return", f"{avg_ann:.1f}%")
        if total_deployed / capital > 0.8:
            st.error("⚠️ >80% deployed — reduce sizes")

    # Sector allocation
    from collections import defaultdict
    sectors = defaultdict(float)
    for rec in recs: sectors[rec.sector] += rec.capital_used
    if sectors:
        st.markdown("**Sector Allocation**")
        sector_df = pd.DataFrame([
            {"Sector": s, "Amount": f"${a:,.0f}", "Percent": f"{a/capital*100:.1f}%"}
            for s, a in sorted(sectors.items(), key=lambda x: -x[1])
        ])
        st.dataframe(sector_df, use_container_width=True, hide_index=True)

    st.success("✅ Results shown above. Run a new scan from the sidebar.")
    st.caption("💡 Data refreshes on each scan. Prices reflect the data provider (yfinance: ~15min delay, Tradier: real-time).")

elif not run_scan:
    # Welcome state
    st.markdown("""
    ## Welcome to ThetaEdge ⚡
    
    **AI-powered options selling recommendation engine** that scans the market and tells you exactly which option to sell.
    
    ### Get Started
    1. **Configure** your capital, duration, and tickers in the sidebar
    2. Click **🚀 Run Scan**
    3. Review ranked recommendations with full Greeks, backtest data, and risk flags
    
    ### Tips
    - **Tradier provider** gives real-time prices (set TRADIER_TOKEN env var)
    - **S&P 500 universe** scans 100+ liquid stocks
    - **Enable Backtesting** to see historical win rates
    - Results persist across page refreshes
    """)
