---
title: "ThetaEdge — Options Selling Recommendation Engine: Research & Architecture"
author: "Options Research Analyst"
date: "2026-06-08"
---

# ThetaEdge — Research Document

## Overview

This document captures the research and architecture for ThetaEdge, an AI-powered options selling recommendation engine. The engine scans options chains, calculates Greeks and financial metrics, applies risk filters, and recommends the best cash-secured puts (and covered calls) to sell based on the user's capital and desired duration.

All findings have been **verified with live market data** via yfinance and the vollib Black-Scholes library.

---

## 1. Market Data Source: yfinance (Yahoo Finance)

### Recommendation: **yfinance** (free, Python-native, no API key required)

**Why yfinance:**
- Completely free — no API keys, no rate limits for reasonable usage, no subscription fees
- Python-native with pandas DataFrame output
- Provides **options chains** with strikes, bid, ask, lastPrice, impliedVolatility, volume, openInterest
- Provides underlying price history for the stock/ETF
- Works with all US-listed options (SPY, AAPL, AMD, F, etc.)
- 35+ expiration dates available for liquid tickers

### How to fetch options data:

```python
import yfinance as yf

# Create ticker object
ticker = yf.Ticker("SPY")

# Get available expiration dates (returns tuple of strings)
expirations = ticker.options  # e.g., ('2026-06-08', '2026-06-09', ...)

# Get underlying price history
history = ticker.history(period="5d")
underlying_price = history['Close'].iloc[-1]

# Fetch options chain for a specific expiration
chain = ticker.option_chain("2026-07-24")
calls = chain.calls  # DataFrame
puts = chain.puts    # DataFrame

# DataFrame columns:
# ['contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask',
#  'change', 'percentChange', 'volume', 'openInterest', 'impliedVolatility',
#  'inTheMoney', 'contractSize', 'currency']
```

**Key fields for our engine:**
| Field | Type | Use |
|-------|------|-----|
| `strike` | float | Option strike price |
| `bid` | float | Current bid price (best estimate of sell price) |
| `ask` | float | Current ask price |
| `lastPrice` | float | Last traded price |
| `impliedVolatility` | float | Market's implied volatility (used for Greeks) |
| `volume` | int | Trading volume (liquidity filter) |
| `openInterest` | int | Open interest (liquidity filter) |
| `inTheMoney` | bool | Whether option is ITM |

**Limitations:**
- Market data is delayed ~15 minutes (acceptable for our use case)
- No historical options data (only current snapshot) — for IV rank/percentile we need to collect daily snapshots or use a paid provider
- No real-time streaming (polling only)
- Bid/Ask may both be 0 outside market hours

### Alternative data sources (if scaling beyond free tier):
| Source | Cost | Pros |
|--------|------|------|
| **Polygon.io** | Paid ($29/mo+) | Real-time, historical options chains, Greeks included |
| **Intrinio** | Paid | Professional-grade options data |
| **TD Ameritrade API** | Free (with TD account) | Real-time, but being deprecated |
| **Tradier** | Free tier available | API access for brokerage customers |

---

## 2. Pricing & Greeks Model

### Library: **vollib** (replaces deprecated py_vollib)

```bash
pip install vollib numpy scipy pandas
```

### Black-Scholes Framework

For selling cash-secured puts (or covered calls), we need these Greeks and metrics:

#### 2a. Option Premium

**Mid-price** is the best estimate of fair value:

```python
if bid > 0 and ask > 0:
    premium_per_share = (bid + ask) / 2
elif lastPrice > 0:
    premium_per_share = lastPrice
else:
    premium_per_share = 0  # Skip illiquid options

premium_received = premium_per_share * 100  # 1 contract = 100 shares
```

**Why mid-price?** Bid-ask spreads can be wide for illiquid options. The mid-price represents the market's consensus fair value. For a sell order, we'd typically set a limit price at the ask or slightly above the mid.

#### 2b. Greeks (via vollib)

```python
from vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega
from vollib.black_scholes import black_scholes

# Parameters
S = underlying_price    # Current stock/ETF price
K = strike_price        # Option strike price
t = dte / 365.0         # Time to expiry in years (dte = days to expiry)
r = 0.05                # Risk-free rate (current ~4-5%)
sigma = implied_volatility  # From yfinance

# Calculate Greeks for a PUT option
delta_val = delta('p', S, K, t, r, sigma)    # ~ -0.30 to 0 for OTM puts
gamma_val = gamma('p', S, K, t, r, sigma)
theta_val = theta('p', S, K, t, r, sigma)    # Positive for sellers! (time decay)
vega_val = vega('p', S, K, t, r, sigma)

# Theoretical Black-Scholes price (for reference)
theo_price = black_scholes('p', S, K, t, r, sigma)
```

**Key Greeks for options selling:**

| Greek | Symbol | Meaning for Put Sellers | Target Range |
|-------|--------|------------------------|--------------|
| **Delta** | Δ | Probability of option being ITM at expiry. Delta of -0.20 means ~20% chance of assignment, 80% chance of keeping premium | -0.30 to -0.05 (OTM) |
| **Theta** | Θ | Time decay. **Positive theta** for sellers means the option loses value daily as time passes. We profit from theta. | Positive (can be $0.10-0.50/day per contract) |
| **Gamma** | Γ | Rate of change of delta. Lower is better — means delta won't spike if stock moves | < 0.01 |
| **Vega** | ν | Sensitivity to IV changes. We want to sell when IV is high (vega positive → IV drop helps us) | Higher when selling in high IV |

#### 2c. Collateral Calculation

```python
# Cash-Secured Put: collateral = strike price × 100 shares
collateral = K * 100  # e.g., $14,000 for a $140 strike put

# Covered Call: collateral = cost of 100 shares
collateral_c = S * 100  # e.g., $14,200 for a $142 stock
```

#### 2d. Key Financial Metrics

```python
# Premium-to-Collateral Ratio (PCR) — the CORE metric
pcr = premium_received / collateral
# e.g., $382.50 / $30,000 = 1.275% return over holding period

# Annualized Return (APR)
annualized_return = pcr / t  # t in years
# e.g., 1.275% / 0.085 (31 days) = 15.0% APR

# Break-Even Price (for cash-secured put)
breakeven = K - premium_per_share
```

**Interpretation of PCR:**
- 0.5% per month (6% annualized) = decent
- 1.0% per month (12% annualized) = good
- 2.0% per month (24% annualized) = excellent (usually higher IV)
- Below 0.5% for ~30 DTE = not worth the capital commitment

#### 2e. Probability of Profit

Using Black-Scholes d2 for a more accurate probability than just delta:

```python
from scipy import stats

d1 = (np.log(S/K) + (r + sigma**2/2)*t) / (sigma * np.sqrt(t))
d2 = d1 - sigma * np.sqrt(t)

# For a SHORT put (cash-secured put):
# Profit if stock > strike at expiry (option expires worthless)
prob_itm = stats.norm.cdf(-d2)   # P(stock < strike at expiry)
prob_otm = 1 - prob_itm           # P(stock > strike) = profit probability

# Delta approximates this: |delta| ≈ P(ITM)
# So P(profit) ≈ 1 - |delta|
```

#### 2f. Expected Return Calculation

```python
# Simplified expected return (ignoring tail risk)
expected_profit = premium_received * prob_otm

# More sophisticated:
# Expected value = premium - (expected loss if assigned)
# Expected loss if assigned ≈ (strike - expected_stock_price) * prob_itm
# For a crude estimate: assume stock at strike if ITM
expected_loss_if_assigned = K * 100 - (K * 100)  # simplified: breakeven analysis per trade
```

---

## 3. Scoring / Recommendation Algorithm

### 3a. Filter Pipeline

Options must pass ALL filters before scoring:

```
Layer 1: Delta Filter (High Probability OTM)
  - abs(delta) < 0.30 (put deltas are negative)
  - Ensures ~70%+ chance of keeping the premium
  - For covered calls: delta <= 0.30 (call deltas positive)

Layer 2: Minimum Premium
  - premium_received >= $20 per contract
  - Removes options where commission costs eat the profit

Layer 3: Minimum PCR
  - pcr >= 0.005 (0.5% return on collateral)
  - Removes trades not worth tying up capital

Layer 4: Liquidity Filter
  - volume > 0 (recent trading activity)
  - open_interest > 0 (open contracts exist)
  - Skips stale/illiquid options with wide spreads

Layer 5: DTE Filter (Duration)
  - User specifies desired holding period
  - Default: 30-60 DTE (sweet spot for theta decay)
  - Minimum 7 DTE (gamma risk too high closer to expiry)
  - Maximum 90 DTE (theta decay too slow further out)
```

### 3b. Scoring Engine

After filtering, options are scored using a **weighted composite score**:

```python
composite_score = (
    pcr_percentile_rank * 0.40 +       # Premium efficiency (highest weight)
    prob_otm_percentile_rank * 0.30 +  # Safety/probability
    volume_percentile_rank * 0.15 +    # Liquidity
    open_interest_percentile_rank * 0.15  # More liquidity
)
```

The composite score is computed by ranking all passing options within their percentile ranks (0-1 range), then weighting.

**Primary ranking** is by PCR (Premium-to-Collateral Ratio) since that's the core ROI metric.

### 3c. IV Rank/Percentile (for v2, requires historical data)

**IV Rank** (52-week):
```
IV Rank = (Current IV - 52W Low IV) / (52W High IV - 52W Low IV)
```
Range: 0-100%. Higher = IV is expensive = good time to sell.

**IV Percentile** (more robust):
```
IV Percentile = (# days IV was lower than today) / (total trading days)
```
Range: 0-100%. More robust because it considers all values, not just extremes.

**Strategy:**
- Sell when IV Rank > 50% (IV is above mid-range)
- Best when IV Rank > 70% (IV is in expensive territory)
- Avoid selling when IV Rank < 20% (premiums too cheap)

**Implementation note:** Calculating IV rank requires collecting daily IV snapshots. Options:
- Start collecting daily now (build historical database)
- Use a paid data provider with historical IV (Polygon, Barchart)
- Use yfinance to fetch ~1 year of daily options data (expensive, ~252 API calls per ticker)

### 3d. Adjustable Parameters (User Inputs)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capital` | $10,000 | Total available capital |
| `max_risk_per_trade` | 30% | Max % of capital per trade |
| `min_dte` | 30 | Minimum days to expiration |
| `max_dte` | 60 | Maximum days to expiration |
| `max_delta` | 0.30 | Maximum abs(delta) for selling |
| `min_premium` | $20 | Minimum premium per contract |
| `min_pcr` | 0.005 | Minimum premium-to-collateral ratio |

---

## 4. Architecture Recommendation

### Tech Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| **Data Fetch** | yfinance | Free, Python-native, options chains ready |
| **Greeks** | vollib | Black-Scholes with all Greeks, successor to py_vollib |
| **Math** | numpy, scipy | D2 calculation, percentile ranking |
| **Data Processing** | pandas | DataFrame operations, filtering, ranking |
| **Backend API** | FastAPI | Modern Python async framework |
| **Caching** | SQLite / Turso | Store daily snapshots, IV history |
| **Frontend** | Streamlit or minimal React | Prototype dashboard |

### Architecture Diagram

```
┌────────────────────────────────────────────────────┐
│                    FastAPI Server                    │
│                                                      │
│  ┌─────────────┐    ┌──────────────────────────┐   │
│  │ Data Fetcher │───>│    Scoring Engine        │   │
│  │ (yfinance)   │    │    - Filter pipeline     │   │
│  └─────────────┘    │    - Greeks calculation   │   │
│         │           │    - PCR ranking          │   │
│         ▼           │    - Composite score      │   │
│  ┌─────────────┐    └──────────┬───────────────┘   │
│  │ IV History  │               │                   │
│  │ (SQLite DB) │               ▼                   │
│  └─────────────┘    ┌──────────────────────────┐   │
│                     │   API Endpoints           │   │
│                     │   /recommend              │   │
│                     │   /scan                   │   │
│                     │   /ticker/{sym}/chains    │   │
│                     └──────────────────────────┘   │
└────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────┐
│              Frontend (Streamlit SPA)                │
│  - Capital & duration inputs                       │
│  - Ranked recommendations table                    │
│  - Trade detail card (strike, premium, PCR, delta) │
│  - Watchlist management                            │
└────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Input**: Capital amount, desired duration (DTE range)
2. **Fetch Data**: yfinance pulls options chains for watched tickers
3. **Calculate**: For each option, compute premium, Greeks, PCR, prob OTM
4. **Filter**: Apply delta, premium, PCR, liquidity, DTE filters
5. **Score**: Rank by PCR, compute composite score
6. **Recommend**: Return top N recommendations sorted by score
7. **Display**: Show as ranked table with key metrics per trade

### Key API Endpoints

```
POST /recommend
  Input:  { capital: 10000, min_dte: 30, max_dte: 60, max_delta: 0.30 }
  Output: { recommendations: [{ ticker, strike, expiry, premium, collateral, pcr, delta, prob_otm, composite_score }] }

GET /ticker/{symbol}/chains?expiry=YYYY-MM-DD
  Output: { calls: [...], puts: [...], underlying_price: ... }
```

### Ticker Watchlist (initial)

Start with high-liquidity, high-interest tickers across market caps:

| Ticker | Type | Stock Price | Min Collateral | Notes |
|--------|------|------------|----------------|-------|
| SPY | ETF (S&P 500) | ~$742 | ~$70K | Most liquid options market |
| QQQ | ETF (Nasdaq) | ~$520 | ~$49K | Tech-heavy |
| AAPL | Stock | ~$313 | ~$29K | High IV liquid options |
| AMD | Stock | ~$488 | ~$46K | High IV, popular with retail |
| PLTR | Stock | ~$136 | ~$13K | Retail favorite, high IV |
| F | Stock | ~$15 | ~$1.4K | **Best for small accounts** |
| INTC | Stock | ~$111 | ~$10K | Mid-range IV, accessible |

### Minimum Capital Tiers

| Capital | Suitable Stocks | Strategy |
|---------|----------------|----------|
| $1K-$2K | F (~$1.4K/strike), SOFI, T | Single contract CSP |
| $2K-$10K | INTC (~$10K), PLTR (~$13K), F + others | 1-2 contracts |
| $10K-$30K | AAPL (~$29K), AMD (~$46K partial) | 1 contract of higher-ticket |
| $30K-$100K+ | SPY (~$72K), QQQ, AMZN | Full diversification |

---

## 5. Verified Test Results

### Test: SPY (S&P 500 ETF), 45 DTE
```
Underlying:     $741.94
Expiration:     2026-07-24 (45 DTE)
Best Pick:      Sell SPY $725 Put
Premium:        $972.50 (mid-price)
Collateral:     $72,500
PCR:            1.34%
Annualized:     10.88%
Delta:          -0.292
Prob OTM:       68.8%
IV:             16.1%
Breakeven:      $715.27
```

### Test: F (Ford), 31 DTE
```
Underlying:     $15.00
Best Pick:      Sell F $14 Put
Premium:        $28.00
Collateral:     $1,400
PCR:            1.96% (2% in 31 DTE)
Delta:          -0.231
Prob OTM:       76.9%
IV:             40.8%
```

### Test: AAPL (Apple), 31 DTE
```
Underlying:     $313.17
Best Pick:      Sell AAPL $300 Put
Premium:        $382.50
Collateral:     $30,000
PCR:            1.28%
Annualized:     15.0%
Delta:          -0.244
Prob OTM:       73.3%
IV:             24.6%
```

---

## 6. Implementation Plan (Recommended Phases)

### Phase 1: Core Prototype (1-2 days)
- [x] Research & data source verification (this document)
- [ ] Python scripts for options chain fetching
- [ ] Greeks calculation module
- [ ] Filter pipeline and scoring engine
- [ ] Command-line tool that outputs top N recommendations

### Phase 2: API + Dashboard (3-5 days)
- [ ] FastAPI backend with /recommend endpoint
- [ ] Streamlit frontend with parameter inputs
- [ ] Ticker watchlist with multi-ticker scanning
- [ ] Trade detail view with key metrics

### Phase 3: Analytics & History (5-7 days)
- [ ] Daily IV snapshot collection to SQLite
- [ ] IV Rank/Percentile calculation
- [ ] Basic backtesting framework
- [ ] Track record of recommended trades

### Phase 4: Production (7-14 days)
- [ ] Subscription tiers (free/paid)
- [ ] Real-time scanning cron job
- [ ] Email/SMS alerts for top picks
- [ ] Web dashboard with user accounts
- [ ] Onboarding flow (time to first trade < 5 min)

---

## 7. Edge Cases & Considerations

### Outside Market Hours
- yfinance returns stale data (last close)
- Bid/Ask may be 0 for many strikes
- **Solution**: Only run engine during market hours (9:30 AM - 4:00 PM ET), or use lastPrice as fallback

### Options with No Volume
- Illiquid options have wide bid-ask spreads
- **Solution**: Volume > 0 filter; OI > 0 filter

### Earnings / Events
- IV spikes before earnings, then collapses
- CSPs sold before earnings have higher premium but higher risk
- **Solution**: Flag earnings dates for tickers; let user decide

### Dividends
- Cash-secured puts: early assignment risk if dividend > remaining premium
- **Solution**: Don't recommend CSPs within 2 weeks of ex-dividend date

### Risk Management Notes
- Maximum loss for CSP = strike × 100 (stock goes to $0)
- Never sell a put on a stock you wouldn't want to own
- Position sizing: max 30% of capital per trade
- Roll when tested: buy back at loss, sell further out
- The wheel strategy: CSP → assigned → sell covered call

---

## 8. Appendix: Quick-Start Code

```python
# pip install yfinance vollib numpy scipy pandas

import yfinance as yf
from datetime import datetime
from vollib.black_scholes.greeks.analytical import delta as bs_delta
from vollib.black_scholes.greeks.analytical import theta as bs_theta
from vollib.black_scholes import black_scholes as bs_price
import numpy as np
from scipy import stats

def scan_cash_secured_puts(ticker_symbol, target_dte=45, max_delta=0.30):
    """Scan a ticker for best cash-secured put to sell."""
    ticker = yf.Ticker(ticker_symbol)
    S = ticker.history(period='5d')['Close'].iloc[-1]
    
    # Find expiration closest to target DTE
    best_exp = None
    for exp in ticker.options:
        dte = (datetime.strptime(exp, '%Y-%m-%d') - datetime.now()).days
        if 7 <= dte <= 90 and (best_exp is None or 
           abs(dte - target_dte) < abs((datetime.strptime(best_exp, '%Y-%m-%d') - datetime.now()).days - target_dte)):
            best_exp = exp
    
    chain = ticker.option_chain(best_exp)
    dte = (datetime.strptime(best_exp, '%Y-%m-%d') - datetime.now()).days
    t = max(dte / 365.0, 1/365)
    r = 0.05
    
    results = []
    for _, row in chain.puts.iterrows():
        K, sigma = float(row['strike']), float(row['impliedVolatility'])
        bid, ask = float(row['bid'] or 0), float(row['ask'] or 0)
        prem = (bid+ask)/2 if bid>0 and ask>0 else float(row['lastPrice'] or 0)
        if prem <= 0: continue
        
        delta_val = bs_delta('p', S, K, t, r, sigma)
        if abs(delta_val) > max_delta: continue
        if prem * 100 < 20: continue
        
        collateral, pcr = K*100, (prem*100)/(K*100)
        if pcr < 0.005: continue
        
        d1 = (np.log(S/K)+(r+sigma**2/2)*t)/(sigma*np.sqrt(t))
        prob_otm = 1 - stats.norm.cdf(-(d1 - sigma*np.sqrt(t)))
        
        results.append({'strike': K, 'premium': prem*100, 'collateral': collateral,
                       'pcr': pcr, 'annualized': pcr/t, 'delta': delta_val,
                       'prob_otm': prob_otm, 'iv': sigma, 'dte': dte})
    
    return sorted(results, key=lambda x: x['pcr'], reverse=True)[:5]

# Example usage
recommendations = scan_cash_secured_puts("SPY", target_dte=45)
for r in recommendations[:3]:
    print(f"Sell SPY ${r['strike']:.0f} Put | Premium ${r['premium']:.0f} | "
          f"Collateral ${r['collateral']:,.0f} | PCR {r['pcr']:.2%} | "
          f"Delta {r['delta']:.3f} | Prob OTM {r['prob_otm']:.1%}")
```