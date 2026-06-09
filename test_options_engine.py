#!/usr/bin/env python3
"""
ThetaEdge — Options Research Test Script
Verifies yfinance options data access, Black-Scholes Greeks, and scoring algorithm.
"""

import sys
sys.path.insert(0, '/home/team/shared/venv/lib/python3.12/site-packages')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from py_vollib.black_scholes import black_scholes as bs_price
from py_vollib.black_scholes.greeks.analytical import delta as bs_delta
from py_vollib.black_scholes.greeks.analytical import gamma as bs_gamma
from py_vollib.black_scholes.greeks.analytical import theta as bs_theta
from py_vollib.black_scholes.greeks.analytical import vega as bs_vega
from py_vollib.black_scholes.implied_volatility import implied_volatility as bs_iv

# =============================================================================
# 1. YFINANCE DATA ACCESS TEST
# =============================================================================
print("=" * 70)
print("1. YFINANCE OPTIONS DATA ACCESS TEST")
print("=" * 70)

ticker = yf.Ticker("SPY")
print(f"Ticker: SPY")

# Get available expiration dates
expirations = ticker.options
print(f"Available expirations (first 5): {expirations[:5]}")
print(f"Number of expirations: {len(expirations)}")

# Fetch options chain for nearest expiration
opt_chain = ticker.option_chain(expirations[0])
calls = opt_chain.calls
puts = opt_chain.puts

print(f"\nCalls columns: {list(calls.columns)}")
print(f"Puts columns: {list(puts.columns)}")
print(f"\nCalls shape: {calls.shape}, Puts shape: {puts.shape}")

# Show a few put entries (relevant for cash-secured puts)
print(f"\nSample puts data (first 5):")
print(puts[['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility', 'volume', 'openInterest']].head(10).to_string())

# Get underlying price from the ticker
spy_info = ticker.history(period="2d")
underlying_price = spy_info['Close'].iloc[-1]
print(f"\nSPY underlying price: ${underlying_price:.2f}")

# =============================================================================
# 2. BLACK-SCHOLES GREEKS CALCULATION
# =============================================================================
print("\n" + "=" * 70)
print("2. BLACK-SCHOLES GREEKS CALCULATION")
print("=" * 70)

# Use a sample option to compute Greeks
sample_put = puts.iloc[5]  # Pick a reasonable OTM put
S = underlying_price
K = float(sample_put['strike'])
expiry = datetime.strptime(expirations[0], '%Y-%m-%d')
today = datetime.now()
t = max((expiry - today).days / 365.0, 1/365)  # Time to expiry in years
r = 0.05  # Risk-free rate (approximate)
sigma = float(sample_put['impliedVolatility'])

print(f"Sample put option:")
print(f"  Strike: ${K:.2f} (Underlying: ${S:.2f})")
print(f"  Expiry: {expirations[0]} (T={t:.4f} years)")
print(f"  Implied Vol: {sigma:.2%}")
print(f"  Bid: ${sample_put['bid']}, Ask: ${sample_put['ask']}, Last: ${sample_put['lastPrice']}")

# Calculate Greeks using py_vollib
delta_val = bs_delta('p', S, K, t, r, sigma)
gamma_val = bs_gamma('p', S, K, t, r, sigma)
theta_val = bs_theta('p', S, K, t, r, sigma)
vega_val = bs_vega('p', S, K, t, r, sigma)

# Theoretical price
theo_price = bs_price('p', S, K, t, r, sigma)

# Collateral for cash-secured put = strike * 100
collateral = K * 100

# Premium received (using mid price or ask price for conservative estimate)
premium_per_share = (sample_put['bid'] + sample_put['ask']) / 2 if sample_put['bid'] > 0 and sample_put['ask'] > 0 else sample_put['lastPrice']
premium_received = premium_per_share * 100

# Key metrics
premium_to_collateral = premium_received / collateral
expected_return = premium_to_collateral * (365 / (t * 365))  # Annualized

print(f"\n  Theo Price (BS): ${theo_price:.2f}")
print(f"  Mid Premium/Share: ${premium_per_share:.2f}")
print(f"  Premium Received (100 shares): ${premium_received:.2f}")
print(f"  Collateral Required: ${collateral:.2f}")
print(f"  Premium/Collateral Ratio: {premium_to_collateral:.4f} ({premium_to_collateral*100:.2f}%)")
print(f"  Expected Return (annualized): {expected_return:.2%}")
print(f"\n  Greeks (put):")
print(f"    Delta: {delta_val:.4f}")
print(f"    Gamma: {gamma_val:.4f}")
print(f"    Theta: {theta_val:.4f}")
print(f"    Vega: {vega_val:.4f}")
print(f"  Probability OTM (delta based): {1 - abs(delta_val):.2%}")

# =============================================================================
# 3. PROBABILITY OF PROFIT ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("3. PROBABILITY OF PROFIT ANALYSIS")
print("=" * 70)

# Delta gives us approximate probability of finishing ITM
# For puts: |delta| ≈ P(ITM). So P(OTM) = 1 - |delta|
# P(profit) ≈ P(stock > strike at expiry) for short put = 1 - |delta|

from scipy import stats

# Calculate probability using Black-Scholes framework
d1 = (np.log(S/K) + (r + sigma**2/2)*t) / (sigma * np.sqrt(t))
d2 = d1 - sigma * np.sqrt(t)

# For a short put: profit if stock > strike at expiry
prob_itm = stats.norm.cdf(-d2)  # P(S < K)
prob_otm = 1 - prob_itm  # P(S > K) -> option expires worthless, seller keeps premium

print(f"  Using Black-Scholes d2:")
print(f"    d2 = {d2:.4f}")
print(f"    P(ITM) = {prob_itm:.2%} (stock below strike at expiry)")
print(f"    P(OTM) = {prob_otm:.2%} (stock above strike -> keep premium)")
print(f"    Delta-based P(OTM): {1 - abs(delta_val):.2%}")

# =============================================================================
# 4. SCORING ALGORITHM PROTOTYPE
# =============================================================================
print("\n" + "=" * 70)
print("4. RECOMMENDATION SCORING ALGORITHM")
print("=" * 70)

def score_cash_secured_put(row, S, t, r, risk_free_rate=0.05):
    """
    Score a cash-secured put option for selling recommendation.
    
    Returns a dict of metrics and a composite score.
    """
    K = float(row['strike'])
    sigma = float(row['impliedVolatility'])
    
    # Calculate Greeks
    delta_val = bs_delta('p', S, K, t, r, sigma)
    theta_val = bs_theta('p', S, K, t, r, sigma)
    
    # Premium (use mid price or ask if bid is 0)
    bid = float(row['bid']) if row['bid'] > 0 else 0
    ask = float(row['ask']) if row['ask'] > 0 else 0
    last = float(row['lastPrice'])
    
    if bid > 0 and ask > 0:
        premium = (bid + ask) / 2
    elif last > 0:
        premium = last
    else:
        premium = 0
    
    # Only consider if premium > 0
    if premium <= 0:
        return None
    
    collateral = K * 100
    premium_received = premium * 100
    
    # Key ratio: premium / collateral
    pcr = premium_received / collateral
    
    # Annualized return if held to expiry
    annualized_return = pcr / t
    
    # Probability OTM = 1 - |delta| for puts
    prob_otm = 1 - abs(delta_val)
    
    # Expected return = premium * prob_otm (simplified, ignoring tail risk)
    expected_return = premium_received * prob_otm
    
    # Risk-adjusted score: we want high premium/collateral, high prob OTM, good theta
    # Normalize components to 0-1 scale, then combine
    
    # Filter criteria checks (these are hard pass/fail in production, for scoring we just note them)
    delta_ok = abs(delta_val) < 0.30  # High probability OTM
    min_premium_ok = premium_received >= 20  # Minimum $20 premium per contract
    min_pcr_ok = pcr >= 0.005  # At least 0.5% return on collateral for the period
    
    return {
        'strike': K,
        'premium_per_share': premium,
        'premium_received': premium_received,
        'collateral': collateral,
        'pcr': pcr,
        'annualized_return': annualized_return,
        'delta': delta_val,
        'theta': theta_val,
        'iv': sigma,
        'prob_otm': prob_otm,
        'expected_return_dollars': expected_return,
        'delta_ok': delta_ok,
        'min_premium_ok': min_premium_ok,
        'min_pcr_ok': min_pcr_ok
    }

# Score all puts in the chain
print("Scoring all puts for nearest expiration...")
scored_puts = []
for i, (_, row) in enumerate(puts.iterrows()):
    result = score_cash_secured_put(row, S, t, r)
    if result:
        # Check DTE (days to expiry)
        dte = max((expiry - today).days, 1)
        result['dte'] = dte
        scored_puts.append(result)

scored_df = pd.DataFrame(scored_puts)

if len(scored_df) > 0:
    print(f"  Scored {len(scored_df)} puts with valid premiums")
    
    # Apply filters
    filtered = scored_df[
        (scored_df['delta_ok']) & 
        (scored_df['min_premium_ok']) & 
        (scored_df['min_pcr_ok'])
    ].copy()
    
    print(f"  After delta < |0.30| filter: {len(scored_df[scored_df['delta_ok']])} options")
    print(f"  After min premium ($20) filter: {len(scored_df[scored_df['min_premium_ok']])} options")
    print(f"  After min PCR (0.5%) filter: {len(scored_df[scored_df['min_pcr_ok']])} options")
    print(f"  Passing all filters: {len(filtered)} options")
    
    if len(filtered) > 0:
        # Sort by PCR (premium-to-collateral ratio) descending
        filtered = filtered.sort_values('pcr', ascending=False)
        
        print(f"\n  Top 5 recommendations by PCR:")
        top_cols = ['strike', 'premium_received', 'collateral', 'pcr', 'annualized_return', 
                    'delta', 'prob_otm', 'iv', 'dte']
        print(filtered.head(5)[top_cols].to_string())
        
        # Also sort by expected return
        filtered = filtered.sort_values('expected_return_dollars', ascending=False)
        print(f"\n  Top 5 by expected return $:")
        print(filtered.head(5)[top_cols].to_string())

# =============================================================================
# 5. MULTI-EXPIRATION SCAN PROTOTYPE
# =============================================================================
print("\n" + "=" * 70)
print("5. MULTI-EXPIRATION SCAN (3 expirations)")
print("=" * 70)

for exp in expirations[:3]:
    exp_date = datetime.strptime(exp, '%Y-%m-%d')
    t_exp = max((exp_date - today).days / 365.0, 1/365)
    
    chain = ticker.option_chain(exp)
    puts_exp = chain.puts
    
    scored = []
    for _, row in puts_exp.iterrows():
        result = score_cash_secured_put(row, S, t_exp, r)
        if result:
            scored.append(result)
    
    if scored:
        sdf = pd.DataFrame(scored)
        sdf['dte'] = (exp_date - today).days
        
        # Filter
        sf = sdf[(sdf['delta_ok']) & (sdf['min_premium_ok']) & (sdf['min_pcr_ok'])]
        
        if len(sf) > 0:
            sf = sf.sort_values('pcr', ascending=False)
            best = sf.iloc[0]
            print(f"  Expiry {exp} ({sdf['dte'].iloc[0]} DTE): Best PCR = {best['pcr']:.2%}, "
                  f"Strike=${best['strike']:.0f}, Premium=${best['premium_received']:.2f}, "
                  f"Delta={best['delta']:.3f}, Prob OTM={best['prob_otm']:.1%}")
        else:
            print(f"  Expiry {exp} ({sdf['dte'].iloc[0]} DTE): No options passing filters")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print("\nSummary of findings:")
print(f"  - yfinance provides reliable options chain data with bid/ask/IV")
print(f"  - py_vollib provides Black-Scholes Greeks (delta, gamma, theta, vega)")
print(f"  - Mid-price premium calculation works with bid/ask spread")
print(f"  - PCR (Premium/Collateral Ratio) is the core scoring metric")
print(f"  - Delta < 0.30 filter keeps ~70%+ probability of profit for puts")
print(f"  - IV from yfinance can be used directly in Greek calculations")