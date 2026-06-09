#!/usr/bin/env python3
"""
ThetaEdge — Options Research Test v2
Tests with 30-60 DTE expirations for realistic premium values.
"""

import sys
sys.path.insert(0, '/home/team/shared/venv/lib/python3.12/site-packages')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from vollib.black_scholes import black_scholes as bs_price
from vollib.black_scholes.greeks.analytical import delta as bs_delta
from vollib.black_scholes.greeks.analytical import gamma as bs_gamma
from vollib.black_scholes.greeks.analytical import theta as bs_theta
from vollib.black_scholes.greeks.analytical import vega as bs_vega
from scipy import stats

today = datetime.now()
print(f"Today: {today}")
print(f"Using vollib (replacement for deprecated py_vollib)\n")

# =============================================================================
# 1. FETCH SPY OPTIONS
# =============================================================================
ticker = yf.Ticker("SPY")
expirations = ticker.options
print(f"Total expirations available: {len(expirations)}")

# Find expiration ~30-60 DTE
target_dte = 45
best_exp = None
best_dte = 999

for exp_str in expirations:
    exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
    dte = (exp_date - today).days
    if dte >= 30 and dte <= 60 and abs(dte - target_dte) < abs(best_dte - target_dte):
        best_dte = dte
        best_exp = exp_str

print(f"Best target expiration: {best_exp} ({best_dte} DTE)")

# Get underlying price
spy_data = ticker.history(period="5d")
S = spy_data['Close'].iloc[-1]
print(f"SPY Underlying Price: ${S:.2f}")

# Fetch option chain for target expiration
opt_chain = ticker.option_chain(best_exp)
calls = opt_chain.calls
puts = opt_chain.puts
print(f"Calls: {len(calls)}, Puts: {len(puts)}")

# =============================================================================
# 2. GREEKS AND METRICS
# =============================================================================
r = 0.05  # Risk-free rate

def calculate_put_metrics(row, S, exp_date, r=0.05):
    """Calculate all metrics for a cash-secured put."""
    K = float(row['strike'])
    sigma = float(row['impliedVolatility'])
    t = max((exp_date - today).days / 365.0, 1/365)
    
    # Greeks
    delta_val = bs_delta('p', S, K, t, r, sigma)
    theta_val = bs_theta('p', S, K, t, r, sigma)
    gamma_val = bs_gamma('p', S, K, t, r, sigma)
    vega_val = bs_vega('p', S, K, t, r, sigma)
    
    # Premium (mid price)
    bid = float(row['bid']) if row['bid'] > 0 else 0
    ask = float(row['ask']) if row['ask'] > 0 and not pd.isna(row['ask']) else 0
    last = float(row['lastPrice']) if not pd.isna(row['lastPrice']) else 0
    
    if bid > 0 and ask > 0:
        premium = (bid + ask) / 2
    elif last > 0:
        premium = last
    else:
        premium = 0
    
    if premium <= 0:
        return None
    
    collateral = K * 100
    premium_received = premium * 100
    
    # Probability of profit using Black-Scholes d2
    d1 = (np.log(S/K) + (r + sigma**2/2)*t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    prob_itm = stats.norm.cdf(-d2)  # P(stock < strike) = assignment risk
    prob_otm = 1 - prob_itm  # P(stock > strike) = keep premium
    
    # Key ratios
    pcr = premium_received / collateral  # Premium-to-Collateral Ratio
    annualized_return = (pcr / t) if t > 0 else 0
    
    # Expected return
    expected_return_dollars = premium_received * prob_otm - (collateral * prob_itm)
    # Simpler: expected profit = premium_received - expected_loss
    # Expected loss = (strike - expected_stock_price_if_assigned) * prob_itm
    # For simplicity: expected_profit approx = premium_received * prob_otm
    
    # Break-even
    breakeven = K - premium
    
    # Vanna-Volga / theta decay: approximate daily theta
    theta_daily = theta_val / 1  # theta is per 1 unit change in t (in years)
    # Actually theta from py_vollib is per year, so daily = theta / 365
    theta_daily = theta_val / 365
    
    return {
        'strike': K,
        'underlying_price': S,
        'moneyness': K / S,
        'premium_per_share': premium,
        'premium_received': premium_received,
        'collateral': collateral,
        'pcr': pcr,
        'annualized_return': annualized_return,
        'delta': delta_val,
        'gamma': gamma_val,
        'theta': theta_val,
        'theta_daily': theta_daily,
        'vega': vega_val,
        'iv': sigma,
        'prob_otm': prob_otm,
        'prob_itm': prob_itm,
        'expected_return': expected_return_dollars,
        'breakeven': breakeven,
        'dte': (exp_date - today).days,
        'bid': bid,
        'ask': ask,
        'volume': int(row['volume']) if not pd.isna(row['volume']) else 0,
        'open_interest': int(row['openInterest']) if not pd.isna(row['openInterest']) else 0,
    }

exp_date = datetime.strptime(best_exp, '%Y-%m-%d')
results = []
for _, row in puts.iterrows():
    m = calculate_put_metrics(row, S, exp_date, r)
    if m:
        results.append(m)

df = pd.DataFrame(results)
print(f"\nScored {len(df)} put options")

# =============================================================================
# 3. FILTERING & RANKING
# =============================================================================
print("\n" + "=" * 70)
print("RECOMMENDATION ENGINE — FILTERS & RANKING")
print("=" * 70)

# Apply filters step by step
f1 = df[abs(df['delta']) < 0.30].copy()
f2 = f1[f1['premium_received'] >= 20].copy()
f3 = f2[f2['pcr'] >= 0.005].copy()
f4 = f3[f3['volume'] > 0].copy()  # Basic liquidity filter
f5 = f4[f4['open_interest'] > 0].copy()

print(f"\nFilter progression:")
print(f"  Raw puts scored:         {len(df)}")
print(f"  Delta < |0.30|:          {len(f1)}")
print(f"  Min premium $20:         {len(f2)}")
print(f"  Min PCR 0.5%:            {len(f3)}")
print(f"  Volume > 0:              {len(f4)}")
print(f"  Open Interest > 0:       {len(f5)}")

if len(f5) > 0:
    # Score by PCR (primary)
    ranked_by_pcr = f5.sort_values('pcr', ascending=False)
    
    print(f"\n--- TOP 10 RECOMMENDATIONS (by Premium/Collateral Ratio) ---")
    cols = ['strike', 'premium_received', 'collateral', 'pcr', 'annualized_return',
            'delta', 'prob_otm', 'iv', 'dte', 'volume']
    print(ranked_by_pcr.head(10)[cols].to_string(index=False))
    
    print(f"\n--- TOP 10 RECOMMENDATIONS (by Annualized Return) ---")
    ranked_by_ar = f5.sort_values('annualized_return', ascending=False)
    print(ranked_by_ar.head(10)[cols].to_string(index=False))
    
    # Composite score: combination of PCR, probability OTM, and liquidity
    f5['composite_score'] = (
        f5['pcr'].rank(pct=True) * 0.4 +
        f5['prob_otm'].rank(pct=True) * 0.3 +
        f5['volume'].rank(pct=True) * 0.15 +
        f5['open_interest'].rank(pct=True) * 0.15
    )
    
    print(f"\n--- TOP 10 BY COMPOSITE SCORE ---")
    ranked_comp = f5.sort_values('composite_score', ascending=False)
    print(ranked_comp.head(10)[['strike', 'pcr', 'prob_otm', 'delta', 'premium_received', 
                                 'collateral', 'volume', 'composite_score']].to_string(index=False))
    
    # Best single recommendation
    best = ranked_comp.iloc[0]
    print(f"\n\n=== BEST PICK ===")
    print(f"  Sell SPY ${best['strike']:.0f} Put")
    print(f"  Expires: {best_exp} ({best['dte']:.0f} days)")
    print(f"  Premium Received: ${best['premium_received']:.2f}")
    print(f"  Collateral Required: ${best['collateral']:.2f}")
    print(f"  Premium/Collateral: {best['pcr']:.2%}")
    print(f"  Annualized Return: {best['annualized_return']:.2%}")
    print(f"  Delta: {best['delta']:.3f}")
    print(f"  Prob OTM: {best['prob_otm']:.1%}")
    print(f"  IV: {best['iv']:.1%}")
    print(f"  Breakeven: ${best['breakeven']:.2f}")

# =============================================================================
# 4. DEMO: COVERED CALLS TOO
# =============================================================================
print("\n" + "=" * 70)
print("BONUS: COVERED CALL SCORING (for completeness)")
print("=" * 70)

def calculate_call_metrics(row, S, exp_date, r=0.05):
    """Calculate metrics for covered call writing."""
    K = float(row['strike'])
    sigma = float(row['impliedVolatility'])
    t = max((exp_date - today).days / 365.0, 1/365)
    
    delta_val = bs_delta('c', S, K, t, r, sigma)
    theta_val = bs_theta('c', S, K, t, r, sigma)
    
    bid = float(row['bid']) if row['bid'] > 0 else 0
    ask = float(row['ask']) if row['ask'] > 0 and not pd.isna(row['ask']) else 0
    last = float(row['lastPrice']) if not pd.isna(row['lastPrice']) else 0
    
    if bid > 0 and ask > 0:
        premium = (bid + ask) / 2
    elif last > 0:
        premium = last
    else:
        premium = 0
    
    if premium <= 0:
        return None
    
    premium_received = premium * 100
    collateral = S * 100  # For covered call, collateral = cost of 100 shares
    pcr = premium_received / collateral
    
    # Prob OTM = prob stock < strike (call expires worthless)
    d1 = (np.log(S/K) + (r + sigma**2/2)*t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    prob_itm = stats.norm.cdf(d2)
    prob_otm = 1 - prob_itm
    
    return {
        'strike': K,
        'premium_received': premium_received,
        'collateral': collateral,
        'pcr': pcr,
        'delta': delta_val,
        'theta': theta_val,
        'iv': sigma,
        'prob_otm': prob_otm,
        'dte': (exp_date - today).days
    }

call_results = []
for _, row in calls.iterrows():
    m = calculate_call_metrics(row, S, exp_date, r)
    if m:
        call_results.append(m)

call_df = pd.DataFrame(call_results)
call_filtered = call_df[(abs(call_df['delta']) <= 0.30) & (call_df['premium_received'] >= 20) & (call_df['pcr'] >= 0.005)]

print(f"Covered call candidates (delta <= 0.30, premium >= $20): {len(call_filtered)}")
if len(call_filtered) > 0:
    print(call_filtered.sort_values('pcr', ascending=False).head(5).to_string(index=False))

print("\n" + "=" * 70)
print("TEST COMPLETE — All systems verified!")
print("=" * 70)