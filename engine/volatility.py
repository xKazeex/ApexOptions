"""
Volatility analysis module for the Apex Options Analytics Engine.

Provides tools for analyzing implied volatility (IV), realized volatility (RV),
IV rank/percentile, volatility surface dynamics, skew patterns, and
volatility regime detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .data_fetcher import OptionContract, OptionsChain, UnderlyingData
from .greeks import black_scholes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class VolatilityAnalysis:
    """Result of volatility analysis for a ticker.

    Attributes:
        ticker: Ticker symbol.
        timestamp: Analysis timestamp.
        underlying_price: Current price of the underlying.
        current_iv: ATM implied volatility (average of call and put ATM IV).
        iv_rank: IV Rank as percentage (0-100).
        iv_percentile: IV Percentile as percentage (0-100).
        realized_vol_10d: 10-day realized volatility (annualized, decimal).
        realized_vol_20d: 20-day realized volatility (annualized, decimal).
        realized_vol_60d: 60-day realized volatility (annualized, decimal).
        iv_rv_ratio_20d: Ratio of current IV to 20-day RV.
        iv_rv_ratio_60d: Ratio of current IV to 60-day RV.
        vol_regime: Identified volatility regime string.
        term_structure: Description of forward volatility term structure.
        skew_description: Description of volatility skew/smile.
        is_premium_selling_opportunity: Boolean flag for opportunity quality.
        opportunity_score: Numeric score (0-100) for opportunity strength.
    """

    ticker: str
    timestamp: datetime = field(default_factory=datetime.now)
    underlying_price: float = 0.0
    current_iv: float = 0.0
    iv_rank: float = 0.0
    iv_percentile: float = 0.0
    realized_vol_10d: float = 0.0
    realized_vol_20d: float = 0.0
    realized_vol_60d: float = 0.0
    iv_rv_ratio_20d: float = 0.0
    iv_rv_ratio_60d: float = 0.0
    vol_regime: str = "unknown"
    term_structure: str = "unknown"
    skew_description: str = "unknown"
    is_premium_selling_opportunity: bool = False
    opportunity_score: float = 0.0


# ---------------------------------------------------------------------------
# Realized Volatility
# ---------------------------------------------------------------------------


def calculate_realized_volatility(
    prices: pd.Series,
    window: int = 20,
    annualize: bool = True,
) -> float:
    """Calculate annualized realized volatility from a price series.

    Uses the standard deviation of log returns over the given window,
    annualized by sqrt(252) for trading days.

    Args:
        prices: Series of price data (must be sorted chronologically).
        window: Rolling window size in trading days.
        annualize: Whether to annualize the result.

    Returns:
        Realized volatility as a decimal (e.g., 0.20 = 20%).
    """
    if len(prices) < window + 1:
        logger.warning(
            "Not enough data points: need at least %d, got %d",
            window + 1,
            len(prices),
        )
        return 0.0

    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if len(log_returns) < window:
        return 0.0

    # Use the most recent window
    recent_returns = log_returns.iloc[-window:]
    rv = float(np.std(recent_returns, ddof=1))

    if annualize:
        rv *= np.sqrt(252)

    return rv


# ---------------------------------------------------------------------------
# IV Rank & Percentile
# ---------------------------------------------------------------------------


def calculate_iv_rank(
    current_iv: float,
    historical_ivs: list[float] | pd.Series,
) -> float:
    """Calculate IV Rank as a percentage.

    IV Rank = (current IV - min IV) / (max IV - min IV) * 100

    Args:
        current_iv: Current ATM implied volatility.
        historical_ivs: List or Series of historical IV values (52-week lookback).

    Returns:
        IV Rank as percentage (0-100). Returns 50.0 if no valid range.
    """
    if historical_ivs is None or len(historical_ivs) == 0:
        return 50.0

    if isinstance(historical_ivs, list):
        historical_ivs = pd.Series(historical_ivs)

    historical_ivs = historical_ivs.dropna()

    if len(historical_ivs) == 0:
        return 50.0

    min_iv = historical_ivs.min()
    max_iv = historical_ivs.max()

    if max_iv == min_iv:
        return 50.0

    rank = (current_iv - min_iv) / (max_iv - min_iv) * 100.0
    return float(np.clip(rank, 0.0, 100.0))


def calculate_iv_percentile(
    current_iv: float,
    historical_ivs: list[float] | pd.Series,
) -> float:
    """Calculate IV Percentile.

    The percentage of days in the lookback period where IV was lower
    than the current IV.

    Args:
        current_iv: Current ATM implied volatility.
        historical_ivs: List or Series of historical IV values.

    Returns:
        IV Percentile as percentage (0-100).
    """
    if historical_ivs is None or len(historical_ivs) == 0:
        return 50.0

    if isinstance(historical_ivs, list):
        historical_ivs = pd.Series(historical_ivs)

    historical_ivs = historical_ivs.dropna()

    if len(historical_ivs) == 0:
        return 50.0

    count_lower = int((historical_ivs < current_iv).sum())
    percentile = (count_lower / len(historical_ivs)) * 100.0
    return float(np.clip(percentile, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Term structure & skew analysis
# ---------------------------------------------------------------------------


@dataclass
class TermStructurePoint:
    """Volatility at a specific expiration."""

    expiration: datetime
    dte: int
    atm_call_iv: float
    atm_put_iv: float
    atm_iv: float  # average


def analyze_term_structure(
    options_chain: OptionsChain,
    underlying_price: float,
) -> list[TermStructurePoint]:
    """Analyze the volatility term structure across expirations.

    For each expiration, finds the ATM straddle IV (average of ATM call
    and put IVs).

    Args:
        options_chain: Full options chain data.
        underlying_price: Current underlying price.

    Returns:
        List of TermStructurePoint sorted by DTE.
    """
    from collections import defaultdict

    expiration_data: dict[datetime, dict[str, list[float]]] = defaultdict(
        lambda: {"call_ivs": [], "put_ivs": []}
    )

    for contract in options_chain.calls:
        expiration_data[contract.expiration]["call_ivs"].append(
            (contract.strike, contract.implied_volatility)
        )

    for contract in options_chain.puts:
        expiration_data[contract.expiration]["put_ivs"].append(
            (contract.strike, contract.implied_volatility)
        )

    term_points: list[TermStructurePoint] = []
    for exp_date in sorted(expiration_data.keys()):
        data = expiration_data[exp_date]

        # Find ATM strikes
        call_ivs = sorted(data["call_ivs"], key=lambda x: abs(x[0] - underlying_price))
        put_ivs = sorted(data["put_ivs"], key=lambda x: abs(x[0] - underlying_price))

        if not call_ivs or not put_ivs:
            continue

        atm_call_iv = call_ivs[0][1]
        atm_put_iv = put_ivs[0][1]
        atm_iv = (atm_call_iv + atm_put_iv) / 2.0

        dte = max(0, (exp_date - datetime.now()).days)

        term_points.append(
            TermStructurePoint(
                expiration=exp_date,
                dte=dte,
                atm_call_iv=atm_call_iv,
                atm_put_iv=atm_put_iv,
                atm_iv=atm_iv,
            )
        )

    return term_points


def describe_term_structure(
    term_points: list[TermStructurePoint],
) -> str:
    """Describe the volatility term structure.

    Returns a human-readable description: 'backwardated', 'contango',
    'flat', or 'mixed'.
    """
    if len(term_points) < 2:
        return "flat (insufficient data)"

    # Compare near-term vs longer-term IV
    near = term_points[0].atm_iv
    far = term_points[-1].atm_iv

    if near > far * 1.05:
        return "backwardated (near-term vol > longer-term vol)"
    elif far > near * 1.05:
        return "contango (longer-term vol > near-term vol)"
    else:
        return "flat (term structure relatively flat)"


def describe_skew(
    options_chain: OptionsChain,
    underlying_price: float,
    target_dte_range: tuple[int, int] = (30, 60),
) -> str:
    """Describe the volatility skew for a target DTE range.

    Analyzes the difference between OTM put IV and OTM call IV to
    determine if the skew is neutral, put-skewed, or call-skewed.

    Args:
        options_chain: Full options chain data.
        underlying_price: Current underlying price.
        target_dte_range: (min_dte, max_dte) to focus analysis.

    Returns:
        Description string.
    """
    now = datetime.now()

    # Filter contracts to target DTE range
    puts = [
        c
        for c in options_chain.puts
        if target_dte_range[0]
        <= (c.expiration - now).days
        <= target_dte_range[1]
        and c.implied_volatility > 0
    ]
    calls = [
        c
        for c in options_chain.calls
        if target_dte_range[0]
        <= (c.expiration - now).days
        <= target_dte_range[1]
        and c.implied_volatility > 0
    ]

    if not puts or not calls:
        return "insufficient data for skew analysis"

    # Find OTM puts (strikes below underlying) and OTM calls (above underlying)
    otm_puts = [p for p in puts if p.strike < underlying_price]
    otm_calls = [c for c in calls if c.strike > underlying_price]

    if not otm_puts or not otm_calls:
        return "insufficient OTM data"

    # Average IV for the first few OTM strikes
    close_otm_puts = sorted(
        otm_puts, key=lambda p: underlying_price - p.strike
    )[:3]
    close_otm_calls = sorted(
        otm_calls, key=lambda c: c.strike - underlying_price
    )[:3]

    avg_put_iv = np.mean([p.implied_volatility for p in close_otm_puts])
    avg_call_iv = np.mean([c.implied_volatility for c in close_otm_calls])

    skew_ratio = avg_put_iv / avg_call_iv if avg_call_iv > 0 else 1.0

    if skew_ratio > 1.3:
        return "strong put skew (OTM puts significantly elevated)"
    elif skew_ratio > 1.15:
        return "moderate put skew (OTM puts modestly elevated)"
    elif skew_ratio < 0.85:
        return "call skew (OTM calls elevated relative to puts)"
    else:
        return "neutral (relatively balanced skew)"


# ---------------------------------------------------------------------------
# Volatility regime detection
# ---------------------------------------------------------------------------


def detect_volatility_regime(
    iv_rank: float,
    iv_rv_ratio: float,
    iv_percentile: float,
) -> str:
    """Detect the current volatility regime.

    Rules:
        - High IV Regime: IVR > 70 → best for premium selling
        - Elevated IV Regime: IVR 50-70 → good for premium selling
        - Normal IV Regime: IVR 20-50 → selective premium selling
        - Low IV Regime: IVR < 20 → cautious with premium selling
        - Crush Opportunity: High IVR + high IV/RV ratio → IV crush likely

    Args:
        iv_rank: IV Rank (0-100).
        iv_rv_ratio: Ratio of IV to RV.
        iv_percentile: IV Percentile (0-100).

    Returns:
        String describing the volatility regime.
    """
    if iv_rank >= 70 and iv_rv_ratio > 1.3:
        return "IV crush opportunity — elevated IV relative to HV"
    elif iv_rank >= 70:
        return "high IV regime — favorable for premium selling"
    elif iv_rank >= 50:
        return "elevated IV regime — good for premium selling"
    elif iv_rank >= 20:
        return "normal IV regime — selective premium selling"
    else:
        return "low IV regime — premiums may not justify risk"


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------


def analyze_volatility(
    underlying: UnderlyingData,
    options_chain: OptionsChain,
    rv_windows: tuple[int, ...] = (10, 20, 60, 120),
    iv_high_threshold: float = 50.0,
) -> VolatilityAnalysis:
    """Run full volatility analysis on a ticker.

    This is the main entry point for volatility analysis. It calculates:
    - Current ATM IV from the options chain
    - IV Rank and IV Percentile from historical data
    - Realized volatility over multiple windows
    - IV/RV ratios
    - Volatility regime classification

    Args:
        underlying: Underlying data including historical prices.
        options_chain: Options chain data.
        rv_windows: Tuple of trading-day windows for RV calculation.
        iv_high_threshold: IVR score above which is considered high IV.

    Returns:
        VolatilityAnalysis with complete results.
    """
    # Get ATM IV
    atm_iv = _get_atm_iv(options_chain)

    # Get historical prices
    prices = underlying.historical_prices["Close"] if underlying.historical_prices is not None else pd.Series(dtype=float)

    # Calculate IV Rank and IV Percentile
    # If we don't have historical IVs, we estimate from ATM IV over time
    # For bootstrapping, we generate a synthetic history based on RV
    historical_ivs = _estimate_historical_ivs(prices, atm_iv)

    iv_rank = calculate_iv_rank(atm_iv, historical_ivs)
    iv_percentile = calculate_iv_percentile(atm_iv, historical_ivs)

    # Calculate RVs
    rv_results = {}
    for window in rv_windows:
        rv_results[window] = calculate_realized_volatility(prices, window=window)

    rv_20d = rv_results.get(20, 0.0)
    rv_60d = rv_results.get(60, 0.0)

    # IV/RV ratios
    iv_rv_20d = atm_iv / rv_20d if rv_20d > 0 else 0.0
    iv_rv_60d = atm_iv / rv_60d if rv_60d > 0 else 0.0

    # Volatility regime
    regime = detect_volatility_regime(iv_rank, iv_rv_20d, iv_percentile)

    # Term structure
    term_points = analyze_term_structure(options_chain, underlying.price)
    term_structure = describe_term_structure(term_points)

    # Skew
    skew = describe_skew(options_chain, underlying.price)

    # Opportunity score (0-100)
    opportunity_score = _calculate_opportunity_score(
        iv_rank=iv_rank,
        iv_percentile=iv_percentile,
        iv_rv_ratio=iv_rv_20d,
        regime=regime,
    )

    is_opportunity = opportunity_score >= 50

    return VolatilityAnalysis(
        ticker=underlying.ticker,
        timestamp=datetime.now(),
        underlying_price=underlying.price,
        current_iv=atm_iv,
        iv_rank=iv_rank,
        iv_percentile=iv_percentile,
        realized_vol_10d=rv_results.get(10, 0.0),
        realized_vol_20d=rv_20d,
        realized_vol_60d=rv_60d,
        iv_rv_ratio_20d=iv_rv_20d,
        iv_rv_ratio_60d=iv_rv_60d,
        vol_regime=regime,
        term_structure=term_structure,
        skew_description=skew,
        is_premium_selling_opportunity=is_opportunity,
        opportunity_score=opportunity_score,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_atm_iv(options_chain: OptionsChain) -> float:
    """Get the ATM implied volatility from the options chain."""
    atm_strike = options_chain.underlying_price

    # Find closest call and put to ATM
    best_call_iv = 0.0
    best_put_iv = 0.0
    min_call_dist = float("inf")
    min_put_dist = float("inf")

    for c in options_chain.calls:
        if c.implied_volatility <= 0:
            continue
        dist = abs(c.strike - atm_strike)
        if dist < min_call_dist:
            min_call_dist = dist
            best_call_iv = c.implied_volatility

    for p in options_chain.puts:
        if p.implied_volatility <= 0:
            continue
        dist = abs(p.strike - atm_strike)
        if dist < min_put_dist:
            min_put_dist = dist
            best_put_iv = p.implied_volatility

    # Average ATM call and put IV
    ivs = [iv for iv in [best_call_iv, best_put_iv] if iv > 0]
    if not ivs:
        return 0.0

    return float(np.mean(ivs))


def _estimate_historical_ivs(
    prices: pd.Series,
    current_iv: float,
) -> pd.Series:
    """Estimate historical IVs from realized volatility.

    In the absence of a direct historical IV feed, we estimate it as
    RV * typical_iv_rv_ratio (1.2-1.3 for equities). This allows us
    to compute IV Rank/Percentile during bootstrapping.

    When paid data sources are integrated, this should be replaced with
    actual historical IV data.
    """
    if len(prices) < 10:
        # Just return a small series around current IV
        return pd.Series([current_iv * 0.8, current_iv, current_iv * 1.2])

    # Calculate rolling 20-day RV and estimate IV as RV * 1.25 (typical premium)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    rolling_rv = log_returns.rolling(window=20).std() * np.sqrt(252)

    # Historical IV ≈ RV * 1.25 (typical IV premium over RV)
    estimated_ivs = rolling_rv * 1.25

    # Drop NaN values
    estimated_ivs = estimated_ivs.dropna()

    if len(estimated_ivs) < 10:
        return pd.Series([current_iv * 0.8, current_iv, current_iv * 1.2])

    return estimated_ivs


def _calculate_opportunity_score(
    iv_rank: float,
    iv_percentile: float,
    iv_rv_ratio: float,
    regime: str,
) -> float:
    """Calculate a composite premium-selling opportunity score (0-100).

    Higher scores indicate better conditions for selling premium.
    """
    score = 0.0

    # IV Rank component (0-40 points)
    if iv_rank >= 70:
        score += 40
    elif iv_rank >= 50:
        score += 30
    elif iv_rank >= 30:
        score += 20
    else:
        score += 10

    # IV/RV ratio component (0-30 points)
    if iv_rv_ratio >= 1.5:
        score += 30
    elif iv_rv_ratio >= 1.3:
        score += 25
    elif iv_rv_ratio >= 1.15:
        score += 20
    elif iv_rv_ratio >= 1.0:
        score += 10
    else:
        score += 5

    # IV Percentile component (0-30 points)
    if iv_percentile >= 70:
        score += 30
    elif iv_percentile >= 50:
        score += 20
    elif iv_percentile >= 30:
        score += 15
    else:
        score += 10

    return float(np.clip(score, 0.0, 100.0))