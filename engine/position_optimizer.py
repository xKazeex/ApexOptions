"""
Position optimizer for the Apex Options Analytics Engine.

Given a selected strategy and risk parameters, determines optimal
strikes, spread width, DTE, and position sizing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from math import floor
from typing import Optional

import numpy as np

from .config import EngineConfig, DEFAULT_CONFIG
from .data_fetcher import OptionContract, OptionsChain
from .greeks import all_greeks, black_scholes
from .strategy_selector import OptionStrategy
from .volatility import VolatilityAnalysis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class PositionParameters:
    """Optimized position parameters for a trade.

    Attributes:
        strategy: The option strategy.
        underlying_ticker: Ticker symbol.
        underlying_price: Current price.
        dte: Days to expiration.
        short_strike: Strike of the short option (sold).
        long_strike: Strike of the long option (bought), if applicable.
        spread_width: Width of the spread.
        short_strike_2: Second short strike (for iron condor / strangle).
        long_strike_2: Second long strike (for iron condor).
        premium_collected: Total premium collected per contract.
        max_loss: Maximum loss per contract.
        max_profit: Maximum profit per contract.
        risk_reward_ratio: Max risk / max profit.
        probability_of_profit: Estimated POP.
        delta_short: Delta of the short option(s).
        theta_short: Theta (daily decay) of the short option.
        vega_short: Vega of the short option.
        contracts: Number of contracts to trade.
        capital_required: Total capital required for the position.
        risk_per_contract: Risk per single contract.
    """

    strategy: OptionStrategy
    underlying_ticker: str
    underlying_price: float
    dte: int
    short_strike: float = 0.0
    long_strike: float = 0.0
    spread_width: float = 0.0
    short_strike_2: float = 0.0
    long_strike_2: float = 0.0
    premium_collected: float = 0.0
    max_loss: float = 0.0
    max_profit: float = 0.0
    risk_reward_ratio: float = 0.0
    probability_of_profit: float = 0.0
    delta_short: float = 0.0
    theta_short: float = 0.0
    vega_short: float = 0.0
    contracts: int = 1
    capital_required: float = 0.0
    risk_per_contract: float = 0.0


# ---------------------------------------------------------------------------
# Optimization helpers
# ---------------------------------------------------------------------------


def _find_strikes_by_delta(
    contracts: list[OptionContract],
    target_delta: float,
    option_type: str,
    tol: float = 0.02,
) -> Optional[OptionContract]:
    """Find an option contract closest to a target delta.

    Args:
        contracts: List of option contracts.
        target_delta: Target delta value (e.g., 0.30 for 30-delta).
        option_type: 'call' or 'put'.
        tol: Tolerance for delta matching.

    Returns:
        Best matching contract, or None if not found.
    """
    if not contracts:
        return None

    # For puts, target_delta is negative; for calls, it's positive
    if option_type == "put":
        target_delta = -abs(target_delta)
    else:
        target_delta = abs(target_delta)

    best_contract = None
    best_diff = float("inf")

    for c in contracts:
        if c.delta is not None:
            diff = abs(c.delta - target_delta)
            if diff < best_diff:
                best_diff = diff
                best_contract = c

    return best_contract


def _filter_by_dte(
    contracts: list[OptionContract],
    min_dte: int,
    max_dte: int,
) -> list[OptionContract]:
    """Filter contracts by days to expiration."""
    now = datetime.now()
    return [
        c
        for c in contracts
        if min_dte <= (c.expiration - now).days <= max_dte
    ]


def _estimate_pop(delta_short: float) -> float:
    """Estimate probability of profit from option delta.

    For short options, POP ≈ 1 - |delta| (a common approximation
    since delta roughly equals probability of expiring ITM for
    ATM/OTM options).

    Args:
        delta_short: Delta of the short option sold.

    Returns:
        Estimated probability of profit (0-1).
    """
    return float(1.0 - abs(delta_short))


# ---------------------------------------------------------------------------
# Strategy-specific optimizers
# ---------------------------------------------------------------------------


def optimize_bull_put_spread(
    chain: OptionsChain,
    vol_analysis: VolatilityAnalysis,
    config: EngineConfig = DEFAULT_CONFIG,
) -> PositionParameters:
    """Optimize a Bull Put Spread (credit spread).

    Sells an OTM put, buys a further OTM put for defined risk.
    Target: ~30 delta short put, ~15 delta long put.
    """
    now = datetime.now()
    ticker = chain.ticker
    underlying = chain.underlying_price
    r = 0.05  # risk-free rate

    # Filter to target DTE range
    puts = _filter_by_dte(chain.puts, config.default_dte_min, config.default_dte_max)
    if not puts:
        logger.warning("No puts found in target DTE range for %s", ticker)
        return PositionParameters(strategy=OptionStrategy.BULL_PUT_SPREAD, underlying_ticker=ticker, underlying_price=underlying, dte=config.default_dte_min)

    # Sort puts by delta proximity to target (30 delta OTM put ~ delta -0.30)
    puts_with_delta = []
    for p in puts:
        if p.implied_volatility > 0:
            T = max((p.expiration - now).days, 1) / 365.0
            g = all_greeks(underlying, p.strike, T, r, p.implied_volatility, "put")
            p.delta = g["delta"]
            p.gamma = g["gamma"]
            p.theta = g["theta"]
            p.vega = g["vega"]
            puts_with_delta.append(p)

    if not puts_with_delta:
        logger.warning("No puts with valid IV for %s", ticker)
        return PositionParameters(strategy=OptionStrategy.BULL_PUT_SPREAD, underlying_ticker=ticker, underlying_price=underlying, dte=config.default_dte_min)

    # Target ~30 delta OTM put (delta ~ -0.30)
    short_put = min(puts_with_delta, key=lambda p: abs(p.delta - (-0.30)) if p.delta is not None else float("inf"))

    if short_put.delta is None or abs(short_put.delta) > 0.50:
        logger.warning("Short put delta too high: %.2f", short_put.delta)
        return PositionParameters(strategy=OptionStrategy.BULL_PUT_SPREAD, underlying_ticker=ticker, underlying_price=underlying, dte=config.default_dte_min)

    # Long put: ~10-15 delta, 1 standard deviation below short strike
    long_put_candidates = [p for p in puts_with_delta if p.strike < short_put.strike]
    if not long_put_candidates:
        long_put = short_put  # fallback
        spread_width = 5.0
    else:
        long_put = min(long_put_candidates, key=lambda p: abs((p.delta or 0.0) - (-0.15)))
        spread_width = abs(short_put.strike - long_put.strike)
        if spread_width < 1.0:
            spread_width = 5.0  # minimum practical width

    # Calculate premium
    T = max((short_put.expiration - now).days, 1) / 365.0

    short_price = max(short_put.bid, 0.05) if short_put.bid > 0 else short_put.mid_price
    long_price = max(long_put.ask, 0.05) if long_put.ask > 0 else long_put.mid_price

    premium = short_price - long_price
    if premium <= 0:
        premium = max(short_put.last - long_put.last, 0.05)

    spread_width = abs(short_put.strike - long_put.strike)
    if spread_width <= 0:
        spread_width = 5.0

    max_loss = spread_width - premium
    max_profit = premium
    rr_ratio = max_loss / max_profit if max_profit > 0 else 0

    pop = _estimate_pop(short_put.delta or 0.0)

    return PositionParameters(
        strategy=OptionStrategy.BULL_PUT_SPREAD,
        underlying_ticker=ticker,
        underlying_price=underlying,
        dte=(short_put.expiration - now).days,
        short_strike=short_put.strike,
        long_strike=long_put.strike,
        spread_width=spread_width,
        premium_collected=premium,
        max_loss=max_loss,
        max_profit=max_profit,
        risk_reward_ratio=rr_ratio,
        probability_of_profit=pop,
        delta_short=short_put.delta or 0.0,
        theta_short=short_put.theta or 0.0,
        vega_short=short_put.vega or 0.0,
        contracts=1,
        capital_required=max_loss * 100,  # per contract
        risk_per_contract=max_loss * 100,
    )


def optimize_bear_call_spread(
    chain: OptionsChain,
    vol_analysis: VolatilityAnalysis,
    config: EngineConfig = DEFAULT_CONFIG,
) -> PositionParameters:
    """Optimize a Bear Call Spread (credit spread).

    Sells an OTM call, buys a further OTM call for defined risk.
    Target: ~30 delta short call, ~15 delta long call.
    """
    now = datetime.now()
    ticker = chain.ticker
    underlying = chain.underlying_price
    r = 0.05

    calls = _filter_by_dte(chain.calls, config.default_dte_min, config.default_dte_max)
    if not calls:
        return PositionParameters(strategy=OptionStrategy.BEAR_CALL_SPREAD, underlying_ticker=ticker, underlying_price=underlying, dte=config.default_dte_min)

    calls_with_delta = []
    for c in calls:
        if c.implied_volatility > 0:
            T = max((c.expiration - now).days, 1) / 365.0
            g = all_greeks(underlying, c.strike, T, r, c.implied_volatility, "call")
            c.delta = g["delta"]
            c.gamma = g["gamma"]
            c.theta = g["theta"]
            c.vega = g["vega"]
            calls_with_delta.append(c)

    if not calls_with_delta:
        return PositionParameters(strategy=OptionStrategy.BEAR_CALL_SPREAD, underlying_ticker=ticker, underlying_price=underlying, dte=config.default_dte_min)

    short_call = min(calls_with_delta, key=lambda c: abs((c.delta or 0.0) - 0.30))

    long_call_candidates = [c for c in calls_with_delta if c.strike > short_call.strike]
    if not long_call_candidates:
        long_call = short_call
    else:
        long_call = min(long_call_candidates, key=lambda c: abs((c.delta or 0.0) - 0.15))

    T = max((short_call.expiration - now).days, 1) / 365.0
    short_price = max(short_call.bid, 0.05) if short_call.bid > 0 else short_call.mid_price
    long_price = max(long_call.ask, 0.05) if long_call.ask > 0 else long_call.mid_price

    premium = short_price - long_price
    if premium <= 0:
        premium = max(short_call.last - long_call.last, 0.05)

    spread_width = abs(long_call.strike - short_call.strike)
    if spread_width <= 0:
        spread_width = 5.0

    max_loss = spread_width - premium
    max_profit = premium
    rr_ratio = max_loss / max_profit if max_profit > 0 else 0
    pop = _estimate_pop(short_call.delta or 0.0)

    return PositionParameters(
        strategy=OptionStrategy.BEAR_CALL_SPREAD,
        underlying_ticker=ticker,
        underlying_price=underlying,
        dte=(short_call.expiration - now).days,
        short_strike=short_call.strike,
        long_strike=long_call.strike,
        spread_width=spread_width,
        premium_collected=premium,
        max_loss=max_loss,
        max_profit=max_profit,
        risk_reward_ratio=rr_ratio,
        probability_of_profit=pop,
        delta_short=short_call.delta or 0.0,
        theta_short=short_call.theta or 0.0,
        vega_short=short_call.vega or 0.0,
        contracts=1,
        capital_required=max_loss * 100,
        risk_per_contract=max_loss * 100,
    )


def optimize_iron_condor(
    chain: OptionsChain,
    vol_analysis: VolatilityAnalysis,
    config: EngineConfig = DEFAULT_CONFIG,
) -> PositionParameters:
    """Optimize an Iron Condor.

    Combines a Bull Put Spread and Bear Call Spread.
    Both spreads at ~30 delta, targeting a neutral outlook.
    """
    now = datetime.now()
    ticker = chain.ticker
    underlying = chain.underlying_price
    r = 0.05

    puts = _filter_by_dte(chain.puts, config.default_dte_min, config.default_dte_max)
    calls = _filter_by_dte(chain.calls, config.default_dte_min, config.default_dte_max)

    if not puts or not calls:
        return PositionParameters(strategy=OptionStrategy.IRON_CONDOR, underlying_ticker=ticker, underlying_price=underlying, dte=config.default_dte_min)

    # Calculate delta for puts and calls
    puts_d = []
    for p in puts:
        if p.implied_volatility > 0:
            T = max((p.expiration - now).days, 1) / 365.0
            g = all_greeks(underlying, p.strike, T, r, p.implied_volatility, "put")
            p.delta = g["delta"]
            puts_d.append(p)

    calls_d = []
    for c in calls:
        if c.implied_volatility > 0:
            T = max((c.expiration - now).days, 1) / 365.0
            g = all_greeks(underlying, c.strike, T, r, c.implied_volatility, "call")
            c.delta = g["delta"]
            calls_d.append(c)

    if not puts_d or not calls_d:
        return PositionParameters(strategy=OptionStrategy.IRON_CONDOR, underlying_ticker=ticker, underlying_price=underlying, dte=config.default_dte_min)

    # ~30 delta on each side
    short_put = min(puts_d, key=lambda p: abs((p.delta or 0.0) - (-0.30)))
    short_call = min(calls_d, key=lambda c: abs((c.delta or 0.0) - 0.30))

    # ~15 delta long wings
    long_put_candidates = [p for p in puts_d if p.strike < short_put.strike]
    long_put = min(long_put_candidates, key=lambda p: abs((p.delta or 0.0) - (-0.15))) if long_put_candidates else short_put

    long_call_candidates = [c for c in calls_d if c.strike > short_call.strike]
    long_call = min(long_call_candidates, key=lambda c: abs((c.delta or 0.0) - 0.15)) if long_call_candidates else short_call

    # Calculate premium
    T_put = max((short_put.expiration - now).days, 1) / 365.0
    T_call = max((short_call.expiration - now).days, 1) / 365.0

    put_credit = short_put.bid if short_put.bid > 0 else short_put.mid_price
    put_debit = long_put.ask if long_put.ask > 0 else long_put.mid_price
    call_credit = short_call.bid if short_call.bid > 0 else short_call.mid_price
    call_debit = long_call.ask if long_call.ask > 0 else long_call.mid_price

    net_premium = (put_credit + call_credit) - (put_debit + call_debit)
    net_premium = max(net_premium, 0.05)

    put_width = abs(long_put.strike - short_put.strike)
    call_width = abs(long_call.strike - short_call.strike)
    max_wing = max(put_width, call_width)

    max_loss = max_wing - net_premium
    max_profit = net_premium
    rr_ratio = max_loss / max_profit if max_profit > 0 else 0

    avg_short_delta = (abs(short_put.delta or 0.0) + abs(short_call.delta or 0.0)) / 2.0
    pop = _estimate_pop(avg_short_delta)

    return PositionParameters(
        strategy=OptionStrategy.IRON_CONDOR,
        underlying_ticker=ticker,
        underlying_price=underlying,
        dte=(short_put.expiration - now).days,
        short_strike=short_call.strike,
        long_strike=long_call.strike,
        short_strike_2=short_put.strike,
        long_strike_2=long_put.strike,
        spread_width=max_wing,
        premium_collected=net_premium,
        max_loss=max_loss,
        max_profit=max_profit,
        risk_reward_ratio=rr_ratio,
        probability_of_profit=pop,
        delta_short=avg_short_delta,
        theta_short=(short_put.theta or 0.0) + (short_call.theta or 0.0),
        vega_short=(short_put.vega or 0.0) + (short_call.vega or 0.0),
        contracts=1,
        capital_required=max_loss * 100,
        risk_per_contract=max_loss * 100,
    )


def optimize_short_strangle(
    chain: OptionsChain,
    vol_analysis: VolatilityAnalysis,
    config: EngineConfig = DEFAULT_CONFIG,
) -> PositionParameters:
    """Optimize a Short Strangle.

    Sells OTM put and OTM call (naked). Higher premium but undefined risk.
    Target: ~25-30 delta on each side, wider wings for safety.
    """
    now = datetime.now()
    ticker = chain.ticker
    underlying = chain.underlying_price
    r = 0.05

    puts = _filter_by_dte(chain.puts, config.default_dte_min, config.default_dte_max)
    calls = _filter_by_dte(chain.calls, config.default_dte_min, config.default_dte_max)

    if not puts or not calls:
        return PositionParameters(strategy=OptionStrategy.SHORT_STRANGLE, underlying_ticker=ticker, underlying_price=underlying, dte=config.default_dte_min)

    # More conservative — target ~25 delta
    puts_d = []
    for p in puts:
        if p.implied_volatility > 0:
            T = max((p.expiration - now).days, 1) / 365.0
            g = all_greeks(underlying, p.strike, T, r, p.implied_volatility, "put")
            p.delta = g["delta"]
            puts_d.append(p)

    calls_d = []
    for c in calls:
        if c.implied_volatility > 0:
            T = max((c.expiration - now).days, 1) / 365.0
            g = all_greeks(underlying, c.strike, T, r, c.implied_volatility, "call")
            c.delta = g["delta"]
            calls_d.append(c)

    if not puts_d or not calls_d:
        return PositionParameters(strategy=OptionStrategy.SHORT_STRANGLE, underlying_ticker=ticker, underlying_price=underlying, dte=config.default_dte_min)

    short_put = min(puts_d, key=lambda p: abs((p.delta or 0.0) - (-0.25)))
    short_call = min(calls_d, key=lambda c: abs((c.delta or 0.0) - 0.25))

    put_price = short_put.bid if short_put.bid > 0 else short_put.mid_price
    call_price = short_call.bid if short_call.bid > 0 else short_call.mid_price

    net_premium = put_price + call_price

    width = abs(short_call.strike - short_put.strike)

    # For strangle, max loss is undefined. Use a proxy: premium collected
    max_loss = width  # proxy: distance between strikes (simplified)
    max_profit = net_premium
    rr_ratio = max_loss / max_profit if max_profit > 0 else 0

    avg_delta = (abs(short_put.delta or 0.0) + abs(short_call.delta or 0.0)) / 2.0
    pop = _estimate_pop(avg_delta)

    return PositionParameters(
        strategy=OptionStrategy.SHORT_STRANGLE,
        underlying_ticker=ticker,
        underlying_price=underlying,
        dte=(short_put.expiration - now).days,
        short_strike=short_call.strike,
        short_strike_2=short_put.strike,
        spread_width=width,
        premium_collected=net_premium,
        max_loss=max_loss,
        max_profit=max_profit,
        risk_reward_ratio=rr_ratio,
        probability_of_profit=pop,
        delta_short=avg_delta,
        theta_short=(short_put.theta or 0.0) + (short_call.theta or 0.0),
        vega_short=(short_put.vega or 0.0) + (short_call.vega or 0.0),
        contracts=1,
        capital_required=max_loss * 100,
        risk_per_contract=max_loss * 100,
    )


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


def calculate_position_size(
    risk_per_contract: float,
    account_size: float,
    risk_per_trade_pct: float,
    max_allocation_pct: float = 0.05,
) -> int:
    """Calculate the number of contracts to trade based on risk management.

    Args:
        risk_per_contract: Dollar risk per single contract.
        account_size: Total account value.
        risk_per_trade_pct: Maximum fraction of account to risk per trade (e.g., 0.02).
        max_allocation_pct: Maximum fraction of account to allocate (e.g., 0.05).

    Returns:
        Number of contracts (rounded down for safety).
    """
    if risk_per_contract <= 0:
        return 1

    risk_budget = account_size * risk_per_trade_pct
    allocation_budget = account_size * max_allocation_pct

    # Risk-based sizing
    from_risk = int(risk_budget / risk_per_contract)

    # Allocation-based sizing
    from_allocation = int(allocation_budget / risk_per_contract)

    # Take the smaller to be conservative
    contracts = max(1, min(from_risk, from_allocation))
    return contracts


# ---------------------------------------------------------------------------
# Main optimizer entry point
# ---------------------------------------------------------------------------

# Strategy → optimizer mapping
_OPTIMIZERS = {
    OptionStrategy.BULL_PUT_SPREAD: optimize_bull_put_spread,
    OptionStrategy.BEAR_CALL_SPREAD: optimize_bear_call_spread,
    OptionStrategy.IRON_CONDOR: optimize_iron_condor,
    OptionStrategy.SHORT_STRANGLE: optimize_short_strangle,
}


def optimize_position(
    strategy: OptionStrategy,
    chain: OptionsChain,
    vol_analysis: VolatilityAnalysis,
    config: EngineConfig = DEFAULT_CONFIG,
) -> PositionParameters:
    """Optimize position parameters for a given strategy.

    Args:
        strategy: The selected option strategy.
        chain: Full options chain data.
        vol_analysis: Volatility analysis results.
        config: Engine configuration.

    Returns:
        Optimized PositionParameters.

    Raises:
        ValueError: If the strategy has no optimizer.
    """
    optimizer = _OPTIMIZERS.get(strategy)
    if optimizer is None:
        raise ValueError(f"No optimizer available for strategy: {strategy.value}")

    params = optimizer(chain, vol_analysis, config)

    # Apply position sizing if contract count is 1 (default from optimizers)
    # The caller should call calculate_position_size separately if needed

    return params