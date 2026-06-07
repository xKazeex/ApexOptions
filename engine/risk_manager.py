"""
Risk management framework for the Apex Options Analytics Engine.

Defines position-level and portfolio-level risk controls, profit targets,
stop-loss triggers, and position correlation limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from .config import EngineConfig, DEFAULT_CONFIG
from .position_optimizer import PositionParameters

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RiskAssessment:
    """Result of risk evaluation for a trade.

    Attributes:
        is_approved: Whether the trade passes all risk checks.
        reason: Human-readable explanation if rejected.
        max_risk_dollars: Maximum dollar risk for the position.
        max_risk_pct: Maximum risk as percentage of account.
        profit_target_dollars: Dollar amount at which to take profit.
        stop_loss_dollars: Dollar loss at which to stop out.
        profit_target_pct: Percentage of max profit to target.
        stop_loss_pct: Percentage of premium collected to trigger stop.
        risk_score: Composite risk score 0-100 (higher = riskier).
        warnings: List of risk warnings.
        suggested_contracts: Adjusted contract count if risk-capped.
    """

    is_approved: bool = True
    reason: str = ""
    max_risk_dollars: float = 0.0
    max_risk_pct: float = 0.0
    profit_target_dollars: float = 0.0
    stop_loss_dollars: float = 0.0
    profit_target_pct: float = 0.50
    stop_loss_pct: float = 2.0
    risk_score: float = 0.0
    warnings: list[str] = field(default_factory=list)
    suggested_contracts: int = 1


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------


def _calculate_risk_score(
    params: PositionParameters,
    account_size: float,
) -> float:
    """Calculate a composite risk score (0-100).

    Higher scores indicate riskier trades.
    """
    score = 0.0

    # DTE risk (very short DTE = more gamma risk, very long = more vega risk)
    if params.dte < 7:
        score += 25
    elif params.dte < 21:
        score += 15
    elif params.dte < 45:
        score += 10
    elif params.dte < 90:
        score += 20
    else:
        score += 25

    # Delta risk
    abs_delta = abs(params.delta_short)
    if abs_delta > 0.40:
        score += 25
    elif abs_delta > 0.30:
        score += 15
    else:
        score += 5

    # Risk/reward ratio
    if params.risk_reward_ratio > 5.0:
        score += 20
    elif params.risk_reward_ratio > 3.0:
        score += 10
    else:
        score += 5

    # Capital at risk
    if account_size > 0:
        risk_pct = (params.risk_per_contract * params.contracts) / account_size
        if risk_pct > 0.05:
            score += 20
        elif risk_pct > 0.03:
            score += 10
        else:
            score += 5

    # POP-based risk
    if params.probability_of_profit < 0.60:
        score += 10

    return min(score, 100.0)


# ---------------------------------------------------------------------------
# Main risk assessment
# ---------------------------------------------------------------------------


def assess_trade_risk(
    params: PositionParameters,
    account_size: float,
    config: EngineConfig = DEFAULT_CONFIG,
    existing_positions: Optional[list[dict]] = None,
) -> RiskAssessment:
    """Run full risk assessment on a proposed trade.

    Checks:
    1. Position sizing (risk per trade ≤ config.risk_per_trade)
    2. Portfolio-level risk (total risk ≤ config.max_portfolio_risk)
    3. Profit target (take profit at config.profit_target_pct of max premium)
    4. Stop-loss trigger (exit when position loses config.stop_loss_pct * premium)
    5. Style checks (min DTE, max DTE)
    6. Existing exposure (if existing_positions provided)

    Args:
        params: Position parameters to assess.
        account_size: Total account value in dollars.
        config: Engine configuration.
        existing_positions: Optional list of existing position dicts with
            keys: 'ticker', 'risk_dollars', 'strategy'.

    Returns:
        RiskAssessment with approval decision and risk details.
    """
    assessment = RiskAssessment()
    assessment.profit_target_pct = config.profit_target_pct
    assessment.stop_loss_pct = config.stop_loss_pct

    risk_per_pos = params.risk_per_contract * params.contracts
    risk_pct = risk_per_pos / account_size if account_size > 0 else 1.0

    assessment.max_risk_dollars = risk_per_pos
    assessment.max_risk_pct = risk_pct

    # --- Position sizing check ---
    if risk_pct > config.risk_per_trade:
        # Try reducing contracts
        max_risk_dollars = account_size * config.risk_per_trade
        max_contracts = max(1, int(max_risk_dollars / params.risk_per_contract))
        assessment.suggested_contracts = max_contracts
        assessment.warnings.append(
            f"Risk {risk_pct:.1%} exceeds max {config.risk_per_trade:.1%} per trade. "
            f"Reducing to {max_contracts} contract(s)."
        )

        risk_per_pos_adj = params.risk_per_contract * max_contracts
        risk_pct_adj = risk_per_pos_adj / account_size if account_size > 0 else 0.0
        assessment.max_risk_dollars = risk_per_pos_adj
        assessment.max_risk_pct = risk_pct_adj

    # --- DTE checks ---
    if params.dte < config.default_dte_min:
        assessment.warnings.append(
            f"DTE of {params.dte} is below minimum of {config.default_dte_min}. "
            "Higher gamma risk."
        )
    if params.dte > config.default_dte_max * 2:
        assessment.warnings.append(
            f"DTE of {params.dte} is long-dated. Higher vega exposure."
        )

    # --- Portfolio-level risk check ---
    if existing_positions:
        total_portfolio_risk = sum(
            pos.get("risk_dollars", 0) for pos in existing_positions
        )
        total_risk_pct = (total_portfolio_risk + risk_per_pos) / account_size

        if total_risk_pct > config.max_portfolio_risk:
            assessment.warnings.append(
                f"Total portfolio risk {total_risk_pct:.1%} exceeds max "
                f"{config.max_portfolio_risk:.1%}. Consider reducing exposure."
            )
            if total_risk_pct > config.max_portfolio_risk * 1.5:
                assessment.is_approved = False
                assessment.reason = (
                    f"Portfolio risk {total_risk_pct:.1%} exceeds "
                    f"{config.max_portfolio_risk:.1%} limit significantly."
                )
                return assessment

        # Check for concentrated ticker exposure
        ticker_exposure = sum(
            pos.get("risk_dollars", 0)
            for pos in existing_positions
            if pos.get("ticker") == params.underlying_ticker
        )
        ticker_risk_pct = (ticker_exposure + risk_per_pos) / account_size
        if ticker_risk_pct > 0.10:
            assessment.warnings.append(
                f"Concentration in {params.underlying_ticker}: {ticker_risk_pct:.1%} "
                "of account. Consider diversifying."
            )

    # --- Profit target ---
    assessment.profit_target_dollars = (
        params.premium_collected * 100 * params.contracts * config.profit_target_pct
    )

    # --- Stop-loss ---
    # Stop-loss when premium decayed or IV expanded
    # Default: stop when position reaches config.stop_loss_pct * premium collected
    assessment.stop_loss_dollars = (
        params.premium_collected * 100 * params.contracts * config.stop_loss_pct
    )

    # --- Risk score ---
    assessment.risk_score = _calculate_risk_score(params, account_size)
    if assessment.risk_score > 70:
        assessment.warnings.append(
            f"High risk score: {assessment.risk_score:.0f}/100. "
            "Consider reducing size or adjusting strikes."
        )
        if assessment.risk_score > 90:
            assessment.is_approved = False
            assessment.reason = (
                f"Risk score {assessment.risk_score:.0f}/100 exceeds "
                "maximum threshold."
            )
            return assessment

    # If only warnings and no hard rejection, keep approved
    if assessment.is_approved and not assessment.reason:
        assessment.reason = "Trade passes all risk checks."

    return assessment


# ---------------------------------------------------------------------------
# Portfolio monitoring helpers
# ---------------------------------------------------------------------------


def check_stop_loss(
    current_premium: float,
    initial_premium: float,
    contracts: int,
    stop_loss_multiple: float = 2.0,
) -> bool:
    """Check if a stop-loss condition has been triggered.

    For a short premium position, a stop-loss triggers when the premium
    has increased by `stop_loss_multiple` times the initial premium
    (i.e., the cost to close has grown unacceptably).

    Args:
        current_premium: Current market premium value (per share).
        initial_premium: Premium collected at entry (per share).
        contracts: Number of contracts traded.
        stop_loss_multiple: Multiple of initial premium increase to trigger.

    Returns:
        True if stop-loss is triggered.
    """
    initial_value = initial_premium * 100 * contracts
    current_value = current_premium * 100 * contracts

    if initial_value <= 0:
        return False

    # For short positions, when current_premium > initial_premium, we're losing money
    # The loss is measured as the multiple of initial premium
    premium_multiple = (current_value - initial_value) / initial_value

    # Trigger when premium has increased by the stop_loss_multiple factor
    return premium_multiple >= stop_loss_multiple


def check_profit_target(
    current_premium: float,
    initial_premium: float,
    contracts: int,
    profit_target_pct: float = 0.50,
) -> bool:
    """Check if profit target has been reached.

    Args:
        current_premium: Current market premium value.
        initial_premium: Premium collected at entry (per share).
        contracts: Number of contracts traded.
        profit_target_pct: Fraction of max profit to lock in.

    Returns:
        True if profit target is hit.
    """
    initial_value = initial_premium * 100 * contracts
    current_value = current_premium * 100 * contracts

    # For credit spreads, profit = premium decay
    profit_taken = initial_value - current_value
    target = initial_value * profit_target_pct

    return profit_taken >= target