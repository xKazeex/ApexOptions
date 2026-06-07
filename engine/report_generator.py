"""
Report generator for the Apex Options Analytics Engine.

Formats analytics results into structured outputs: JSON for API consumption
and Markdown for human-readable trade recommendations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from typing import Any, Optional

from .position_optimizer import PositionParameters
from .risk_manager import RiskAssessment
from .strategy_selector import StrategyRecommendation
from .volatility import VolatilityAnalysis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Complete trade recommendation
# ---------------------------------------------------------------------------


@dataclass
class ApexStrategyReport:
    """Complete trade recommendation output (the product).

    This is the final output delivered to users — a complete,
    actionable options premium-selling recommendation.

    Attributes:
        generated_at: Timestamp when the report was generated.
        ticker: Underlying ticker.
        underlying_price: Current price.
        market_thesis: One-sentence thesis for the trade.
        strategy_name: Display name of the selected strategy.
        strategy_description: Description of how the strategy works.
        volatility_summary: Key volatility metrics summary.
        strikes: Trade structure details.
        risk_parameters: Risk management parameters.
        position_sizing: Position sizing recommendation.
        trade_rationale: Why this trade is being recommended.
        disclaimers: Standard options trading disclaimers.
    """

    generated_at: str
    ticker: str
    underlying_price: float
    market_thesis: str
    strategy_name: str
    strategy_description: str
    volatility_summary: dict[str, Any]
    strikes: dict[str, Any]
    risk_parameters: dict[str, Any]
    position_sizing: dict[str, Any]
    trade_rationale: str
    disclaimers: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_serialize(obj: Any) -> Any:
    """Recursively serialize a dataclass or object to JSON-safe types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _safe_serialize(value) for key, value in obj.items()}
    if is_dataclass(obj):
        return _safe_serialize(asdict(obj))
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return str(obj)


def _format_pct(value: float, decimals: int = 1) -> str:
    """Format a decimal as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def _format_dollar(value: float) -> str:
    """Format a dollar value."""
    return f"${value:.2f}"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_json_report(
    vol_analysis: VolatilityAnalysis,
    strategy_rec: StrategyRecommendation,
    position_params: PositionParameters,
    risk_assessment: RiskAssessment,
    account_size: float = 100_000,
) -> str:
    """Generate a complete strategy report in JSON format.

    Args:
        vol_analysis: Volatility analysis results.
        strategy_rec: Strategy recommendation.
        position_params: Optimized position parameters.
        risk_assessment: Risk assessment results.
        account_size: Account size for position sizing.

    Returns:
        JSON-formatted string of the complete report.
    """
    report = _build_report(
        vol_analysis, strategy_rec, position_params, risk_assessment, account_size
    )
    return json.dumps(_safe_serialize(report), indent=2, default=str)


def generate_markdown_report(
    vol_analysis: VolatilityAnalysis,
    strategy_rec: StrategyRecommendation,
    position_params: PositionParameters,
    risk_assessment: RiskAssessment,
    account_size: float = 100_000,
) -> str:
    """Generate a complete strategy report in Markdown format.

    Args:
        vol_analysis: Volatility analysis results.
        strategy_rec: Strategy recommendation.
        position_params: Optimized position parameters.
        risk_assessment: Risk assessment results.
        account_size: Account size for position sizing.

    Returns:
        Markdown-formatted string of the complete report.
    """
    report = _build_report(
        vol_analysis, strategy_rec, position_params, risk_assessment, account_size
    )

    lines: list[str] = []
    lines.append(f"# Apex Strategy Report: ${report.ticker}")
    lines.append(f"*Generated: {report.generated_at}*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Market Thesis
    lines.append("## 🎯 Market Thesis")
    lines.append(f"> {report.market_thesis}")
    lines.append("")

    # Strategy
    lines.append(f"## 📊 Recommended Strategy: {report.strategy_name}")
    lines.append("")
    lines.append(f"*{report.strategy_description}*")
    lines.append("")

    # Volatility Summary
    lines.append("## 🌋 Volatility Analysis")
    lines.append("")
    vs = report.volatility_summary
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Current IV | {_format_pct(vs.get('current_iv', 0))} |")
    lines.append(f"| IV Rank | {vs.get('iv_rank', 0):.1f}% |")
    lines.append(f"| IV Percentile | {vs.get('iv_percentile', 0):.1f}% |")
    lines.append(f"| RV (20-day) | {_format_pct(vs.get('realized_vol_20d', 0))} |")
    lines.append(f"| RV (60-day) | {_format_pct(vs.get('realized_vol_60d', 0))} |")
    lines.append(f"| IV/RV Ratio (20d) | {vs.get('iv_rv_ratio_20d', 0):.2f}x |")
    lines.append(f"| Regime | {vs.get('vol_regime', 'unknown')} |")
    lines.append(f"| Term Structure | {vs.get('term_structure', 'unknown')} |")
    lines.append(f"| Skew | {vs.get('skew_description', 'unknown')} |")
    lines.append(f"| Opportunity Score | {vs.get('opportunity_score', 0):.0f}/100 |")
    lines.append("")

    # Strike Structure
    lines.append("## 🏗️ Trade Structure")
    lines.append("")
    sk = report.strikes
    lines.append(f"| Parameter | Value |")
    lines.append(f"|--------|-------|")
    if sk.get("short_strike"):
        lines.append(f"| Short Strike | {_format_dollar(sk['short_strike'])} |")
    if sk.get("long_strike"):
        lines.append(f"| Long Strike | {_format_dollar(sk['long_strike'])} |")
    if sk.get("short_strike_2"):
        lines.append(f"| Short Strike (Put Side) | {_format_dollar(sk['short_strike_2'])} |")
    if sk.get("long_strike_2"):
        lines.append(f"| Long Strike (Put Side) | {_format_dollar(sk['long_strike_2'])} |")
    lines.append(f"| Spread Width | {_format_dollar(sk.get('spread_width', 0))} |")
    lines.append(f"| DTE | {sk.get('dte', 0)} days |")
    lines.append(f"| Underlying Price | {_format_dollar(sk.get('underlying_price', 0))} |")
    lines.append("")

    # Trade Economics
    lines.append("## 💰 Trade Economics (Per Contract)")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Premium Collected | {_format_dollar(position_params.premium_collected * 100)} |")
    lines.append(f"| Max Profit | {_format_dollar(position_params.max_profit * 100)} |")
    lines.append(f"| Max Loss | {_format_dollar(position_params.max_loss * 100)} |")
    lines.append(f"| Risk/Reward | {position_params.risk_reward_ratio:.2f}:1 |")
    lines.append(f"| Prob. of Profit | {_format_pct(position_params.probability_of_profit)} |")
    lines.append(f"| Short Delta | {position_params.delta_short:.3f} |")
    lines.append(f"| Daily Theta | {_format_dollar(position_params.theta_short * 100)} |")
    lines.append(f"| Vega | {_format_dollar(position_params.vega_short * 100)} |")
    lines.append("")

    # Risk Parameters
    lines.append("## ⚠️ Risk Management")
    lines.append("")
    rp = report.risk_parameters
    lines.append(f"| Parameter | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Max Risk | {_format_dollar(rp.get('max_risk_dollars', 0))} ({_format_pct(rp.get('max_risk_pct', 0))}) |")
    lines.append(f"| Profit Target | {_format_dollar(rp.get('profit_target_dollars', 0))} ({_format_pct(rp.get('profit_target_pct', 0))}) |")
    lines.append(f"| Stop-Loss | {_format_dollar(rp.get('stop_loss_dollars', 0))} |")
    lines.append(f"| Risk Score | {rp.get('risk_score', 0):.0f}/100 |")
    if rp.get("warnings"):
        for w in rp["warnings"]:
            lines.append(f"| ⚠️ Warning | {w} |")
    lines.append("")

    # Position Sizing
    lines.append("## 📐 Position Sizing")
    lines.append("")
    ps = report.position_sizing
    lines.append(f"| Parameter | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Account Size | {_format_dollar(ps.get('account_size', account_size))} |")
    lines.append(f"| Contracts | {ps.get('contracts', position_params.contracts)} |")
    lines.append(f"| Capital Required | {_format_dollar(ps.get('capital_required', 0))} |")
    lines.append(f"| Risk per Trade | {_format_pct(ps.get('risk_per_trade_pct', 0))} |")
    lines.append("")

    # Trade Rationale
    lines.append("## 📝 Trade Rationale")
    lines.append("")
    lines.append(report.trade_rationale)
    lines.append("")

    # Disclaimers
    lines.append("---")
    lines.append("")
    lines.append("### ⚠️ Disclaimers")
    lines.append("")
    for d in report.disclaimers:
        lines.append(f"- {d}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------


def _build_report(
    vol_analysis: VolatilityAnalysis,
    strategy_rec: StrategyRecommendation,
    position_params: PositionParameters,
    risk_assessment: RiskAssessment,
    account_size: float,
) -> ApexStrategyReport:
    """Build the complete ApexStrategyReport dataclass."""
    now = datetime.now()

    # Market thesis
    iv_rank_str = f"{vol_analysis.iv_rank:.0f}%"
    thesis_parts = [
        f"{vol_analysis.ticker} at ${vol_analysis.underlying_price:.2f}",
        f"with IV Rank {iv_rank_str}",
        f"({vol_analysis.vol_regime}).",
        f"Recommended: {strategy_rec.strategy.display_name}.",
    ]
    market_thesis = " ".join(thesis_parts)

    # Strategy description
    desc_map = {
        "bull_put_spread": (
            "Sell an OTM Put and buy a further OTM Put. "
            "Collects premium with defined risk on the downside. "
            "Profits if the underlying stays above the short strike."
        ),
        "bear_call_spread": (
            "Sell an OTM Call and buy a further OTM Call. "
            "Collects premium with defined risk on the upside. "
            "Profits if the underlying stays below the short strike."
        ),
        "iron_condor": (
            "Combines a Bull Put Spread and a Bear Call Spread. "
            "Collects premium with defined risk on both sides. "
            "Profits if the underlying stays between the short strikes."
        ),
        "short_strangle": (
            "Sell an OTM Put and an OTM Call simultaneously. "
            "Collects premium with undefined risk. "
            "Profits if the underlying stays between the strikes."
        ),
        "calendar_spread": (
            "Sell a near-term option and buy a longer-term option at the same strike. "
            "Captures time decay with limited risk."
        ),
    }

    strategy_description = desc_map.get(
        strategy_rec.strategy.value,
        "Premium-selling strategy with defined risk parameters."
    )

    # Volatility summary dict
    vol_summary = _safe_serialize(
        {
            "current_iv": vol_analysis.current_iv,
            "iv_rank": vol_analysis.iv_rank,
            "iv_percentile": vol_analysis.iv_percentile,
            "realized_vol_10d": vol_analysis.realized_vol_10d,
            "realized_vol_20d": vol_analysis.realized_vol_20d,
            "realized_vol_60d": vol_analysis.realized_vol_60d,
            "iv_rv_ratio_20d": vol_analysis.iv_rv_ratio_20d,
            "iv_rv_ratio_60d": vol_analysis.iv_rv_ratio_60d,
            "vol_regime": vol_analysis.vol_regime,
            "term_structure": vol_analysis.term_structure,
            "skew_description": vol_analysis.skew_description,
            "opportunity_score": vol_analysis.opportunity_score,
        }
    )

    # Strikes dict
    strikes_dict = _safe_serialize(
        {
            "underlying_price": position_params.underlying_price,
            "short_strike": position_params.short_strike,
            "long_strike": position_params.long_strike,
            "short_strike_2": position_params.short_strike_2,
            "long_strike_2": position_params.long_strike_2,
            "spread_width": position_params.spread_width,
            "dte": position_params.dte,
        }
    )

    # Risk parameters dict
    risk_params = _safe_serialize(
        {
            "max_risk_dollars": risk_assessment.max_risk_dollars,
            "max_risk_pct": risk_assessment.max_risk_pct,
            "profit_target_dollars": risk_assessment.profit_target_dollars,
            "profit_target_pct": risk_assessment.profit_target_pct,
            "stop_loss_dollars": risk_assessment.stop_loss_dollars,
            "stop_loss_pct": risk_assessment.stop_loss_pct,
            "risk_score": risk_assessment.risk_score,
            "is_approved": risk_assessment.is_approved,
            "warnings": risk_assessment.warnings,
        }
    )

    # Position sizing dict
    contracts = risk_assessment.suggested_contracts if risk_assessment.suggested_contracts > 0 else position_params.contracts
    capital_req = position_params.risk_per_contract * contracts
    risk_per_trade_pct = risk_assessment.max_risk_dollars / account_size if account_size > 0 else 0

    pos_sizing = _safe_serialize(
        {
            "account_size": account_size,
            "contracts": contracts,
            "capital_required": capital_req,
            "risk_per_trade_pct": risk_per_trade_pct,
        }
    )

    # Trade rationale
    rationale_parts = [
        f"**Why {strategy_rec.strategy.display_name}?**",
        f"{strategy_rec.rationale}",
        "",
        f"**Confidence Score:** {strategy_rec.confidence_score:.0f}/100",
    ]
    if strategy_rec.alternative_strategies:
        rationale_parts.append(
            f"**Alternatives:** {', '.join(strategy_rec.alternative_strategies)}"
        )
    trade_rationale = "\n\n".join(rationale_parts)

    # Disclaimers
    disclaimers = [
        "Options trading involves substantial risk and is not suitable for all investors.",
        "Past performance is not indicative of future results.",
        "This report is for informational purposes only and does not constitute financial advice.",
        "Position sizing should be adjusted based on individual risk tolerance and account size.",
        "Always monitor positions and adjust stop-losses as market conditions change.",
    ]

    return ApexStrategyReport(
        generated_at=now.isoformat(),
        ticker=vol_analysis.ticker,
        underlying_price=vol_analysis.underlying_price,
        market_thesis=market_thesis,
        strategy_name=strategy_rec.strategy.display_name,
        strategy_description=strategy_description,
        volatility_summary=vol_summary,
        strikes=strikes_dict,
        risk_parameters=risk_params,
        position_sizing=pos_sizing,
        trade_rationale=trade_rationale,
        disclaimers=disclaimers,
    )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def generate_full_report(
    vol_analysis: VolatilityAnalysis,
    strategy_rec: StrategyRecommendation,
    position_params: PositionParameters,
    risk_assessment: RiskAssessment,
    account_size: float = 100_000,
    output_format: str = "markdown",
) -> str:
    """Generate a report in the requested format.

    Args:
        vol_analysis: Volatility analysis results.
        strategy_rec: Strategy recommendation.
        position_params: Optimized position parameters.
        risk_assessment: Risk assessment results.
        account_size: Account size for position sizing.
        output_format: 'markdown' or 'json'.

    Returns:
        Formatted report string.
    """
    if output_format == "json":
        return generate_json_report(
            vol_analysis, strategy_rec, position_params, risk_assessment, account_size
        )
    else:
        return generate_markdown_report(
            vol_analysis, strategy_rec, position_params, risk_assessment, account_size
        )