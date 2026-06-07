"""
Strategy selection engine for the Apex Options Analytics Engine.

Rule-based logic that maps volatility regime + market conditions to
optimal premium-selling strategy recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np

from .volatility import VolatilityAnalysis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------


class OptionStrategy(Enum):
    """Supported option premium-selling strategies."""

    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"
    IRON_CONDOR = "iron_condor"
    SHORT_STRANGLE = "short_strangle"
    CALENDAR_SPREAD = "calendar_spread"

    @property
    def display_name(self) -> str:
        names = {
            "bull_put_spread": "Bull Put Spread (Credit)",
            "bear_call_spread": "Bear Call Spread (Credit)",
            "iron_condor": "Iron Condor",
            "short_strangle": "Short Strangle",
            "calendar_spread": "Calendar Spread (Short Vol)",
        }
        return names.get(self.value, self.value)


@dataclass
class StrategyRecommendation:
    """A single strategy recommendation from the selector.

    Attributes:
        strategy: The recommended option strategy.
        rationale: Human-readable explanation of why this strategy was selected.
        confidence_score: Score 0-100 indicating how well the strategy fits conditions.
        market_outlook: Bullish, bearish, or neutral.
        is_viable: Whether this strategy is viable given current conditions.
        alternative_strategies: List of alternative strategies to consider.
    """

    strategy: OptionStrategy
    rationale: str
    confidence_score: float = 50.0
    market_outlook: str = "neutral"
    is_viable: bool = True
    alternative_strategies: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Market condition classifier
# ---------------------------------------------------------------------------


@dataclass
class MarketConditions:
    """Classified market conditions derived from volatility analysis.

    This is the intermediate representation the strategy selector uses
    to make decisions.
    """

    trend_direction: str = "neutral"  # 'bullish', 'bearish', 'neutral'
    trend_strength: float = 0.0  # 0-1
    volatility_regime: str = "normal"
    iv_rank_category: str = "medium"  # 'low', 'medium', 'high', 'very_high'
    skew_bias: str = "neutral"  # 'put_skew', 'call_skew', 'neutral'
    term_structure: str = "flat"
    iv_rv_premium: float = 0.0  # how much IV exceeds RV
    composite_risk_level: str = "moderate"


def classify_market_conditions(
    vol_analysis: VolatilityAnalysis,
) -> MarketConditions:
    """Classify current market conditions from volatility analysis.

    Args:
        vol_analysis: Complete volatility analysis output.

    Returns:
        MarketConditions classified for strategy selection.
    """
    mc = MarketConditions()

    # Trend direction from skew + price action proxy
    if vol_analysis.skew_description and "put skew" in vol_analysis.skew_description.lower():
        if "strong" in vol_analysis.skew_description.lower():
            mc.trend_direction = "bearish"
        else:
            mc.trend_direction = "slightly_bearish"
    elif vol_analysis.skew_description and "call skew" in vol_analysis.skew_description.lower():
        mc.trend_direction = "bullish"
    else:
        mc.trend_direction = "neutral"

    # IV Rank category
    if vol_analysis.iv_rank >= 70:
        mc.iv_rank_category = "very_high"
    elif vol_analysis.iv_rank >= 50:
        mc.iv_rank_category = "high"
    elif vol_analysis.iv_rank >= 20:
        mc.iv_rank_category = "medium"
    else:
        mc.iv_rank_category = "low"

    # Volatility regime
    mc.volatility_regime = vol_analysis.vol_regime

    # Skew bias
    if vol_analysis.skew_description and "put skew" in vol_analysis.skew_description.lower():
        mc.skew_bias = "put_skew"
    elif vol_analysis.skew_description and "call skew" in vol_analysis.skew_description.lower():
        mc.skew_bias = "call_skew"
    else:
        mc.skew_bias = "neutral"

    # Term structure
    if "backwardated" in vol_analysis.term_structure:
        mc.term_structure = "backwardated"
    elif "contango" in vol_analysis.term_structure:
        mc.term_structure = "contango"
    else:
        mc.term_structure = "flat"

    # IV/RV premium
    mc.iv_rv_premium = max(0.0, vol_analysis.iv_rv_ratio_20d - 1.0)

    # Composite risk level
    if vol_analysis.iv_rank >= 70 and mc.iv_rv_premium > 0.5:
        mc.composite_risk_level = "high"
    elif vol_analysis.iv_rank >= 50:
        mc.composite_risk_level = "elevated"
    elif vol_analysis.iv_rank >= 20:
        mc.composite_risk_level = "moderate"
    else:
        mc.composite_risk_level = "low"

    return mc


# ---------------------------------------------------------------------------
# Strategy selection logic
# ---------------------------------------------------------------------------


def _evaluate_bull_put_spread(mc: MarketConditions) -> StrategyRecommendation:
    """Evaluate Bull Put Spread (bullish credit spread).

    Best when: neutral-to-bullish outlook, elevated IV, put skew present.
    Sell OTM put, buy further OTM put for defined risk.
    """
    score = 0.0
    reasons: list[str] = []

    if mc.trend_direction in ("bullish", "slightly_bearish", "neutral"):
        score += 25
        reasons.append(f"outlook is {mc.trend_direction}")
    else:
        score += 5
        reasons.append("bearish outlook reduces suitability")

    if mc.iv_rank_category in ("high", "very_high"):
        score += 25
        reasons.append("elevated IV boosts premium collected")
    else:
        score += 10
        reasons.append("moderate IV limits premium potential")

    if mc.skew_bias == "put_skew":
        score += 20
        reasons.append("put skew makes put spreads more attractive")
    else:
        score += 10

    if mc.composite_risk_level in ("elevated", "high"):
        score += 15
        reasons.append("defined-risk spread good for volatile markets")
    else:
        score += 10

    score = min(score, 100.0)

    return StrategyRecommendation(
        strategy=OptionStrategy.BULL_PUT_SPREAD,
        rationale="; ".join(reasons) if reasons else "Standard bull put spread setup",
        confidence_score=score,
        market_outlook="bullish / neutral",
        is_viable=score >= 30,
        alternative_strategies=["iron_condor", "short_strangle"],
    )


def _evaluate_bear_call_spread(mc: MarketConditions) -> StrategyRecommendation:
    """Evaluate Bear Call Spread (bearish credit spread).

    Best when: neutral-to-bearish outlook, elevated IV.
    Sell OTM call, buy further OTM call for defined risk.
    """
    score = 0.0
    reasons: list[str] = []

    if mc.trend_direction in ("bearish", "slightly_bearish", "neutral"):
        score += 25
        reasons.append(f"outlook is {mc.trend_direction}")
    else:
        score += 5
        reasons.append("bullish outlook reduces suitability")

    if mc.iv_rank_category in ("high", "very_high"):
        score += 25
        reasons.append("elevated IV boosts premium collected")
    else:
        score += 10

    if mc.skew_bias == "call_skew":
        score += 20
        reasons.append("call skew makes call spreads more attractive")
    else:
        score += 10

    if mc.composite_risk_level in ("elevated", "high"):
        score += 15
    else:
        score += 10

    score = min(score, 100.0)

    return StrategyRecommendation(
        strategy=OptionStrategy.BEAR_CALL_SPREAD,
        rationale="; ".join(reasons) if reasons else "Standard bear call spread setup",
        confidence_score=score,
        market_outlook="bearish / neutral",
        is_viable=score >= 30,
        alternative_strategies=["iron_condor", "short_strangle"],
    )


def _evaluate_iron_condor(mc: MarketConditions) -> StrategyRecommendation:
    """Evaluate Iron Condor (neutral defined-risk).

    Best when: neutral outlook, elevated IV, balanced skew.
    Combines a bull put spread and bear call spread.
    """
    score = 0.0
    reasons: list[str] = []

    if mc.trend_direction in ("neutral", "slightly_bearish", "slightly_bullish"):
        score += 30
        reasons.append(f"neutral-ish outlook ({mc.trend_direction})")
    else:
        score += 5
        reasons.append("directional trend reduces suitability")

    if mc.iv_rank_category in ("high", "very_high"):
        score += 25
        reasons.append("elevated IV maximizes credit collected")
    else:
        score += 10

    if mc.skew_bias == "neutral":
        score += 15
        reasons.append("balanced skew supports iron condor")
    elif mc.skew_bias == "put_skew":
        score += 10
        reasons.append("put skew — consider wider put side")
    else:
        score += 5

    if "contango" in mc.term_structure:
        score += 15
        reasons.append("contango beneficial for short vol strategies")
    else:
        score += 10

    score = min(score, 100.0)

    return StrategyRecommendation(
        strategy=OptionStrategy.IRON_CONDOR,
        rationale="; ".join(reasons) if reasons else "Standard iron condor setup",
        confidence_score=score,
        market_outlook="neutral",
        is_viable=score >= 35,
        alternative_strategies=["short_strangle", "bull_put_spread", "bear_call_spread"],
    )


def _evaluate_short_strangle(mc: MarketConditions) -> StrategyRecommendation:
    """Evaluate Short Strangle (naked options, unlimited risk).

    Best when: very high IV, strong IV/RV premium, neutral outlook.
    Sells OTM put and OTM call. Uncapped risk — requires margin.
    """
    score = 0.0
    reasons: list[str] = []

    if mc.trend_direction == "neutral":
        score += 20
        reasons.append("neutral outlook is ideal")
    else:
        score += 5
        reasons.append(f"directional trend ({mc.trend_direction}) adds risk")

    if mc.iv_rank_category in ("very_high",):
        score += 30
        reasons.append("very high IV makes premium attractive")
    elif mc.iv_rank_category == "high":
        score += 20
        reasons.append("elevated IV")
    else:
        score += 5
        reasons.append("low IV limits premium")

    if mc.iv_rv_premium > 0.3:
        score += 20
        reasons.append("large IV/RV premium suggests overpricing")
    else:
        score += 10

    if mc.skew_bias == "neutral":
        score += 15
        reasons.append("balanced skew supports symmetrical strangle")
    else:
        score += 5
        reasons.append("skew makes asymmetric adjustment needed")

    # Risk: short strangle has undefined risk
    score -= 10  # Penalty for unlimited risk in retail context

    score = min(score, 100.0)

    return StrategyRecommendation(
        strategy=OptionStrategy.SHORT_STRANGLE,
        rationale="; ".join(reasons) if reasons else "Standard short strangle setup",
        confidence_score=score,
        market_outlook="neutral",
        is_viable=score >= 40,
        alternative_strategies=["iron_condor", "bull_put_spread", "bear_call_spread"],
    )


def _evaluate_calendar_spread(mc: MarketConditions) -> StrategyRecommendation:
    """Evaluate Calendar Spread (short vol, time decay).

    Best when: contango term structure, flat skew, normal-to-elevated IV.
    Sell short-term option, buy longer-term option with same strike.
    """
    score = 0.0
    reasons: list[str] = []

    if mc.term_structure == "contango":
        score += 30
        reasons.append("contango term structure is ideal")
    elif mc.term_structure == "backwardated":
        score += 5
        reasons.append("backwardation hurts calendar spreads")
    else:
        score += 15

    if mc.iv_rank_category in ("high", "very_high"):
        score += 15
        reasons.append("elevated IV helps short calendar")
    else:
        score += 10

    if mc.trend_direction == "neutral":
        score += 15
        reasons.append("neutral outlook suits calendar")
    else:
        score += 5

    if mc.composite_risk_level in ("moderate", "low"):
        score += 15
        reasons.append("lower vol environments suit calendar spreads")
    else:
        score += 5

    score = min(score, 100.0)

    return StrategyRecommendation(
        strategy=OptionStrategy.CALENDAR_SPREAD,
        rationale="; ".join(reasons) if reasons else "Standard calendar spread setup",
        confidence_score=score,
        market_outlook="neutral",
        is_viable=score >= 30,
        alternative_strategies=["iron_condor", "short_strangle"],
    )


# ---------------------------------------------------------------------------
# Main strategy selection entry point
# ---------------------------------------------------------------------------


def select_strategy(
    vol_analysis: VolatilityAnalysis,
) -> StrategyRecommendation:
    """Select the optimal premium-selling strategy based on market conditions.

    Evaluates all available strategies and returns the highest-confidence
    recommendation that is viable.

    Args:
        vol_analysis: Complete volatility analysis output.

    Returns:
        The highest-ranked StrategyRecommendation.
    """
    mc = classify_market_conditions(vol_analysis)

    evaluators = [
        _evaluate_bull_put_spread,
        _evaluate_bear_call_spread,
        _evaluate_iron_condor,
        _evaluate_short_strangle,
        _evaluate_calendar_spread,
    ]

    recommendations: list[StrategyRecommendation] = []

    for evaluator in evaluators:
        try:
            rec = evaluator(mc)
            recommendations.append(rec)
        except Exception as e:
            logger.error("Strategy evaluation failed: %s", e)
            continue

    if not recommendations:
        return StrategyRecommendation(
            strategy=OptionStrategy.IRON_CONDOR,
            rationale="Default recommendation — insufficient data for analysis",
            confidence_score=30.0,
            is_viable=False,
        )

    # Sort by confidence (highest first), preferring viable
    recommendations.sort(key=lambda r: (r.is_viable, r.confidence_score), reverse=True)

    best = recommendations[0]

    # Collect alternatives
    alts = []
    for r in recommendations[1:4]:
        if r.is_viable and r.strategy != best.strategy:
            alts.append(r.strategy.display_name)
    best.alternative_strategies = alts[:3]

    logger.info(
        "Selected strategy: %s (score=%.1f, viable=%s)",
        best.strategy.value,
        best.confidence_score,
        best.is_viable,
    )

    return best