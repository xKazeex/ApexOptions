"""
Apex Options Analytics Engine.

A quantitative engine for analyzing market microstructure, volatility dynamics,
and generating institutional-grade options premium-selling recommendations.

Public API
----------
The engine exposes a clean, single-entry-point interface:

    from engine import run_analysis, ApexStrategyReport

    # Full pipeline: data → volatility → strategy → position → risk → report
    report = run_analysis(ticker="SPY", account_size=100_000, output_format="markdown")

Individual module components are also available for advanced use.

Modules:
    config:            Configuration management (EngineConfig dataclass)
    data_fetcher:      Market data abstraction (DataSource ABC, YahooFinanceDataSource)
    greeks:            Black-Scholes pricing, Greeks, implied volatility solver
    volatility:        IV/RV analysis, IV rank/percentile, volatility surface, regime detection
    strategy_selector: Rule-based strategy selection logic (5 strategies)
    position_optimizer: Strike/DTE/sizing optimization
    risk_manager:      Risk assessment, profit targets, stop-loss monitoring
    report_generator:  Structured report output (JSON + Markdown)
"""

from __future__ import annotations

from typing import Optional

from .config import EngineConfig, DEFAULT_CONFIG
from .data_fetcher import (
    DataSource,
    OptionContract,
    OptionsChain,
    UnderlyingData,
    YahooFinanceDataSource,
    create_data_source,
)
from .greeks import (
    all_greeks,
    black_scholes,
    delta,
    gamma,
    implied_volatility,
    price,
    rho,
    theta,
    vega,
)
from .position_optimizer import (
    PositionParameters,
    calculate_position_size,
    optimize_position,
)
from .report_generator import (
    ApexStrategyReport,
    generate_full_report,
    generate_json_report,
    generate_markdown_report,
)
from .risk_manager import (
    RiskAssessment,
    assess_trade_risk,
    check_profit_target,
    check_stop_loss,
)
from .strategy_selector import (
    MarketConditions,
    OptionStrategy,
    StrategyRecommendation,
    select_strategy,
)
from .volatility import (
    VolatilityAnalysis,
    analyze_volatility,
    calculate_iv_percentile,
    calculate_iv_rank,
    calculate_realized_volatility,
    detect_volatility_regime,
)

__all__ = [
    # High-level pipeline
    "run_analysis",
    # Config
    "EngineConfig",
    "DEFAULT_CONFIG",
    # Data
    "DataSource",
    "OptionContract",
    "OptionsChain",
    "UnderlyingData",
    "YahooFinanceDataSource",
    "create_data_source",
    # Greeks
    "black_scholes",
    "price",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "all_greeks",
    "implied_volatility",
    # Volatility
    "VolatilityAnalysis",
    "analyze_volatility",
    "calculate_realized_volatility",
    "calculate_iv_rank",
    "calculate_iv_percentile",
    "detect_volatility_regime",
    # Strategy
    "OptionStrategy",
    "StrategyRecommendation",
    "MarketConditions",
    "select_strategy",
    # Position
    "PositionParameters",
    "optimize_position",
    "calculate_position_size",
    # Risk
    "RiskAssessment",
    "assess_trade_risk",
    "check_stop_loss",
    "check_profit_target",
    # Reports
    "ApexStrategyReport",
    "generate_full_report",
    "generate_json_report",
    "generate_markdown_report",
]

__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# High-level pipeline (single entry point)
# ---------------------------------------------------------------------------


def run_analysis(
    ticker: str,
    account_size: float = 100_000,
    output_format: str = "markdown",
    data_source: str = "yahoo",
    config: Optional[EngineConfig] = None,
) -> str:
    """Run the full Apex analytics pipeline on a ticker.

    This is the primary entry point for the engine. It:
    1. Fetches live market data (options chain + underlying)
    2. Analyzes volatility (IV rank, IV percentile, RV, regime)
    3. Selects the optimal premium-selling strategy
    4. Optimizes position parameters (strikes, DTE, sizing)
    5. Assesses risk (per-trade, portfolio, stop-loss, profit targets)
    6. Generates a structured report (Markdown or JSON)

    Args:
        ticker: Stock ticker symbol (e.g., "SPY", "AAPL").
        account_size: Portfolio account value in dollars.
        output_format: 'markdown' (default) or 'json'.
        data_source: Data provider ('yahoo' for bootstrapping).
        config: Optional EngineConfig override.

    Returns:
        Formatted report string (Markdown or JSON).

    Raises:
        ValueError: If data cannot be fetched or strategy selection fails.
    """
    import logging

    logger = logging.getLogger(__name__)
    cfg = config or DEFAULT_CONFIG

    # 1. Fetch data
    logger.info("Fetching data for %s...", ticker)
    source = create_data_source(data_source, timeout=cfg.yfinance_timeout)
    underlying = source.fetch_underlying(ticker)
    chain = source.fetch_options_chain(
        ticker, max_strikes=cfg.max_strikes_per_side
    )

    # 2. Analyze volatility
    logger.info("Analyzing volatility for %s...", ticker)
    vol = analyze_volatility(underlying, chain, rv_windows=cfg.rv_windows)

    # 3. Select strategy
    logger.info("Selecting strategy for %s...", ticker)
    strategy_rec = select_strategy(vol)

    if not strategy_rec.is_viable:
        logger.warning(
            "No viable strategy found for %s (score=%.1f)",
            ticker,
            strategy_rec.confidence_score,
        )

    # 4. Optimize position
    logger.info("Optimizing position for %s...", ticker)
    position = optimize_position(strategy_rec.strategy, chain, vol, cfg)

    # Apply position sizing
    position.contracts = calculate_position_size(
        risk_per_contract=position.risk_per_contract,
        account_size=account_size,
        risk_per_trade_pct=cfg.risk_per_trade,
    )

    # 5. Assess risk
    logger.info("Assessing risk for %s...", ticker)
    risk = assess_trade_risk(position, account_size, cfg)

    # 6. Generate report
    logger.info("Generating report for %s...", ticker)
    report = generate_full_report(
        vol_analysis=vol,
        strategy_rec=strategy_rec,
        position_params=position,
        risk_assessment=risk,
        account_size=account_size,
        output_format=output_format,
    )

    return report