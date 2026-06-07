# Apex Options Analytics Engine

**Institutional-grade quantitative analytics for options premium-selling strategies.**

A Python engine that analyzes market microstructure, volatility dynamics, and historical data to generate risk-managed options premium-selling recommendations. Takes a stock ticker → fetches live market data → analyzes volatility → selects strategy → optimizes parameters → outputs structured trade recommendations.

---

## Architecture

```
apex-options/
├── engine/
│   ├── __init__.py          # Package init, version
│   ├── config.py            # Configuration (dataclass + env vars)
│   ├── data_fetcher.py      # Market data abstraction (DataSource ABC + YahooFinance impl)
│   ├── greeks.py            # Black-Scholes-Merton pricing, Greeks, IV solver
│   ├── volatility.py        # IV/RV analysis, IV Rank/Percentile, regime detection
│   ├── strategy_selector.py # Rule-based strategy selection (5 strategies)
│   ├── position_optimizer.py# Strike/DTE/sizing optimization, position parameters
│   ├── risk_manager.py      # Risk assessment, profit targets, stop-loss checks
│   └── report_generator.py  # Report output (JSON + Markdown)
├── tests/
│   ├── test_greeks.py       # BSM pricing, Greeks, IV solver tests
│   ├── test_volatility.py   # RV calc, IV rank/percentile, regime tests
│   └── test_risk_manager.py # Risk assessment, stop-loss, profit target tests
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=engine -v
```

## Usage Example

```python
from engine.data_fetcher import create_data_source
from engine.volatility import analyze_volatility
from engine.strategy_selector import select_strategy
from engine.position_optimizer import optimize_position
from engine.risk_manager import assess_trade_risk
from engine.report_generator import generate_full_report

# 1. Fetch data
source = create_data_source("yahoo")
underlying = source.fetch_underlying("SPY")
chain = source.fetch_options_chain("SPY")

# 2. Analyze volatility
vol = analyze_volatility(underlying, chain)
print(f"IV Rank: {vol.iv_rank:.1f}%")
print(f"Regime: {vol.vol_regime}")

# 3. Select strategy
strategy = select_strategy(vol)
print(f"Best: {strategy.strategy.display_name} (confidence: {strategy.confidence_score:.0f}/100)")

# 4. Optimize position
position = optimize_position(strategy.strategy, chain, vol)
print(f"Short strike: {position.short_strike}, Premium: ${position.premium_collected:.2f}")

# 5. Assess risk
risk = assess_trade_risk(position, account_size=100_000)
print(f"Risk score: {risk.risk_score:.0f}/100")

# 6. Generate report
report = generate_full_report(vol, strategy, position, risk, output_format="markdown")
print(report)
```

## Supported Strategies

| Strategy | Best When | Risk Type |
|----------|-----------|-----------|
| **Bull Put Spread** | Neutral-to-bullish, elevated IV | Defined risk |
| **Bear Call Spread** | Neutral-to-bearish, elevated IV | Defined risk |
| **Iron Condor** | Neutral, high IV, balanced skew | Defined risk |
| **Short Strangle** | Neutral, very high IV/RV premium | Undefined risk |
| **Calendar Spread** | Neutral, contango term structure | Defined risk |

## Configuration

Configure via environment variables or the `EngineConfig` dataclass:

| Variable | Default | Description |
|----------|---------|-------------|
| `APEX_RISK_PER_TRADE` | `0.02` (2%) | Max risk per trade |
| `APEX_MAX_PORTFOLIO_RISK` | `0.15` (15%) | Max portfolio risk |
| `APEX_DTE_MIN` | `30` | Min days to expiration |
| `APEX_DTE_MAX` | `60` | Max days to expiration |
| `APEX_PROFIT_TARGET_PCT` | `0.50` (50%) | Profit target |
| `APEX_IVR_HIGH` | `50.0` | High IV threshold |
| `APEX_IVR_LOW` | `20.0` | Low IV threshold |

## Output Formats

- **Markdown**: Human-readable strategy report with tables for volatility metrics, trade structure, risk parameters, and position sizing
- **JSON**: Machine-readable output for API consumption and integration

## Data Sources

Currently uses **Yahoo Finance** (via `yfinance`) for free market data. The `DataSource` abstract base class allows swapping to paid providers (Polygon, Tradier, etc.) by implementing the interface.

## Development

```bash
# Type checking
mypy engine/ --strict

# Run all tests
pytest tests/ -v --tb=short

# Test with coverage
pytest tests/ --cov=engine --cov-report=term-missing
```

## License

Proprietary — Apex Options Analytics
