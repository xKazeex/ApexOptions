"""
Configuration management for the Apex Options Analytics Engine.

Configuration is loaded from environment variables with sensible defaults
suitable for bootstrapping. Can be overridden via a configuration dict or
.env file in the future.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EngineConfig:
    """Central configuration for the options analytics engine.

    All settings have sensible defaults for bootstrapping with free data
    sources. Override via environment variables or by constructing a custom
    Config object and passing it to engine components.

    Attributes:
        risk_per_trade: Maximum risk per trade as fraction of portfolio (0.0–1.0).
        max_portfolio_risk: Maximum total portfolio risk at any time (0.0–1.0).
        default_dte_min: Minimum days to expiration for recommendations.
        default_dte_max: Maximum days to expiration for recommendations.
        profit_target_pct: Fraction of max premium to target for profit (0.0–1.0).
        stop_loss_pct: Fraction of max premium collected to trigger stop-loss (0.0–1.0).
        ivr_high_threshold: IV Rank above which is considered "high IV" for selling.
        ivr_low_threshold: IV Rank below which is considered "low IV".
        min_premium_collected: Minimum premium (as fraction of width) to consider a trade.
        option_iv_window: Number of trading days for IV history (≈ 1 year = 252).
        rv_windows: List of windows (in trading days) for RV calculation.
        max_strikes_per_side: Maximum number of strikes to evaluate per side.
        yfinance_timeout: HTTP request timeout for yfinance calls (seconds).
    """

    # Risk parameters
    risk_per_trade: float = field(
        default_factory=lambda: float(os.environ.get("APEX_RISK_PER_TRADE", "0.02"))
    )
    max_portfolio_risk: float = field(
        default_factory=lambda: float(os.environ.get("APEX_MAX_PORTFOLIO_RISK", "0.15"))
    )

    # Time parameters
    default_dte_min: int = int(os.environ.get("APEX_DTE_MIN", "7"))
    default_dte_max: int = int(os.environ.get("APEX_DTE_MAX", "180"))

    # Profit / loss targets
    profit_target_pct: float = field(
        default_factory=lambda: float(os.environ.get("APEX_PROFIT_TARGET_PCT", "0.50"))
    )
    stop_loss_pct: float = field(
        default_factory=lambda: float(os.environ.get("APEX_STOP_LOSS_PCT", "2.0"))
    )

    # Volatility thresholds
    ivr_high_threshold: float = field(
        default_factory=lambda: float(os.environ.get("APEX_IVR_HIGH", "50.0"))
    )
    ivr_low_threshold: float = field(
        default_factory=lambda: float(os.environ.get("APEX_IVR_LOW", "20.0"))
    )

    # Trade quality
    min_premium_collected: float = field(
        default_factory=lambda: float(os.environ.get("APEX_MIN_PREMIUM", "0.10"))
    )

    # Analysis windows
    option_iv_window: int = int(os.environ.get("APEX_IV_WINDOW", "252"))
    rv_windows: tuple[int, ...] = (10, 20, 60, 120)

    # Performance
    max_strikes_per_side: int = int(os.environ.get("APEX_MAX_STRIKES", "100"))
    yfinance_timeout: int = int(os.environ.get("APEX_YFINANCE_TIMEOUT", "10"))

    def validate(self) -> None:
        """Validate configuration values, raising ValueError on invalid inputs."""
        if not 0.0 < self.risk_per_trade <= 1.0:
            raise ValueError(f"risk_per_trade must be in (0, 1], got {self.risk_per_trade}")
        if not 0.0 < self.max_portfolio_risk <= 1.0:
            raise ValueError(
                f"max_portfolio_risk must be in (0, 1], got {self.max_portfolio_risk}"
            )
        if self.default_dte_min < 1:
            raise ValueError(f"default_dte_min must be >= 1, got {self.default_dte_min}")
        if self.default_dte_max < self.default_dte_min:
            raise ValueError(
                f"default_dte_max ({self.default_dte_max}) must be >= "
                f"default_dte_min ({self.default_dte_min})"
            )
        if not 0.0 < self.profit_target_pct <= 1.0:
            raise ValueError(
                f"profit_target_pct must be in (0, 1], got {self.profit_target_pct}"
            )
        if self.stop_loss_pct <= 0:
            raise ValueError(f"stop_loss_pct must be > 0, got {self.stop_loss_pct}")


# Module-level singleton for convenience
DEFAULT_CONFIG = EngineConfig()
DEFAULT_CONFIG.validate()