"""
Tests for the volatility analysis module.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from engine.strategy_selector import classify_market_conditions
from engine.volatility import (
    VolatilityAnalysis,
    analyze_term_structure,
    calculate_iv_percentile,
    calculate_iv_rank,
    calculate_realized_volatility,
    describe_skew,
    describe_term_structure,
    detect_volatility_regime,
)


class TestRealizedVolatility:
    """Tests for realized volatility calculation."""

    def test_constant_prices(self):
        """Constant prices should give 0% RV."""
        prices = pd.Series([100.0] * 100)
        rv = calculate_realized_volatility(prices, window=20)
        assert abs(rv) < 1e-10

    def test_steady_trend(self):
        """Steady linear trend should give deterministic RV."""
        prices = pd.Series([100.0 + i * 0.1 for i in range(100)])
        rv = calculate_realized_volatility(prices, window=20)
        assert rv > 0
        assert rv < 1.0  # Reasonable range

    def test_high_volatility(self):
        """High amplitude price changes should give higher RV."""
        np.random.seed(42)
        prices = pd.Series(100.0 * np.exp(np.random.randn(100) * 0.02).cumprod())
        rv = calculate_realized_volatility(prices, window=20)
        assert 0.05 < rv < 1.0

    def test_insufficient_data(self):
        """Should return 0.0 with insufficient data."""
        prices = pd.Series([100.0, 101.0])
        rv = calculate_realized_volatility(prices, window=20)
        assert rv == 0.0

    def test_empty_series(self):
        """Should return 0.0 for empty series."""
        rv = calculate_realized_volatility(pd.Series(dtype=float), window=20)
        assert rv == 0.0


class TestIVRankAndPercentile:
    """Tests for IV Rank and IV Percentile."""

    def test_iv_rank_at_min(self):
        """IV Rank should be 0 when current IV equals min."""
        result = calculate_iv_rank(0.20, [0.20, 0.30, 0.40, 0.50])
        assert abs(result) < 1e-10

    def test_iv_rank_at_max(self):
        """IV Rank should be 100 when current IV equals max."""
        result = calculate_iv_rank(0.50, [0.20, 0.30, 0.40, 0.50])
        assert abs(result - 100.0) < 1e-10

    def test_iv_rank_midpoint(self):
        """IV Rank should be ~50 when current IV is at midpoint."""
        result = calculate_iv_rank(0.35, [0.20, 0.30, 0.40, 0.50])
        assert 45.0 < result < 55.0

    def test_iv_rank_empty(self):
        """IV Rank should return 50 for empty data."""
        result = calculate_iv_rank(0.30, [])
        assert abs(result - 50.0) < 1e-10

    def test_iv_rank_single_value(self):
        """IV Rank should return 50 for single value."""
        result = calculate_iv_rank(0.30, [0.30])
        assert abs(result - 50.0) < 1e-10

    def test_iv_percentile(self):
        """IV Percentile should correctly count lower values."""
        result = calculate_iv_percentile(0.35, [0.20, 0.25, 0.30, 0.40, 0.50])
        # 3 values are lower, total = 5, so 60%
        assert abs(result - 60.0) < 1e-10

    def test_iv_percentile_all_higher(self):
        """IV Percentile should be 0 when all historical values are higher."""
        result = calculate_iv_percentile(0.20, [0.30, 0.40, 0.50])
        assert abs(result) < 1e-10

    def test_iv_percentile_all_lower(self):
        """IV Percentile should be 100 when all historical values are lower."""
        result = calculate_iv_percentile(0.50, [0.20, 0.30, 0.40])
        assert abs(result - 100.0) < 1e-10


class TestRegimeDetection:
    """Tests for volatility regime detection."""

    def test_high_iv_regime(self):
        """IVR >= 70 should indicate high IV regime."""
        regime = detect_volatility_regime(75.0, 1.2, 80.0)
        assert "high" in regime.lower()

    def test_crush_opportunity(self):
        """High IVR + high IV/RV should indicate crush opportunity."""
        regime = detect_volatility_regime(80.0, 1.5, 85.0)
        assert "crush" in regime.lower()

    def test_elevated_iv_regime(self):
        """IVR 50-70 should indicate elevated regime."""
        regime = detect_volatility_regime(60.0, 1.1, 60.0)
        assert "elevated" in regime.lower()

    def test_normal_iv_regime(self):
        """IVR 20-50 should indicate normal regime."""
        regime = detect_volatility_regime(35.0, 1.0, 40.0)
        assert "normal" in regime.lower()

    def test_low_iv_regime(self):
        """IVR < 20 should indicate low regime."""
        regime = detect_volatility_regime(10.0, 0.8, 15.0)
        assert "low" in regime.lower()


class TestMarketConditions:
    """Tests for market condition classification."""

    def test_basic_classification(self):
        """Happy path classification should not raise errors."""
        vol = VolatilityAnalysis(
            ticker="SPY",
            underlying_price=500.0,
            current_iv=0.25,
            iv_rank=65.0,
            iv_percentile=70.0,
            realized_vol_20d=0.18,
            realized_vol_60d=0.17,
            iv_rv_ratio_20d=1.39,
            iv_rv_ratio_60d=1.47,
            vol_regime="elevated IV regime",
            term_structure="contango (longer-term vol > near-term vol)",
            skew_description="moderate put skew",
            is_premium_selling_opportunity=True,
            opportunity_score=75.0,
        )
        mc = classify_market_conditions(vol)
        assert mc.iv_rank_category == "high"
        assert "elevated" in mc.composite_risk_level

    def test_very_high_iv_category(self):
        """IVR >= 70 should be 'very_high'."""
        vol = VolatilityAnalysis(
            ticker="SPY",
            iv_rank=85.0,
            iv_percentile=90.0,
            iv_rv_ratio_20d=1.5,
            iv_rv_ratio_60d=1.4,
        )
        mc = classify_market_conditions(vol)
        assert mc.iv_rank_category == "very_high"