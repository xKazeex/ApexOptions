"""
Tests for the risk manager module.
"""

from __future__ import annotations

import pytest

from engine.position_optimizer import PositionParameters
from engine.risk_manager import (
    RiskAssessment,
    assess_trade_risk,
    check_profit_target,
    check_stop_loss,
)
from engine.strategy_selector import OptionStrategy


class TestAssessTradeRisk:
    """Tests for trade risk assessment."""

    def test_basic_assessment(self):
        """Basic trade should pass risk checks."""
        params = PositionParameters(
            strategy=OptionStrategy.IRON_CONDOR,
            underlying_ticker="SPY",
            underlying_price=500.0,
            dte=45,
            short_strike=520.0,
            long_strike=530.0,
            short_strike_2=480.0,
            long_strike_2=470.0,
            spread_width=10.0,
            premium_collected=2.50,
            max_loss=7.50,
            max_profit=2.50,
            risk_reward_ratio=3.0,
            probability_of_profit=0.70,
            delta_short=0.30,
            theta_short=0.05,
            vega_short=0.10,
            contracts=1,
            capital_required=750.0,
            risk_per_contract=750.0,
        )
        assessment = assess_trade_risk(params, account_size=100_000)
        assert assessment.is_approved
        assert assessment.max_risk_dollars == 750.0
        assert assessment.profit_target_dollars > 0
        assert assessment.stop_loss_dollars > 0

    def test_excessive_risk(self):
        """Trade exceeding risk-per-trade should be flagged."""
        params = PositionParameters(
            strategy=OptionStrategy.IRON_CONDOR,
            underlying_ticker="SPY",
            underlying_price=500.0,
            dte=45,
            short_strike=520.0,
            long_strike=530.0,
            short_strike_2=480.0,
            long_strike_2=470.0,
            spread_width=10.0,
            premium_collected=2.50,
            max_loss=7.50,
            max_profit=2.50,
            risk_reward_ratio=3.0,
            probability_of_profit=0.70,
            delta_short=0.30,
            theta_short=0.05,
            vega_short=0.10,
            contracts=5,
            capital_required=750.0,
            risk_per_contract=750.0,
        )
        # $3750 risk on $50000 account = 7.5% > 2%
        assessment = assess_trade_risk(params, account_size=50_000)
        assert len(assessment.warnings) > 0
        assert assessment.suggested_contracts < 5

    def test_small_account(self):
        """Trade risk should be capped for small accounts."""
        params = PositionParameters(
            strategy=OptionStrategy.BULL_PUT_SPREAD,
            underlying_ticker="AAPL",
            underlying_price=200.0,
            dte=45,
            short_strike=190.0,
            long_strike=185.0,
            spread_width=5.0,
            premium_collected=1.20,
            max_loss=3.80,
            max_profit=1.20,
            risk_reward_ratio=3.17,
            probability_of_profit=0.68,
            delta_short=-0.30,
            theta_short=0.03,
            vega_short=0.08,
            contracts=1,
            capital_required=380.0,
            risk_per_contract=380.0,
        )
        # $380 risk on $10000 account = 3.8% > 2%
        assessment = assess_trade_risk(params, account_size=10_000)
        assert len(assessment.warnings) > 0
        assert assessment.suggested_contracts == 1  # Can't go below 1
        assert assessment.max_risk_dollars == 380.0

    def test_existing_positions(self):
        """Portfolio risk check with existing positions."""
        params = PositionParameters(
            strategy=OptionStrategy.IRON_CONDOR,
            underlying_ticker="SPY",
            underlying_price=500.0,
            dte=45,
            short_strike=520.0,
            long_strike=530.0,
            spread_width=10.0,
            premium_collected=2.50,
            max_loss=7.50,
            max_profit=2.50,
            risk_reward_ratio=3.0,
            probability_of_profit=0.70,
            delta_short=0.30,
            theta_short=0.05,
            vega_short=0.10,
            contracts=1,
            capital_required=750.0,
            risk_per_contract=750.0,
        )
        existing = [
            {"ticker": "QQQ", "risk_dollars": 1200, "strategy": "iron_condor"},
            {"ticker": "IWM", "risk_dollars": 800, "strategy": "bull_put_spread"},
        ]
        # Total risk would be 750 + 2000 = 2750 / 20000 = 13.75% < 15% max
        assessment = assess_trade_risk(params, account_size=20_000, existing_positions=existing)
        # Should pass since 13.75% < 15%
        assert assessment.is_approved

    def test_excessive_portfolio_risk(self):
        """Trade should be rejected if it pushes portfolio risk too high."""
        params = PositionParameters(
            strategy=OptionStrategy.IRON_CONDOR,
            underlying_ticker="SPY",
            underlying_price=500.0,
            dte=45,
            spread_width=10.0,
            premium_collected=2.50,
            max_loss=7.50,
            max_profit=2.50,
            risk_reward_ratio=3.0,
            probability_of_profit=0.70,
            delta_short=0.30,
            theta_short=0.05,
            vega_short=0.10,
            contracts=3,
            capital_required=2250.0,
            risk_per_contract=750.0,
        )
        existing = [
            {"ticker": "QQQ", "risk_dollars": 3000, "strategy": "iron_condor"},
            {"ticker": "IWM", "risk_dollars": 2500, "strategy": "short_strangle"},
        ]
        # Total risk = 2250 + 5500 = 7750 / 25000 = 31% > 15% max
        assessment = assess_trade_risk(params, account_size=25_000, existing_positions=existing)
        assert not assessment.is_approved

    def test_ticker_concentration(self):
        """Should warn on ticker concentration."""
        params = PositionParameters(
            strategy=OptionStrategy.BULL_PUT_SPREAD,
            underlying_ticker="AAPL",
            underlying_price=200.0,
            dte=45,
            spread_width=5.0,
            premium_collected=1.20,
            max_loss=3.80,
            max_profit=1.20,
            risk_reward_ratio=3.17,
            probability_of_profit=0.68,
            delta_short=-0.30,
            theta_short=0.03,
            vega_short=0.08,
            contracts=1,
            capital_required=380.0,
            risk_per_contract=380.0,
        )
        existing = [
            {"ticker": "AAPL", "risk_dollars": 3500, "strategy": "bear_call_spread"},
        ]
        # Total AAPL risk = 380 + 3500 = 3880 / 20000 = 19.4% > 10%
        assessment = assess_trade_risk(params, account_size=20_000, existing_positions=existing)
        assert any("AAPL" in w for w in assessment.warnings)

    def test_short_dte_warning(self):
        """Short DTE should generate a warning."""
        params = PositionParameters(
            strategy=OptionStrategy.IRON_CONDOR,
            underlying_ticker="SPY",
            underlying_price=500.0,
            dte=5,
            spread_width=10.0,
            premium_collected=2.50,
            max_loss=7.50,
            max_profit=2.50,
            risk_reward_ratio=3.0,
            probability_of_profit=0.70,
            delta_short=0.30,
            theta_short=0.05,
            vega_short=0.10,
            contracts=1,
            capital_required=750.0,
            risk_per_contract=750.0,
        )
        assessment = assess_trade_risk(params, account_size=100_000)
        assert any("DTE" in w for w in assessment.warnings)


class TestStopLoss:
    """Tests for stop-loss monitoring."""

    def test_stop_not_triggered(self):
        """Should not trigger stop when premium is stable."""
        triggered = check_stop_loss(
            current_premium=2.00,
            initial_premium=2.50,
            contracts=1,
            stop_loss_multiple=2.0,
        )
        assert not triggered

    def test_stop_triggered(self):
        """Should trigger stop when premium triples."""
        triggered = check_stop_loss(
            current_premium=7.60,  # Loss = (760-250)/250 = 2.04x >= 2.0x
            initial_premium=2.50,
            contracts=1,
            stop_loss_multiple=2.0,
        )
        assert triggered

    def test_stop_edge_case(self):
        """Should trigger stop exactly at boundary."""
        triggered = check_stop_loss(
            current_premium=7.50,
            initial_premium=2.50,
            contracts=1,
            stop_loss_multiple=2.0,
        )
        # loss_pct = (750 - 250) / 250 = 2.0 = exactly at threshold
        assert triggered


class TestProfitTarget:
    """Tests for profit target monitoring."""

    def test_profit_not_reached(self):
        """Should not trigger when profit is below target."""
        hit = check_profit_target(
            current_premium=2.00,
            initial_premium=2.50,
            contracts=1,
            profit_target_pct=0.50,
        )
        # profit_taken = 250 - 200 = 50, target = 125
        assert not hit

    def test_profit_reached(self):
        """Should trigger when profit target is hit."""
        hit = check_profit_target(
            current_premium=1.00,
            initial_premium=2.50,
            contracts=1,
            profit_target_pct=0.50,
        )
        # profit_taken = 250 - 100 = 150, target = 125
        assert hit


class TestRiskScore:
    """Tests for risk score calculation."""

    def test_low_risk_score(self):
        """A conservative trade should have a low risk score."""
        params = PositionParameters(
            strategy=OptionStrategy.IRON_CONDOR,
            underlying_ticker="SPY",
            underlying_price=500.0,
            dte=45,
            spread_width=10.0,
            premium_collected=2.50,
            max_loss=7.50,
            max_profit=2.50,
            risk_reward_ratio=3.0,
            probability_of_profit=0.75,
            delta_short=0.25,
            theta_short=0.05,
            vega_short=0.10,
            contracts=1,
            capital_required=750.0,
            risk_per_contract=750.0,
        )
        assessment = assess_trade_risk(params, account_size=200_000)
        assert assessment.risk_score < 50

    def test_high_risk_score(self):
        """An aggressive trade should have a higher risk score."""
        params = PositionParameters(
            strategy=OptionStrategy.SHORT_STRANGLE,
            underlying_ticker="SPY",
            underlying_price=500.0,
            dte=10,
            spread_width=40.0,
            premium_collected=8.00,
            max_loss=32.00,
            max_profit=8.00,
            risk_reward_ratio=4.0,
            probability_of_profit=0.55,
            delta_short=0.40,
            theta_short=0.15,
            vega_short=0.25,
            contracts=5,
            capital_required=16000.0,
            risk_per_contract=3200.0,
        )
        assessment = assess_trade_risk(params, account_size=100_000)
        assert assessment.risk_score > 50