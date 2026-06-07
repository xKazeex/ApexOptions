"""
Tests for the Greeks calculator module.

Tests Black-Scholes pricing, Greeks, and implied volatility solver.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from engine.greeks import (
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


class TestBlackScholes:
    """Tests for Black-Scholes option pricing."""

    def test_call_price_atm(self):
        """ATM call should be roughly 0.4 * S * sigma * sqrt(T) approximation."""
        S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.20
        result = black_scholes(S, K, T, r, sigma, "call")
        assert 4.0 < result < 6.0  # Approximate range

    def test_put_price_atm(self):
        """ATM put should be close to ATM call when r ≈ 0."""
        S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.0, 0.20
        call_price = black_scholes(S, K, T, r, sigma, "call")
        put_price = black_scholes(S, K, T, r, sigma, "put")
        assert abs(call_price - put_price) < 0.10  # Put-Call parity

    def test_deep_itm_call(self):
        """Deep ITM call should be worth approximately S - K."""
        S, K, T, r, sigma = 150.0, 100.0, 0.25, 0.05, 0.20
        result = black_scholes(S, K, T, r, sigma, "call")
        assert abs(result - (S - K * math.exp(-r * T))) < 0.50

    def test_deep_otm_call(self):
        """Deep OTM call should be close to zero."""
        S, K, T, r, sigma = 100.0, 200.0, 0.25, 0.05, 0.30
        result = black_scholes(S, K, T, r, sigma, "call")
        assert result < 0.01

    def test_deep_itm_put(self):
        """Deep ITM put should be worth approximately K - S."""
        S, K, T, r, sigma = 100.0, 150.0, 0.25, 0.05, 0.20
        result = black_scholes(S, K, T, r, sigma, "put")
        assert abs(result - (K * math.exp(-r * T) - S)) < 0.50

    def test_deep_otm_put(self):
        """Deep OTM put should be close to zero."""
        S, K, T, r, sigma = 200.0, 100.0, 0.25, 0.05, 0.30
        result = black_scholes(S, K, T, r, sigma, "put")
        assert result < 0.01

    def test_zero_time_to_expiry_itm_call(self):
        """At expiration, ITM call = S - K."""
        S, K, T, r, sigma = 110.0, 100.0, 0.0, 0.05, 0.20
        result = black_scholes(S, K, T, r, sigma, "call")
        assert abs(result - 10.0) < 0.01

    def test_zero_time_to_expiry_otm_call(self):
        """At expiration, OTM call = 0."""
        S, K, T, r, sigma = 90.0, 100.0, 0.0, 0.05, 0.20
        result = black_scholes(S, K, T, r, sigma, "call")
        assert abs(result) < 0.01

    def test_zero_time_to_expiry_itm_put(self):
        """At expiration, ITM put = K - S."""
        S, K, T, r, sigma = 90.0, 100.0, 0.0, 0.05, 0.20
        result = black_scholes(S, K, T, r, sigma, "put")
        assert abs(result - 10.0) < 0.01

    def test_invalid_option_type(self):
        """Should raise ValueError for invalid option type."""
        with pytest.raises(ValueError):
            black_scholes(100, 100, 0.25, 0.05, 0.20, "invalid")

    def test_negative_price(self):
        """Should raise ValueError for negative underlying price."""
        with pytest.raises(ValueError):
            black_scholes(-100, 100, 0.25, 0.05, 0.20, "call")


class TestGreeks:
    """Tests for Greeks calculations."""

    def test_call_delta_range(self):
        """Call delta should be between 0 and 1."""
        for strike in [80, 90, 100, 110, 120]:
            d = delta(100, strike, 0.25, 0.05, 0.20, "call")
            assert 0.0 <= d <= 1.0, f"Call delta {d} out of range at strike {strike}"

    def test_put_delta_range(self):
        """Put delta should be between -1 and 0."""
        for strike in [80, 90, 100, 110, 120]:
            d = delta(100, strike, 0.25, 0.05, 0.20, "put")
            assert -1.0 <= d <= 0.0, f"Put delta {d} out of range at strike {strike}"

    def test_atm_delta_call(self):
        """ATM call delta should be approximately 0.5 (adjusting for r>0)."""
        d = delta(100, 100, 0.25, 0.0, 0.20, "call")
        assert 0.45 < d < 0.55

    def test_gamma_positive(self):
        """Gamma should always be positive."""
        g = gamma(100, 100, 0.25, 0.05, 0.20)
        assert g > 0

    def test_gamma_same_call_put(self):
        """Gamma should be the same for calls and puts (same params)."""
        S, K, T, r, sigma, q = 100.0, 100.0, 0.25, 0.05, 0.20, 0.0
        g_call = gamma(S, K, T, r, sigma, q)
        g_put = gamma(S, K, T, r, sigma, q)
        assert abs(g_call - g_put) < 1e-10

    def test_theta_negative_long_call(self):
        """Theta for a long call should typically be negative."""
        t = theta(100, 100, 0.25, 0.05, 0.20, "call")
        assert t < 0

    def test_vega_positive(self):
        """Vega should be positive for both calls and puts."""
        v_call = vega(100, 100, 0.25, 0.05, 0.20)
        v_put = vega(100, 100, 0.25, 0.05, 0.20)
        assert v_call > 0
        assert v_put > 0

    def test_all_greeks(self):
        """all_greeks should return all 6 values."""
        result = all_greeks(100, 100, 0.25, 0.05, 0.20, "call")
        expected_keys = {"price", "delta", "gamma", "theta", "vega", "rho"}
        assert set(result.keys()) == expected_keys
        for key in expected_keys:
            assert isinstance(result[key], float)

    def test_itm_call_delta_high(self):
        """Deep ITM call delta should approach 1."""
        d = delta(150, 100, 0.25, 0.05, 0.30, "call")
        assert d > 0.85

    def test_otm_call_delta_low(self):
        """Deep OTM call delta should approach 0."""
        d = delta(100, 150, 0.25, 0.05, 0.30, "call")
        assert d < 0.15

    def test_atm_gamma_highest(self):
        """Gamma should be highest near the ATM strike."""
        g_atm = gamma(100, 100, 0.25, 0.05, 0.20)
        g_itm = gamma(100, 120, 0.25, 0.05, 0.20)
        g_otm = gamma(100, 80, 0.25, 0.05, 0.20)
        assert g_atm > g_itm
        assert g_atm > g_otm


class TestImpliedVolatility:
    """Tests for the implied volatility solver."""

    def test_iv_recovers_input(self):
        """IV solver should recover the input volatility from a BSM price."""
        S, K, T, r, sigma, q = 100.0, 100.0, 0.25, 0.05, 0.25, 0.0
        price_val = black_scholes(S, K, T, r, sigma, "call", q)
        iv = implied_volatility(price_val, S, K, T, r, "call", q)
        assert abs(iv - sigma) < 1e-6

    def test_iv_put(self):
        """IV solver should work for puts too."""
        S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.05, 0.35
        price_val = black_scholes(S, K, T, r, sigma, "put")
        iv = implied_volatility(price_val, S, K, T, r, "put")
        assert abs(iv - sigma) < 1e-6

    def test_iv_high_vol(self):
        """IV solver should handle high volatility."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 1.0
        price_val = black_scholes(S, K, T, r, sigma, "call")
        iv = implied_volatility(price_val, S, K, T, r, "call")
        assert abs(iv - sigma) < 1e-6

    def test_iv_short_dated(self):
        """IV solver should handle short-dated options."""
        S, K, T, r, sigma = 100.0, 100.0, 0.05, 0.05, 0.20
        price_val = black_scholes(S, K, T, r, sigma, "call")
        iv = implied_volatility(price_val, S, K, T, r, "call")
        assert abs(iv - sigma) < 1e-6

    def test_iv_negative_price(self):
        """Should raise ValueError for negative price."""
        with pytest.raises(ValueError):
            implied_volatility(-1.0, 100, 100, 0.25, 0.05, "call")

    def test_iv_zero_tte(self):
        """Should raise ValueError for zero time to expiry."""
        with pytest.raises(ValueError):
            implied_volatility(5.0, 100, 100, 0.0, 0.05, "call")

    def test_iv_below_intrinsic(self):
        """Should raise ValueError if price below intrinsic value."""
        with pytest.raises(ValueError):
            implied_volatility(1.0, 110, 100, 0.25, 0.05, "call")


class TestPutCallParity:
    """Tests put-call parity relationship."""

    def test_put_call_parity(self):
        """S + P = K*exp(-rT) + C should hold within numerical tolerance."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.03, 0.25
        call_price = black_scholes(S, K, T, r, sigma, "call")
        put_price = black_scholes(S, K, T, r, sigma, "put")

        lhs = S + put_price
        rhs = K * math.exp(-r * T) + call_price

        assert abs(lhs - rhs) < 1e-10

    def test_put_call_parity_with_dividends(self):
        """S*exp(-qT) + P = K*exp(-rT) + C (with dividends)."""
        S, K, T, r, sigma, q = 100.0, 100.0, 0.5, 0.03, 0.25, 0.02
        call_price = black_scholes(S, K, T, r, sigma, "call", q)
        put_price = black_scholes(S, K, T, r, sigma, "put", q)

        lhs = S * math.exp(-q * T) + put_price
        rhs = K * math.exp(-r * T) + call_price

        assert abs(lhs - rhs) < 1e-10