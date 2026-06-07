"""
Greeks calculator for the Apex Options Analytics Engine.

Implements Black-Scholes-Merton option pricing, first-order Greeks
(Delta, Gamma, Theta, Vega, Rho), and implied volatility calculation
via Newton-Raphson root-finding.

All functions accept and return float values. Volatility is expressed
as a decimal (e.g., 0.25 = 25%). Time-to-expiration is in years.
"""

from __future__ import annotations

import logging
from math import exp, log, sqrt
from typing import Tuple

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Black-Scholes-Merton pricing
# ---------------------------------------------------------------------------


def _d1(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """Calculate d1 in the Black-Scholes formula.

    Args:
        S: Current price of the underlying.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (decimal).
        sigma: Implied volatility (decimal).
        q: Dividend yield (decimal).

    Returns:
        d1 value.
    """
    if T <= 0:
        return 0.0
    return (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))


def _d2(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """Calculate d2 in the Black-Scholes formula.

    d2 = d1 - sigma * sqrt(T)
    """
    return _d1(S, K, T, r, sigma, q) - sigma * sqrt(T)


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float = 0.0,
) -> float:
    """Calculate the Black-Scholes-Merton price of a European option.

    Args:
        S: Current price of the underlying.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (decimal).
        sigma: Implied volatility (decimal).
        option_type: 'call' or 'put'.
        q: Dividend yield (decimal).

    Returns:
        Option price.

    Raises:
        ValueError: If option_type is not 'call' or 'put'.
    """
    _validate_inputs(S, K, T, sigma)

    if T <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))

    d1 = _d1(S, K, T, r, sigma, q)
    d2 = _d2(S, K, T, r, sigma, q)

    if option_type == "call":
        price = S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    return float(price)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------


def delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float = 0.0,
) -> float:
    """Calculate the Delta of a European option.

    Delta measures the rate of change of the option price with respect
    to changes in the underlying asset price.

    Returns:
        Delta value. Call delta is in [0, 1]; put delta is in [-1, 0].
    """
    _validate_inputs(S, K, T, sigma)

    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else (0.5 if S == K else 0.0)
        else:
            return -1.0 if S < K else (-0.5 if S == K else 0.0)

    d1 = _d1(S, K, T, r, sigma, q)

    if option_type == "call":
        return float(exp(-q * T) * norm.cdf(d1))
    elif option_type == "put":
        return float(exp(-q * T) * (norm.cdf(d1) - 1.0))
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """Calculate the Gamma of a European option.

    Gamma measures the rate of change of Delta with respect to changes
    in the underlying price. It is the same for calls and puts.

    Returns:
        Gamma value (always positive).
    """
    _validate_inputs(S, K, T, sigma)

    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma, q)
    gamma_val = exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt(T))
    return float(gamma_val)


def theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float = 0.0,
) -> float:
    """Calculate the Theta of a European option (per calendar day).

    Theta measures the rate of change of the option price with respect
    to the passage of time (time decay). Returned as the *daily* decay
    (divided by 365).

    Returns:
        Theta value (typically negative for long options).
    """
    _validate_inputs(S, K, T, sigma)

    if T <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma, q)
    d2 = _d2(S, K, T, r, sigma, q)

    term1 = -(S * exp(-q * T) * norm.pdf(d1) * sigma) / (2 * sqrt(T))

    if option_type == "call":
        term2 = r * K * exp(-r * T) * norm.cdf(d2)
        term3 = q * S * exp(-q * T) * norm.cdf(d1)
        theta_val = term1 - term2 + term3
    elif option_type == "put":
        term2 = r * K * exp(-r * T) * norm.cdf(-d2)
        term3 = q * S * exp(-q * T) * norm.cdf(-d1)
        theta_val = term1 + term2 - term3
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    # Convert from annual to daily
    return float(theta_val / 365.0)


def vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """Calculate the Vega of a European option.

    Vega measures the rate of change of the option price with respect
    to a 1% change in implied volatility. Returned as sensitivity per
    1 percentage point (i.e., a 1% absolute change in vol).

    Returns:
        Vega value (option price change per 1% vol change).
    """
    _validate_inputs(S, K, T, sigma)

    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma, q)
    vega_val = S * exp(-q * T) * norm.pdf(d1) * sqrt(T)
    # Scale: change per 1% (0.01) change in volatility
    return float(vega_val / 100.0)


def rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float = 0.0,
) -> float:
    """Calculate the Rho of a European option.

    Rho measures the rate of change of the option price with respect
    to a 1% change in the risk-free interest rate.

    Returns:
        Rho value (option price change per 1% rate change).
    """
    _validate_inputs(S, K, T, sigma)

    if T <= 0:
        return 0.0

    d2 = _d2(S, K, T, r, sigma, q)

    if option_type == "call":
        rho_val = K * T * exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        rho_val = -K * T * exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    # Scale: change per 1% (0.01) change in rate
    return float(rho_val / 100.0)


def all_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float = 0.0,
) -> dict[str, float]:
    """Calculate all Greeks for a European option at once.

    More efficient than calling each function individually.

    Returns:
        Dictionary with keys: 'price', 'delta', 'gamma', 'theta', 'vega', 'rho'.
    """
    _validate_inputs(S, K, T, sigma)

    if T <= 0:
        intrinsic = max(0.0, (S - K) if option_type == "call" else (K - S))
        d = 1.0 if S > K else (0.5 if S == K else 0.0)
        return {
            "price": intrinsic,
            "delta": d if option_type == "call" else -d,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0,
        }

    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * sqrt(T)
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        price = S * exp(-q * T) * nd1 - K * exp(-r * T) * nd2
        delta_val = exp(-q * T) * nd1
        theta_term = r * K * exp(-r * T) * nd2
        theta_d1_term = q * S * exp(-q * T) * nd1
        rho_val = K * T * exp(-r * T) * nd2
    else:
        price = K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)
        delta_val = exp(-q * T) * (nd1 - 1.0)
        theta_term = r * K * exp(-r * T) * norm.cdf(-d2)
        theta_d1_term = q * S * exp(-q * T) * norm.cdf(-d1)
        rho_val = -K * T * exp(-r * T) * norm.cdf(-d2)

    gamma_val = exp(-q * T) * pdf_d1 / (S * sigma * sqrt(T))
    theta_val = (
        -(S * exp(-q * T) * pdf_d1 * sigma) / (2 * sqrt(T))
        + (q * S * exp(-q * T) * (
            nd1 if option_type == "call" else norm.cdf(-d1)
        ))
        - (r * K * exp(-r * T) * (
            nd2 if option_type == "call" else norm.cdf(-d2)
        ))
    ) / 365.0

    vega_val = S * exp(-q * T) * pdf_d1 * sqrt(T) / 100.0
    rho_val = rho_val / 100.0

    return {
        "price": float(price),
        "delta": float(delta_val),
        "gamma": float(gamma_val),
        "theta": float(theta_val),
        "vega": float(vega_val),
        "rho": float(rho_val),
    }


# ---------------------------------------------------------------------------
# Implied Volatility (Newton-Raphson solver)
# ---------------------------------------------------------------------------


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    q: float = 0.0,
    initial_guess: float = 0.3,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> float:
    """Calculate implied volatility using Newton-Raphson root-finding.

    Finds the volatility sigma such that black_scholes(S, K, T, r, sigma, ...)
    equals the given market_price.

    Args:
        market_price: Observed market price of the option.
        S: Current price of the underlying.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (decimal).
        option_type: 'call' or 'put'.
        q: Dividend yield (decimal).
        initial_guess: Starting volatility guess (decimal).
        max_iterations: Maximum Newton-Raphson iterations.
        tolerance: Convergence tolerance for price difference.

    Returns:
        Implied volatility as a decimal (e.g., 0.25 = 25%).

    Raises:
        ValueError: If the solver does not converge or inputs are invalid.
    """
    # Validate inputs
    if market_price < 0:
        raise ValueError(f"market_price cannot be negative, got {market_price}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if S <= 0 or K <= 0:
        raise ValueError(f"S and K must be positive, got S={S}, K={K}")

    # Check for arbitrage bounds
    intrinsic = max(0.0, (S - K) if option_type == "call" else (K - S))
    if market_price < intrinsic:
        raise ValueError(
            f"market_price ({market_price:.4f}) is below intrinsic value ({intrinsic:.4f})"
        )

    sigma = initial_guess

    for i in range(max_iterations):
        try:
            price = black_scholes(S, K, T, r, sigma, option_type, q)
        except Exception as e:
            raise ValueError(f"BSM pricing failed at iteration {i}: {e}") from e

        diff = price - market_price

        if abs(diff) < tolerance:
            logger.debug(
                "IV converged in %d iterations: sigma=%.6f", i + 1, sigma
            )
            return sigma

        # Calculate vega (derivative of price w.r.t. volatility)
        # dPrice/dSigma = S * sqrt(T) * n(d1) * exp(-q*T)
        d1 = _d1(S, K, T, r, sigma, q)
        vega_raw = S * exp(-q * T) * norm.pdf(d1) * sqrt(T)

        if abs(vega_raw) < 1e-12:
            # Vega is too small; if far from solution, adjust guess
            if diff > 0:
                sigma *= 1.5
            else:
                sigma *= 0.7
            continue

        # Newton-Raphson step: sigma_new = sigma - diff / vega
        sigma_new = sigma - diff / vega_raw

        # Clamp to [0.001, 5.0] to prevent divergence
        sigma_new = max(0.001, min(5.0, sigma_new))

        if abs(sigma_new - sigma) < tolerance:
            logger.debug(
                "IV converged (step) in %d iterations: sigma=%.6f", i + 1, sigma_new
            )
            return sigma_new

        sigma = sigma_new

    # Fallback: try bisection if Newton-Raphson doesn't converge
    logger.warning(
        "Newton-Raphson did not converge in %d iterations, "
        "falling back to bisection for IV of %s option (S=%.2f, K=%.2f, T=%.4f, price=%.4f)",
        max_iterations, option_type, S, K, T, market_price,
    )
    return _implied_volatility_bisection(
        market_price, S, K, T, r, option_type, q
    )


def _implied_volatility_bisection(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    q: float = 0.0,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> float:
    """Fallback IV calculation using bisection method."""
    low, high = 0.001, 5.0

    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        try:
            price = black_scholes(S, K, T, r, mid, option_type, q)
        except Exception:
            return mid

        diff = price - market_price

        if abs(diff) < tolerance:
            return mid

        if diff > 0:
            high = mid
        else:
            low = mid

    return (low + high) / 2.0


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _validate_inputs(S: float, K: float, T: float, sigma: float) -> None:
    """Validate common inputs for pricing and Greeks functions."""
    if S <= 0:
        raise ValueError(f"Underlying price S must be > 0, got {S}")
    if K <= 0:
        raise ValueError(f"Strike price K must be > 0, got {K}")
    if T < 0:
        raise ValueError(f"Time to expiration T must be >= 0, got {T}")
    if sigma < 0:
        raise ValueError(f"Volatility sigma must be >= 0, got {sigma}")


# Public alias
price = black_scholes