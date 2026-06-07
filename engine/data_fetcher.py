"""
Market data abstraction layer for the Apex Options Analytics Engine.

Provides a clean interface for fetching options chain data, underlying prices,
and historical data. The base implementation uses Yahoo Finance via yfinance
for bootstrapping; swap in a paid provider (Polygon, Tradier) by implementing
the abstract DataSource interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class OptionContract:
    """Represents a single options contract (call or put).

    Attributes:
        ticker: Underlying ticker symbol.
        strike: Strike price.
        expiration: Expiration date.
        option_type: 'call' or 'put'.
        bid: Current bid price.
        ask: Current ask price.
        last: Last traded price.
        volume: Trading volume.
        open_interest: Open interest.
        implied_volatility: Implied volatility (decimal, e.g. 0.25 = 25%).
        delta: Option delta.
        gamma: Option gamma.
        theta: Option theta.
        vega: Option vega.
        rho: Option rho.
    """

    ticker: str
    strike: float
    expiration: datetime
    option_type: str  # 'call' or 'put'
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_volatility: float = 0.0
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

    @property
    def mid_price(self) -> float:
        """Mid-market price (average of bid and ask)."""
        return (self.bid + self.ask) / 2.0

    @property
    def days_to_expiration(self) -> int:
        """Calendar days until expiration."""
        delta = self.expiration - datetime.now()
        return max(0, delta.days)

    @property
    def trading_days_to_expiration(self) -> float:
        """Approximate trading days until expiration (~252/year)."""
        return self.days_to_expiration * (252.0 / 365.0)


@dataclass
class UnderlyingData:
    """Current state of the underlying asset.

    Attributes:
        ticker: Ticker symbol.
        price: Current price.
        timestamp: Timestamp of the data.
        historical_prices: DataFrame of historical adjusted close prices.
        dividend_yield: Dividend yield (decimal).
        risk_free_rate: Risk-free interest rate (decimal).
    """

    ticker: str
    price: float
    timestamp: datetime = field(default_factory=datetime.now)
    historical_prices: Optional[pd.DataFrame] = None
    dividend_yield: float = 0.0
    risk_free_rate: float = 0.05  # approximate; could fetch from treasury yield


@dataclass
class OptionsChain:
    """Full options chain for a ticker.

    Attributes:
        ticker: Underlying ticker.
        underlying_price: Current price of the underlying.
        timestamp: Timestamp of the chain data.
        expirations: List of available expiration dates.
        calls: List of OptionContract for call options.
        puts: List of OptionContract for put options.
    """

    ticker: str
    underlying_price: float
    timestamp: datetime = field(default_factory=datetime.now)
    expirations: list[datetime] = field(default_factory=list)
    calls: list[OptionContract] = field(default_factory=list)
    puts: list[OptionContract] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract data source
# ---------------------------------------------------------------------------


class DataSource(ABC):
    """Abstract base class for market data providers.

    Implement this interface to swap between Yahoo Finance, Polygon, Tradier,
    or any other data source.
    """

    @abstractmethod
    def fetch_underlying(self, ticker: str) -> UnderlyingData:
        """Fetch current underlying price and historical data."""
        ...

    @abstractmethod
    def fetch_options_chain(
        self,
        ticker: str,
        expiration: Optional[datetime] = None,
        max_strikes: int = 20,
    ) -> OptionsChain:
        """Fetch the options chain for a ticker, optionally filtered by expiration.

        Args:
            ticker: Ticker symbol.
            expiration: If provided, only fetch contracts for this expiration.
            max_strikes: Maximum number of strikes to return per side.

        Returns:
            An OptionsChain containing calls, puts, and metadata.
        """
        ...

    @abstractmethod
    def fetch_historical_prices(
        self, ticker: str, period: str = "1y"
    ) -> pd.DataFrame:
        """Fetch historical price data (adjusted close).

        Args:
            ticker: Ticker symbol.
            period: Time period string (e.g. '1y', '6mo', '3mo').

        Returns:
            DataFrame with columns: Date, Close (adjusted close).
        """
        ...


# ---------------------------------------------------------------------------
# Yahoo Finance implementation
# ---------------------------------------------------------------------------


class YahooFinanceDataSource(DataSource):
    """Data source implementation using Yahoo Finance via yfinance.

    Free and suitable for bootstrapping. Will be upgraded to a paid
    provider (Polygon, Tradier) in production.
    """

    def __init__(self, timeout: int = 10) -> None:
        self._timeout = timeout
        self._ticker_cache: dict[str, "yfinance.Ticker"] = {}

    def _get_ticker(self, ticker: str) -> "yfinance.Ticker":
        """Get or create a yfinance Ticker object, with caching."""
        import yfinance as yf

        ticker_upper = ticker.upper()
        if ticker_upper not in self._ticker_cache:
            self._ticker_cache[ticker_upper] = yf.Ticker(ticker_upper)
        return self._ticker_cache[ticker_upper]

    def fetch_underlying(self, ticker: str) -> UnderlyingData:
        """Fetch current underlying price and 1 year of historical prices."""
        yf_ticker = self._get_ticker(ticker)

        # Get current price
        try:
            info = yf_ticker.info if hasattr(yf_ticker, "info") else {}
        except Exception:
            info = {}

        # Fast price lookup
        hist = self.fetch_historical_prices(ticker, period="1y")
        if hist.empty:
            raise ValueError(f"No historical data available for {ticker}")

        current_price = float(hist["Close"].iloc[-1])

        # Try to get dividend yield from info
        div_yield = float(info.get("dividendYield", 0.0) or 0.0)

        # Use a proxy risk-free rate (could fetch 13-week treasury yield)
        risk_free_rate = 0.05

        return UnderlyingData(
            ticker=ticker.upper(),
            price=current_price,
            timestamp=datetime.now(),
            historical_prices=hist,
            dividend_yield=div_yield,
            risk_free_rate=risk_free_rate,
        )

    def fetch_options_chain(
        self,
        ticker: str,
        expiration: Optional[datetime] = None,
        max_strikes: int = 20,
    ) -> OptionsChain:
        """Fetch the options chain."""
        yf_ticker = self._get_ticker(ticker)

        # Get current underlying price
        underlying = self.fetch_underlying(ticker)

        # Get available expirations
        try:
            expirations = yf_ticker.options
        except Exception as e:
            raise ValueError(f"Failed to fetch options for {ticker}: {e}") from e

        if not expirations:
            raise ValueError(f"No options available for {ticker}")

        expiration_dates = [
            datetime.strptime(exp_str, "%Y-%m-%d") for exp_str in expirations
        ]

        if expiration is not None:
            # Find the closest expiration to the requested date
            closest = min(
                expiration_dates, key=lambda d: abs((d - expiration).total_seconds())
            )
            target_expirations = [closest]
        else:
            # Return all expirations but only fetch the nearest ones
            target_expirations = expiration_dates[:10]

        all_calls: list[OptionContract] = []
        all_puts: list[OptionContract] = []

        for exp_date in target_expirations:
            exp_str = exp_date.strftime("%Y-%m-%d")
            try:
                opt_chain = yf_ticker.option_chain(exp_str)
            except Exception as e:
                logger.warning(
                    "Failed to fetch chain for %s %s: %s", ticker, exp_str, e
                )
                continue

            # Process calls
            for _, row in opt_chain.calls.iterrows():
                volume = row.get("volume", 0)
                if pd.isna(volume): volume = 0
                oi = row.get("openInterest", 0)
                if pd.isna(oi): oi = 0
                
                all_calls.append(
                    OptionContract(
                        ticker=ticker.upper(),
                        strike=float(row["strike"]),
                        expiration=exp_date,
                        option_type="call",
                        bid=float(row.get("bid", 0.0) or 0.0),
                        ask=float(row.get("ask", 0.0) or 0.0),
                        last=float(row.get("lastPrice", 0.0) or 0.0),
                        volume=int(volume),
                        open_interest=int(oi),
                        implied_volatility=float(
                            row.get("impliedVolatility", 0.0) or 0.0
                        ),
                    )
                )

            # Process puts
            for _, row in opt_chain.puts.iterrows():
                volume = row.get("volume", 0)
                if pd.isna(volume): volume = 0
                oi = row.get("openInterest", 0)
                if pd.isna(oi): oi = 0
                
                all_puts.append(
                    OptionContract(
                        ticker=ticker.upper(),
                        strike=float(row["strike"]),
                        expiration=exp_date,
                        option_type="put",
                        bid=float(row.get("bid", 0.0) or 0.0),
                        ask=float(row.get("ask", 0.0) or 0.0),
                        last=float(row.get("lastPrice", 0.0) or 0.0),
                        volume=int(volume),
                        open_interest=int(oi),
                        implied_volatility=float(
                            row.get("impliedVolatility", 0.0) or 0.0
                        ),
                    )
                )

        # Filter to ATM + surrounding strikes if max_strikes is set
        all_calls = self._filter_strikes(all_calls, underlying.price, max_strikes)
        all_puts = self._filter_strikes(all_puts, underlying.price, max_strikes)

        return OptionsChain(
            ticker=ticker.upper(),
            underlying_price=underlying.price,
            timestamp=datetime.now(),
            expirations=expiration_dates,
            calls=all_calls,
            puts=all_puts,
        )

    def fetch_historical_prices(
        self, ticker: str, period: str = "1y"
    ) -> pd.DataFrame:
        """Fetch historical adjusted close prices."""
        yf_ticker = self._get_ticker(ticker)
        try:
            hist = yf_ticker.history(period=period)
        except Exception as e:
            raise ValueError(
                f"Failed to fetch historical prices for {ticker}: {e}"
            ) from e

        if hist.empty:
            # Try daily data with a longer period
            hist = yf_ticker.history(period="max")

        if hist.empty:
            return pd.DataFrame()

        # Return DataFrame with just adjusted close
        df = hist[["Close"]].copy()
        df.index.name = "Date"
        return df

    @staticmethod
    def _filter_strikes(
        contracts: list[OptionContract],
        underlying_price: float,
        max_strikes: int,
    ) -> list[OptionContract]:
        """Filter to the N strikes closest to ATM, balanced OTM/ITM."""
        if len(contracts) <= max_strikes:
            return contracts

        sorted_contracts = sorted(
            contracts, key=lambda c: abs(c.strike - underlying_price)
        )
        return sorted_contracts[:max_strikes]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_data_source(source_type: str = "yahoo", **kwargs) -> DataSource:
    """Factory function to create a data source.

    Args:
        source_type: One of 'yahoo' (default, free), 'polygon' (future), 'tradier' (future).
        **kwargs: Additional arguments passed to the data source constructor.

    Returns:
        An initialized DataSource instance.
    """
    source_map: dict[str, type[DataSource]] = {
        "yahoo": YahooFinanceDataSource,
    }
    if source_type not in source_map:
        raise ValueError(
            f"Unknown data source '{source_type}'. "
            f"Available: {list(source_map.keys())}"
        )
    return source_map[source_type](**kwargs)