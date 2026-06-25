"""Tradier Data Source for Apex Options Analytics.
Provides real-time market data via the Tradier API.
Supports both sandbox (free, delayed) and production (real-time) environments.

Requires env var: TRADIER_API_KEY (or pass via constructor)
Sandbox: https://sandbox.tradier.com/v1/
Production: https://api.tradier.com/v1/
"""
from __future__ import annotations
import logging
import os
from datetime import datetime, timedelta
from typing import Optional
import requests
import pandas as pd
import numpy as np

from .data_fetcher import DataSource, OptionContract, UnderlyingData, OptionsChain

logger = logging.getLogger(__name__)

TRADIER_SANDBOX = "https://sandbox.tradier.com/v1/"
TRADIER_PROD = "https://api.tradier.com/v1/"


class TradierDataSource(DataSource):
    """Data source using Tradier API for options and market data."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        sandbox: bool = True,
        timeout: int = 10,
    ):
        api_key = api_key or os.environ.get("TRADIER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Tradier API key required. Set TRADIER_API_KEY env var or pass api_key."
            )
        self.api_key = api_key
        self.base_url = TRADIER_SANDBOX if sandbox else TRADIER_PROD
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        })
        logger.info(
            "TradierDataSource initialized (%s mode)",
            "sandbox" if sandbox else "production",
        )

    def fetch_underlying(self, ticker: str) -> UnderlyingData:
        """Fetch current price for a ticker."""
        url = f"{self.base_url}markets/quotes"
        params = {"symbols": ticker, "greeks": "false"}
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise ValueError(f"Tradier quote failed for {ticker}: {e}")

        quotes = data.get("quotes", {}).get("quote", [])
        if isinstance(quotes, dict):
            quotes = [quotes]

        if not quotes:
            raise ValueError(f"No quote data for {ticker}")

        q = quotes[0]
        price = q.get("last", q.get("close", 0.0)) or 0.0
        symbol = q.get("symbol", ticker)
        name = q.get("description", symbol)
        change = q.get("change", 0.0) or 0.0
        change_pct = q.get("change_percentage", 0.0) or 0.0

        return UnderlyingData(
            ticker=symbol,
            price=float(price),
            name=name,
            change=float(change),
            change_pct=float(change_pct),
        )

    def fetch_options_chain(
        self,
        ticker: str,
        expiration: Optional[datetime] = None,
        max_strikes: int = 100,
    ) -> OptionsChain:
        """Fetch options chain from Tradier."""
        underlying = self.fetch_underlying(ticker)

        # Get available expirations
        url = f"{self.base_url}markets/options/expirations"
        params = {"symbol": ticker, "includeAllRoots": "true", "strikes": "true"}
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            exp_data = resp.json()
        except Exception as e:
            raise ValueError(f"Tradier expirations failed for {ticker}: {e}")

        expirations = exp_data.get("expirations", {}).get("expiration", [])
        if isinstance(expirations, dict):
            expirations = [expirations]
        if not expirations:
            raise ValueError(f"No expirations found for {ticker}")

        expiration_dates = []
        all_strikes = set()
        for exp in expirations:
            date_str = exp.get("date", "")
            if date_str:
                expiration_dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
            strikes = exp.get("strikes", {}).get("strike", [])
            if isinstance(strikes, (int, float)):
                strikes = [float(strikes)]
            elif isinstance(strikes, list):
                strikes = [float(s) for s in strikes]
            for s in strikes:
                all_strikes.add(s)

        expiration_dates.sort()
        all_strikes = sorted(all_strikes)

        # Filter to target expiration (or use nearest within 7-180 DTE)
        now = datetime.now()
        if expiration:
            target_exp = min(expiration_dates, key=lambda d: abs((d - expiration).total_seconds()))
        else:
            # Pick expiration closest to 45 DTE
            target_exp = min(expiration_dates, key=lambda d: abs((d - now).days - 45))

        # Fetch option chain for target expiration
        chain_url = f"{self.base_url}markets/options/chains"
        chain_params = {
            "symbol": ticker,
            "expiration": target_exp.strftime("%Y-%m-%d"),
            "greeks": "true",
        }
        try:
            resp = self._session.get(chain_url, params=chain_params, timeout=self.timeout)
            resp.raise_for_status()
            chain_data = resp.json()
        except Exception as e:
            raise ValueError(f"Tradier chain failed for {ticker}: {e}")

        options = chain_data.get("options", {}).get("option", [])
        if isinstance(options, dict):
            options = [options]
        if not options:
            raise ValueError(f"No options data for {ticker} at {target_exp.date()}")

        all_calls: list[OptionContract] = []
        all_puts: list[OptionContract] = []

        for opt in options:
            opt_type = opt.get("option_type", "").lower()
            strike = float(opt.get("strike", 0))
            bid = float(opt.get("bid", 0) or 0)
            ask = float(opt.get("ask", 0) or 0)
            last = float(opt.get("last", 0) or 0)
            volume = int(opt.get("volume", 0) or 0)
            oi = int(opt.get("open_interest", 0) or 0)
            iv = float(opt.get("greeks", {}).get("mid_iv", 0) or 0)

            contract = OptionContract(
                ticker=ticker.upper(),
                strike=strike,
                expiration=target_exp,
                option_type=opt_type,
                bid=bid,
                ask=ask,
                last=last,
                volume=volume,
                open_interest=oi,
                implied_volatility=iv,
            )

            if opt_type == "call":
                all_calls.append(contract)
            elif opt_type == "put":
                all_puts.append(contract)

        # Sort by strike
        all_calls.sort(key=lambda c: c.strike)
        all_puts.sort(key=lambda c: c.strike)

        # Filter to ATM + surrounding strikes
        if len(all_calls) > max_strikes:
            all_calls = self._filter_strikes(all_calls, underlying.price, max_strikes)
        if len(all_puts) > max_strikes:
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
        """Fetch historical daily prices from Tradier."""
        # Map period to Tradier interval
        interval_map = {
            "1mo": "daily",
            "3mo": "daily",
            "6mo": "daily",
            "1y": "daily",
            "2y": "daily",
            "5y": "weekly",
        }
        interval = interval_map.get(period, "daily")

        # Calculate start date based on period
        period_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
        days = period_days.get(period, 365)
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        url = f"{self.base_url}markets/history"
        params = {"symbol": ticker, "interval": interval, "start": start}
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise ValueError(f"Tradier history failed for {ticker}: {e}")

        history = data.get("history", {}).get("day", [])
        if isinstance(history, dict):
            history = [history]
        if not history:
            return pd.DataFrame()

        records = []
        for day in history:
            records.append({
                "Date": day.get("date", ""),
                "Close": float(day.get("close", 0) or 0),
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)

        return df

    @staticmethod
    def _filter_strikes(contracts, underlying_price, max_strikes):
        """Filter to N strikes closest to ATM."""
        if len(contracts) <= max_strikes:
            return contracts
        sorted_c = sorted(contracts, key=lambda c: abs(c.strike - underlying_price))
        return sorted_c[:max_strikes]


def create_tradier_source(api_key: Optional[str] = None, sandbox: bool = True, **kwargs) -> TradierDataSource:
    """Factory function for Tradier data source."""
    return TradierDataSource(api_key=api_key, sandbox=sandbox, **kwargs)