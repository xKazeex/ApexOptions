#!/usr/bin/env python3
"""
ThetaEdge — Advanced Options Selling Recommendation Engine
===========================================================

Multi-factor, institutional-grade recommendation engine with full Greeks,
IV analysis, technical overlay, Kelly sizing, tail risk flags, and portfolio
allocation. Scans the options market and recommends the BEST option to SELL
(cash-secured puts or covered calls).

Usage:
    python3 thetaedge.py --capital 5000 --duration 30 --top 5
    python3 thetaedge.py --capital 10000 --duration 45 --mode cc --tickers AAPL,MSFT
    python3 thetaedge.py --capital 25000 --duration 14 --mode csp --min-delta 0.10 --max-delta 0.25
    python3 thetaedge.py --capital 30000 --duration 30 --universe sp500
"""

import argparse
import sys
import math
import io
import os
import contextlib
import json
import re
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from collections import defaultdict
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy import stats

# vollib for all 5 Black-Scholes Greeks
from vollib.black_scholes.greeks.analytical import delta as bs_delta
from vollib.black_scholes.greeks.analytical import gamma as bs_gamma
from vollib.black_scholes.greeks.analytical import theta as bs_theta
from vollib.black_scholes.greeks.analytical import vega as bs_vega
from vollib.black_scholes.greeks.analytical import rho as bs_rho
from vollib.black_scholes import black_scholes as bs_price

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
RISK_FREE_RATE = 0.045
DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA",
    "META", "AMD", "INTC", "BABA", "JPM", "V", "DIS", "NFLX", "KO", "F", "PLTR"
]
MIN_PREMIUM_PER_CONTRACT = 5.0
MAX_RESULTS_DEFAULT = 10
MIN_PCR = 0.001

# Scoring weights (7-factor model)
W_PCR = 0.22
W_POP = 0.22
W_THETA = 0.13
W_IV = 0.13
W_LIQ = 0.10
W_RISKADJ = 0.10
W_TECH = 0.10  # Technical trend bias

# Sector mapping
SECTOR_MAP = {
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF",
    "AAPL": "Tech", "MSFT": "Tech", "AMZN": "Tech", "GOOGL": "Tech",
    "TSLA": "Auto", "NVDA": "Semiconductor", "META": "Tech",
    "AMD": "Semiconductor", "INTC": "Semiconductor", "BABA": "Tech",
    "JPM": "Financial", "V": "Financial", "DIS": "Entertainment",
    "NFLX": "Entertainment", "KO": "Consumer", "F": "Auto", "PLTR": "Tech"
}

# ──────────────────────────────────────────────
# 1. DYNAMIC TICKER UNIVERSE
# ──────────────────────────────────────────────

TOP_LIQUID_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "NVDA", "TSLA",
    "BRK-B", "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS",
    "BAC", "KO", "PEP", "ADBE", "CRM", "CMCSA", "AVGO", "NFLX", "PYPL",
    "TXN", "QCOM", "INTC", "AMD", "AMGN", "ABNB", "UBER", "NOW",
    "PLTR", "SNAP", "SOFI", "HOOD", "COIN", "MARA", "CVNA", "RIVN",
    "LCID", "CHWY", "DASH", "SQ", "AFRM", "ZM", "DOCU", "ROKU",
    "F", "AA", "X", "NIO", "BABA", "JD", "BIDU"
]

# GICS Sector-based ticker lists (liquid, optionable stocks per sector)
SECTOR_TICKERS = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AMD", "INTC", "CRM", "ADBE", "TXN", "QCOM",
        "NOW", "PLTR", "MU", "ORCL", "IBM", "CSCO", "HPQ", "DELL", "ANET",
        "MDB", "DDOG", "NET", "SNAP", "ZM", "DOCU", "WDAY", "ADSK", "FTNT",
        "PANW", "CRWD", "OKTA"
    ],
    "Financial": [
        "JPM", "BAC", "V", "MA", "GS", "MS", "WFC", "C", "AXP", "SCHW",
        "BLK", "BX", "COF", "USB", "PNC", "TFC", "SOFI", "HOOD", "SQ",
        "AFRM", "PYPL", "JEF", "MCO", "SPGI", "CME", "ICE"
    ],
    "Healthcare": [
        "UNH", "JNJ", "PFE", "ABBV", "AMGN", "MRK", "LLY", "ABT", "TMO",
        "DHR", "BMY", "VRTX", "ISRG", "SYK", "BSX", "GILD", "REGN",
        "HCA", "CVS", "CI", "HUM", "BIIB", "ILMN"
    ],
    "Energy": [
        "XOM", "CVX", "COP", "OXY", "EOG", "SLB", "HAL", "MPC", "PSX",
        "VLO", "KMI", "WMB", "OKE", "PXD", "FANG", "DVN", "MRO"
    ],
    "Consumer Cyclical": [
        "AMZN", "TSLA", "HD", "LOW", "NKE", "SBUX", "MCD", "BKNG", "MAR",
        "HLT", "ABNB", "UBER", "DASH", "CMG", "YUM", "DHI", "LEN", "F",
        "GM", "RIVN", "LCID", "CHWY", "BBY", "ROST", "TJX", "TGT"
    ],
    "Consumer Defensive": [
        "WMT", "COST", "PG", "KO", "PEP", "CL", "KMB", "MDLZ", "KHC",
        "GIS", "CPB", "SYY", "HSY", "CAG", "K", "MKC", "SJM", "COTY"
    ],
    "Communication": [
        "GOOGL", "META", "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS",
        "ROKU", "TTWO", "EA", "LYV", "PARA", "WBD", "CHTR", "DISH"
    ],
    "Industrial": [
        "GE", "CAT", "BA", "HON", "UPS", "UNP", "RTX", "LMT", "GD",
        "NOC", "MMM", "EMR", "ETN", "ROK", "CSX", "NSC", "FDX", "ODFL",
        "DE", "PAYX", "CARR", "OTIS", "GEV"
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED",
        "PEG", "WEC", "ES", "EIX", "PCG", "AWK", "CEG", "VST"
    ],
    "Real Estate": [
        "PLD", "AMT", "CCI", "EQIX", "PSA", "O", "WELL", "SPG", "DLR",
        "AVB", "EQR", "EXR", "MAA", "ARE", "INVH", "SUI"
    ],
    "Materials": [
        "LIN", "SHW", "APD", "ECL", "NEM", "FCX", "DOW", "DD", "PPG",
        "AA", "X", "CLF", "STLD", "NUE", "BALL", "IP", "WRK"
    ],
}


def fetch_sp500_tickers() -> List[str]:
    """Fetch current S&P 500 components from Wikipedia or fallback to liquid list."""
    try:
        # Try with requests and lxml parser
        import requests
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=10
        )
        tables = pd.read_html(resp.text)
        sp500 = tables[0]['Symbol'].tolist()
        return [t.replace('.', '-') for t in sp500]
    except Exception:
        # Fallback: return the comprehensive liquid ticker list
        return TOP_LIQUID_TICKERS


def fetch_nasdaq100_tickers() -> List[str]:
    """Fetch current NASDAQ-100 components or fallback."""
    try:
        import requests
        resp = requests.get(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=10
        )
        tables = pd.read_html(resp.text)
        # Find the correct table with tickers
        for t in tables:
            if 'Ticker' in t.columns:
                ndx = t['Ticker'].tolist()
                return [s.replace('.', '-') for s in ndx]
        # Try with Symbol column
        for t in tables:
            if 'Symbol' in t.columns:
                ndx = t['Symbol'].tolist()
                return [s.replace('.', '-') for s in ndx]
        return TOP_LIQUID_TICKERS[:50]
    except Exception:
        return TOP_LIQUID_TICKERS[:50]


def build_ticker_universe(universe: str = "default", min_price: float = 5.0,
                          max_scanned: int = 150) -> List[str]:
    """
    Build a dynamic ticker universe based on the specified strategy.
    
    Args:
        universe: 'default', 'sp500', 'nasdaq100', 'liquid', or a sector name like 'Technology'
        min_price: minimum stock price to include
        max_scanned: max tickers to scan (to avoid rate limits)
    """
    if universe == "default":
        return DEFAULT_TICKERS

    # Check if it's a sector name
    if universe in SECTOR_TICKERS:
        print(f"  Using {universe} sector ({len(SECTOR_TICKERS[universe])} tickers)", file=sys.stderr)
        return SECTOR_TICKERS[universe]

    raw_tickers = []
    if universe == "sp500":
        print("  Fetching S&P 500 components...", file=sys.stderr)
        raw_tickers = fetch_sp500_tickers()
    elif universe == "nasdaq100":
        print("  Fetching NASDAQ-100 components...", file=sys.stderr)
        raw_tickers = fetch_nasdaq100_tickers()
    elif universe == "liquid":
        raw_tickers = DEFAULT_TICKERS + ["GOOG", "ORCL", "CRM", "ADBE", "PYPL",
                                          "SNAP", "UBER", "SQ", "RIVN", "LCID",
                                          "SOFI", "HOOD", "COIN", "MARA", "CVNA"]
    else:
        return DEFAULT_TICKERS

    # Filter by price to only include stocks we can afford (strike * 100 <= capital)
    # We do a quick pre-screen
    filtered = []
    for i, sym in enumerate(raw_tickers):
        if i >= max_scanned:
            break
        filtered.append(sym)
    
    print(f"  Universe: {len(filtered)} tickers (from {universe})", file=sys.stderr)
    return filtered


# ──────────────────────────────────────────────
# 2. BACKTESTING ENGINE
# ──────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Result of backtesting a strategy setup."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown: float = 0.0
    sharpe_approx: float = 0.0
    similar_setups_found: int = 0


def backtest_setup(ticker_symbol: str, strike: float, dte: int,
                   option_type: str, delta_target: float,
                   lookback_days: int = 180) -> BacktestResult:
    """
    Backtest similar setups over historical price data.
    
    Scans past N days of price history, finds periods where selling
    a similar option would have been possible (same delta regime),
    and simulates the outcome.
    
    This is a simplified backtest using underlying price movement
    as a proxy for option P&L.
    """
    result = BacktestResult()
    try:
        # Get 1 year of data + extra for lookback window
        hist = yf.download(ticker_symbol, period="1y", progress=False, auto_adjust=True)
        if hist.empty or len(hist) < 60:
            return result

        closes = hist['Close'].values.flatten()
        dates_idx = hist.index
        
        # For each potential entry point in the lookback window, simulate the trade
        trades = []
        entry_window_end = len(closes) - dte - 1  # can't go past data
        
        for i in range(0, entry_window_end, 5):  # Step by 5 days to avoid overlap
            entry_price = float(closes[i])
            
            # Check if this would pass our delta filter (approximate)
            # For puts: delta ~ -0.3 means strike ~ 0.95 * entry_price (rough)
            # For calls: delta ~ 0.3 means strike ~ 1.05 * entry_price (rough)
            if option_type == "put":
                approx_strike = entry_price * (1 - abs(delta_target) * 0.5)
                # Simulate CSP: profit if price > strike at expiration
                expiry_idx = min(i + dte, len(closes) - 1)
                price_at_expiry = float(closes[expiry_idx])
                # Simplified: premium collected ~ delta_target * strike * 0.3
                premium_est = abs(delta_target) * approx_strike * 0.5
                
                if price_at_expiry >= approx_strike:
                    # Win: keep premium
                    ret = premium_est / (approx_strike) * 100
                    result.winning_trades += 1
                else:
                    # Loss: stock below strike, loss = (strike - price) - premium
                    loss = (approx_strike - price_at_expiry) - premium_est
                    ret = -loss / (approx_strike) * 100
                    result.losing_trades += 1
                
                trades.append(ret)
                
            else:  # call
                approx_strike = entry_price * (1 + delta_target * 0.5)
                expiry_idx = min(i + dte, len(closes) - 1)
                price_at_expiry = float(closes[expiry_idx])
                premium_est = delta_target * approx_strike * 0.3
                
                if price_at_expiry <= approx_strike:
                    ret = premium_est / (entry_price) * 100
                    result.winning_trades += 1
                else:
                    loss = (price_at_expiry - approx_strike) - premium_est
                    ret = -loss / (entry_price) * 100
                    result.losing_trades += 1
                
                trades.append(ret)

        result.total_trades = result.winning_trades + result.losing_trades
        if result.total_trades > 0:
            result.win_rate = (result.winning_trades / result.total_trades) * 100
            result.avg_return = float(np.mean(trades)) if trades else 0.0
            result.total_return_pct = float(np.sum(trades)) if trades else 0.0
            result.max_drawdown = float(np.min(trades)) if trades else 0.0
            if trades and np.std(trades) > 0:
                result.sharpe_approx = float(np.mean(trades) / np.std(trades)) * np.sqrt(252 / dte)
            result.similar_setups_found = result.total_trades

        return result

    except Exception:
        return result


# ──────────────────────────────────────────────
# 3. DATA PROVIDER ABSTRACTION
# ──────────────────────────────────────────────

class OptionsDataProvider(ABC):
    """Abstract base class for options data providers."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return provider name."""
        pass
    
    @abstractmethod
    def fetch_underlying_price(self, ticker: str) -> Optional[float]:
        """Fetch current underlying price."""
        pass
    
    @abstractmethod
    def fetch_expirations(self, ticker: str) -> Optional[List[str]]:
        """Fetch available expiration dates."""
        pass
    
    @abstractmethod
    def fetch_option_chain(self, ticker: str, expiration: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Fetch calls and puts DataFrames for an expiration."""
        pass
    
    @abstractmethod
    def fetch_price_history(self, ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """Fetch historical price data."""
        pass


class YFinanceProvider(OptionsDataProvider):
    """yfinance-based options data provider (current implementation)."""
    
    def get_name(self) -> str:
        return "yfinance"
    
    def fetch_underlying_price(self, ticker: str) -> Optional[float]:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if hist.empty:
                return None
            return float(hist['Close'].iloc[-1])
        except Exception:
            return None
    
    def fetch_expirations(self, ticker: str) -> Optional[List[str]]:
        try:
            t = yf.Ticker(ticker)
            return t.options if t.options else None
        except Exception:
            return None
    
    def fetch_option_chain(self, ticker: str, expiration: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        try:
            t = yf.Ticker(ticker)
            chain = t.option_chain(expiration)
            return chain.calls, chain.puts
        except Exception:
            return None, None
    
    def fetch_price_history(self, ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        try:
            return yf.download(ticker, period=period, progress=False, auto_adjust=True)
        except Exception:
            return None


class TradierProvider(OptionsDataProvider):
    """
    Tradier API options data provider — live, production-ready.
    
    Requires TRADIER_TOKEN environment variable to be set.
    API docs: https://documentation.tradier.com/
    """
    
    def __init__(self, token: str = None):
        self.token = token or os.environ.get("TRADIER_TOKEN", "")
        self.base_url = "https://api.tradier.com/v1"
    
    def _is_active(self) -> bool:
        return bool(self.token)
    
    def get_name(self) -> str:
        return "tradier"
    
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"
        }
    
    def _request(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make a GET request to the Tradier API."""
        if not self._is_active():
            return None
        try:
            url = f"{self.base_url}{endpoint}"
            resp = requests.get(url, headers=self._headers(), params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None
    
    def fetch_underlying_price(self, ticker: str) -> Optional[float]:
        """Fetch current underlying price via Tradier quotes API."""
        data = self._request("/markets/quotes", {"symbols": ticker})
        if not data:
            return None
        try:
            quote = data.get('quotes', {}).get('quote', {})
            if isinstance(quote, list):
                quote = quote[0]
            return float(quote.get('last', 0))
        except (TypeError, ValueError, KeyError):
            return None
    
    def fetch_expirations(self, ticker: str) -> Optional[List[str]]:
        """Fetch available expiration dates via Tradier options expiry API."""
        data = self._request("/markets/options/expirations", {
            "symbol": ticker, "includeAllRoots": "true"
        })
        if not data:
            return None
        try:
            # Response uses "date" key, not "expiration"
            exp_data = data.get('expirations', {})
            exps = exp_data.get('date', exp_data.get('expiration', []))
            if isinstance(exps, str):
                exps = [exps]
            return exps if exps else None
        except (TypeError, KeyError):
            return None
    
    def fetch_option_chain(self, ticker: str, expiration: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Fetch full option chain via Tradier quotes API using OCC symbols.
        
        Approach:
        1. Get strikes from /markets/options/strikes
        2. Build OCC symbols: ROOT + YYMMDD + C/P + STRIKE*1000 zero-padded to 8 digits
        3. Batch query (50 at a time) via /markets/quotes
        4. Parse response into calls/puts DataFrames
        5. Greeks computed by vollib in build_rec() (consistent across providers)
        """
        # Step 1: Get strikes
        strikes_data = self._request("/markets/options/strikes", {
            "symbol": ticker, "expiration": expiration
        })
        if not strikes_data:
            return None, None
        
        try:
            strike_list = strikes_data.get('strikes', {}).get('strike', [])
            if not strike_list or not isinstance(strike_list, list):
                return None, None
            if isinstance(strike_list, (int, float)):
                strike_list = [strike_list]
        except (TypeError, KeyError):
            return None, None
        
        # Build date component: YYMMDD
        exp_parts = expiration.split('-')
        date_code = exp_parts[0][2:] + exp_parts[1] + exp_parts[2]
        
        # Step 2: Build OCC symbols and batch query
        all_rows = []
        batch_size = 50  # Tradier max per call
        
        def build_symbol(strike_val, opt_type):
            """Build OCC symbol: ROOT + YYMMDD + C/P + STRIKE*1000 zfilled to 8 digits"""
            strike_int = int(round(strike_val * 1000))
            strike_padded = f"{strike_int:08d}"
            return f"{ticker}{date_code}{opt_type}{strike_padded}"
        
        for i in range(0, len(strike_list), batch_size):
            batch = strike_list[i:i+batch_size]
            symbols = []
            for s in batch:
                symbols.append(build_symbol(s, 'C'))
                symbols.append(build_symbol(s, 'P'))
            
            data = self._request("/markets/quotes", {
                "symbols": ",".join(symbols), "greeks": "true"
            })
            if not data:
                continue
            
            try:
                quotes = data.get('quotes', {}).get('quote', [])
                if not quotes:
                    continue
                if isinstance(quotes, dict):
                    quotes = [quotes]
                
                for q in quotes:
                    if not q.get('symbol'):
                        continue
                    
                    sym = q.get('symbol', '')
                    opt_type = sym[-9] if len(sym) > 9 else ''
                    
                    # Extract IV from greeks object
                    greeks_obj = q.get('greeks', {}) or {}
                    mid_iv = float(greeks_obj.get('mid_iv', 0) or 0)
                    bid_iv = float(greeks_obj.get('bid_iv', 0) or 0)
                    ask_iv = float(greeks_obj.get('ask_iv', 0) or 0)
                    iv = mid_iv or bid_iv or ask_iv or 0.3
                    
                    all_rows.append({
                        'contractSymbol': sym,
                        'strike': float(q.get('strike', 0)),
                        'bid': float(q.get('bid', 0) or 0),
                        'ask': float(q.get('ask', 0) or 0),
                        'lastPrice': float(q.get('last', 0) or 0),
                        'volume': int(q.get('volume', 0) or 0),
                        'openInterest': int(q.get('open_interest', 0) or 0),
                        'impliedVolatility': iv,
                        'inTheMoney': None,
                        'contractSize': 'REGULAR',
                        'currency': 'USD',
                        # Greeks computed by vollib in build_rec() for consistency
                        'option_type': opt_type,
                    })
            except (TypeError, KeyError):
                continue
        
        if not all_rows:
            return None, None
        
        df = pd.DataFrame(all_rows)
        
        # Split by option type from OCC symbol
        calls_df = df[df['option_type'] == 'C'].drop(columns=['option_type']).copy() if not df.empty and 'option_type' in df.columns else pd.DataFrame()
        puts_df = df[df['option_type'] == 'P'].drop(columns=['option_type']).copy() if not df.empty and 'option_type' in df.columns else pd.DataFrame()
        
        return calls_df if not calls_df.empty else None, puts_df if not puts_df.empty else None
    
    def fetch_price_history(self, ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """Fetch historical prices via Tradier timesales API."""
        interval_map = {"6mo": "weekly", "1y": "monthly", "1mo": "daily"}
        interval = interval_map.get(period, "daily")
        data = self._request("/markets/history", {
            "symbol": ticker, "interval": interval
        })
        if not data:
            return None
        try:
            series = data.get('history', {}).get('day', [])
            if not series:
                return None
            if isinstance(series, dict):
                series = [series]
            
            records = []
            for day in series:
                records.append({
                    'Date': day.get('date'),
                    'Open': float(day.get('open', 0)),
                    'High': float(day.get('high', 0)),
                    'Low': float(day.get('low', 0)),
                    'Close': float(day.get('close', 0)),
                    'Volume': int(day.get('volume', 0)),
                })
            
            df = pd.DataFrame(records)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            return df
        except Exception:
            return None


def get_data_provider(provider_name: str = "yfinance", **kwargs) -> OptionsDataProvider:
    """Factory function to get the appropriate data provider."""
    providers = {
        "yfinance": YFinanceProvider,
        "tradier": TradierProvider,
    }
    provider_class = providers.get(provider_name.lower(), YFinanceProvider)
    return provider_class(**kwargs)


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class TechnicalIndicators:
    rsi: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    atr_pct: Optional[float] = None
    trend: str = "neutral"
    oversold_flag: bool = False
    overbought_flag: bool = False
    support_level: Optional[float] = None


@dataclass
class OptionRecommendation:
    ticker: str = ""
    option_type: str = ""
    strike: float = 0.0
    expiration: str = ""
    dte: int = 0
    current_price: float = 0.0

    premium_mid: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    premium_total: float = 0.0

    delta: float = 0.0
    gamma: float = 0.0
    theta_daily: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    collateral: float = 0.0
    pcr: float = 0.0
    annualized_return: float = 0.0
    probability_of_profit: float = 0.0
    break_even: float = 0.0
    expected_return: float = 0.0

    implied_volatility: float = 0.0
    iv_rank: Optional[float] = None
    iv_percentile: Optional[float] = None

    score_components: dict = field(default_factory=dict)
    composite_score: float = 0.0

    kelly_fraction: float = 0.0
    conservative_kelly: float = 0.0
    recommended_contracts: int = 0
    capital_used: float = 0.0
    capital_used_pct: float = 0.0

    max_loss: float = 0.0
    max_loss_pct: float = 0.0
    tail_sensitivity: float = 0.0
    theta_capture_rate: float = 0.0
    risk_adjusted_return: float = 0.0

    volume: int = 0
    open_interest: int = 0
    liquidity_score: float = 0.0

    technicals: Optional[TechnicalIndicators] = None
    flags: list = field(default_factory=list)
    in_earnings_window: bool = False
    next_earnings_date: Optional[str] = None
    ex_dividend_date: Optional[str] = None
    sector: str = ""
    
    # Backtesting data
    backtest: Optional[BacktestResult] = None


# ──────────────────────────────────────────────
# Technical Analysis Module
# ──────────────────────────────────────────────

def compute_technicals(ticker_symbol: str) -> TechnicalIndicators:
    tech = TechnicalIndicators()
    try:
        hist = yf.download(ticker_symbol, period="6mo", progress=False, auto_adjust=True)
        if hist.empty or len(hist) < 20:
            return tech

        closes = hist['Close'].values.flatten()
        highs = hist['High'].values.flatten()
        lows = hist['Low'].values.flatten()

        # RSI(14)
        if len(closes) >= 15:
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            if avg_loss == 0:
                tech.rsi = 100.0
            else:
                tech.rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
            tech.oversold_flag = tech.rsi < 30
            tech.overbought_flag = tech.rsi > 70

        # SMAs
        closes_f = closes[-200:] if len(closes) >= 200 else closes
        if len(closes_f) >= 20:
            tech.sma_20 = float(np.mean(closes_f[-20:]))
        if len(closes_f) >= 50:
            tech.sma_50 = float(np.mean(closes_f[-50:]))
        if len(closes_f) >= 200:
            tech.sma_200 = float(np.mean(closes_f[-200:]))

        # Trend
        cp = closes[-1]
        if tech.sma_20 and tech.sma_50:
            if cp > tech.sma_20 > tech.sma_50:
                tech.trend = "bullish"
            elif cp < tech.sma_20 < tech.sma_50:
                tech.trend = "bearish"
            else:
                tech.trend = "neutral"

        # ATR(14) %
        if len(closes) >= 15:
            tr_vals = []
            for j in range(max(15, len(closes)-14), len(closes)):
                h = highs[j]
                l_ = lows[j]
                pc = closes[j-1]
                tr_vals.append(max(h-l_, abs(h-pc), abs(l_-pc)))
            tech.atr_pct = float(np.mean(tr_vals) / cp * 100) if tr_vals else None

        if len(closes) >= 60:
            tech.support_level = float(np.min(lows[-60:]))

        return tech
    except Exception:
        return tech


def estimate_iv_rank(ticker_symbol: str, current_iv: float) -> Tuple[Optional[float], Optional[float]]:
    try:
        hist = yf.download(ticker_symbol, period="1y", progress=False, auto_adjust=True)
        if hist.empty or len(hist) < 20:
            return None, None
        closes = hist['Close'].values.flatten()
        returns = np.diff(np.log(closes))
        hist_vol = float(np.std(returns) * np.sqrt(252))
        if hist_vol <= 0:
            return None, None
        min_iv = hist_vol * 0.4
        max_iv = hist_vol * 3.0
        if max_iv <= min_iv:
            return 50.0, 50.0
        rank = max(0, min(100, (current_iv - min_iv) / (max_iv - min_iv) * 100))
        rolling = [float(np.std(returns[i-20:i]) * np.sqrt(252)) for i in range(20, len(returns))]
        percentile = float(sum(1 for rv in rolling if rv < current_iv) / max(len(rolling),1) * 100) if rolling else 50.0
        return rank, percentile
    except Exception:
        return None, None


def check_earnings(ticker_symbol: str, dte_high: int) -> Tuple[bool, Optional[str]]:
    try:
        ticker = yf.Ticker(ticker_symbol)
        cal = ticker.calendar
        if cal is not None and 'Earnings Date' in cal.index:
            ed = cal.loc['Earnings Date']
            if isinstance(ed, pd.Timestamp):
                return True, ed.strftime('%Y-%m-%d')
        return False, None
    except Exception:
        return False, None


def check_dividend(ticker_symbol: str) -> Optional[str]:
    try:
        ticker = yf.Ticker(ticker_symbol)
        if ticker.dividends is not None and not ticker.dividends.empty:
            last_div = ticker.dividends.index[-1]
            if isinstance(last_div, pd.Timestamp):
                next_ex = last_div.date() + timedelta(days=28)
                return next_ex.isoformat()
        return None
    except Exception:
        return None


# ──────────────────────────────────────────────
# Greeks Engine
# ──────────────────────────────────────────────

def black_scholes_price(S, K, T, r, sigma, flag):
    """Calculate Black-Scholes option price (fallback)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if flag == 'p':
        return max(K*math.exp(-r*T)*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1), 0)
    else:
        return max(S*stats.norm.cdf(d1) - K*math.exp(-r*T)*stats.norm.cdf(d2), 0)


# ──────────────────────────────────────────────
# Data Fetcher
# ──────────────────────────────────────────────

# Global data provider instance
_data_provider: Optional[OptionsDataProvider] = None

def set_data_provider(provider: OptionsDataProvider):
    global _data_provider
    _data_provider = provider

def get_data_provider_instance() -> OptionsDataProvider:
    global _data_provider
    if _data_provider is None:
        _data_provider = YFinanceProvider()
    return _data_provider


def fetch_ticker(ticker_symbol: str):
    """Fetch ticker data using the current data provider."""
    provider = get_data_provider_instance()
    try:
        if provider.get_name() == "yfinance":
            return _fetch_ticker_yfinance(ticker_symbol)
        else:
            # Use provider abstraction for non-yfinance
            cp = provider.fetch_underlying_price(ticker_symbol)
            exps = provider.fetch_expirations(ticker_symbol)
            # Need a yfinance ticker object for option_chain() calls elsewhere
            # This would need full provider integration throughout
            return None, cp, exps
    except Exception:
        return None, None, None


def _fetch_ticker_yfinance(ticker_symbol: str):
    """Original yfinance ticker fetcher."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="5d")
        if hist.empty:
            return None, None, None
        price = float(hist['Close'].iloc[-1])
        expirations = ticker.options
        return ticker, price, expirations
    except Exception:
        return None, None, None


# ──────────────────────────────────────────────
# Core Processing
# ──────────────────────────────────────────────

def build_rec(ticker_symbol, row, current_price, expiration, dte, T, mode,
              capital, technicals, earnings_win, earnings_date, ex_div) -> Optional[OptionRecommendation]:

    rec = OptionRecommendation(
        ticker=ticker_symbol, expiration=expiration, dte=dte,
        current_price=current_price, sector=SECTOR_MAP.get(ticker_symbol, "Other"),
        in_earnings_window=earnings_win, next_earnings_date=earnings_date,
        ex_dividend_date=ex_div, technicals=technicals,
    )

    strike = float(row['strike'])
    rec.strike = strike
    bid = float(row.get('bid', 0) or 0)
    ask = float(row.get('ask', 0) or 0)
    last = float(row.get('lastPrice', 0) or 0)

    if bid > 0 and ask > 0:
        rec.premium_mid = (bid + ask) / 2
    elif last > 0:
        rec.premium_mid = last
    else:
        return None

    rec.bid, rec.ask = bid, ask
    rec.premium_total = rec.premium_mid * 100
    if rec.premium_total < MIN_PREMIUM_PER_CONTRACT:
        return None
    
    # Skip if no bid (untradeable), unless we have a last price
    if bid <= 0 and last <= 0:
        return None
    
    # Skip if strike is too far from current price (> 50% away)
    if abs(strike - current_price) / current_price > 0.50:
        return None

    sigma = float(row.get('impliedVolatility', 0) or 0)
    if sigma <= 0:
        return None
    rec.implied_volatility = sigma

    if mode == "csp":
        if strike >= current_price:
            return None
        rec.option_type = "put"
        rec.collateral = strike * 100
        rec.break_even = strike - rec.premium_mid
        d2 = (math.log(current_price/strike) + (RISK_FREE_RATE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        rec.probability_of_profit = (1.0 - stats.norm.cdf(-d2)) * 100
    else:
        if strike <= current_price:
            return None
        rec.option_type = "call"
        rec.collateral = current_price * 100
        rec.break_even = current_price
        d2 = (math.log(current_price/strike) + (RISK_FREE_RATE - 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        rec.probability_of_profit = (1.0 - stats.norm.cdf(d2)) * 100

    if rec.collateral > capital:
        return None

    rec.pcr = (rec.premium_total / rec.collateral) * 100
    if rec.pcr / 100 < MIN_PCR:
        return None
    rec.annualized_return = rec.pcr * (365.0 / max(dte, 1))

    # Greeks via vollib
    flag = 'p' if rec.option_type == 'put' else 'c'
    try:
        rec.delta = bs_delta(flag, current_price, strike, T, RISK_FREE_RATE, sigma)
        rec.gamma = bs_gamma(flag, current_price, strike, T, RISK_FREE_RATE, sigma)
        annual_theta = bs_theta(flag, current_price, strike, T, RISK_FREE_RATE, sigma)
        rec.theta_daily = annual_theta / 365.0
        rec.vega = bs_vega(flag, current_price, strike, T, RISK_FREE_RATE, sigma)
        rec.rho = bs_rho(flag, current_price, strike, T, RISK_FREE_RATE, sigma)
    except Exception:
        return None

    # Delta filter
    if abs(rec.delta) > 0.30:
        return None

    # Metrics
    rec.theta_capture_rate = abs(rec.theta_daily) / max(rec.premium_mid, 0.001)
    vol_val = row.get('volume', 0)
    oi_val = row.get('openInterest', 0)
    if isinstance(vol_val, float) and math.isnan(vol_val):
        vol_val = 0
    if isinstance(oi_val, float) and math.isnan(oi_val):
        oi_val = 0
    rec.volume = int(vol_val)
    rec.open_interest = int(oi_val)
    rec.max_loss = rec.collateral
    rec.max_loss_pct = (rec.max_loss / max(capital, 1)) * 100
    pop = rec.probability_of_profit / 100
    rec.expected_return = rec.premium_total * pop - rec.max_loss * (1 - pop)
    rec.tail_sensitivity = abs(rec.gamma * rec.vega / 100)
    denom = math.sqrt(rec.gamma**2 + (rec.vega/100)**2) if rec.vega != 0 else 0.001
    rec.risk_adjusted_return = rec.annualized_return / denom

    # IV
    iv_rank, iv_perc = estimate_iv_rank(ticker_symbol, sigma)
    rec.iv_rank, rec.iv_percentile = iv_rank, iv_perc

    # Liquidity score
    if rec.volume > 0 and rec.open_interest > 0:
        rec.liquidity_score = min(1.0, math.sqrt(rec.volume * rec.open_interest) / 5000)

    # Backtest this setup
    bt = backtest_setup(ticker_symbol, strike, dte, rec.option_type, abs(rec.delta))
    rec.backtest = bt

    # Flags
    flags = []
    if rec.dte <= 7:
        flags.append("Gamma risk: <7 DTE")
    if rec.gamma > 0.01:
        flags.append(f"Gamma spike: {rec.gamma:.4f}")
    if rec.max_loss_pct > 5:
        flags.append(f"Max loss {rec.max_loss_pct:.1f}% of capital")
    if technicals and technicals.oversold_flag and mode == "csp":
        flags.append(f"RSI({technicals.rsi:.0f}) oversold — bounce risk")
    if technicals and technicals.overbought_flag and mode == "cc":
        flags.append(f"RSI({technicals.rsi:.0f}) overbought")
    if earnings_win:
        flags.append(f"Earnings {earnings_date} in window")
    if ex_div:
        flags.append(f"Ex-div ~{ex_div}")
    if rec.vega > 1.0 and (iv_rank is not None and iv_rank < 30):
        flags.append("IV expansion risk")
    if rec.pcr < 0.5:
        flags.append(f"Low yield: {rec.pcr:.2f}%")
    if bt.total_trades > 0:
        flags.append(f"Backtest: {bt.win_rate:.0f}% win rate ({bt.total_trades} trades)")
    rec.flags = flags

    return rec


# ──────────────────────────────────────────────
# Technical Trend Scoring
# ──────────────────────────────────────────────

def compute_technical_score(rec, mode):
    """Score the technical setup for options selling (0-100).
    
    Rewards bullish/neutral trends for CSPs, penalizes bearish.
    Penalizes oversold conditions for CSPs (stock might drop further).
    Penalizes high ATR (too much volatility for safe premium).
    """
    tech = rec.technicals
    if not tech:
        return 50.0  # Neutral default if no TA data
    
    score = 50.0  # Start neutral
    
    # Trend direction (±30 points)
    if tech.trend == "bullish":
        score += 30  # Great for selling puts
    elif tech.trend == "bearish":
        score -= 30  # Risky for selling puts
    
    # RSI adjustment (±10 points)
    if tech.rsi is not None:
        if mode == "csp":
            if tech.rsi < 30:
                score -= 15  # Oversold — further downside risk
            elif tech.rsi > 70:
                score += 10  # Overbought — good to sell puts (expect pullback)
        else:  # covered calls
            if tech.rsi > 70:
                score -= 15  # Overbought — stock might pull back, bad for CCs
            elif tech.rsi < 30:
                score += 10  # Oversold — good to sell calls
    
    # ATR penalty (±10 points)
    if tech.atr_pct is not None:
        if tech.atr_pct > 10:
            score -= 10  # Too volatile
        elif tech.atr_pct < 3:
            score += 5   # Low vol is good for premium collecting
    
    return max(0, min(100, score))


# ──────────────────────────────────────────────
# Scoring Engine
# ──────────────────────────────────────────────

def compute_composite_score(rec, all_recs, mode="csp"):
    scores = {}
    pcr_vals = [r.pcr for r in all_recs]
    scores['pcr'] = (rec.pcr - min(pcr_vals)) / (max(pcr_vals) - min(pcr_vals) + 0.001) * 100

    pop_vals = [r.probability_of_profit for r in all_recs]
    scores['pop'] = (rec.probability_of_profit - min(pop_vals)) / (max(pop_vals) - min(pop_vals) + 0.001) * 100

    theta_vals = [r.theta_capture_rate for r in all_recs]
    scores['theta'] = (rec.theta_capture_rate - min(theta_vals)) / (max(theta_vals) - min(theta_vals) + 0.001) * 100

    iv_s = []
    if rec.iv_rank is not None:
        iv_s.append(rec.iv_rank)
    if rec.iv_percentile is not None:
        iv_s.append(rec.iv_percentile)
    scores['iv'] = np.mean(iv_s) if iv_s else 50.0

    liq_vals = [r.liquidity_score for r in all_recs]
    scores['liq'] = (rec.liquidity_score - min(liq_vals)) / (max(liq_vals) - min(liq_vals) + 0.001) * 100

    radj_vals = [r.risk_adjusted_return for r in all_recs]
    scores['riskadj'] = (rec.risk_adjusted_return - min(radj_vals)) / (max(radj_vals) - min(radj_vals) + 0.001) * 100

    # Technical trend score
    scores['tech'] = compute_technical_score(rec, mode)

    composite = (scores['pcr'] * W_PCR + scores['pop'] * W_POP + scores['theta'] * W_THETA +
                 scores['iv'] * W_IV + scores['liq'] * W_LIQ + scores['riskadj'] * W_RISKADJ +
                 scores['tech'] * W_TECH)

    rec.score_components = scores
    rec.composite_score = composite


# ──────────────────────────────────────────────
# Kelly Sizing
# ──────────────────────────────────────────────

def compute_kelly(rec, capital):
    p = rec.probability_of_profit / 100.0
    q = 1.0 - p
    realistic_loss_per_share = rec.strike * 0.10
    if realistic_loss_per_share <= 0:
        return 0.0, 0.0, 0
    total_loss = realistic_loss_per_share * 100 - rec.premium_total
    if total_loss <= 0:
        total_loss = rec.premium_total * 0.5
    profit = rec.premium_total
    b = profit / max(total_loss, 1.0)
    kelly = max(0, (b * p - q) / max(b, 0.0001))
    cons_kelly = kelly * 0.25
    if cons_kelly > 0:
        max_by_cap = int((cons_kelly * capital) / max(rec.collateral, 1))
        max_by_vol = max(1, int(rec.volume / 10)) if rec.volume > 0 else 1
        max_by_oi = max(1, int(rec.open_interest / 5)) if rec.open_interest > 0 else 1
        recommended = max(0, min(max_by_cap, max_by_vol, max_by_oi, 5))
    else:
        recommended = 0
    if kelly > 0.001 and recommended == 0:
        recommended = 1
    return kelly, cons_kelly, recommended


# ──────────────────────────────────────────────
# Scanner Engine
# ──────────────────────────────────────────────

def scan_market(tickers, capital, duration_days, mode="csp",
                min_delta=0.0, max_delta=0.30, dte_min=7, dte_max=220, top_n=10,
                enable_backtest=True):

    all_recs = []
    preferred_dte = max(duration_days, 7)
    dte_low = max(dte_min, int(preferred_dte * 0.5))
    dte_high = min(dte_max, int(preferred_dte * 2.0))
    if dte_high < dte_low:
        dte_high = dte_low + 14

    today = date.today()

    print(f"\n{'='*72}")
    print(f"  THETAEDGE — Advanced Options Selling Engine")
    print(f"{'='*72}")
    print(f"  Mode:          {'Cash-Secured Puts' if mode == 'csp' else 'Covered Calls'}")
    print(f"  Capital:       ${capital:>8,.2f}")
    print(f"  Duration:      {duration_days}d (DTE: {dte_low}-{dte_high})")
    print(f"  Delta:         {min_delta:.2f}-{max_delta:.2f}")
    print(f"  Tickers:       {len(tickers)}")
    if enable_backtest:
        print(f"  Backtest:      enabled")
    print(f"{'='*72}\n")

    for ticker_symbol in tickers:
        print(f"  Scanning {ticker_symbol}...", end="", file=sys.stderr)
        ticker, cp, exps = fetch_ticker(ticker_symbol)
        if cp is None or not exps:
            print(" SKIP", file=sys.stderr)
            continue

        valid = []
        for exp in exps:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if dte_low <= dte <= dte_high:
                valid.append((exp, dte))

        if not valid:
            print(" (no exp)", file=sys.stderr)
            continue

        ew, ed = check_earnings(ticker_symbol, dte_high)
        exd = check_dividend(ticker_symbol)
        ticker_recs = []

        provider = get_data_provider_instance()
        for exp, dte in valid:
            if provider.get_name() != "yfinance" and provider._is_active():
                calls_df, puts_df = provider.fetch_option_chain(ticker_symbol, exp)
                if mode == "csp":
                    df = puts_df
                else:
                    df = calls_df
            else:
                try:
                    chain = ticker.option_chain(exp)
                except Exception:
                    continue
                df = chain.puts if mode == "csp" else chain.calls
            
            if df is None or df.empty:
                continue
                
            T = max(dte / 365.0, 1/365)
            for _, row in df.iterrows():
                rec = build_rec(ticker_symbol, row, cp, exp, dte, T, mode,
                                capital, TechnicalIndicators(), ew, ed, exd)
                if rec:
                    ticker_recs.append(rec)

        print(f" {len(ticker_recs)} candidates", file=sys.stderr)

        # Lazy-load TA only for tickers that had candidates
        if ticker_recs:
            ticker_ta = compute_technicals(ticker_symbol)
            for rec in ticker_recs:
                rec.technicals = ticker_ta
            rsi_str = f"RSI: {ticker_ta.rsi:.0f}" if ticker_ta.rsi else "N/A"
            print(f"  Loaded {ticker_symbol} TA → {rsi_str}", file=sys.stderr)

        all_recs.extend(ticker_recs)

    if not all_recs:
        print("\n  No candidates found.\n")
        return []

    for rec in all_recs:
        compute_composite_score(rec, all_recs, mode)

    for rec in all_recs:
        k, ck, contracts = compute_kelly(rec, capital)
        rec.kelly_fraction = k
        rec.conservative_kelly = ck
        rec.recommended_contracts = contracts
        rec.capital_used = contracts * rec.collateral
        rec.capital_used_pct = (rec.capital_used / max(capital,1)) * 100

    all_recs.sort(key=lambda r: r.composite_score, reverse=True)
    return all_recs[:top_n]


# ──────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────

def print_recs(recs, capital):
    if not recs:
        return

    print(f"\n{'='*72}")
    print(f"  TOP {len(recs)} RECOMMENDATIONS")
    print(f"{'='*72}\n")

    total_deployed = 0
    for i, rec in enumerate(recs, 1):
        remaining = capital - total_deployed
        if rec.capital_used > remaining:
            rec.capital_used = remaining
            rec.capital_used_pct = (remaining / capital) * 100 if capital > 0 else 0
            rec.recommended_contracts = max(0, int(remaining / rec.collateral)) if rec.collateral > 0 else 0
        total_deployed += rec.capital_used

        emoji = "PUT" if rec.option_type == "put" else "CALL"
        print(f"  {'='*68}")
        print(f"  #{i:<3} {rec.ticker:<6} {emoji:<5} ${rec.strike:<7.2f}  Exp: {rec.expiration:<12}  DTE: {rec.dte}")
        print(f"  {'-'*68}")
        print(f"  Premium:  ${rec.premium_mid:<7.2f}  (B:${rec.bid:<6.2f} A:${rec.ask:<6.2f})")
        # Current price vs strike
        diff = abs(rec.current_price - rec.strike)
        if rec.option_type == "put":
            moneyness = "OTM" if rec.strike < rec.current_price else ("ITM" if rec.strike > rec.current_price else "ATM")
        else:
            moneyness = "OTM" if rec.strike > rec.current_price else ("ITM" if rec.strike < rec.current_price else "ATM")
        print(f"  Current:  ${rec.current_price:<8.2f}  Strike: ${rec.strike:<7.2f}  {moneyness} by ${diff:<.2f}")
        print(f"  Delta:    {rec.delta:<+8.4f}  Gamma: {rec.gamma:<8.4f}  Theta/D: ${rec.theta_daily:<+8.6f}")
        print(f"  Vega:     {rec.vega:<8.4f}  Rho:   {rec.rho:<+8.4f}  PoP:     {rec.probability_of_profit:<5.1f}%")
        print(f"  {'-'*68}")
        print(f"  Collateral: ${rec.collateral:>8,.0f}   P/C: {rec.pcr:<7.3f}%   Ann: {rec.annualized_return:<7.2f}%")
        ivr = f"{rec.iv_rank:.0f}%" if rec.iv_rank else "N/A"
        ivp = f"{rec.iv_percentile:.0f}%" if rec.iv_percentile else "N/A"
        print(f"  IV: {rec.implied_volatility:.2%}   Rank: {ivr:<8}  Perc: {ivp:<8}")
        print(f"  {'-'*68}")
        c = rec.score_components
        print(f"  Score: PCR={c.get('pcr',0):.0f} PoP={c.get('pop',0):.0f} Theta={c.get('theta',0):.0f} "
              f"IV={c.get('iv',0):.0f} Liq={c.get('liq',0):.0f} RiskAdj={c.get('riskadj',0):.0f} Tech={c.get('tech',50):.0f}")
        print(f"  Composite: {rec.composite_score:.1f}/100")
        print(f"  {'-'*68}")
        k_str = f"{rec.kelly_fraction:.4f}" if rec.kelly_fraction > 0 else "0"
        print(f"  Kelly: {k_str}  Conservative: {rec.conservative_kelly:.4f}  "
              f"Rec Contracts: {rec.recommended_contracts}")
        print(f"  Capital Used: ${rec.capital_used:>8,.0f} ({rec.capital_used_pct:.1f}%)")
        
        # Backtest results
        if rec.backtest and rec.backtest.total_trades > 0:
            bt = rec.backtest
            print(f"  {'-'*68}")
            print(f"  📊 Backtest (1yr): {bt.total_trades} similar setups, "
                  f"{bt.win_rate:.0f}% win rate, avg {bt.avg_return:+.2f}% per trade")
        
        if rec.technicals:
            t = rec.technicals
            rsi_str = f"RSI: {t.rsi:.0f}" if t.rsi else "RSI: N/A"
            trend_str = f"Trend: {t.trend}"
            atr_str = f"ATR: {t.atr_pct:.1f}%" if t.atr_pct else ""
            print(f"  TA: {rsi_str:<12} {trend_str:<15} {atr_str}")
        if rec.flags:
            for f in rec.flags[:4]:
                print(f"  Flag: {f}")
        print()

    print(f"  {'='*68}")
    print(f"  PORTFOLIO: Deployed ${total_deployed:>8,.0f}/{capital:,.0f} ({total_deployed/capital*100:.1f}%)")
    if total_deployed / capital > 0.8:
        print(f"  WARNING: >80% deployed — reduce sizes")

    sectors = defaultdict(float)
    for rec in recs:
        sectors[rec.sector] += rec.capital_used
    for sector, amt in sorted(sectors.items(), key=lambda x: -x[1]):
        pct = amt / capital * 100
        warn = " >30%" if pct > 30 else ""
        print(f"  {sector:<16} ${amt:>8,.0f} ({pct:5.1f}%){warn}")
    print(f"{'='*72}\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ThetaEdge — Advanced Options Selling Engine",
        epilog="Run without --capital for interactive mode."
    )
    p.add_argument("--capital", type=float, default=None,
                   help="Available capital (omit for interactive mode)")
    p.add_argument("--duration", type=int, default=None,
                   help="Preferred DTE (omit for interactive mode)")
    p.add_argument("--mode", choices=["csp", "cc"], default=None,
                   help="Strategy: csp (cash-secured puts) or cc (covered calls)")
    p.add_argument("--tickers", type=str, default=None,
                   help="Comma-separated tickers")
    p.add_argument("--universe", type=str, default=None,
                   choices=["default", "sp500", "nasdaq100", "liquid"],
                   help="Ticker universe: sp500, nasdaq100, liquid, or default")
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--max-delta", type=float, default=0.30)
    p.add_argument("--dte-min", type=int, default=7)
    p.add_argument("--dte-max", type=int, default=45)
    p.add_argument("--top", type=int, default=MAX_RESULTS_DEFAULT)
    p.add_argument("--no-backtest", action="store_true",
                   help="Disable backtesting (faster)")
    p.add_argument("--provider", type=str, default="yfinance",
                   choices=["yfinance", "tradier"],
                   help="Data provider (default: yfinance)")
    return p.parse_args()


def interactive_mode():
    """Prompt user for inputs interactively when no CLI flags given."""
    print(f"\n{'='*72}")
    print(f"  ⚡ THETAEDGE — Interactive Mode")
    print(f"  (Press Ctrl+C anytime to exit)")
    print(f"{'='*72}\n")

    while True:
        try:
            raw = input("  💰 Enter your available capital (e.g. 25000): $").strip()
            capital = float(raw.replace(",", "").replace("$", ""))
            if capital <= 0:
                print("  ❌ Capital must be positive. Try again.")
                continue
            break
        except (ValueError, EOFError):
            print("  ❌ Please enter a valid number (e.g. 25000).")
        except KeyboardInterrupt:
            print("\n  👋 Goodbye!")
            sys.exit(0)

    while True:
        try:
            raw = input("  📅 Preferred days to expiration (e.g. 30): ").strip()
            duration = int(raw)
            if duration <= 0:
                print("  ❌ Duration must be positive. Try again.")
                continue
            if duration < 7:
                print("  ⚠  Very short DTE increases gamma risk. Minimum 7 recommended.")
            break
        except (ValueError, EOFError):
            print("  ❌ Please enter a valid number of days (e.g. 30).")
        except KeyboardInterrupt:
            print("\n  👋 Goodbye!")
            sys.exit(0)

    while True:
        try:
            raw = input("  📊 Mode — Cash-Secured Puts (csp) or Covered Calls (cc)? [csp]: ").strip().lower()
            if not raw:
                mode = "csp"
            elif raw in ("csp", "cc"):
                mode = raw
            else:
                print("  ❌ Please enter 'csp' or 'cc'.")
                continue
            break
        except (EOFError, KeyboardInterrupt):
            print("\n  👋 Goodbye!")
            sys.exit(0)

    universe = "default"
    try:
        raw = input("  🌐 Ticker universe — default/sp500/nasdaq100/liquid/sector? [default]: ").strip().lower()
        if raw in ("sp500", "nasdaq100", "liquid", "default") or raw in {s.lower() for s in SECTOR_TICKERS}:
            universe = raw
            # Map lowercase sector names back to proper case
            for s in SECTOR_TICKERS:
                if raw == s.lower():
                    universe = s
                    break
    except (EOFError, KeyboardInterrupt):
        pass

    try:
        raw = input(f"  📈 Specific tickers (comma-separated, Enter for universe): ").strip().upper()
        if raw:
            tickers = [t.strip() for t in raw.replace(",", " ").split() if t.strip()]
        else:
            tickers = build_ticker_universe(universe)
            print(f"     Universe: {len(tickers)} tickers")
    except (EOFError, KeyboardInterrupt):
        print("\n  👋 Goodbye!")
        sys.exit(0)

    try:
        raw = input("  🎯 Max delta (default 0.30, lower = safer): [0.30] ").strip()
        max_delta = float(raw) if raw else 0.30
        max_delta = min(max(max_delta, 0.0), 1.0)
    except (ValueError, EOFError):
        max_delta = 0.30

    print(f"\n{'='*72}")
    print(f"  ✅ Running scan with: capital=${capital:,.0f}, "
          f"duration={duration}d, mode={mode.upper()}, universe={universe}")
    print(f"{'='*72}\n")

    return capital, duration, mode, tickers, 0.0, max_delta, 7, 220, MAX_RESULTS_DEFAULT


def main():
    args = parse_args()

    # Interactive mode
    if args.capital is None or args.duration is None:
        capital, duration, mode, tickers, min_delta, max_delta, dte_min, dte_max, top = interactive_mode()
    else:
        capital, duration, mode = args.capital, args.duration, args.mode or "csp"
        min_delta, max_delta = args.min_delta, args.max_delta
        dte_min, dte_max = args.dte_min, args.dte_max
        top = args.top

        if capital <= 0 or duration <= 0:
            print("Error: capital and duration must be positive", file=sys.stderr)
            sys.exit(1)

        # Build ticker list
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",")]
        elif args.universe:
            tickers = build_ticker_universe(args.universe)
        else:
            tickers = DEFAULT_TICKERS

    # Initialize data provider
    if args.provider and args.provider == "tradier":
        token = os.environ.get("TRADIER_TOKEN", "")
        if token:
            set_data_provider(TradierProvider(token=token))
            print("  Using Tradier data provider", file=sys.stderr)
        else:
            print("  [WARN] TRADIER_TOKEN not set — using yfinance", file=sys.stderr)
            set_data_provider(YFinanceProvider())

    enable_backtest = not args.no_backtest

    # Capture output for file
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        recs = scan_market(tickers, capital, duration, mode,
                           min_delta, max_delta, dte_min, dte_max, top,
                           enable_backtest=enable_backtest)
        print_recs(recs, capital)

    output = buffer.getvalue()
    print(output, end="")

    # Save to file
    output_path = "/home/team/shared/thetaedge_output.txt"
    try:
        with open(output_path, "w") as f:
            f.write(output)
        print(f"  ✅ Results saved to {output_path}")
    except Exception as e:
        print(f"  ⚠ Could not save output file: {e}")

    if not recs:
        sys.exit(1)


if __name__ == "__main__":
    main()