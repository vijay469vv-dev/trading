import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional
import httpx
import yfinance as yf
from config import config

BINANCE_BASE = "https://api.binance.com"
BINANCE_TIMEFRAME_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1h": "1h", "4h": "4h", "1d": "1d",
}


class DataFetcher:
    """Unified data fetcher for BTC (Binance REST) and XAU/USD (yfinance)."""

    def __init__(self):
        self._http: Optional[httpx.AsyncClient] = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=15)
        return self._http

    # ── BTC ──────────────────────────────────────────────────────────────────

    async def fetch_btc_ohlcv(
        self,
        timeframe: str = "15m",
        limit: int = 500,
    ) -> pd.DataFrame:
        interval = BINANCE_TIMEFRAME_MAP.get(timeframe, "15m")
        http = await self._get_http()
        resp = await http.get(
            f"{BINANCE_BASE}/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": interval, "limit": limit},
        )
        resp.raise_for_status()
        raw = [
            [row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]
            for row in resp.json()
        ]
        return self._to_df(raw)

    async def fetch_btc_ticker(self) -> dict:
        http = await self._get_http()
        resp = await http.get(
            f"{BINANCE_BASE}/api/v3/ticker/bookTicker",
            params={"symbol": "BTCUSDT"},
        )
        resp.raise_for_status()
        data = resp.json()
        mid = (float(data["bidPrice"]) + float(data["askPrice"])) / 2
        return {
            "symbol": "BTC/USDT",
            "bid": float(data["bidPrice"]),
            "ask": float(data["askPrice"]),
            "last": mid,
            "timestamp": datetime.now(timezone.utc),
        }

    # ── XAU/USD ──────────────────────────────────────────────────────────────

    def fetch_xauusd_ohlcv(
        self,
        timeframe: str = "15m",
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch gold OHLCV using yfinance (GC=F futures as proxy)."""
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m",
            "1h": "1h", "4h": "1h",  # yfinance has no 4h, use 1h
            "1d": "1d",
        }
        period_map = {
            "1m": "1d", "5m": "5d", "15m": "30d",
            "1h": "60d", "4h": "180d", "1d": "2y",
        }
        interval = interval_map.get(timeframe, "15m")
        period = period_map.get(timeframe, "30d")

        ticker = yf.Ticker(config.XAUUSD_YFINANCE)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[["open", "high", "low", "close", "volume"]].tail(limit)
        df.index.name = "timestamp"
        return df

    def fetch_xauusd_ticker(self) -> dict:
        ticker = yf.Ticker(config.XAUUSD_YFINANCE)
        info = ticker.fast_info
        price = info.last_price
        return {
            "symbol": "XAU/USD",
            "bid": round(price - 0.10, 2),
            "ask": round(price + 0.10, 2),
            "last": price,
            "timestamp": datetime.now(timezone.utc),
        }

    # ── Generic ───────────────────────────────────────────────────────────────

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 500) -> pd.DataFrame:
        if symbol == "BTC/USDT":
            return await self.fetch_btc_ohlcv(timeframe, limit)
        elif symbol in ("XAU/USD", "XAUUSD"):
            return self.fetch_xauusd_ohlcv(timeframe, limit)
        raise ValueError(f"Unknown symbol: {symbol}")

    async def fetch_ticker(self, symbol: str) -> dict:
        if symbol == "BTC/USDT":
            return await self.fetch_btc_ticker()
        elif symbol in ("XAU/USD", "XAUUSD"):
            return self.fetch_xauusd_ticker()
        raise ValueError(f"Unknown symbol: {symbol}")

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    @staticmethod
    def _to_df(raw: list) -> pd.DataFrame:
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df
