"""
Real-time price streaming
- BTC/USDT : Binance WebSocket (tick-by-tick, ~100ms updates)
- XAU/USD  : httpx polling every 3 seconds (yfinance has no WebSocket)
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import httpx
import websockets
import yfinance as yf

# Shared live price store — updated by stream tasks, read by everything else
live_prices: dict[str, dict] = {
    "BTC/USDT": {"bid": 0, "ask": 0, "last": 0, "change_pct": 0, "ts": 0},
    "XAU/USD":  {"bid": 0, "ask": 0, "last": 0, "change_pct": 0, "ts": 0},
}

_callbacks: list[Callable] = []   # called on every price update


def on_price_update(fn: Callable):
    """Register a callback that fires whenever any price updates."""
    _callbacks.append(fn)


def _notify(symbol: str):
    for fn in _callbacks:
        try:
            fn(symbol, live_prices[symbol])
        except Exception:
            pass


# ── BTC — Binance WebSocket ───────────────────────────────────────────────────

async def stream_btc():
    """Connect to Binance miniTicker WebSocket. Auto-reconnects on disconnect."""
    url = "wss://stream.binance.com:9443/ws/btcusdt@miniTicker"
    open_24h: Optional[float] = None

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                async for raw in ws:
                    data = json.loads(raw)
                    last = float(data["c"])
                    if open_24h is None:
                        open_24h = float(data["o"])
                    change_pct = ((last - open_24h) / open_24h * 100) if open_24h else 0
                    spread = last * 0.0001  # ~0.01% synthetic spread
                    live_prices["BTC/USDT"] = {
                        "bid": round(last - spread, 2),
                        "ask": round(last + spread, 2),
                        "last": last,
                        "change_pct": round(change_pct, 2),
                        "ts": time.time(),
                    }
                    _notify("BTC/USDT")
        except Exception as e:
            # Reconnect after 5s on any error
            await asyncio.sleep(5)


# ── XAU/USD — fast HTTP poll ──────────────────────────────────────────────────

async def stream_xauusd(interval_seconds: float = 3.0):
    """Poll Binance for XAU/USD equivalent via Gold spot price."""
    open_price: Optional[float] = None

    while True:
        try:
            price = await _fetch_gold_price()
            if price and price > 100:
                if open_price is None:
                    open_price = price
                change_pct = ((price - open_price) / open_price * 100) if open_price else 0
                live_prices["XAU/USD"] = {
                    "bid": round(price - 0.15, 2),
                    "ask": round(price + 0.15, 2),
                    "last": price,
                    "change_pct": round(change_pct, 2),
                    "ts": time.time(),
                }
                _notify("XAU/USD")
        except Exception:
            pass
        await asyncio.sleep(interval_seconds)


async def _fetch_gold_price() -> Optional[float]:
    """Fetch gold spot price from public APIs with fallback."""
    # Try Binance PAXGUSDT (tokenized gold) as a fast proxy
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": "PAXGUSDT"},
            )
            if r.status_code == 200:
                return float(r.json()["price"])
    except Exception:
        pass

    # Fallback: yfinance (slower but reliable)
    try:
        ticker = yf.Ticker("GC=F")
        price = ticker.fast_info.last_price
        if price and price > 100:
            return float(price)
    except Exception:
        pass

    return None


# ── Start all streams ─────────────────────────────────────────────────────────

async def start_streams():
    """Launch all price streams as background tasks."""
    asyncio.create_task(stream_btc(), name="btc_stream")
    asyncio.create_task(stream_xauusd(), name="xau_stream")
