"""
Market Structure Analysis
Detects: Break of Structure (BOS), Change of Character (ChoCH), trend direction.
"""

from dataclasses import dataclass
from typing import Literal, Optional
import pandas as pd
from .liquidity import SwingPoint


@dataclass
class StructureEvent:
    timestamp: pd.Timestamp
    price: float
    kind: Literal["BOS_bullish", "BOS_bearish", "ChoCH_bullish", "ChoCH_bearish"]


class MarketStructure:
    def __init__(self, lookback: int = 10):
        self.lookback = lookback

    def analyze(
        self,
        df: pd.DataFrame,
        swing_highs: list[SwingPoint],
        swing_lows: list[SwingPoint],
    ) -> dict:
        trend = self._determine_trend(swing_highs, swing_lows)
        events = self._detect_structure_events(df, swing_highs, swing_lows)
        return {
            "trend": trend,
            "events": events,
            "last_event": events[-1] if events else None,
        }

    def _determine_trend(
        self,
        swing_highs: list[SwingPoint],
        swing_lows: list[SwingPoint],
    ) -> Literal["bullish", "bearish", "ranging"]:
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "ranging"

        recent_highs = swing_highs[-3:]
        recent_lows = swing_lows[-3:]

        hh = all(recent_highs[i].price < recent_highs[i + 1].price for i in range(len(recent_highs) - 1))
        hl = all(recent_lows[i].price < recent_lows[i + 1].price for i in range(len(recent_lows) - 1))
        lh = all(recent_highs[i].price > recent_highs[i + 1].price for i in range(len(recent_highs) - 1))
        ll = all(recent_lows[i].price > recent_lows[i + 1].price for i in range(len(recent_lows) - 1))

        if hh and hl:
            return "bullish"
        if lh and ll:
            return "bearish"
        return "ranging"

    def _detect_structure_events(
        self,
        df: pd.DataFrame,
        swing_highs: list[SwingPoint],
        swing_lows: list[SwingPoint],
    ) -> list[StructureEvent]:
        events = []
        last_close = df["close"].iloc[-1]
        ts = df.index[-1]

        if len(swing_highs) >= 2:
            prev_high = swing_highs[-2].price
            last_high = swing_highs[-1].price

            # BOS bullish: close breaks above previous swing high (continuation up)
            if last_close > prev_high and last_high > prev_high:
                events.append(StructureEvent(ts, last_close, "BOS_bullish"))

            # ChoCH bullish: price was in downtrend, now breaks a swing high (reversal)
            if last_close > last_high:
                events.append(StructureEvent(ts, last_close, "ChoCH_bullish"))

        if len(swing_lows) >= 2:
            prev_low = swing_lows[-2].price
            last_low = swing_lows[-1].price

            # BOS bearish: close breaks below previous swing low (continuation down)
            if last_close < prev_low and last_low < prev_low:
                events.append(StructureEvent(ts, last_close, "BOS_bearish"))

            # ChoCH bearish: price was in uptrend, now breaks a swing low (reversal)
            if last_close < last_low:
                events.append(StructureEvent(ts, last_close, "ChoCH_bearish"))

        return events
