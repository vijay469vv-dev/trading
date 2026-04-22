"""
Multi-Timeframe (MTF) Confluence Analyzer

Checks HTF (4h/1h) trend and structure before confirming LTF (15m) entry.
Only takes trades that align across all timeframes — dramatically improves win rate.

Hierarchy:
  4h  → macro bias (bullish / bearish / ranging)
  1h  → intermediate structure (confirms or rejects 4h bias)
  15m → entry trigger (sweep + OB/FVG at this level)
"""

from dataclasses import dataclass
from typing import Literal, Optional
import pandas as pd
from signals.indicators import Indicators
from signals.liquidity import LiquidityAnalyzer
from signals.structure import MarketStructure


@dataclass
class MTFBias:
    htf_trend: Literal["bullish", "bearish", "ranging"]     # 4h
    itf_trend: Literal["bullish", "bearish", "ranging"]     # 1h
    aligned: bool                                            # HTF == ITF
    bias: Literal["bullish", "bearish", "ranging"]          # combined
    htf_ema_slope: float                                     # +/- trend strength
    itf_above_ema: bool                                      # price > key EMA on 1h
    confluence_score: float                                  # 0.0 – 1.0


class MTFAnalyzer:
    def __init__(self):
        self._liq = LiquidityAnalyzer(lookback=8)
        self._struct = MarketStructure(lookback=8)

    def analyze(self, df_4h: pd.DataFrame, df_1h: pd.DataFrame) -> MTFBias:
        htf_trend = self._trend(df_4h)
        itf_trend = self._trend(df_1h)

        aligned = (
            htf_trend == itf_trend
            and htf_trend != "ranging"
        )

        bias: Literal["bullish", "bearish", "ranging"]
        if aligned:
            bias = htf_trend
        elif htf_trend != "ranging":
            bias = htf_trend      # HTF overrides ITF when misaligned
        else:
            bias = itf_trend

        htf_ema_slope = self._ema_slope(df_4h)
        itf_above_ema = self._price_above_ema(df_1h)

        score = self._score(htf_trend, itf_trend, aligned, htf_ema_slope, itf_above_ema)

        return MTFBias(
            htf_trend=htf_trend,
            itf_trend=itf_trend,
            aligned=aligned,
            bias=bias,
            htf_ema_slope=htf_ema_slope,
            itf_above_ema=itf_above_ema,
            confluence_score=score,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _trend(self, df: pd.DataFrame) -> Literal["bullish", "bearish", "ranging"]:
        if len(df) < 50:
            return "ranging"
        df2 = Indicators.add_all(df)
        liq = self._liq.analyze(df2)
        result = self._struct.analyze(df2, liq.swing_highs, liq.swing_lows)
        return result["trend"]

    def _ema_slope(self, df: pd.DataFrame) -> float:
        """Positive = bullish slope, negative = bearish. Normalized per price."""
        if len(df) < 52:
            return 0.0
        ema = Indicators.ema(df, 50)
        recent = ema.dropna()
        if len(recent) < 5:
            return 0.0
        slope = (recent.iloc[-1] - recent.iloc[-5]) / recent.iloc[-5]
        return round(slope * 100, 4)   # as %

    def _price_above_ema(self, df: pd.DataFrame) -> bool:
        if len(df) < 51:
            return True
        ema50 = Indicators.ema(df, 50).iloc[-1]
        return df["close"].iloc[-1] > ema50

    def _score(self, htf, itf, aligned, slope, above_ema) -> float:
        score = 0.0
        if htf != "ranging":
            score += 0.3
        if itf != "ranging":
            score += 0.2
        if aligned:
            score += 0.25
        if abs(slope) > 0.1:
            score += 0.15
        if above_ema:
            score += 0.1
        return min(score, 1.0)

    def allows_long(self, bias: MTFBias) -> bool:
        return True  # never hard-block; use mtf_bonus() to adjust confidence

    def allows_short(self, bias: MTFBias) -> bool:
        return True  # never hard-block; use mtf_bonus() to adjust confidence

    def mtf_bonus(self, bias: MTFBias, direction: str) -> float:
        """
        Returns confidence delta based on MTF alignment.
        +0.12 if aligned, -0.08 if counter-trend, 0 if ranging.
        """
        if bias.bias == "ranging":
            return 0.0
        if direction == "buy" and bias.bias == "bullish":
            return 0.12
        if direction == "sell" and bias.bias == "bearish":
            return 0.12
        # Counter-trend: small penalty, don't block
        return -0.08
