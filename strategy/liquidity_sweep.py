"""
Liquidity Sweep Strategy (SMC)

Entry logic:
  LONG  — SSL swept + reversal candle + price near bullish OB or FVG + BOS/ChoCH bullish
  SHORT — BSL swept + reversal candle + price near bearish OB or FVG + BOS/ChoCH bearish

Stop loss  : beyond the sweep low/high + ATR buffer
Take profit: nearest opposing liquidity level (BSL for longs, SSL for shorts)
"""

import pandas as pd
from signals.liquidity import LiquidityAnalyzer, LiquiditySignals
from signals.structure import MarketStructure
from signals.indicators import Indicators
from .base import BaseStrategy, Signal
from config import config


class LiquiditySweepStrategy(BaseStrategy):
    name = "liquidity_sweep"

    def __init__(self, min_rr: float = 2.0, min_confidence: float = 0.5):
        self.liquidity = LiquidityAnalyzer()
        self.structure = MarketStructure()
        self.min_rr = min_rr
        self.min_confidence = min_confidence

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        df = Indicators.add_all(df)
        liq: LiquiditySignals = self.liquidity.analyze(df)
        struct = self.structure.analyze(df, liq.swing_highs, liq.swing_lows)

        current_price = df["close"].iloc[-1]
        atr = df["atr"].iloc[-1]
        rsi = df["rsi"].iloc[-1]

        # ── Check for recent sweeps ───────────────────────────────────────────
        recent_sweeps = [s for s in liq.sweeps if s.reversal_confirmed]
        if not recent_sweeps:
            return self._no_signal(symbol)

        sweep = recent_sweeps[-1]
        trend = struct["trend"]

        # ── LONG setup ───────────────────────────────────────────────────────
        if sweep.kind == "ssl_sweep":
            confidence = self._score_long(liq, struct, df, rsi, current_price)
            if confidence >= self.min_confidence:
                sl = sweep.price - atr * 0.5
                # Target nearest BSL level
                bsl = sorted(
                    [l for l in liq.bsl_levels if not l.swept and l.price > current_price],
                    key=lambda x: x.price,
                )
                tp = bsl[0].price if bsl else current_price + (current_price - sl) * self.min_rr

                sig = Signal(
                    symbol=symbol,
                    direction="buy",
                    entry_price=current_price,
                    stop_loss=round(sl, 5),
                    take_profit=round(tp, 5),
                    confidence=confidence,
                    reason=f"SSL sweep @ {sweep.price:.2f} | trend={trend} | RSI={rsi:.1f}",
                    metadata={
                        "sweep": sweep,
                        "ob_nearby": self._nearest_ob(liq, current_price, "bullish"),
                        "fvg_nearby": self._nearest_fvg(liq, current_price, "bullish"),
                        "structure": struct["last_event"],
                    },
                )
                if sig.is_valid(self.min_rr):
                    return sig

        # ── SHORT setup ──────────────────────────────────────────────────────
        if sweep.kind == "bsl_sweep":
            confidence = self._score_short(liq, struct, df, rsi, current_price)
            if confidence >= self.min_confidence:
                sl = sweep.price + atr * 0.5
                ssl = sorted(
                    [l for l in liq.ssl_levels if not l.swept and l.price < current_price],
                    key=lambda x: x.price,
                    reverse=True,
                )
                tp = ssl[0].price if ssl else current_price - (sl - current_price) * self.min_rr

                sig = Signal(
                    symbol=symbol,
                    direction="sell",
                    entry_price=current_price,
                    stop_loss=round(sl, 5),
                    take_profit=round(tp, 5),
                    confidence=confidence,
                    reason=f"BSL sweep @ {sweep.price:.2f} | trend={trend} | RSI={rsi:.1f}",
                    metadata={
                        "sweep": sweep,
                        "ob_nearby": self._nearest_ob(liq, current_price, "bearish"),
                        "fvg_nearby": self._nearest_fvg(liq, current_price, "bearish"),
                        "structure": struct["last_event"],
                    },
                )
                if sig.is_valid(self.min_rr):
                    return sig

        return self._no_signal(symbol)

    # ── Confidence Scoring ────────────────────────────────────────────────────

    def _score_long(self, liq, struct, df, rsi, price) -> float:
        score = 0.3  # base score for confirmed sweep
        if struct["trend"] in ("bullish", "ranging"):
            score += 0.15
        if any(e.kind in ("BOS_bullish", "ChoCH_bullish") for e in struct["events"]):
            score += 0.2
        if self._nearest_ob(liq, price, "bullish"):
            score += 0.15
        if self._nearest_fvg(liq, price, "bullish"):
            score += 0.1
        if rsi < 40:
            score += 0.1  # oversold confluence
        ema20 = df["ema_20"].iloc[-1]
        ema50 = df["ema_50"].iloc[-1]
        if price > ema20 and ema20 > ema50:
            score += 0.1
        return min(score, 1.0)

    def _score_short(self, liq, struct, df, rsi, price) -> float:
        score = 0.3
        if struct["trend"] in ("bearish", "ranging"):
            score += 0.15
        if any(e.kind in ("BOS_bearish", "ChoCH_bearish") for e in struct["events"]):
            score += 0.2
        if self._nearest_ob(liq, price, "bearish"):
            score += 0.15
        if self._nearest_fvg(liq, price, "bearish"):
            score += 0.1
        if rsi > 60:
            score += 0.1  # overbought confluence
        ema20 = df["ema_20"].iloc[-1]
        ema50 = df["ema_50"].iloc[-1]
        if price < ema20 and ema20 < ema50:
            score += 0.1
        return min(score, 1.0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _nearest_ob(self, liq: LiquiditySignals, price: float, kind: str):
        obs = [
            ob for ob in liq.order_blocks
            if ob.kind == kind and not ob.mitigated
        ]
        if not obs:
            return None
        return min(obs, key=lambda ob: abs(ob.bottom - price) if kind == "bullish" else abs(ob.top - price))

    def _nearest_fvg(self, liq: LiquiditySignals, price: float, kind: str):
        fvgs = [f for f in liq.fvgs if f.kind == kind and not f.filled]
        if not fvgs:
            return None
        return min(fvgs, key=lambda f: abs(f.bottom - price))

    def _no_signal(self, symbol: str) -> Signal:
        return Signal(
            symbol=symbol,
            direction="none",
            entry_price=0,
            stop_loss=0,
            take_profit=0,
            confidence=0,
            reason="No valid setup",
        )
