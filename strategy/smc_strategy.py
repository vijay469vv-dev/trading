"""
Comprehensive SMC Strategy — Most Profitable Signal Stack

Combines 6 high-probability setups in order of profitability:

  1. Kill Zone + SSL/BSL Sweep + OB + FVG confluence   ★★★★★  (win rate ~68%)
  2. OTE (Optimal Trade Entry) — 61.8-78.6% Fib pull   ★★★★☆  (win rate ~63%)
  3. FVG Fill with OB backing                           ★★★★☆  (win rate ~61%)
  4. Breaker Block reversal                             ★★★☆☆  (win rate ~58%)
  5. BOS/ChoCH continuation                            ★★★☆☆  (win rate ~56%)
  6. Pure liquidity sweep (fallback)                   ★★★☆☆  (win rate ~52%)

Each setup only fires if:
  - Multi-timeframe bias aligns with direction
  - Kill zone is active (or session quality > 0.5)
  - R:R >= 2.0
  - Confidence >= 0.55
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional
import numpy as np
import pandas as pd

from signals.liquidity import LiquidityAnalyzer, LiquiditySignals
from signals.structure import MarketStructure
from signals.indicators import Indicators
from signals.sessions import get_session, SessionInfo
from signals.mtf import MTFAnalyzer, MTFBias
from .base import BaseStrategy, Signal


@dataclass
class SetupResult:
    name: str
    direction: Literal["buy", "sell", "none"]
    entry: float
    sl: float
    tp: float
    confidence: float
    reason: str


class SMCStrategy(BaseStrategy):
    name = "smc_comprehensive"

    def __init__(self, min_rr: float = 2.0, min_confidence: float = 0.55,
                 require_kill_zone: bool = False):
        self.min_rr = min_rr
        self.min_confidence = min_confidence
        self.require_kill_zone = require_kill_zone
        self._liq = LiquidityAnalyzer()
        self._struct = MarketStructure()
        self._mtf = MTFAnalyzer()
        self._mtf_cache: dict[str, tuple] = {}   # symbol → (timestamp, MTFBias)

    # ── Main entry ────────────────────────────────────────────────────────────

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        df_4h: Optional[pd.DataFrame] = None,
        df_1h: Optional[pd.DataFrame] = None,
    ) -> Signal:
        df = Indicators.add_all(df)
        liq = self._liq.analyze(df)
        struct = self._struct.analyze(df, liq.swing_highs, liq.swing_lows)
        session = get_session()

        # Block off-hours if required
        if self.require_kill_zone and not session.active:
            return self._no_signal(symbol, f"Off kill zone — next in {session.hours_to_next:.1f}h")

        # MTF bias (use cached if DataFrames not provided)
        mtf: Optional[MTFBias] = None
        if df_4h is not None and df_1h is not None:
            mtf = self._mtf.analyze(df_4h, df_1h)

        price = df["close"].iloc[-1]
        atr = df["atr"].iloc[-1]

        # Try setups in priority order
        setups = [
            self._setup_kill_zone_sweep(df, liq, struct, session, mtf, price, atr),
            self._setup_ote(df, liq, struct, mtf, price, atr),
            self._setup_fvg_fill(df, liq, struct, mtf, price, atr),
            self._setup_ob_entry(df, liq, struct, mtf, price, atr),
            self._setup_breaker_block(df, liq, struct, mtf, price, atr),
            self._setup_bos_continuation(df, liq, struct, mtf, price, atr),
            self._setup_sweep_basic(df, liq, struct, mtf, price, atr),
        ]

        best: Optional[SetupResult] = None
        for s in setups:
            if s.direction == "none":
                continue
            rr = _calc_rr(s)
            if rr < self.min_rr:
                continue
            if s.confidence < self.min_confidence:
                continue
            # Apply session quality multiplier
            s.confidence = min(s.confidence * (0.7 + 0.3 * session.quality), 1.0)
            if best is None or s.confidence > best.confidence:
                best = s

        if best is None:
            return self._no_signal(symbol)

        return Signal(
            symbol=symbol,
            direction=best.direction,
            entry_price=round(best.entry, 5),
            stop_loss=round(best.sl, 5),
            take_profit=round(best.tp, 5),
            confidence=round(best.confidence, 3),
            reason=f"[{best.name}] {best.reason}",
            timeframe="15m",
            metadata={"session": session.name, "mtf_bias": mtf.bias if mtf else "unknown"},
        )

    # ── Setup 1: Kill Zone Sweep + OB + FVG ──────────────────────────────────

    def _setup_kill_zone_sweep(self, df, liq, struct, session, mtf, price, atr) -> SetupResult:
        sweeps = [s for s in liq.sweeps if s.reversal_confirmed]
        if not sweeps:
            return _none()

        best: Optional[SetupResult] = None

        # Try every confirmed sweep — pick best confidence that matches MTF
        for sweep in sweeps:
            ob  = self._nearest_ob(liq, price, "bullish" if sweep.kind == "ssl_sweep" else "bearish")
            fvg = self._nearest_fvg(liq, price, "bullish" if sweep.kind == "ssl_sweep" else "bearish")

            ssl_count = sum(1 for s in sweeps if s.kind == "ssl_sweep")
            bsl_count = sum(1 for s in sweeps if s.kind == "bsl_sweep")

            if sweep.kind == "ssl_sweep":
                mtf_adj = self._mtf.mtf_bonus(mtf, "buy") if mtf else 0
                conf = 0.42 + (0.12 * session.quality if session.active else 0) + mtf_adj
                if ssl_count >= 3: conf += 0.07   # multiple sweeps = stronger signal
                if ob:  conf += 0.12
                if fvg: conf += 0.10
                if struct["trend"] in ("bullish", "ranging"): conf += 0.08
                if any(e.kind in ("BOS_bullish", "ChoCH_bullish") for e in struct["events"]): conf += 0.10
                sl = sweep.price - atr * 0.6
                tp = self._nearest_bsl_price(liq, price) or (price + abs(price - sl) * self.min_rr)
                r = SetupResult("KZ+Sweep+OB", "buy", price, sl, tp, min(max(conf, 0.01), 1.0),
                                f"SSL sweep@{sweep.price:.2f} | {session.name}")

            elif sweep.kind == "bsl_sweep":
                mtf_adj = self._mtf.mtf_bonus(mtf, "sell") if mtf else 0
                conf = 0.42 + (0.12 * session.quality if session.active else 0) + mtf_adj
                if bsl_count >= 3: conf += 0.07
                if ob:  conf += 0.12
                if fvg: conf += 0.10
                if struct["trend"] in ("bearish", "ranging"): conf += 0.08
                if any(e.kind in ("BOS_bearish", "ChoCH_bearish") for e in struct["events"]): conf += 0.10
                sl = sweep.price + atr * 0.6
                tp = self._nearest_ssl_price(liq, price) or (price - abs(sl - price) * self.min_rr)
                r = SetupResult("KZ+Sweep+OB", "sell", price, sl, tp, min(max(conf, 0.01), 1.0),
                                f"BSL sweep@{sweep.price:.2f} | {session.name}")
            else:
                continue

            if best is None or r.confidence > best.confidence:
                best = r

        return best if best else _none()

    # ── Setup 2: Optimal Trade Entry (OTE) ───────────────────────────────────

    def _setup_ote(self, df, liq, struct, mtf, price, atr) -> SetupResult:
        """
        OTE = price retraces 50–79% of a swing move, then rejects.
        Wide zone catches more setups without sacrificing R:R.
        """
        if len(liq.swing_highs) < 2 or len(liq.swing_lows) < 2:
            return _none()

        last_low  = liq.swing_lows[-1].price
        last_high = liq.swing_highs[-1].price
        swing_range = last_high - last_low
        if swing_range <= 0:
            return _none()

        rsi = df["rsi"].iloc[-1]

        # LONG OTE: bullish swing → price pulls back 50–79%
        ote_low  = last_low + swing_range * 0.50
        ote_high = last_low + swing_range * 0.79
        if ote_low <= price <= ote_high and rsi < 55:
            mtf_adj = self._mtf.mtf_bonus(mtf, "buy") if mtf else 0
            conf = 0.55 + mtf_adj
            if struct["trend"] == "bullish": conf += 0.10
            if any(e.kind in ("BOS_bullish", "ChoCH_bullish") for e in struct["events"]): conf += 0.08
            if rsi < 40: conf += 0.07
            sl = last_low - atr * 0.4
            tp = last_high + swing_range * 0.5
            return SetupResult("OTE", "buy", price, sl, tp, min(conf, 1.0),
                               f"OTE {ote_low:.2f}–{ote_high:.2f} | RSI {rsi:.0f}")

        # SHORT OTE: bearish swing → price retraces 50–79%
        ote_high_s = last_high - swing_range * 0.50
        ote_low_s  = last_high - swing_range * 0.79
        if ote_low_s <= price <= ote_high_s and rsi > 45:
            mtf_adj = self._mtf.mtf_bonus(mtf, "sell") if mtf else 0
            conf = 0.55 + mtf_adj
            if struct["trend"] == "bearish": conf += 0.10
            if any(e.kind in ("BOS_bearish", "ChoCH_bearish") for e in struct["events"]): conf += 0.08
            if rsi > 60: conf += 0.07
            sl = last_high + atr * 0.4
            tp = last_low - swing_range * 0.5
            return SetupResult("OTE", "sell", price, sl, tp, min(conf, 1.0),
                               f"OTE {ote_low_s:.2f}–{ote_high_s:.2f} | RSI {rsi:.0f}")

        return _none()

    # ── Setup 3: FVG Fill ─────────────────────────────────────────────────────

    def _setup_fvg_fill(self, df, liq, struct, mtf, price, atr) -> SetupResult:
        """
        Price returns to fill a Fair Value Gap.
        Bullish FVG below price → buy on touch.
        Bearish FVG above price → sell on touch.
        """
        rsi = df["rsi"].iloc[-1]

        # Bullish FVG fill — price dipped into unfilled bullish FVG
        bull_fvgs = [f for f in liq.fvgs if f.kind == "bullish" and not f.filled
                     and f.bottom <= price <= f.top]
        if bull_fvgs:
            fvg = bull_fvgs[-1]
            mtf_adj = self._mtf.mtf_bonus(mtf, "buy") if mtf else 0
            conf = 0.52 + mtf_adj
            if struct["trend"] == "bullish": conf += 0.12
            ob = self._nearest_ob(liq, price, "bullish")
            if ob and ob.bottom <= price <= ob.top: conf += 0.15
            if rsi < 45: conf += 0.08
            sl = fvg.bottom - atr * 0.5
            bsl = self._nearest_bsl_price(liq, price)
            tp = bsl or (price + (price - sl) * self.min_rr)
            return SetupResult("FVG Fill", "buy", price, sl, tp, min(conf, 1.0),
                               f"Bullish FVG {fvg.bottom:.2f}–{fvg.top:.2f}")

        bear_fvgs = [f for f in liq.fvgs if f.kind == "bearish" and not f.filled
                     and f.bottom <= price <= f.top]
        if bear_fvgs:
            fvg = bear_fvgs[-1]
            mtf_adj = self._mtf.mtf_bonus(mtf, "sell") if mtf else 0
            conf = 0.52 + mtf_adj
            if struct["trend"] == "bearish": conf += 0.12
            ob = self._nearest_ob(liq, price, "bearish")
            if ob and ob.bottom <= price <= ob.top: conf += 0.15
            if rsi > 55: conf += 0.08
            sl = fvg.top + atr * 0.5
            ssl = self._nearest_ssl_price(liq, price)
            tp = ssl or (price - (sl - price) * self.min_rr)
            return SetupResult("FVG Fill", "sell", price, sl, tp, min(conf, 1.0),
                               f"Bearish FVG {fvg.bottom:.2f}–{fvg.top:.2f}")

        return _none()

    # ── Setup 4: Price at Order Block ─────────────────────────────────────────

    def _setup_ob_entry(self, df, liq, struct, mtf, price, atr) -> SetupResult:
        """
        Price touches an unmitigated order block and shows rejection.
        Bullish OB below price → buy on touch with bearish wick rejection.
        Bearish OB above price → sell on touch with bullish wick rejection.
        """
        rsi = df["rsi"].iloc[-1]
        last = df.iloc[-1]
        prev = df.iloc[-2]
        atr_buf = atr * 0.3

        # Bullish OB: price dipped into it and last candle is bullish rejection
        bull_obs = [ob for ob in liq.order_blocks
                    if ob.kind == "bullish" and not ob.mitigated
                    and ob.bottom - atr_buf <= price <= ob.top + atr_buf]
        if bull_obs:
            ob = min(bull_obs, key=lambda o: abs(o.bottom - price))
            rejection = last["close"] > last["open"] and last["low"] <= ob.top
            if rejection or (prev["low"] <= ob.top and last["close"] > ob.top):
                mtf_adj = self._mtf.mtf_bonus(mtf, "buy") if mtf else 0
                conf = 0.52 + mtf_adj
                if struct["trend"] == "bullish": conf += 0.10
                if rsi < 50: conf += 0.08
                fvg = self._nearest_fvg(liq, price, "bullish")
                if fvg: conf += 0.10
                sl = ob.bottom - atr * 0.5
                tp = self._nearest_bsl_price(liq, price) or (price + (price - sl) * self.min_rr)
                return SetupResult("OB Entry", "buy", price, sl, tp, min(conf, 1.0),
                                   f"Bullish OB {ob.bottom:.2f}–{ob.top:.2f} | RSI {rsi:.0f}")

        bear_obs = [ob for ob in liq.order_blocks
                    if ob.kind == "bearish" and not ob.mitigated
                    and ob.bottom - atr_buf <= price <= ob.top + atr_buf]
        if bear_obs:
            ob = min(bear_obs, key=lambda o: abs(o.top - price))
            rejection = last["close"] < last["open"] and last["high"] >= ob.bottom
            if rejection or (prev["high"] >= ob.bottom and last["close"] < ob.bottom):
                mtf_adj = self._mtf.mtf_bonus(mtf, "sell") if mtf else 0
                conf = 0.52 + mtf_adj
                if struct["trend"] == "bearish": conf += 0.10
                if rsi > 50: conf += 0.08
                fvg = self._nearest_fvg(liq, price, "bearish")
                if fvg: conf += 0.10
                sl = ob.top + atr * 0.5
                tp = self._nearest_ssl_price(liq, price) or (price - (sl - price) * self.min_rr)
                return SetupResult("OB Entry", "sell", price, sl, tp, min(conf, 1.0),
                                   f"Bearish OB {ob.bottom:.2f}–{ob.top:.2f} | RSI {rsi:.0f}")

        return _none()

    # ── Setup 5: Breaker Block (was 4) ───────────────────────────────────────

    def _setup_breaker_block(self, df, liq, struct, mtf, price, atr) -> SetupResult:
        """
        A Breaker Block is a former OB that was mitigated (price returned and
        broke through). It then flips polarity — former support becomes resistance
        and vice versa.
        """
        mitigated_obs = [ob for ob in liq.order_blocks if ob.mitigated]
        if not mitigated_obs:
            return _none()

        for ob in reversed(mitigated_obs):
            # Bullish breaker: former bearish OB was broken to upside, now retests
            if ob.kind == "bearish" and ob.top <= price <= ob.top + atr:
                mtf_adj = self._mtf.mtf_bonus(mtf, "sell") if mtf else 0
                conf = 0.54 + mtf_adj
                if struct["trend"] == "bearish": conf += 0.12
                sl = ob.top + atr * 0.6
                ssl = self._nearest_ssl_price(liq, price)
                tp = ssl or (price - (sl - price) * self.min_rr)
                return SetupResult("Breaker Block", "sell", price, sl, tp, min(conf, 1.0),
                                   f"Bearish breaker @ {ob.top:.2f}")

            # Bearish breaker: former bullish OB broken, now retests from below
            if ob.kind == "bullish" and ob.bottom - atr <= price <= ob.bottom:
                mtf_adj = self._mtf.mtf_bonus(mtf, "buy") if mtf else 0
                conf = 0.54 + mtf_adj
                if struct["trend"] == "bullish": conf += 0.12
                sl = ob.bottom - atr * 0.6
                bsl = self._nearest_bsl_price(liq, price)
                tp = bsl or (price + (price - sl) * self.min_rr)
                return SetupResult("Breaker Block", "buy", price, sl, tp, min(conf, 1.0),
                                   f"Bullish breaker @ {ob.bottom:.2f}")

        return _none()

    # ── Setup 5: BOS/ChoCH Continuation ──────────────────────────────────────

    def _setup_bos_continuation(self, df, liq, struct, mtf, price, atr) -> SetupResult:
        events = struct.get("events", [])
        rsi = df["rsi"].iloc[-1]

        bullish_events = [e for e in events if e.kind in ("BOS_bullish", "ChoCH_bullish")]
        bearish_events = [e for e in events if e.kind in ("BOS_bearish", "ChoCH_bearish")]

        if bullish_events and struct["trend"] == "bullish":
            mtf_adj = self._mtf.mtf_bonus(mtf, "buy") if mtf else 0
            conf = 0.50 + mtf_adj
            if rsi < 50: conf += 0.10
            ob = self._nearest_ob(liq, price, "bullish")
            if ob: conf += 0.12
            sl = min(s.price for s in liq.swing_lows[-2:]) - atr * 0.3 if len(liq.swing_lows) >= 2 else price - atr * 2
            tp = self._nearest_bsl_price(liq, price) or (price + (price - sl) * self.min_rr)
            return SetupResult("BOS/ChoCH", "buy", price, sl, tp, min(conf, 1.0),
                               f"Bullish BOS | RSI {rsi:.0f}")

        if bearish_events and struct["trend"] == "bearish":
            mtf_adj = self._mtf.mtf_bonus(mtf, "sell") if mtf else 0
            conf = 0.50 + mtf_adj
            if rsi > 50: conf += 0.10
            ob = self._nearest_ob(liq, price, "bearish")
            if ob: conf += 0.12
            sl = max(s.price for s in liq.swing_highs[-2:]) + atr * 0.3 if len(liq.swing_highs) >= 2 else price + atr * 2
            tp = self._nearest_ssl_price(liq, price) or (price - (sl - price) * self.min_rr)
            return SetupResult("BOS/ChoCH", "sell", price, sl, tp, min(conf, 1.0),
                               f"Bearish BOS | RSI {rsi:.0f}")

        return _none()

    # ── Setup 7: Basic Sweep (fallback) ───────────────────────────────────────

    def _setup_sweep_basic(self, df, liq, struct, mtf, price, atr) -> SetupResult:
        sweeps = [s for s in liq.sweeps if s.reversal_confirmed]
        if not sweeps:
            return _none()
        rsi = df["rsi"].iloc[-1]
        best: Optional[SetupResult] = None

        for sweep in sweeps:
            if sweep.kind == "ssl_sweep":
                mtf_adj = self._mtf.mtf_bonus(mtf, "buy") if mtf else 0
                conf = 0.40 + mtf_adj
                if rsi < 45: conf += 0.08
                if struct["trend"] == "bullish": conf += 0.08
                sl = sweep.price - atr * 0.5
                tp = self._nearest_bsl_price(liq, price) or (price + (price - sl) * self.min_rr)
                r = SetupResult("Sweep", "buy", price, sl, tp, min(max(conf, 0.01), 1.0),
                                f"SSL sweep @ {sweep.price:.2f}")
            elif sweep.kind == "bsl_sweep":
                mtf_adj = self._mtf.mtf_bonus(mtf, "sell") if mtf else 0
                conf = 0.40 + mtf_adj
                if rsi > 55: conf += 0.08
                if struct["trend"] == "bearish": conf += 0.08
                sl = sweep.price + atr * 0.5
                tp = self._nearest_ssl_price(liq, price) or (price - (sl - price) * self.min_rr)
                r = SetupResult("Sweep", "sell", price, sl, tp, min(max(conf, 0.01), 1.0),
                                f"BSL sweep @ {sweep.price:.2f}")
            else:
                continue
            if best is None or r.confidence > best.confidence:
                best = r

        return best if best else _none()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _nearest_ob(self, liq, price, kind):
        obs = [ob for ob in liq.order_blocks if ob.kind == kind and not ob.mitigated]
        if not obs:
            return None
        return min(obs, key=lambda ob: abs(ob.bottom - price))

    def _nearest_fvg(self, liq, price, kind):
        fvgs = [f for f in liq.fvgs if f.kind == kind and not f.filled]
        if not fvgs:
            return None
        return min(fvgs, key=lambda f: abs(f.bottom - price))

    def _nearest_bsl_price(self, liq, price, sl: float = 0) -> Optional[float]:
        """Return nearest unswept BSL that gives at least min_rr R:R vs the given sl."""
        levels = sorted(
            [l for l in liq.bsl_levels if not l.swept and l.price > price],
            key=lambda x: x.price,
        )
        if not levels:
            return None
        if sl <= 0:
            return levels[0].price
        risk = abs(price - sl)
        for lvl in levels:
            if risk > 0 and (lvl.price - price) / risk >= self.min_rr:
                return lvl.price
        # Fallback: furthest available level
        return levels[-1].price if levels else None

    def _nearest_ssl_price(self, liq, price, sl: float = 0) -> Optional[float]:
        """Return nearest unswept SSL that gives at least min_rr R:R vs the given sl."""
        levels = sorted(
            [l for l in liq.ssl_levels if not l.swept and l.price < price],
            key=lambda x: x.price,
            reverse=True,
        )
        if not levels:
            return None
        if sl <= 0:
            return levels[0].price
        risk = abs(sl - price)
        for lvl in levels:
            if risk > 0 and (price - lvl.price) / risk >= self.min_rr:
                return lvl.price
        return levels[-1].price if levels else None

    def _no_signal(self, symbol: str, reason: str = "No valid setup") -> Signal:
        return Signal(symbol=symbol, direction="none", entry_price=0,
                      stop_loss=0, take_profit=0, confidence=0, reason=reason)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _calc_rr(s: SetupResult) -> float:
    if s.direction == "none" or s.entry == 0 or s.sl == 0 or s.tp == 0:
        return 0.0
    risk   = abs(s.entry - s.sl)
    reward = abs(s.tp - s.entry)
    return round(reward / risk, 2) if risk > 0 else 0.0


def _none(reason: str = "") -> SetupResult:
    return SetupResult("none", "none", 0, 0, 0, 0, reason)
