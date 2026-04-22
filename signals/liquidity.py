"""
SMC Liquidity Signal Detection
Detects: swing highs/lows, buy/sell-side liquidity, sweeps,
         order blocks, fair value gaps (FVG), inducement levels.
"""

from dataclasses import dataclass, field
from typing import Literal
import pandas as pd
import numpy as np
from config import config


@dataclass
class SwingPoint:
    index: int
    timestamp: pd.Timestamp
    price: float
    kind: Literal["high", "low"]


@dataclass
class OrderBlock:
    timestamp: pd.Timestamp
    top: float
    bottom: float
    kind: Literal["bullish", "bearish"]   # bullish OB = demand, bearish OB = supply
    mitigated: bool = False


@dataclass
class FairValueGap:
    timestamp: pd.Timestamp
    top: float
    bottom: float
    kind: Literal["bullish", "bearish"]
    filled: bool = False


@dataclass
class LiquidityLevel:
    price: float
    kind: Literal["bsl", "ssl"]           # buy-side or sell-side liquidity
    touches: int = 1
    swept: bool = False


@dataclass
class LiquiditySweep:
    timestamp: pd.Timestamp
    price: float
    kind: Literal["bsl_sweep", "ssl_sweep"]
    reversal_confirmed: bool = False


@dataclass
class LiquiditySignals:
    swing_highs: list[SwingPoint] = field(default_factory=list)
    swing_lows: list[SwingPoint] = field(default_factory=list)
    bsl_levels: list[LiquidityLevel] = field(default_factory=list)   # buy-side liquidity (above highs)
    ssl_levels: list[LiquidityLevel] = field(default_factory=list)   # sell-side liquidity (below lows)
    sweeps: list[LiquiditySweep] = field(default_factory=list)
    order_blocks: list[OrderBlock] = field(default_factory=list)
    fvgs: list[FairValueGap] = field(default_factory=list)
    inducement_levels: list[float] = field(default_factory=list)


class LiquidityAnalyzer:
    def __init__(self, lookback: int = None, fvg_threshold: float = None):
        self.lookback = lookback or config.SMC_SWING_LOOKBACK
        self.fvg_threshold = fvg_threshold or config.SMC_FVG_THRESHOLD
        self.ob_lookback = config.SMC_OB_LOOKBACK

    def analyze(self, df: pd.DataFrame) -> LiquiditySignals:
        signals = LiquiditySignals()
        if len(df) < self.lookback * 2 + 1:
            return signals

        signals.swing_highs, signals.swing_lows = self._detect_swings(df)
        signals.bsl_levels = self._build_bsl(signals.swing_highs)
        signals.ssl_levels = self._build_ssl(signals.swing_lows)
        signals.sweeps = self._detect_sweeps(df, signals.bsl_levels, signals.ssl_levels)
        signals.order_blocks = self._detect_order_blocks(df)
        signals.fvgs = self._detect_fvgs(df)
        signals.inducement_levels = self._detect_inducement(
            signals.swing_highs, signals.swing_lows
        )
        return signals

    # ── Swing Detection ───────────────────────────────────────────────────────

    def _detect_swings(
        self, df: pd.DataFrame
    ) -> tuple[list[SwingPoint], list[SwingPoint]]:
        highs, lows = [], []
        n = len(df)
        lb = self.lookback

        for i in range(lb, n - lb):
            window_h = df["high"].iloc[i - lb : i + lb + 1]
            window_l = df["low"].iloc[i - lb : i + lb + 1]

            if df["high"].iloc[i] == window_h.max():
                highs.append(
                    SwingPoint(
                        index=i,
                        timestamp=df.index[i],
                        price=df["high"].iloc[i],
                        kind="high",
                    )
                )
            if df["low"].iloc[i] == window_l.min():
                lows.append(
                    SwingPoint(
                        index=i,
                        timestamp=df.index[i],
                        price=df["low"].iloc[i],
                        kind="low",
                    )
                )
        return highs, lows

    # ── Liquidity Levels ──────────────────────────────────────────────────────

    def _build_bsl(self, swing_highs: list[SwingPoint]) -> list[LiquidityLevel]:
        """Buy-side liquidity sits above swing highs (stop hunts target these)."""
        levels: dict[float, LiquidityLevel] = {}
        for sp in swing_highs:
            rounded = round(sp.price, 2)
            if rounded in levels:
                levels[rounded].touches += 1
            else:
                levels[rounded] = LiquidityLevel(price=rounded, kind="bsl")
        return list(levels.values())

    def _build_ssl(self, swing_lows: list[SwingPoint]) -> list[LiquidityLevel]:
        """Sell-side liquidity sits below swing lows."""
        levels: dict[float, LiquidityLevel] = {}
        for sp in swing_lows:
            rounded = round(sp.price, 2)
            if rounded in levels:
                levels[rounded].touches += 1
            else:
                levels[rounded] = LiquidityLevel(price=rounded, kind="ssl")
        return list(levels.values())

    # ── Sweep Detection ───────────────────────────────────────────────────────

    def _detect_sweeps(
        self,
        df: pd.DataFrame,
        bsl_levels: list[LiquidityLevel],
        ssl_levels: list[LiquidityLevel],
    ) -> list[LiquiditySweep]:
        """
        Look back SWEEP_LOOKBACK candles for sweeps, not just the last candle.
        A sweep = wick through a liquidity level + close back inside = reversal.
        """
        SWEEP_LOOKBACK = 10
        sweeps = []
        n = len(df)
        lookback = min(SWEEP_LOOKBACK, n - 1)

        for lvl in bsl_levels:
            for i in range(lookback):
                idx = n - 1 - i
                candle = df.iloc[idx]
                next_candle = df.iloc[idx + 1] if idx + 1 < n else candle
                # Wick above level but close below → sweep
                if candle["high"] > lvl.price and candle["close"] < lvl.price:
                    reversal = candle["close"] < candle["open"]   # bearish body
                    # Extra confirmation: next candle continues lower
                    strong = reversal and (next_candle["close"] < candle["close"])
                    sweeps.append(LiquiditySweep(
                        timestamp=df.index[idx],
                        price=lvl.price,
                        kind="bsl_sweep",
                        reversal_confirmed=reversal or strong,
                    ))
                    lvl.swept = True
                    break  # one sweep per level

        for lvl in ssl_levels:
            for i in range(lookback):
                idx = n - 1 - i
                candle = df.iloc[idx]
                next_candle = df.iloc[idx + 1] if idx + 1 < n else candle
                # Wick below level but close above → sweep
                if candle["low"] < lvl.price and candle["close"] > lvl.price:
                    reversal = candle["close"] > candle["open"]   # bullish body
                    strong = reversal and (next_candle["close"] > candle["close"])
                    sweeps.append(LiquiditySweep(
                        timestamp=df.index[idx],
                        price=lvl.price,
                        kind="ssl_sweep",
                        reversal_confirmed=reversal or strong,
                    ))
                    lvl.swept = True
                    break

        return sweeps

    # ── Order Blocks ──────────────────────────────────────────────────────────

    def _detect_order_blocks(self, df: pd.DataFrame) -> list[OrderBlock]:
        """
        Bearish OB: last bullish candle before a strong bearish impulse.
        Bullish OB: last bearish candle before a strong bullish impulse.
        """
        obs = []
        n = len(df)
        lookback = min(self.ob_lookback, n - 2)

        for i in range(1, lookback):
            idx = n - 1 - i
            if idx < 1:
                break
            curr = df.iloc[idx]
            next_c = df.iloc[idx + 1]

            bullish = curr["close"] > curr["open"]
            bearish_impulse = (
                next_c["close"] < next_c["open"]
                and (next_c["open"] - next_c["close"]) > (curr["high"] - curr["low"]) * 0.5
            )
            if bullish and bearish_impulse:
                obs.append(
                    OrderBlock(
                        timestamp=df.index[idx],
                        top=curr["high"],
                        bottom=curr["low"],
                        kind="bearish",
                    )
                )

            bearish = curr["close"] < curr["open"]
            bullish_impulse = (
                next_c["close"] > next_c["open"]
                and (next_c["close"] - next_c["open"]) > (curr["high"] - curr["low"]) * 0.5
            )
            if bearish and bullish_impulse:
                obs.append(
                    OrderBlock(
                        timestamp=df.index[idx],
                        top=curr["high"],
                        bottom=curr["low"],
                        kind="bullish",
                    )
                )

        # Mark mitigated OBs (price has returned into them)
        last_close = df["close"].iloc[-1]
        for ob in obs:
            if ob.bottom <= last_close <= ob.top:
                ob.mitigated = True

        return obs

    # ── Fair Value Gaps ───────────────────────────────────────────────────────

    def _detect_fvgs(self, df: pd.DataFrame) -> list[FairValueGap]:
        """
        Bullish FVG: candle[i-1].high < candle[i+1].low  (gap up)
        Bearish FVG: candle[i-1].low  > candle[i+1].high (gap down)
        """
        fvgs = []
        n = len(df)
        lookback = min(50, n - 2)

        for i in range(1, lookback + 1):
            idx = n - 1 - i
            if idx < 1 or idx + 1 >= n:
                continue
            prev = df.iloc[idx - 1]
            curr = df.iloc[idx]
            nxt = df.iloc[idx + 1]

            gap_size = abs(nxt["low"] - prev["high"]) / curr["close"]

            if nxt["low"] > prev["high"] and gap_size >= self.fvg_threshold:
                fvg = FairValueGap(
                    timestamp=df.index[idx],
                    top=nxt["low"],
                    bottom=prev["high"],
                    kind="bullish",
                )
                # Check if filled
                recent_low = df["low"].iloc[idx + 1 :].min()
                if recent_low <= fvg.bottom:
                    fvg.filled = True
                fvgs.append(fvg)

            elif prev["low"] > nxt["high"] and gap_size >= self.fvg_threshold:
                fvg = FairValueGap(
                    timestamp=df.index[idx],
                    top=prev["low"],
                    bottom=nxt["high"],
                    kind="bearish",
                )
                recent_high = df["high"].iloc[idx + 1 :].max()
                if recent_high >= fvg.top:
                    fvg.filled = True
                fvgs.append(fvg)

        return fvgs

    # ── Inducement ────────────────────────────────────────────────────────────

    def _detect_inducement(
        self,
        swing_highs: list[SwingPoint],
        swing_lows: list[SwingPoint],
    ) -> list[float]:
        """
        Inducement = minor swing point between two major swings.
        These are liquidity targets used to trap retail traders.
        """
        inducements = []

        if len(swing_highs) >= 3:
            # Minor high between two lower highs
            for i in range(1, len(swing_highs) - 1):
                prev_h = swing_highs[i - 1].price
                curr_h = swing_highs[i].price
                next_h = swing_highs[i + 1].price
                if prev_h > curr_h and next_h > curr_h:
                    inducements.append(curr_h)

        if len(swing_lows) >= 3:
            # Minor low between two higher lows
            for i in range(1, len(swing_lows) - 1):
                prev_l = swing_lows[i - 1].price
                curr_l = swing_lows[i].price
                next_l = swing_lows[i + 1].price
                if prev_l < curr_l and next_l < curr_l:
                    inducements.append(curr_l)

        return inducements

    # ── Summary ───────────────────────────────────────────────────────────────

    def get_nearest_levels(
        self, signals: LiquiditySignals, current_price: float, n: int = 3
    ) -> dict:
        """Return the n nearest BSL and SSL levels to current price."""
        bsl = sorted(
            [l for l in signals.bsl_levels if not l.swept and l.price > current_price],
            key=lambda x: x.price,
        )[:n]
        ssl = sorted(
            [l for l in signals.ssl_levels if not l.swept and l.price < current_price],
            key=lambda x: x.price,
            reverse=True,
        )[:n]
        return {"bsl": bsl, "ssl": ssl}
