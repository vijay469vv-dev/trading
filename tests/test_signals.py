"""Quick smoke tests for the SMC signal engine."""

import pandas as pd
import numpy as np
import pytest
from signals.liquidity import LiquidityAnalyzer
from signals.structure import MarketStructure
from signals.indicators import Indicators


def make_df(n=200, trend="up") -> pd.DataFrame:
    np.random.seed(42)
    base = 2000.0
    prices = []
    for i in range(n):
        if trend == "up":
            base += np.random.normal(0.5, 2)
        else:
            base -= np.random.normal(0.5, 2)
        prices.append(base)

    df = pd.DataFrame({
        "open":   [p - abs(np.random.normal(0, 1)) for p in prices],
        "high":   [p + abs(np.random.normal(0, 2)) for p in prices],
        "low":    [p - abs(np.random.normal(0, 2)) for p in prices],
        "close":  prices,
        "volume": [np.random.uniform(100, 1000) for _ in prices],
    }, index=pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"))
    return df


def test_swing_detection():
    df = make_df()
    analyzer = LiquidityAnalyzer(lookback=5)
    sigs = analyzer.analyze(df)
    assert len(sigs.swing_highs) > 0
    assert len(sigs.swing_lows) > 0


def test_liquidity_levels():
    df = make_df()
    analyzer = LiquidityAnalyzer(lookback=5)
    sigs = analyzer.analyze(df)
    assert len(sigs.bsl_levels) > 0
    assert len(sigs.ssl_levels) > 0
    assert all(l.kind == "bsl" for l in sigs.bsl_levels)
    assert all(l.kind == "ssl" for l in sigs.ssl_levels)


def test_fvg_detection():
    df = make_df()
    analyzer = LiquidityAnalyzer(lookback=5, fvg_threshold=0.0001)
    sigs = analyzer.analyze(df)
    # FVGs may or may not exist depending on random data
    assert isinstance(sigs.fvgs, list)


def test_order_blocks():
    df = make_df()
    analyzer = LiquidityAnalyzer(lookback=5)
    sigs = analyzer.analyze(df)
    assert isinstance(sigs.order_blocks, list)


def test_market_structure_trend():
    df = make_df(trend="up")
    analyzer = LiquidityAnalyzer(lookback=5)
    sigs = analyzer.analyze(df)
    ms = MarketStructure(lookback=5)
    result = ms.analyze(df, sigs.swing_highs, sigs.swing_lows)
    assert result["trend"] in ("bullish", "bearish", "ranging")


def test_indicators():
    df = make_df()
    df = Indicators.add_all(df)
    assert "ema_20" in df.columns
    assert "rsi" in df.columns
    assert "atr" in df.columns
    assert not df["ema_20"].isna().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
