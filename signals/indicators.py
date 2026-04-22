"""Standard technical indicators as confluence for SMC signals."""

import pandas as pd
import numpy as np


class Indicators:
    @staticmethod
    def ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
        return df[col].ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, min_periods=period).mean()

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical * df["volume"]).cumsum() / df["volume"].cumsum()
        return vwap

    @staticmethod
    def bollinger_bands(
        df: pd.DataFrame, period: int = 20, std: float = 2.0
    ) -> dict[str, pd.Series]:
        mid = df["close"].rolling(period).mean()
        sigma = df["close"].rolling(period).std()
        return {"upper": mid + std * sigma, "mid": mid, "lower": mid - std * sigma}

    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_20"] = Indicators.ema(df, 20)
        df["ema_50"] = Indicators.ema(df, 50)
        df["ema_200"] = Indicators.ema(df, 200)
        df["rsi"] = Indicators.rsi(df)
        df["atr"] = Indicators.atr(df)
        df["vwap"] = Indicators.vwap(df)
        bb = Indicators.bollinger_bands(df)
        df["bb_upper"] = bb["upper"]
        df["bb_mid"] = bb["mid"]
        df["bb_lower"] = bb["lower"]
        return df
