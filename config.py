from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    # Binance
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET: str = os.getenv("BINANCE_SECRET", "")
    BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

    # OANDA
    OANDA_API_KEY: str = os.getenv("OANDA_API_KEY", "")
    OANDA_ACCOUNT_ID: str = os.getenv("OANDA_ACCOUNT_ID", "")
    OANDA_ENVIRONMENT: str = os.getenv("OANDA_ENVIRONMENT", "practice")

    # Paper Trading
    PAPER_TRADING: bool = os.getenv("PAPER_TRADING", "true").lower() == "true"
    INITIAL_BALANCE_USD: float = float(os.getenv("INITIAL_BALANCE_USD", "150"))

    # Risk (placeholder — will be overridden by RiskManager when user provides details)
    MAX_RISK_PER_TRADE_PCT: float = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "1.0"))
    MAX_DAILY_LOSS_PCT: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "3.0"))
    MAX_OPEN_TRADES: int = int(os.getenv("MAX_OPEN_TRADES", "3"))

    # Dashboard
    DASHBOARD_HOST: str = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "8000"))

    # Symbols
    BTC_SYMBOL_BINANCE: str = "BTC/USDT"
    XAUUSD_SYMBOL: str = "XAU/USD"
    XAUUSD_YFINANCE: str = "GC=F"  # Gold futures as proxy

    # Timeframes
    TIMEFRAMES: list = ["1m", "5m", "15m", "1h", "4h", "1d"]
    DEFAULT_TIMEFRAME: str = "15m"

    # SMC Signal Settings
    SMC_SWING_LOOKBACK: int = 8        # bars to look back for swing detection
    SMC_FVG_THRESHOLD: float = 0.0003  # min gap size as % of price (lowered for more signals)
    SMC_OB_LOOKBACK: int = 30          # order block lookback


config = Config()
