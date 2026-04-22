"""
Risk Manager
Calculates position size, enforces daily loss limits, and validates signals.
All parameters are placeholders — update when user provides their risk details.
"""

from dataclasses import dataclass
from typing import Optional
from strategy.base import Signal
from config import config


@dataclass
class RiskConfig:
    """Editable risk parameters — replace with user-provided values."""
    max_risk_per_trade_pct: float = config.MAX_RISK_PER_TRADE_PCT
    max_daily_loss_pct: float = config.MAX_DAILY_LOSS_PCT
    max_open_trades: int = config.MAX_OPEN_TRADES
    min_risk_reward: float = 2.0
    max_position_pct: float = 10.0
    leverage: int = 1000                # 1:1000 leverage
    btc_max_size_usd: float = 5000.0
    xauusd_pip_value: float = 1.0


class RiskManager:
    def __init__(self, risk_config: Optional[RiskConfig] = None):
        self.cfg = risk_config or RiskConfig()
        self._daily_pnl: float = 0.0
        self._daily_reset_date: Optional[str] = None

    def update_daily_pnl(self, pnl: float):
        from datetime import date
        today = str(date.today())
        if self._daily_reset_date != today:
            self._daily_pnl = 0.0
            self._daily_reset_date = today
        self._daily_pnl += pnl

    def is_daily_limit_hit(self, balance: float) -> bool:
        max_loss = balance * (self.cfg.max_daily_loss_pct / 100)
        return self._daily_pnl <= -max_loss

    def calculate_position_size(
        self,
        signal: Signal,
        balance: float,
    ) -> float:
        """
        Returns notional position size in USD (market exposure).
        With leverage, margin used = notional / leverage.
        Formula: notional = (balance * risk_pct) / sl_distance_pct
        Leverage amplifies exposure but risk is still capped at risk_pct of balance.
        """
        if signal.direction == "none":
            return 0.0

        risk_amount = balance * (self.cfg.max_risk_per_trade_pct / 100)
        sl_distance_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price

        if sl_distance_pct <= 0:
            return 0.0

        # Notional exposure
        notional = risk_amount / sl_distance_pct

        # Hard cap: can't use more margin than balance allows
        max_notional = balance * self.cfg.leverage
        notional = min(notional, max_notional)

        # Cap at max_position_pct of leveraged exposure
        max_by_pct = balance * self.cfg.leverage * (self.cfg.max_position_pct / 100)
        notional = min(notional, max_by_pct)

        if "BTC" in signal.symbol:
            notional = min(notional, self.cfg.btc_max_size_usd * self.cfg.leverage)

        return round(notional, 2)

    def validate_signal(
        self,
        signal: Signal,
        balance: float,
        open_trade_count: int,
    ) -> tuple[bool, str]:
        """Returns (is_valid, rejection_reason)."""
        if signal.direction == "none":
            return False, "No signal"
        if self.is_daily_limit_hit(balance):
            return False, "Daily loss limit reached"
        if open_trade_count >= self.cfg.max_open_trades:
            return False, f"Max open trades ({self.cfg.max_open_trades}) reached"
        if signal.risk_reward < self.cfg.min_risk_reward:
            return False, f"R:R {signal.risk_reward:.1f} below minimum {self.cfg.min_risk_reward}"
        size = self.calculate_position_size(signal, balance)
        if size <= 0:
            return False, "Position size is zero"
        return True, "OK"

    def update_config(self, **kwargs):
        """Dynamically update risk parameters."""
        for key, value in kwargs.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, value)
