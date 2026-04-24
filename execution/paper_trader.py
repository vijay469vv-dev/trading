"""
Paper Trading Engine
Simulates order fills, tracks open/closed trades, calculates P&L.
All operations are synchronous and in-memory (persisted to JSON on each update).
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from strategy.base import Signal
from config import config


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass
class Trade:
    id: str
    symbol: str
    direction: Literal["buy", "sell"]
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float               # base currency units (notional / entry_price)
    size_usd: float           # notional USD exposure
    margin: float             # margin locked (size_usd / leverage)
    confidence: float
    reason: str
    timeframe: str
    leverage: int = 1
    status: TradeStatus = TradeStatus.OPEN
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    opened_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    closed_at: Optional[str] = None

    def close(self, exit_price: float):
        self.exit_price = exit_price
        self.status = TradeStatus.CLOSED
        self.closed_at = datetime.now(timezone.utc).isoformat()
        if self.direction == "buy":
            self.pnl = (exit_price - self.entry_price) * self.size
        else:
            self.pnl = (self.entry_price - exit_price) * self.size
        # PnL % relative to margin (not notional) to show real leverage return
        self.pnl_pct = round((self.pnl / self.margin) * 100, 2) if self.margin > 0 else 0.0


class PaperTrader:
    TRADES_FILE = Path("logs/paper_trades.json")

    def __init__(self, leverage: int = 1000):
        self.leverage: int = leverage
        self.balance: float = config.INITIAL_BALANCE_USD
        self.initial_balance: float = config.INITIAL_BALANCE_USD
        self.open_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self._load()

    # ── Order Management ──────────────────────────────────────────────────────

    def open_trade(self, signal: Signal, size_usd: float) -> Optional[Trade]:
        if signal.direction == "none":
            return None
        if len(self.open_trades) >= config.MAX_OPEN_TRADES:
            return None

        # With leverage: margin = notional / leverage
        margin = round(size_usd / self.leverage, 4)
        if margin > self.balance:
            return None

        size = size_usd / signal.entry_price

        trade = Trade(
            id=str(uuid.uuid4())[:8],
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            size=round(size, 6),
            size_usd=round(size_usd, 2),
            margin=margin,
            confidence=signal.confidence,
            reason=signal.reason,
            timeframe=signal.timeframe,
            leverage=self.leverage,
        )
        self.balance -= margin      # only deduct margin, not full notional
        self.open_trades.append(trade)
        self._save()
        return trade

    PROFIT_TARGET_USD: float = 1.0   # close trade as soon as unrealized profit hits $1

    def update_trades(self, symbol: str, current_price: float) -> list[Trade]:
        """Check open trades for SL/TP hits. Returns list of closed trades."""
        just_closed = []
        remaining = []

        for trade in self.open_trades:
            if trade.symbol != symbol:
                remaining.append(trade)
                continue

            # Unrealized P&L at current price
            if trade.direction == "buy":
                unrealized = (current_price - trade.entry_price) * trade.size
                hit_tp = current_price >= trade.take_profit or unrealized >= self.PROFIT_TARGET_USD
                hit_sl = current_price <= trade.stop_loss
            else:
                unrealized = (trade.entry_price - current_price) * trade.size
                hit_tp = current_price <= trade.take_profit or unrealized >= self.PROFIT_TARGET_USD
                hit_sl = current_price >= trade.stop_loss

            if hit_tp or hit_sl:
                # $1 profit cap exits at market price; price-based exits use exact level
                if hit_sl:
                    exit_price = trade.stop_loss
                elif unrealized >= self.PROFIT_TARGET_USD:
                    exit_price = current_price
                else:
                    exit_price = trade.take_profit
                trade.close(exit_price)
                self.balance += trade.margin + trade.pnl
                self.closed_trades.append(trade)
                just_closed.append(trade)
            else:
                remaining.append(trade)

        self.open_trades = remaining
        if just_closed:
            self._save()
        return just_closed

    def cancel_trade(self, trade_id: str) -> bool:
        for trade in self.open_trades:
            if trade.id == trade_id:
                trade.status = TradeStatus.CANCELLED
                self.balance += trade.margin
                self.open_trades.remove(trade)
                self.closed_trades.append(trade)
                self._save()
                return True
        return False

    # ── P&L and Stats ─────────────────────────────────────────────────────────

    def get_unrealized_pnl(self, prices: dict[str, float]) -> float:
        pnl = 0.0
        for trade in self.open_trades:
            price = prices.get(trade.symbol, trade.entry_price)
            if trade.direction == "buy":
                pnl += (price - trade.entry_price) * trade.size
            else:
                pnl += (trade.entry_price - price) * trade.size
        return round(pnl, 2)

    def get_stats(self) -> dict:
        if not self.closed_trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_pnl": 0,
                "best_trade": 0,
                "worst_trade": 0,
                "balance": round(self.balance, 2),
                "equity_change_pct": 0,
            }

        pnls = [t.pnl for t in self.closed_trades]
        wins = [p for p in pnls if p > 0]
        return {
            "total_trades": len(pnls),
            "win_rate": round(len(wins) / len(pnls) * 100, 1),
            "total_pnl": round(sum(pnls), 2),
            "avg_pnl": round(sum(pnls) / len(pnls), 2),
            "best_trade": round(max(pnls), 2),
            "worst_trade": round(min(pnls), 2),
            "balance": round(self.balance, 2),
            "equity_change_pct": round(
                (self.balance - self.initial_balance) / self.initial_balance * 100, 2
            ),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        self.TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "open_trades": [asdict(t) for t in self.open_trades],
            "closed_trades": [asdict(t) for t in self.closed_trades],
        }
        self.TRADES_FILE.write_text(json.dumps(data, indent=2))

    def _load(self):
        if not self.TRADES_FILE.exists():
            return
        try:
            data = json.loads(self.TRADES_FILE.read_text())
            self.balance = data.get("balance", self.balance)
            self.initial_balance = data.get("initial_balance", self.initial_balance)
            for t in data.get("open_trades", []):
                self.open_trades.append(self._dict_to_trade(t))
            for t in data.get("closed_trades", []):
                self.closed_trades.append(self._dict_to_trade(t))
        except Exception:
            pass

    @staticmethod
    def _dict_to_trade(d: dict) -> Trade:
        d["status"] = TradeStatus(d["status"])
        return Trade(**d)
