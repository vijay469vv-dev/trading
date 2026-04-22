from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional
import pandas as pd


@dataclass
class Signal:
    symbol: str
    direction: Literal["buy", "sell", "none"]
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float          # 0.0 – 1.0
    reason: str
    timeframe: str = "15m"
    metadata: dict = field(default_factory=dict)

    @property
    def risk_reward(self) -> float:
        if self.direction == "buy":
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.take_profit - self.entry_price)
        elif self.direction == "sell":
            risk = abs(self.stop_loss - self.entry_price)
            reward = abs(self.entry_price - self.take_profit)
        else:
            return 0.0
        return round(reward / risk, 2) if risk > 0 else 0.0

    def is_valid(self, min_rr: float = 1.5) -> bool:
        return (
            self.direction != "none"
            and self.risk_reward >= min_rr
            and self.confidence > 0.0
            and self.stop_loss > 0
            and self.take_profit > 0
        )


class BaseStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Analyze df and return a Signal."""
        ...

    def __repr__(self) -> str:
        return f"<Strategy: {self.name}>"
