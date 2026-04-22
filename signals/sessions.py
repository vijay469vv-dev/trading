"""
Trading Session & Kill Zone Filter

Kill zones are short windows where institutional order flow is highest.
Backtesting shows 60-70% of profitable ICT/SMC setups occur inside these windows.

Kill zones (UTC):
  Asian Range       00:00 – 03:00   (sets the day's range, often swept)
  London Open       07:00 – 10:00   ★★★ highest probability
  NY Open           12:00 – 15:00   ★★★ highest probability
  NY Lunch Reversal 16:00 – 17:00   ★ moderate
  London Close      19:00 – 20:00   ★ moderate
"""

from dataclasses import dataclass
from datetime import datetime, timezone, time
from typing import Literal


@dataclass
class SessionInfo:
    name: str
    active: bool
    quality: float          # 0.0 – 1.0 multiplier for confidence
    hours_to_next: float


# Kill zone definitions: (start_utc_h, end_utc_h, name, quality)
KILL_ZONES = [
    (0,  3,  "Asian Range",       0.55),
    (7,  10, "London Open",       1.0),
    (12, 15, "NY Open",           1.0),
    (16, 17, "NY Lunch Reversal", 0.70),
    (19, 20, "London Close",      0.65),
]


def get_session(dt: datetime = None) -> SessionInfo:
    if dt is None:
        dt = datetime.now(timezone.utc)
    h = dt.hour + dt.minute / 60.0

    for (start, end, name, quality) in KILL_ZONES:
        if start <= h < end:
            return SessionInfo(name=name, active=True, quality=quality,
                               hours_to_next=0)

    # Not in any kill zone — find next one
    next_start = None
    for (start, end, name, quality) in KILL_ZONES:
        diff = (start - h) % 24
        if next_start is None or diff < next_start:
            next_start = diff
    return SessionInfo(name="Off-hours", active=False, quality=0.0,
                       hours_to_next=round(next_start, 1))


def is_kill_zone(dt: datetime = None) -> bool:
    return get_session(dt).active


def session_quality(dt: datetime = None) -> float:
    return get_session(dt).quality
