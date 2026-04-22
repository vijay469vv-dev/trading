"""
Walk-forward Backtest — 5-day SMC Strategy Replay

Methodology:
  - Fetches 5 days of 15m OHLCV + warmup period
  - Steps bar by bar; at each bar generates signal using only historical data
  - Simulates SL/TP on subsequent candles (no lookahead bias)
  - Reports full performance breakdown per symbol and combined
"""

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from data.fetcher import DataFetcher
from strategy.smc_strategy import SMCStrategy
from signals.indicators import Indicators

console = Console()

SYMBOLS      = ["BTC/USDT", "XAU/USD"]
DAYS         = 5
WARMUP_BARS  = 300          # bars fed to strategy before backtest window
TF           = "15m"
BARS_PER_DAY = 4 * 24       # 96 bars/day on 15m
TOTAL_BARS   = DAYS * BARS_PER_DAY + WARMUP_BARS
INITIAL_BAL  = 10_000.0
RISK_PCT     = 0.01          # 1% risk per trade
MAX_TRADES   = 3             # max concurrent open trades (not used in simple bt)
SLIPPAGE_PCT = 0.0002        # 0.02% slippage on entry


@dataclass
class BTrade:
    symbol: str
    direction: str
    entry: float
    sl: float
    tp: float
    size_usd: float
    entry_bar: int
    exit_bar: int = 0
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    confidence: float = 0.0
    rr: float = 0.0
    reason: str = ""


@dataclass
class SymbolResult:
    symbol: str
    trades: list[BTrade] = field(default_factory=list)

    @property
    def closed(self):
        return [t for t in self.trades if t.exit_bar > 0]

    def stats(self) -> dict:
        c = self.closed
        if not c:
            return {}
        wins = [t for t in c if t.pnl > 0]
        losses = [t for t in c if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in c)
        win_rate = len(wins) / len(c) * 100 if c else 0
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        avg_rr = sum(t.rr for t in c) / len(c)
        profit_factor = (
            abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in losses))
            if losses and sum(t.pnl for t in losses) != 0 else float("inf")
        )
        return {
            "trades": len(c),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_rr": avg_rr,
            "profit_factor": profit_factor,
        }


def _position_size(balance: float, entry: float, sl: float) -> float:
    risk_amount = balance * RISK_PCT
    sl_dist = abs(entry - sl) / entry
    if sl_dist == 0:
        return 0.0
    return min(risk_amount / sl_dist, balance * 0.25)


def _simulate_trade(
    df: pd.DataFrame, trade: BTrade, entry_bar: int
) -> BTrade:
    """Step through future bars and check if SL or TP is hit."""
    for i in range(entry_bar + 1, len(df)):
        candle = df.iloc[i]
        if trade.direction == "buy":
            if candle["low"] <= trade.sl:
                trade.exit_price = trade.sl
                trade.exit_reason = "SL"
                trade.exit_bar = i
                break
            if candle["high"] >= trade.tp:
                trade.exit_price = trade.tp
                trade.exit_reason = "TP"
                trade.exit_bar = i
                break
        else:  # sell
            if candle["high"] >= trade.sl:
                trade.exit_price = trade.sl
                trade.exit_reason = "SL"
                trade.exit_bar = i
                break
            if candle["low"] <= trade.tp:
                trade.exit_price = trade.tp
                trade.exit_reason = "TP"
                trade.exit_bar = i
                break
    else:
        # Trade still open at end — close at last price
        trade.exit_price = df["close"].iloc[-1]
        trade.exit_reason = "EOD"
        trade.exit_bar = len(df) - 1

    # PnL calculation
    if trade.direction == "buy":
        ret = (trade.exit_price - trade.entry) / trade.entry
    else:
        ret = (trade.entry - trade.exit_price) / trade.entry
    trade.pnl = ret * trade.size_usd
    trade.pnl_pct = ret * 100
    return trade


def _equity_curve(trades: list[BTrade], initial: float) -> list[float]:
    curve = [initial]
    bal = initial
    for t in sorted(trades, key=lambda x: x.entry_bar):
        if t.exit_bar > 0:
            bal += t.pnl
            curve.append(bal)
    return curve


def _max_drawdown(curve: list[float]) -> float:
    peak = curve[0]
    max_dd = 0.0
    for v in curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd


async def run_backtest():
    fetcher = DataFetcher()
    strategy = SMCStrategy(min_rr=1.8, min_confidence=0.45, require_kill_zone=False)

    console.print(f"\n[bold cyan]SMC Backtest — {DAYS} days @ 15m[/bold cyan]")
    console.print(f"  Symbols:  {', '.join(SYMBOLS)}")
    console.print(f"  Risk/trade: {RISK_PCT*100:.1f}%  |  Start balance: ${INITIAL_BAL:,.0f}\n")

    all_results: list[SymbolResult] = []

    for sym in SYMBOLS:
        console.print(f"[yellow]Fetching {sym} ({TOTAL_BARS} bars)...[/yellow]")
        df = await fetcher.fetch_ohlcv(sym, timeframe=TF, limit=TOTAL_BARS)
        df_4h = await fetcher.fetch_ohlcv(sym, timeframe="4h", limit=100)
        df_1h = await fetcher.fetch_ohlcv(sym, timeframe="1h", limit=200)

        if df.empty or len(df) < WARMUP_BARS + 10:
            console.print(f"[red]{sym}: insufficient data[/red]")
            continue

        result = SymbolResult(symbol=sym)
        balance = INITIAL_BAL
        open_trade: Optional[BTrade] = None
        signals_tried = 0
        signals_skipped_rr = 0
        last_entry_bar = -10  # avoid re-entering immediately after close

        # Walk forward — start after warmup
        for bar_i in range(WARMUP_BARS, len(df)):
            window = df.iloc[: bar_i + 1]
            price = window["close"].iloc[-1]

            # Check if open trade is closed by this bar
            if open_trade and open_trade.exit_bar > 0 and open_trade.exit_bar <= bar_i:
                balance += open_trade.pnl
                result.trades.append(open_trade)
                status = "TP" if open_trade.pnl > 0 else "SL"
                color = "green" if open_trade.pnl > 0 else "red"
                ts = df.index[open_trade.exit_bar]
                console.print(
                    f"  [{color}]{status}[/{color}] {sym} {open_trade.direction.upper()} "
                    f"@ {open_trade.exit_price:.2f} | PnL: ${open_trade.pnl:+.2f} "
                    f"(RR={open_trade.rr:.1f}) | {ts.strftime('%m-%d %H:%M')}"
                )
                open_trade = None
                last_entry_bar = bar_i

            # Only look for new signal if no open trade and enough gap since last
            if open_trade is not None:
                continue
            if bar_i - last_entry_bar < 4:   # 1h gap min between entries
                continue

            # Generate signal on current window
            sig = strategy.generate_signal(
                window.copy(),
                sym,
                df_4h=df_4h if not df_4h.empty else None,
                df_1h=df_1h if not df_1h.empty else None,
            )

            if sig.direction == "none":
                continue

            signals_tried += 1

            # Apply slippage
            if sig.direction == "buy":
                entry = sig.entry_price * (1 + SLIPPAGE_PCT)
            else:
                entry = sig.entry_price * (1 - SLIPPAGE_PCT)

            size = _position_size(balance, entry, sig.stop_loss)
            if size <= 0:
                continue

            rr = abs(sig.take_profit - entry) / abs(entry - sig.stop_loss) if abs(entry - sig.stop_loss) > 0 else 0
            if rr < 1.8:
                signals_skipped_rr += 1
                continue

            trade = BTrade(
                symbol=sym,
                direction=sig.direction,
                entry=entry,
                sl=sig.stop_loss,
                tp=sig.take_profit,
                size_usd=size,
                entry_bar=bar_i,
                confidence=sig.confidence,
                rr=round(rr, 2),
                reason=sig.reason,
            )

            # Simulate forward SL/TP from entry bar
            trade = _simulate_trade(df, trade, bar_i)
            open_trade = trade

            ts = df.index[bar_i]
            console.print(
                f"  [cyan]ENTRY[/cyan] {sym} {sig.direction.upper()} @ {entry:.2f} "
                f"SL={sig.stop_loss:.2f} TP={sig.take_profit:.2f} "
                f"RR={rr:.1f} conf={sig.confidence:.2f} | {ts.strftime('%m-%d %H:%M')}"
            )

        # Close any still-open trade
        if open_trade and open_trade.exit_bar == 0:
            open_trade.exit_price = df["close"].iloc[-1]
            open_trade.exit_reason = "EOD"
            open_trade.exit_bar = len(df) - 1
            if open_trade.direction == "buy":
                ret = (open_trade.exit_price - open_trade.entry) / open_trade.entry
            else:
                ret = (open_trade.entry - open_trade.exit_price) / open_trade.entry
            open_trade.pnl = ret * open_trade.size_usd
            open_trade.pnl_pct = ret * 100
            result.trades.append(open_trade)

        console.print(f"  [dim]Signals generated: {signals_tried} | Skipped (RR): {signals_skipped_rr}[/dim]\n")
        all_results.append(result)

    await fetcher.close()

    # ── Print results ─────────────────────────────────────────────────────────
    _print_results(all_results)


def _print_results(results: list[SymbolResult]):
    all_trades: list[BTrade] = []

    for r in results:
        st = r.stats()
        if not st:
            console.print(f"[dim]{r.symbol}: no closed trades[/dim]")
            continue

        all_trades.extend(r.closed)

        t = Table(title=f"[bold]{r.symbol}[/bold]", box=box.ROUNDED, show_header=True)
        t.add_column("Metric", style="dim")
        t.add_column("Value", justify="right")

        wr_color = "green" if st["win_rate"] >= 50 else "red"
        pnl_color = "green" if st["total_pnl"] >= 0 else "red"

        t.add_row("Trades", str(st["trades"]))
        t.add_row("Wins / Losses", f"{st['wins']} / {st['losses']}")
        t.add_row("Win Rate", f"[{wr_color}]{st['win_rate']:.1f}%[/{wr_color}]")
        t.add_row("Total PnL", f"[{pnl_color}]${st['total_pnl']:+.2f}[/{pnl_color}]")
        t.add_row("Avg Win", f"${st['avg_win']:+.2f}")
        t.add_row("Avg Loss", f"${st['avg_loss']:+.2f}")
        t.add_row("Avg R:R", f"{st['avg_rr']:.2f}")
        t.add_row("Profit Factor", f"{st['profit_factor']:.2f}")

        curve = _equity_curve(r.closed, INITIAL_BAL)
        dd = _max_drawdown(curve)
        final = curve[-1] if len(curve) > 1 else INITIAL_BAL
        t.add_row("Max Drawdown", f"[red]{dd:.2f}%[/red]")
        t.add_row("End Balance", f"${final:,.2f}")

        console.print(t)
        console.print()

    # Combined summary
    if len(all_trades) >= 2:
        all_closed = [t for t in all_trades if t.exit_bar > 0]
        wins = [t for t in all_closed if t.pnl > 0]
        total_pnl = sum(t.pnl for t in all_closed)
        curve = _equity_curve(all_closed, INITIAL_BAL)
        dd = _max_drawdown(curve)
        pf_num = sum(t.pnl for t in wins)
        pf_den = abs(sum(t.pnl for t in all_closed if t.pnl <= 0))

        t = Table(title="[bold white]COMBINED[/bold white]", box=box.HEAVY_EDGE, show_header=True)
        t.add_column("Metric", style="dim")
        t.add_column("Value", justify="right")

        wr = len(wins)/len(all_closed)*100 if all_closed else 0
        pnl_c = "green" if total_pnl >= 0 else "red"
        t.add_row("Total Trades", str(len(all_closed)))
        t.add_row("Win Rate", f"{'green' and '[green]' if wr>=50 else '[red]'}{wr:.1f}%{'[/green]' if wr>=50 else '[/red]'}")
        t.add_row("Total PnL", f"[{pnl_c}]${total_pnl:+.2f}[/{pnl_c}]")
        t.add_row("Profit Factor", f"{pf_num/pf_den:.2f}" if pf_den > 0 else "∞")
        t.add_row("Max Drawdown", f"[red]{dd:.2f}%[/red]")
        t.add_row("Return on $10k", f"[{pnl_c}]{total_pnl/INITIAL_BAL*100:+.2f}%[/{pnl_c}]")

        console.print(t)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_backtest())
