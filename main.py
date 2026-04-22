"""
Automated Trading Bot — Main Entry Point

Three concurrent loops:
  1. price_broadcast_loop — pushes live prices to dashboard every second
  2. strategy_loop        — scans for setups every 60 seconds
  3. sl_tp_monitor_loop  — checks SL/TP on every price tick
"""

import asyncio
import signal
import sys
import threading
from datetime import datetime, timezone

import uvicorn
from rich.console import Console
from rich.table import Table
from rich import box

from config import config
from data.fetcher import DataFetcher
from data.price_stream import live_prices, start_streams
from strategy.smc_strategy import SMCStrategy
from signals.sessions import get_session
from execution.paper_trader import PaperTrader
from risk.risk_manager import RiskManager
from dashboard.app import app as dashboard_app, update_state, broadcast, push_trade_event, _state as dash_state, on_start, on_stop, on_risk_update

console = Console()

SYMBOLS = ["BTC/USDT", "XAU/USD"]
STRATEGY_INTERVAL = 60      # seconds between strategy scans
BROADCAST_INTERVAL = 1      # seconds between dashboard pushes
SLTP_INTERVAL = 2           # seconds between SL/TP checks

fetcher = DataFetcher()
strategy = SMCStrategy(min_rr=1.8, min_confidence=0.45, require_kill_zone=False)
risk = RiskManager()
paper = PaperTrader(leverage=risk.cfg.leverage)

_running = True
_last_signals: dict = {}

def _is_trading() -> bool:
    return dash_state.get("trading_active", False)


# ── Price broadcast loop ──────────────────────────────────────────────────────

async def price_broadcast_loop():
    """Push live prices to all WebSocket clients every second."""
    while _running:
        prices = {
            sym: live_prices[sym].copy()
            for sym in SYMBOLS
            if live_prices[sym]["last"] > 0
        }
        if prices:
            unrealized = paper.get_unrealized_pnl(
                {sym: v["last"] for sym, v in prices.items()}
            )
            update_state({
                "status": "running",
                "balance": round(paper.balance, 2),
                "unrealized_pnl": unrealized,
                "open_trades": [_t(t) for t in paper.open_trades],
                "closed_trades": [_t(t) for t in paper.closed_trades],
                "stats": paper.get_stats(),
                "last_signals": _last_signals,
                "prices": prices,
            })
            await broadcast({"type": "tick", **_current_state()})
        await asyncio.sleep(BROADCAST_INTERVAL)


# ── SL / TP monitor ───────────────────────────────────────────────────────────

async def sltp_monitor_loop():
    """Check SL/TP on every tick for fast exits."""
    while _running:
        for sym in SYMBOLS:
            price = live_prices[sym]["last"]
            if price <= 0:
                continue
            closed = paper.update_trades(sym, price)
            for t in closed:
                risk.update_daily_pnl(t.pnl)
                win = t.pnl > 0
                label = "WIN" if win else "LOSS"
                color = "green" if win else "red"
                console.print(
                    f"[{color}]CLOSED [{label}] {sym} {t.direction.upper()} "
                    f"@ {t.exit_price:.2f} | P&L: ${t.pnl:+.2f} ({t.pnl_pct:+.2f}%)[/{color}]"
                )
                push_trade_event({
                    "type": "exit",
                    "symbol": sym,
                    "direction": t.direction,
                    "exit_price": t.exit_price,
                    "entry_price": t.entry_price,
                    "pnl": round(t.pnl, 2),
                    "pnl_pct": round(t.pnl_pct, 2),
                    "reason": "TP hit" if win else "SL hit",
                    "ts": datetime.now(timezone.utc).isoformat(),
                })
        await asyncio.sleep(SLTP_INTERVAL)


# ── Strategy scan loop ────────────────────────────────────────────────────────

async def strategy_loop():
    """Scan for setups every STRATEGY_INTERVAL seconds — only when trading is active."""
    global _last_signals

    while _running:
        for sym in SYMBOLS:
            try:
                daily_hit = risk.is_daily_limit_hit(paper.balance)
                if daily_hit and _is_trading():
                    console.print("[red]Daily loss limit hit — no new trades[/red]")

                df = await fetcher.fetch_ohlcv(sym, timeframe=config.DEFAULT_TIMEFRAME, limit=300)
                if df.empty:
                    continue

                live = live_prices[sym]["last"]
                if live > 0:
                    df.iloc[-1, df.columns.get_loc("close")] = live

                # Fetch HTF data for MTF confluence
                df_4h = await fetcher.fetch_ohlcv(sym, timeframe="4h", limit=100)
                df_1h = await fetcher.fetch_ohlcv(sym, timeframe="1h", limit=200)

                sig = strategy.generate_signal(
                    df, sym,
                    df_4h=df_4h if not df_4h.empty else None,
                    df_1h=df_1h if not df_1h.empty else None,
                )
                session = get_session()
                _last_signals[sym] = {
                    "direction": sig.direction,
                    "confidence": round(sig.confidence, 2),
                    "reason": sig.reason,
                    "entry": sig.entry_price,
                    "sl": sig.stop_loss,
                    "tp": sig.take_profit,
                    "rr": sig.risk_reward,
                    "session": session.name,
                    "session_active": session.active,
                    "mtf_bias": sig.metadata.get("mtf_bias", "unknown") if sig.metadata else "unknown",
                }

                # Only execute trades when user has pressed Start
                if _is_trading() and not daily_hit:
                    valid, reason = risk.validate_signal(
                        sig, paper.balance, len(paper.open_trades)
                    )
                    if valid:
                        size_usd = risk.calculate_position_size(sig, paper.balance)
                        trade = paper.open_trade(sig, size_usd)
                        if trade:
                            console.print(
                                f"[cyan]ENTRY {sym} {trade.direction.upper()} "
                                f"@ {trade.entry_price:.2f} | SL:{trade.stop_loss:.2f} "
                                f"TP:{trade.take_profit:.2f} | ${trade.size_usd:.0f} "
                                f"| {trade.confidence:.0%}[/cyan]"
                            )
                            push_trade_event({
                                "type": "entry",
                                "symbol": sym,
                                "direction": trade.direction,
                                "price": trade.entry_price,
                                "sl": trade.stop_loss,
                                "tp": trade.take_profit,
                                "size_usd": trade.size_usd,
                                "rr": sig.risk_reward,
                                "reason": trade.reason,
                                "ts": datetime.now(timezone.utc).isoformat(),
                            })
                    elif sig.direction != "none":
                        console.print(f"[dim]{sym} {sig.direction} skipped: {reason}[/dim]")

            except Exception as e:
                console.print(f"[red]Strategy error {sym}: {e}[/red]")

        _print_status()
        await asyncio.sleep(STRATEGY_INTERVAL)


# ── Console status ────────────────────────────────────────────────────────────

def _print_status():
    prices = {sym: live_prices[sym]["last"] for sym in SYMBOLS if live_prices[sym]["last"] > 0}
    upnl = paper.get_unrealized_pnl(prices)
    stats = paper.get_stats()
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_row("[dim]Time[/dim]", datetime.now(timezone.utc).strftime("%H:%M:%S UTC"))
    for sym, price in prices.items():
        chg = live_prices[sym].get("change_pct", 0)
        color = "green" if chg >= 0 else "red"
        t.add_row(f"[dim]{sym}[/dim]", f"${price:,.2f} [{color}]{chg:+.2f}%[/{color}]")
    t.add_row("[dim]Balance[/dim]", f"${paper.balance:,.2f}")
    t.add_row("[dim]Unrealized[/dim]", f"${upnl:+.2f}")
    t.add_row("[dim]Open[/dim]", str(len(paper.open_trades)))
    t.add_row("[dim]Win Rate[/dim]", f"{stats.get('win_rate', 0):.1f}%")
    console.print(t)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _t(trade) -> dict:
    from dataclasses import asdict
    d = asdict(trade)
    d["status"] = d["status"].value if hasattr(d["status"], "value") else d["status"]
    return d


def _current_state() -> dict:
    from dashboard.app import _state
    return _state


# ── Dashboard server ──────────────────────────────────────────────────────────

def run_dashboard():
    uvicorn.run(
        dashboard_app,
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
        log_level="warning",
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def handle_shutdown(sig, frame):
    global _running
    console.print("\n[yellow]Shutting down...[/yellow]")
    _running = False
    sys.exit(0)


async def main():
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    console.print("[bold cyan]Trading Bot Starting...[/bold cyan]")
    console.print(f"  Mode:     [yellow]PAPER TRADING[/yellow]")
    console.print(f"  Balance:  [green]${paper.balance:,.2f}[/green]")
    console.print(f"  Dashboard:[blue] http://localhost:{config.DASHBOARD_PORT}[/blue]\n")

    # Start dashboard in background thread
    dash_thread = threading.Thread(target=run_dashboard, daemon=True)
    dash_thread.start()
    await asyncio.sleep(1)

    # Wire start/stop console logs
    def _on_start_cb():
        console.print("[green bold]▶ Trading STARTED by user[/green bold]")
        push_trade_event({"type": "control", "msg": "Trading started", "ts": datetime.now(timezone.utc).isoformat()})

    def _on_stop_cb():
        console.print("[yellow bold]⏹ Trading STOPPED by user[/yellow bold]")
        push_trade_event({"type": "control", "msg": "Trading stopped", "ts": datetime.now(timezone.utc).isoformat()})

    on_start(_on_start_cb)
    on_stop(_on_stop_cb)

    def _on_risk_cb(updated: dict):
        risk.update_config(**updated)
        if "leverage" in updated:
            paper.leverage = int(updated["leverage"])
        console.print(f"[yellow]Risk updated: {updated}[/yellow]")

    on_risk_update(_on_risk_cb)

    # Push initial risk config to dashboard state
    from dataclasses import asdict
    update_state({"risk": {k: v for k, v in vars(risk.cfg).items()}})

    # Start real-time price streams
    await start_streams()
    await asyncio.sleep(2)  # let first prices arrive

    # Run all loops concurrently
    await asyncio.gather(
        price_broadcast_loop(),
        sltp_monitor_loop(),
        strategy_loop(),
    )

    await fetcher.close()


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
