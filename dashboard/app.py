"""
FastAPI Dashboard — Real-time prices, live entry/exit alerts, trade feed.
"""

import asyncio
import json
from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Trading Bot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Shared state ──────────────────────────────────────────────────────────────

_state: dict = {
    "status": "stopped",
    "trading_active": False,
    "balance": 0,
    "unrealized_pnl": 0,
    "open_trades": [],
    "closed_trades": [],
    "stats": {},
    "prices": {},
    "last_signals": {},
    "trade_log": [],
    "last_update": None,
    "risk": {
        "max_risk_per_trade_pct": 1.0,
        "max_daily_loss_pct": 3.0,
        "max_open_trades": 3,
        "min_risk_reward": 2.0,
        "max_position_pct": 10.0,
        "leverage": 1000,
        "btc_max_size_usd": 5000.0,
    },
}
_ws_clients: list[WebSocket] = []

# Callback hooks set by main.py
_on_start: list = []
_on_stop:  list = []
_on_risk_update: list = []


def on_start(fn):
    _on_start.append(fn)

def on_stop(fn):
    _on_stop.append(fn)

def on_risk_update(fn):
    _on_risk_update.append(fn)


def update_state(data: dict):
    _state.update(data)
    _state["last_update"] = datetime.now(timezone.utc).isoformat()


def push_trade_event(event: dict):
    """Add an entry or exit event to the trade log (keep last 100)."""
    _state["trade_log"].insert(0, event)
    _state["trade_log"] = _state["trade_log"][:100]


async def broadcast(data: dict):
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(json.dumps(data, default=str))
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# ── REST ──────────────────────────────────────────────────────────────────────

@app.get("/api/state")
def get_state():
    return _state

@app.get("/api/prices")
def get_prices():
    return _state.get("prices", {})

@app.get("/api/trades/open")
def get_open_trades():
    return _state.get("open_trades", [])

@app.get("/api/trades/closed")
def get_closed_trades():
    return _state.get("closed_trades", [])

@app.get("/api/log")
def get_log():
    return _state.get("trade_log", [])

@app.get("/api/stats")
def get_stats():
    return _state.get("stats", {})

@app.post("/api/start")
async def start_trading():
    if _state["trading_active"]:
        return {"ok": False, "msg": "Already running"}
    _state["trading_active"] = True
    _state["status"] = "running"
    for fn in _on_start:
        fn()
    await broadcast({"type": "control", "trading_active": True})
    return {"ok": True, "msg": "Trading started"}

@app.post("/api/stop")
async def stop_trading():
    if not _state["trading_active"]:
        return {"ok": False, "msg": "Already stopped"}
    _state["trading_active"] = False
    _state["status"] = "stopped"
    for fn in _on_stop:
        fn()
    await broadcast({"type": "control", "trading_active": False})
    return {"ok": True, "msg": "Trading stopped"}


@app.get("/api/risk")
def get_risk():
    return _state["risk"]


@app.post("/api/risk")
async def set_risk(payload: dict):
    allowed = {
        "max_risk_per_trade_pct", "max_daily_loss_pct", "max_open_trades",
        "min_risk_reward", "max_position_pct", "leverage", "btc_max_size_usd",
    }
    updated = {}
    for k, v in payload.items():
        if k not in allowed:
            continue
        try:
            # int fields
            updated[k] = int(v) if k in ("max_open_trades", "leverage") else float(v)
        except (ValueError, TypeError):
            return {"ok": False, "msg": f"Invalid value for {k}: {v}"}

    if not updated:
        return {"ok": False, "msg": "No valid fields"}

    _state["risk"].update(updated)
    for fn in _on_risk_update:
        fn(updated)
    await broadcast({"type": "risk_update", "risk": _state["risk"]})
    return {"ok": True, "risk": _state["risk"]}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    await websocket.send_text(json.dumps(_state, default=str))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ── HTML Dashboard ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>XAU/USD & BTC — Live Trading</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',monospace;font-size:14px}
header{background:#161b22;padding:14px 24px;border-bottom:1px solid #30363d;
       display:flex;align-items:center;gap:12px;flex-wrap:wrap}
h1{font-size:1.1rem;color:#58a6ff;flex:1}
.badge{padding:3px 10px;border-radius:10px;font-size:.7rem;font-weight:700;
       background:#1f6feb33;color:#58a6ff}
#conn-dot{width:9px;height:9px;border-radius:50%;background:#f85149;transition:.3s}
#conn-dot.on{background:#3fb950;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

/* Ticker bar */
#ticker{display:flex;gap:24px;padding:12px 24px;background:#0d1117;
        border-bottom:1px solid #21262d;flex-wrap:wrap}
.tick-sym{display:flex;flex-direction:column;gap:2px;min-width:160px}
.tick-name{font-size:.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em}
.tick-price{font-size:1.5rem;font-weight:700;font-family:monospace;
            transition:color .3s}
.tick-price.flash-up{color:#3fb950}
.tick-price.flash-dn{color:#f85149}
.tick-chg{font-size:.75rem;margin-top:2px}
.tick-spread{font-size:.7rem;color:#8b949e}
.tick-ts{font-size:.65rem;color:#484f58;margin-top:2px}

/* Stats cards */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));
       gap:12px;padding:16px 24px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}
.card h3{font-size:.65rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
.card .val{font-size:1.3rem;font-weight:700}
.pos{color:#3fb950}.neg{color:#f85149}.neu{color:#e6edf3}

/* Sections */
.section{padding:0 24px 20px}
.section h2{font-size:.85rem;color:#58a6ff;text-transform:uppercase;
            letter-spacing:.08em;margin-bottom:10px;padding-top:16px}
table{width:100%;border-collapse:collapse;font-size:.8rem}
th{color:#8b949e;text-align:left;padding:7px 8px;border-bottom:1px solid #30363d;
   font-size:.7rem;text-transform:uppercase}
td{padding:7px 8px;border-bottom:1px solid #21262d;vertical-align:middle}
.buy{color:#3fb950;font-weight:700}.sell{color:#f85149;font-weight:700}
.dir-badge{padding:2px 8px;border-radius:4px;font-size:.7rem;font-weight:700}
.dir-badge.buy{background:#3fb95022}.dir-badge.sell{background:#f8514922}
.rr{color:#e3b341;font-weight:600}
.conf-bar{background:#21262d;border-radius:3px;height:6px;width:60px;display:inline-block;vertical-align:middle;margin-left:6px}
.conf-fill{background:#58a6ff;height:100%;border-radius:3px}

/* Trade log */
#log-feed{max-height:280px;overflow-y:auto;display:flex;flex-direction:column;gap:6px}
.log-item{display:flex;align-items:flex-start;gap:10px;padding:10px 12px;
          border-radius:6px;font-size:.8rem;animation:slide-in .3s ease}
@keyframes slide-in{from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:none}}
.log-item.entry{background:#1f6feb15;border-left:3px solid #58a6ff}
.log-item.exit-win{background:#3fb95015;border-left:3px solid #3fb950}
.log-item.exit-loss{background:#f8514915;border-left:3px solid #f85149}
.log-icon{font-size:1rem;flex-shrink:0}
.log-body{flex:1}
.log-title{font-weight:700;margin-bottom:2px}
.log-detail{color:#8b949e;font-size:.75rem}
.log-time{font-size:.65rem;color:#484f58;flex-shrink:0;padding-top:2px}

/* Signal panel */
#signal-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.sig-card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}
.sig-card.buy-sig{border-color:#3fb95055}
.sig-card.sell-sig{border-color:#f8514955}
.sig-sym{font-weight:700;font-size:.9rem;margin-bottom:8px;display:flex;
         align-items:center;justify-content:space-between}
.sig-row{display:flex;justify-content:space-between;padding:3px 0;
         border-bottom:1px solid #21262d;font-size:.78rem}
.sig-row:last-child{border:none}
.sig-label{color:#8b949e}
.sig-val{font-weight:600}
.sig-reason{font-size:.7rem;color:#8b949e;margin-top:6px;padding-top:6px;
            border-top:1px solid #21262d}

/* Risk panel */
#risk-overlay{display:none;position:fixed;inset:0;background:#0008;z-index:100;align-items:center;justify-content:center}
#risk-overlay.open{display:flex}
#risk-modal{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:28px;
            width:min(520px,95vw);max-height:90vh;overflow-y:auto}
#risk-modal h2{font-size:1rem;color:#58a6ff;margin-bottom:20px;display:flex;
               align-items:center;justify-content:space-between}
.risk-group{margin-bottom:18px}
.risk-group label{display:block;font-size:.75rem;color:#8b949e;text-transform:uppercase;
                  letter-spacing:.06em;margin-bottom:6px}
.risk-row{display:flex;align-items:center;gap:10px}
.risk-row input[type=range]{flex:1;accent-color:#58a6ff;height:4px}
.risk-num{width:72px;background:#0d1117;border:1px solid #30363d;border-radius:6px;
          color:#e6edf3;padding:6px 8px;font-size:.85rem;text-align:right}
.risk-unit{font-size:.75rem;color:#8b949e;min-width:20px}
.risk-hint{font-size:.7rem;color:#484f58;margin-top:4px}
.risk-save{width:100%;padding:10px;margin-top:8px;background:#1f6feb;border:none;
           border-radius:6px;color:#fff;font-weight:700;font-size:.9rem;cursor:pointer;transition:.2s}
.risk-save:hover{background:#388bfd}
.risk-save:disabled{background:#21262d;color:#8b949e;cursor:not-allowed}
.risk-tags{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}
.risk-tag{padding:3px 10px;border-radius:10px;font-size:.7rem;background:#21262d;color:#8b949e}
.risk-tag.warn{background:#bb800033;color:#e3b341}
.risk-tag.good{background:#3fb95022;color:#3fb950}

/* Responsive */
@media(max-width:600px){
  .tick-price{font-size:1.1rem}
  .cards{grid-template-columns:repeat(2,1fr)}
}
</style>
</head>
<body>

<header>
  <div id="conn-dot"></div>
  <h1>XAU/USD &amp; BTC — Paper Trading</h1>
  <span class="badge">PAPER MODE</span>
  <button id="trade-btn" onclick="toggleTrading()" style="
    margin-left:auto;padding:8px 22px;border-radius:6px;border:none;
    font-size:.85rem;font-weight:700;cursor:pointer;transition:.2s;
    background:#238636;color:#fff;letter-spacing:.03em">
    ▶ START TRADING
  </button>
  <span id="trade-status" style="font-size:.75rem;color:#8b949e;padding:0 8px">Idle</span>
  <span id="last-upd" style="color:#8b949e;font-size:.7rem"></span>
  <button onclick="openRisk()" title="Risk Settings" style="
    margin-left:8px;padding:7px 14px;border-radius:6px;border:1px solid #30363d;
    background:#21262d;color:#8b949e;font-size:.85rem;cursor:pointer;transition:.2s"
    onmouseover="this.style.borderColor='#58a6ff'" onmouseout="this.style.borderColor='#30363d'">
    ⚙ Risk
  </button>
</header>

<!-- Risk Management Modal -->
<div id="risk-overlay" onclick="if(event.target===this)closeRisk()">
<div id="risk-modal">
  <h2>Risk Management
    <button onclick="closeRisk()" style="background:none;border:none;color:#8b949e;
      font-size:1.2rem;cursor:pointer;padding:0 4px">✕</button>
  </h2>

  <div class="risk-group">
    <label>Risk Per Trade</label>
    <div class="risk-row">
      <input type="range" id="r-risk-pct" min="0.1" max="5" step="0.1"
             oninput="syncNum(this,'n-risk-pct')">
      <input type="number" id="n-risk-pct" class="risk-num" min="0.1" max="5" step="0.1"
             oninput="syncRange(this,'r-risk-pct')">
      <span class="risk-unit">%</span>
    </div>
    <div class="risk-hint">% of account balance risked on each trade</div>
  </div>

  <div class="risk-group">
    <label>Max Daily Loss</label>
    <div class="risk-row">
      <input type="range" id="r-daily-loss" min="0.5" max="10" step="0.5"
             oninput="syncNum(this,'n-daily-loss')">
      <input type="number" id="n-daily-loss" class="risk-num" min="0.5" max="10" step="0.5"
             oninput="syncRange(this,'r-daily-loss')">
      <span class="risk-unit">%</span>
    </div>
    <div class="risk-hint">Bot stops trading when daily drawdown hits this limit</div>
  </div>

  <div class="risk-group">
    <label>Max Open Trades</label>
    <div class="risk-row">
      <input type="range" id="r-max-trades" min="1" max="10" step="1"
             oninput="syncNum(this,'n-max-trades')">
      <input type="number" id="n-max-trades" class="risk-num" min="1" max="10" step="1"
             oninput="syncRange(this,'r-max-trades')">
      <span class="risk-unit"></span>
    </div>
    <div class="risk-hint">Maximum simultaneous open positions across all symbols</div>
  </div>

  <div class="risk-group">
    <label>Minimum R:R Ratio</label>
    <div class="risk-row">
      <input type="range" id="r-min-rr" min="1" max="5" step="0.1"
             oninput="syncNum(this,'n-min-rr')">
      <input type="number" id="n-min-rr" class="risk-num" min="1" max="5" step="0.1"
             oninput="syncRange(this,'r-min-rr')">
      <span class="risk-unit">:1</span>
    </div>
    <div class="risk-hint">Signals with lower R:R will be skipped</div>
  </div>

  <div class="risk-group">
    <label>Max Position Size</label>
    <div class="risk-row">
      <input type="range" id="r-max-pos" min="5" max="50" step="1"
             oninput="syncNum(this,'n-max-pos')">
      <input type="number" id="n-max-pos" class="risk-num" min="5" max="50" step="1"
             oninput="syncRange(this,'r-max-pos')">
      <span class="risk-unit">%</span>
    </div>
    <div class="risk-hint">Single position cap as % of balance (overrides risk % calculation)</div>
  </div>

  <div class="risk-group">
    <label>Leverage</label>
    <div class="risk-row">
      <select id="n-leverage" class="risk-num" style="width:120px;text-align:left"
              onchange="updateRiskTags()">
        <option value="1">1:1 (No leverage)</option>
        <option value="10">1:10</option>
        <option value="50">1:50</option>
        <option value="100">1:100</option>
        <option value="200">1:200</option>
        <option value="500">1:500</option>
        <option value="1000" selected>1:1000</option>
      </select>
    </div>
    <div class="risk-hint">Margin multiplier — 1:1000 means $1 controls $1,000 in market exposure</div>
  </div>

  <div class="risk-group">
    <label>BTC Max Position (USD)</label>
    <div class="risk-row">
      <input type="range" id="r-btc-max" min="100" max="20000" step="100"
             oninput="syncNum(this,'n-btc-max')">
      <input type="number" id="n-btc-max" class="risk-num" min="100" max="20000" step="100"
             oninput="syncRange(this,'r-btc-max')">
      <span class="risk-unit">$</span>
    </div>
    <div class="risk-hint">Hard cap on BTC/USDT position size in USD</div>
  </div>

  <div class="risk-tags" id="risk-tags"></div>

  <button class="risk-save" id="risk-save-btn" onclick="saveRisk()">Save Settings</button>
  <div id="risk-msg" style="text-align:center;font-size:.78rem;margin-top:10px;height:18px"></div>
</div>
</div>

<!-- Live Ticker -->
<div id="ticker">
  <div class="tick-sym" id="tick-btc">
    <span class="tick-name">BTC / USDT</span>
    <span class="tick-price neu" id="btc-price">—</span>
    <span class="tick-chg neu" id="btc-chg"></span>
    <span class="tick-spread" id="btc-spread"></span>
    <span class="tick-ts" id="btc-ts"></span>
  </div>
  <div class="tick-sym" id="tick-xau">
    <span class="tick-name">XAU / USD</span>
    <span class="tick-price neu" id="xau-price">—</span>
    <span class="tick-chg neu" id="xau-chg"></span>
    <span class="tick-spread" id="xau-spread"></span>
    <span class="tick-ts" id="xau-ts"></span>
  </div>
</div>

<!-- Stats Cards -->
<div class="cards">
  <div class="card"><h3>Balance</h3><div class="val neu" id="s-balance">—</div></div>
  <div class="card"><h3>Unrealized P&amp;L</h3><div class="val neu" id="s-upnl">—</div></div>
  <div class="card"><h3>Realized P&amp;L</h3><div class="val neu" id="s-rpnl">—</div></div>
  <div class="card"><h3>Win Rate</h3><div class="val neu" id="s-wr">—</div></div>
  <div class="card"><h3>Total Trades</h3><div class="val neu" id="s-tt">—</div></div>
  <div class="card"><h3>Open Trades</h3><div class="val neu" id="s-ot">—</div></div>
  <div class="card"><h3>Equity Δ</h3><div class="val neu" id="s-eq">—</div></div>
</div>

<!-- Live Signals -->
<div class="section">
  <h2>Latest Signals</h2>
  <div id="signal-grid"><p style="color:#8b949e;font-size:.8rem">Waiting for first scan…</p></div>
</div>

<!-- Open Trades -->
<div class="section">
  <h2>Open Trades</h2>
  <table>
    <thead><tr>
      <th>ID</th><th>Symbol</th><th>Direction</th><th>Entry</th>
      <th>SL</th><th>TP</th><th>Size USD</th><th>R:R</th><th>Confidence</th><th>Reason</th>
    </tr></thead>
    <tbody id="open-tbody"><tr><td colspan="10" style="color:#8b949e">No open trades</td></tr></tbody>
  </table>
</div>

<!-- Trade Log (Entry / Exit feed) -->
<div class="section">
  <h2>Trade Activity Log</h2>
  <div id="log-feed"><p style="color:#8b949e;font-size:.8rem;padding:8px 0">No activity yet…</p></div>
</div>

<!-- Closed Trades -->
<div class="section">
  <h2>Closed Trades</h2>
  <table>
    <thead><tr>
      <th>ID</th><th>Symbol</th><th>Dir</th><th>Entry</th>
      <th>Exit</th><th>P&amp;L</th><th>P&amp;L %</th><th>Closed At</th>
    </tr></thead>
    <tbody id="closed-tbody"><tr><td colspan="8" style="color:#8b949e">No closed trades</td></tr></tbody>
  </table>
</div>

<script>
// ── Trading control ───────────────────────────────────────────────────────────
let _tradingActive = false;

function setTradingUI(active) {
  _tradingActive = active;
  const btn = document.getElementById('trade-btn');
  const status = document.getElementById('trade-status');
  if (active) {
    btn.textContent = '⏹ STOP TRADING';
    btn.style.background = '#b62324';
    status.textContent = 'Trading LIVE';
    status.style.color = '#3fb950';
  } else {
    btn.textContent = '▶ START TRADING';
    btn.style.background = '#238636';
    status.textContent = 'Idle';
    status.style.color = '#8b949e';
  }
}

async function toggleTrading() {
  const btn = document.getElementById('trade-btn');
  btn.disabled = true;
  try {
    const endpoint = _tradingActive ? '/api/stop' : '/api/start';
    const r = await fetch(endpoint, { method: 'POST' });
    const data = await r.json();
    if (data.ok) setTradingUI(!_tradingActive);
    else alert(data.msg);
  } catch(e) {
    alert('Connection error: ' + e);
  } finally {
    btn.disabled = false;
  }
}

// ── WebSocket connection ───────────────────────────────────────────────────────
const dot = document.getElementById('conn-dot');
let ws, reconnectTimer;
let prevPrices = {};

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen  = () => { dot.classList.add('on'); clearTimeout(reconnectTimer); };
  ws.onclose = () => { dot.classList.remove('on'); reconnectTimer = setTimeout(connect, 3000); };
  ws.onerror = () => ws.close();
  ws.onmessage = e => {
    try {
      const d = JSON.parse(e.data);
      if (d.type === 'control') { setTradingUI(d.trading_active); return; }
      if (d.type === 'risk_update') { applyRiskUpdate(d.risk); return; }
      if (d.risk) { _risk = d.risk; }
      render(d);
    } catch(err) { console.error(err); }
  };
}
connect();

// ── Helpers ───────────────────────────────────────────────────────────────────
const f  = (n, d=2) => n == null ? '—' : Number(n).toFixed(d);
const f0 = n => f(n, 0);
const fSign = (n, d=2) => n == null ? '—' : (n >= 0 ? '+' : '') + f(n, d);
const clr = n => n > 0 ? 'pos' : n < 0 ? 'neg' : 'neu';
const ts  = iso => iso ? new Date(iso).toLocaleTimeString() : '—';

function flashPrice(el, prev, curr) {
  if (!prev || prev === curr) return;
  el.classList.remove('flash-up','flash-dn');
  void el.offsetWidth;
  el.classList.add(curr > prev ? 'flash-up' : 'flash-dn');
  setTimeout(() => el.classList.remove('flash-up','flash-dn'), 600);
}

// ── Render ────────────────────────────────────────────────────────────────────
function render(d) {
  if (d.trading_active !== undefined) setTradingUI(d.trading_active);
  renderPrices(d.prices || {});
  renderStats(d);
  renderSignals(d.last_signals || {});
  renderOpenTrades(d.open_trades || []);
  renderClosedTrades(d.closed_trades || []);
  renderLog(d.trade_log || []);
  if (d.last_update) {
    document.getElementById('last-upd').textContent = 'Updated ' + ts(d.last_update);
  }
}

function renderPrices(prices) {
  // BTC
  if (prices['BTC/USDT'] != null) {
    const curr = prices['BTC/USDT'];
    const prev = prevPrices['BTC/USDT'];
    const el = document.getElementById('btc-price');
    el.textContent = '$' + f0(curr.last);
    flashPrice(el, prev?.last, curr.last);
    const chgEl = document.getElementById('btc-chg');
    const chg = curr.change_pct || 0;
    chgEl.textContent = fSign(chg, 2) + '% (24h)';
    chgEl.className = 'tick-chg ' + clr(chg);
    document.getElementById('btc-spread').textContent =
      `Bid ${f0(curr.bid)}  Ask ${f0(curr.ask)}`;
    document.getElementById('btc-ts').textContent =
      curr.ts ? new Date(curr.ts * 1000).toLocaleTimeString() : '';
  }
  // XAU
  if (prices['XAU/USD'] != null) {
    const curr = prices['XAU/USD'];
    const prev = prevPrices['XAU/USD'];
    const el = document.getElementById('xau-price');
    el.textContent = '$' + f(curr.last, 2);
    flashPrice(el, prev?.last, curr.last);
    const chgEl = document.getElementById('xau-chg');
    const chg = curr.change_pct || 0;
    chgEl.textContent = fSign(chg, 2) + '%';
    chgEl.className = 'tick-chg ' + clr(chg);
    document.getElementById('xau-spread').textContent =
      `Bid ${f(curr.bid, 2)}  Ask ${f(curr.ask, 2)}`;
    document.getElementById('xau-ts').textContent =
      curr.ts ? new Date(curr.ts * 1000).toLocaleTimeString() : '';
  }
  prevPrices = JSON.parse(JSON.stringify(prices));
}

function renderStats(d) {
  const s = d.stats || {};
  document.getElementById('s-balance').textContent = '$' + f(d.balance);
  const upnl = d.unrealized_pnl || 0;
  const upEl = document.getElementById('s-upnl');
  upEl.textContent = '$' + fSign(upnl); upEl.className = 'val ' + clr(upnl);
  const rpnl = s.total_pnl || 0;
  const rpEl = document.getElementById('s-rpnl');
  rpEl.textContent = '$' + fSign(rpnl); rpEl.className = 'val ' + clr(rpnl);
  document.getElementById('s-wr').textContent = f(s.win_rate, 1) + '%';
  document.getElementById('s-tt').textContent = s.total_trades || 0;
  document.getElementById('s-ot').textContent = (d.open_trades || []).length;
  const eq = s.equity_change_pct || 0;
  const eqEl = document.getElementById('s-eq');
  eqEl.textContent = fSign(eq, 2) + '%'; eqEl.className = 'val ' + clr(eq);
}

function renderSignals(signals) {
  const grid = document.getElementById('signal-grid');
  const syms = Object.keys(signals);
  if (!syms.length) return;
  grid.innerHTML = syms.map(sym => {
    const s = signals[sym];
    if (s.direction === 'none') {
      return `<div class="sig-card">
        <div class="sig-sym">${sym} <span style="color:#8b949e;font-size:.75rem">No Setup</span></div>
        <div class="sig-reason">${s.reason || ''}</div>
      </div>`;
    }
    const cls = s.direction === 'buy' ? 'buy-sig' : 'sell-sig';
    const dirCls = s.direction === 'buy' ? 'buy' : 'sell';
    return `<div class="sig-card ${cls}">
      <div class="sig-sym">${sym}
        <span class="dir-badge ${dirCls}">${s.direction.toUpperCase()}</span>
      </div>
      <div class="sig-row"><span class="sig-label">Entry</span><span class="sig-val">$${f(s.entry)}</span></div>
      <div class="sig-row"><span class="sig-label">Stop Loss</span><span class="sig-val neg">$${f(s.sl)}</span></div>
      <div class="sig-row"><span class="sig-label">Take Profit</span><span class="sig-val pos">$${f(s.tp)}</span></div>
      <div class="sig-row"><span class="sig-label">R:R</span><span class="sig-val rr">${f(s.rr, 1)}</span></div>
      <div class="sig-row"><span class="sig-label">Confidence</span>
        <span class="sig-val">${f(s.confidence * 100, 0)}%
          <span class="conf-bar"><span class="conf-fill" style="width:${s.confidence*100}%"></span></span>
        </span>
      </div>
      <div class="sig-row"><span class="sig-label">Session</span>
        <span class="sig-val" style="color:${s.session_active?'#3fb950':'#8b949e'}">${s.session||'—'} ${s.session_active?'●':''}</span>
      </div>
      <div class="sig-row"><span class="sig-label">MTF Bias</span>
        <span class="sig-val" style="color:${s.mtf_bias==='bullish'?'#3fb950':s.mtf_bias==='bearish'?'#f85149':'#8b949e'}">${s.mtf_bias||'—'}</span>
      </div>
      <div class="sig-reason">${s.reason || ''}</div>
    </div>`;
  }).join('');
}

function renderOpenTrades(trades) {
  const tb = document.getElementById('open-tbody');
  if (!trades.length) {
    tb.innerHTML = '<tr><td colspan="10" style="color:#8b949e">No open trades</td></tr>';
    return;
  }
  tb.innerHTML = trades.map(t => {
    const rr = t.stop_loss && t.entry_price ?
      (t.direction === 'buy'
        ? Math.abs(t.take_profit - t.entry_price) / Math.abs(t.entry_price - t.stop_loss)
        : Math.abs(t.entry_price - t.take_profit) / Math.abs(t.stop_loss - t.entry_price)
      ).toFixed(1) : '—';
    return `<tr>
      <td style="font-family:monospace;color:#8b949e">${t.id}</td>
      <td><strong>${t.symbol}</strong></td>
      <td><span class="dir-badge ${t.direction}">${t.direction.toUpperCase()}</span></td>
      <td>$${f(t.entry_price)}</td>
      <td class="neg">$${f(t.stop_loss)}</td>
      <td class="pos">$${f(t.take_profit)}</td>
      <td>$${f0(t.size_usd)}</td>
      <td class="rr">${rr}</td>
      <td>${f(t.confidence*100,0)}%</td>
      <td style="color:#8b949e;font-size:.72rem;max-width:180px;overflow:hidden">${t.reason}</td>
    </tr>`;
  }).join('');
}

function renderClosedTrades(trades) {
  const tb = document.getElementById('closed-tbody');
  const recent = trades.slice().reverse().slice(0, 30);
  if (!recent.length) {
    tb.innerHTML = '<tr><td colspan="8" style="color:#8b949e">No closed trades</td></tr>';
    return;
  }
  tb.innerHTML = recent.map(t => `<tr>
    <td style="font-family:monospace;color:#8b949e">${t.id}</td>
    <td><strong>${t.symbol}</strong></td>
    <td><span class="dir-badge ${t.direction}">${t.direction.toUpperCase()}</span></td>
    <td>$${f(t.entry_price)}</td>
    <td>$${f(t.exit_price)}</td>
    <td class="${clr(t.pnl)}"><strong>$${fSign(t.pnl)}</strong></td>
    <td class="${clr(t.pnl_pct)}">${fSign(t.pnl_pct)}%</td>
    <td style="font-size:.72rem">${t.closed_at ? new Date(t.closed_at).toLocaleString() : '—'}</td>
  </tr>`).join('');
}

// ── Risk Management ───────────────────────────────────────────────────────────
let _risk = {};

function openRisk() {
  populateRisk(_risk);
  document.getElementById('risk-overlay').classList.add('open');
  document.getElementById('risk-msg').textContent = '';
}
function closeRisk() {
  document.getElementById('risk-overlay').classList.remove('open');
}
function syncNum(range, numId) {
  document.getElementById(numId).value = range.value;
  updateRiskTags();
}
function syncRange(num, rangeId) {
  document.getElementById(rangeId).value = num.value;
  updateRiskTags();
}

function populateRisk(r) {
  if (!r || !Object.keys(r).length) return;
  const set = (rangeId, numId, val) => {
    document.getElementById(rangeId).value = val;
    document.getElementById(numId).value = val;
  };
  set('r-risk-pct',   'n-risk-pct',   r.max_risk_per_trade_pct ?? 1);
  set('r-daily-loss', 'n-daily-loss',  r.max_daily_loss_pct ?? 3);
  set('r-max-trades', 'n-max-trades',  r.max_open_trades ?? 3);
  set('r-min-rr',     'n-min-rr',      r.min_risk_reward ?? 2);
  set('r-max-pos',    'n-max-pos',     r.max_position_pct ?? 10);
  set('r-btc-max',    'n-btc-max',     r.btc_max_size_usd ?? 5000);
  const lev = document.getElementById('n-leverage');
  if (lev) lev.value = r.leverage ?? 1000;
  updateRiskTags();
}

function updateRiskTags() {
  const riskPct = parseFloat(document.getElementById('n-risk-pct').value);
  const dailyLoss = parseFloat(document.getElementById('n-daily-loss').value);
  const maxTrades = parseInt(document.getElementById('n-max-trades').value);
  const minRR = parseFloat(document.getElementById('n-min-rr').value);
  const leverage = parseInt(document.getElementById('n-leverage').value) || 1;
  const tags = [];
  const bal = parseFloat(document.getElementById('s-balance').textContent.replace(/[$,]/g,'')) || 150;

  const perTrade = (bal * riskPct / 100).toFixed(2);
  const exposure = (bal * leverage * riskPct / 100).toFixed(0);
  tags.push({text: `$${perTrade} margin risked / $${exposure} exposure`, cls: riskPct > 2 ? 'warn' : 'good'});

  const maxDay = (bal * dailyLoss / 100).toFixed(2);
  tags.push({text: `Max loss/day: $${maxDay}`, cls: dailyLoss > 5 ? 'warn' : 'good'});

  tags.push({text: `Up to ${maxTrades} concurrent trades`, cls: maxTrades > 5 ? 'warn' : 'good'});
  tags.push({text: `Min R:R ${minRR}:1`, cls: minRR < 1.5 ? 'warn' : 'good'});
  tags.push({text: `Leverage 1:${leverage}`, cls: leverage >= 500 ? 'warn' : 'good'});

  document.getElementById('risk-tags').innerHTML =
    tags.map(t => `<span class="risk-tag ${t.cls}">${t.text}</span>`).join('');
}

async function saveRisk() {
  const btn = document.getElementById('risk-save-btn');
  const msg = document.getElementById('risk-msg');
  btn.disabled = true;
  btn.textContent = 'Saving…';
  try {
    const payload = {
      max_risk_per_trade_pct: parseFloat(document.getElementById('n-risk-pct').value),
      max_daily_loss_pct:     parseFloat(document.getElementById('n-daily-loss').value),
      max_open_trades:        parseInt(document.getElementById('n-max-trades').value),
      min_risk_reward:        parseFloat(document.getElementById('n-min-rr').value),
      max_position_pct:       parseFloat(document.getElementById('n-max-pos').value),
      leverage:               parseInt(document.getElementById('n-leverage').value),
      btc_max_size_usd:       parseFloat(document.getElementById('n-btc-max').value),
    };
    const r = await fetch('/api/risk', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await r.json();
    if (data.ok) {
      _risk = data.risk;
      msg.style.color = '#3fb950';
      msg.textContent = '✓ Risk settings saved';
      setTimeout(() => { msg.textContent = ''; closeRisk(); }, 1200);
    } else {
      msg.style.color = '#f85149';
      msg.textContent = '✗ ' + data.msg;
    }
  } catch(e) {
    msg.style.color = '#f85149';
    msg.textContent = '✗ Connection error';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Save Settings';
  }
}

// Handle risk_update broadcast from server
function applyRiskUpdate(r) {
  _risk = r;
}

let _lastLogLen = 0;
function renderLog(log) {
  const feed = document.getElementById('log-feed');
  if (!log.length) return;
  if (log.length === _lastLogLen) return;
  _lastLogLen = log.length;

  feed.innerHTML = log.slice(0, 30).map(e => {
    let cls, icon, title, detail;
    if (e.type === 'control') {
      const started = e.msg && e.msg.includes('started');
      cls = 'entry'; icon = started ? '▶' : '⏹';
      title = e.msg || 'Control event'; detail = '';
    } else if (e.type === 'entry') {
      cls = 'entry';
      icon = e.direction === 'buy' ? '📈' : '📉';
      title = `${e.direction.toUpperCase()} ${e.symbol} @ $${f(e.price)}`;
      detail = `SL: $${f(e.sl)} | TP: $${f(e.tp)} | Size: $${f0(e.size_usd)} | R:R ${f(e.rr,1)} | ${e.reason}`;
    } else {
      const win = e.pnl > 0;
      cls = win ? 'exit-win' : 'exit-loss';
      icon = win ? '✅' : '❌';
      title = `CLOSED ${e.symbol} — ${win ? 'WIN' : 'LOSS'} $${fSign(e.pnl)}`;
      detail = `Exit: $${f(e.exit_price)} | P&L: ${fSign(e.pnl_pct)}% | ${e.reason || ''}`;
    }
    return `<div class="log-item ${cls}">
      <span class="log-icon">${icon}</span>
      <div class="log-body">
        <div class="log-title">${title}</div>
        <div class="log-detail">${detail}</div>
      </div>
      <span class="log-time">${e.ts ? new Date(e.ts).toLocaleTimeString() : ''}</span>
    </div>`;
  }).join('');
}
</script>
</body>
</html>"""
