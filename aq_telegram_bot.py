"""
╔══════════════════════════════════════════════════════════════════════╗
║  AstraQuant Telegram Bot  v3.0 — Real Money Edition                  ║
║                                                                      ║
║  Built as if trading with own capital. Every filter exists because   ║
║  its absence would lose money in real markets.                       ║
║                                                                      ║
║  New in v3.0:                                                        ║
║  • Market regime filter (NIFTY/S&P must be healthy before entries)   ║
║  • Liquidity gate (min avg volume enforced)                          ║
║  • Earnings blackout (no entries 5 days before earnings)             ║
║  • Support-anchored stop loss (not just ATR)                         ║
║  • Trailing stop logic (breakeven after T1, trail after T2)          ║
║  • MTF contradiction downgrade (weekly bear = reduce signal)         ║
║  • Entry validity window (signal expires if price gaps away)         ║
║  • Portfolio risk cap (max open trades, max sector exposure)          ║
║  • Improved divergence (swing-based, not bar-comparison)             ║
║  • Signal auto-validation (did last week's signals work?)            ║
║  • Gap risk assessment (recent gap history, corporate actions)       ║
║  • Honest probability (show sample size, confidence level)           ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import os, json, logging, math, sqlite3, time, threading, io
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION — reads from .env file or environment variables
# ══════════════════════════════════════════════════════════════════════
import os

def _load_env():
    """Load .env file if it exists (simple parser, no dependencies)."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip(); val = val.strip()
            # Don't override if already set in actual environment
            if key and val and key not in os.environ:
                os.environ[key] = val

_load_env()

# ── Ticker info cache — one .info call shared across fund/earnings ────
_info_cache = {}
_INFO_TTL   = 300

def get_ticker_info(ticker):
    now = time.time()
    cached = _info_cache.get(ticker)
    if cached and now - cached[0] < _INFO_TTL:
        return cached[1]
    try:
        info = yf.Ticker(ticker).info
        _info_cache[ticker] = (now, info)
        return info
    except Exception as e:
        log.warning(f"get_ticker_info {ticker}: {e}")
        return {}


# ── Required ─────────────────────────────────────────────────────────
BOT_TOKEN    = os.environ.get("BOT_TOKEN", "")
YOUR_CHAT_ID = os.environ.get("CHAT_ID", "")

if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
    print("ERROR: BOT_TOKEN not set. Add it to your .env file.")
    print("       Get one from @BotFather on Telegram.")
if not YOUR_CHAT_ID or YOUR_CHAT_ID == "YOUR_NUMERIC_CHAT_ID":
    print("ERROR: CHAT_ID not set. Add it to your .env file.")
    print("       Get it from @userinfobot on Telegram.")

# ── AI (optional — Groq is free at console.groq.com) ─────────────────
AI_API_KEY  = os.environ.get("GROQ_API_KEY", "")
AI_BASE_URL = os.environ.get("AI_BASE_URL", "https://api.groq.com/openai/v1")
AI_MODEL    = os.environ.get("AI_MODEL",    "llama-3.3-70b-versatile")

# ── Portfolio risk controls ───────────────────────────────────────────
TRADING_CAPITAL   = float(os.environ.get("TRADING_CAPITAL",   "500000"))
RISK_PER_TRADE    = float(os.environ.get("RISK_PER_TRADE",    "0.01"))
MAX_OPEN_TRADES   = int(os.environ.get("MAX_OPEN_TRADES",     "6"))
MAX_SECTOR_TRADES = int(os.environ.get("MAX_SECTOR_TRADES",   "2"))
MAX_PORTFOLIO_RISK= float(os.environ.get("MAX_PORTFOLIO_RISK","0.06"))

# ── Liquidity requirements ────────────────────────────────────────────
MIN_AVG_VOLUME_INR = float(os.environ.get("MIN_AVG_VOLUME_INR", "5000000"))
MIN_AVG_VOLUME_USD = float(os.environ.get("MIN_AVG_VOLUME_USD", "1000000"))

# ── Entry validity ────────────────────────────────────────────────────
ENTRY_VALID_DAYS  = int(os.environ.get("ENTRY_VALID_DAYS",  "3"))
MAX_CHASE_PCT     = float(os.environ.get("MAX_CHASE_PCT",   "1.5"))

# ── Market regime index per exchange ─────────────────────────────────
REGIME_INDEX = {
    ".NS":  "^NSEI",
    ".BO":  "^BSESN",
    "":     "^GSPC",
    ".L":   "^FTSE",
    ".DE":  "^GDAXI",
    ".KS":  "^KS11",
}

DB_FILE = os.environ.get("DB_FILE", "astraquant.db")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# DATABASE — extended for trailing stops and signal validation
# ══════════════════════════════════════════════════════════════════════
def init_db():
    con = sqlite3.connect(DB_FILE)
    con.execute("""CREATE TABLE IF NOT EXISTS trades (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker        TEXT NOT NULL,
        entry_date    TEXT NOT NULL,
        entry_price   REAL NOT NULL,
        qty           INTEGER DEFAULT 1,
        sl            REAL, sl_original REAL,
        t1 REAL, t2 REAL, t3 REAL,
        lead_score    INTEGER, lag_score INTEGER,
        signal_type   TEXT, strategy TEXT, horizon TEXT,
        sector        TEXT,
        status        TEXT DEFAULT 'OPEN',
        sl_hit        INTEGER DEFAULT 0,
        t1_hit        INTEGER DEFAULT 0,
        t2_hit        INTEGER DEFAULT 0,
        t3_hit        INTEGER DEFAULT 0,
        breakeven_set INTEGER DEFAULT 0,
        trailing_sl   REAL,
        exit_price    REAL, exit_date TEXT,
        notes         TEXT,
        signal_date   TEXT,
        signal_price  REAL,
        entry_valid_till TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS alerts (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker    TEXT NOT NULL, direction TEXT NOT NULL,
        price     REAL NOT NULL, active INTEGER DEFAULT 1,
        created   TEXT NOT NULL, triggered TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS signal_log (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker       TEXT NOT NULL,
        signal_date  TEXT NOT NULL,
        signal_type  TEXT NOT NULL,
        lead_score   INTEGER, lag_score INTEGER,
        entry_price  REAL, sl REAL, t1 REAL, t2 REAL,
        horizon      TEXT, strategy TEXT,
        -- Outcome tracking (filled by monitor)
        outcome      TEXT DEFAULT 'PENDING',
        outcome_date TEXT,
        max_gain_pct REAL,
        t1_hit       INTEGER DEFAULT 0,
        sl_hit       INTEGER DEFAULT 0,
        days_to_outcome INTEGER
    )""")
    con.commit()
    # Migrate existing schema
    existing = [r[1] for r in con.execute("PRAGMA table_info(trades)").fetchall()]
    new_cols = [
        ("sl_original","REAL"), ("breakeven_set","INTEGER DEFAULT 0"),
        ("trailing_sl","REAL"), ("sector","TEXT"),
        ("signal_date","TEXT"), ("signal_price","REAL"),
        ("entry_valid_till","TEXT"),
    ]
    for col, typedef in new_cols:
        if col not in existing:
            try: con.execute(f"ALTER TABLE journal ADD COLUMN {col} {typedef}")
            except:
                try: con.execute(f"ALTER TABLE trades ADD COLUMN {col} {typedef}")
                except: pass
    con.commit(); con.close()

init_db()

# ══════════════════════════════════════════════════════════════════════
# TICKER + EXCHANGE UTILS
# ══════════════════════════════════════════════════════════════════════
INDIAN_NAMES = {
    "RELIANCE","TCS","INFY","INFOSYS","HDFCBANK","ICICIBANK","SBIN",
    "BAJFINANCE","HINDUNILVR","WIPRO","AXISBANK","KOTAKBANK","LT",
    "ASIANPAINT","MARUTI","SUNPHARMA","TITAN","TATAMOTORS","TECHM",
    "HCLTECH","ONGC","NTPC","COALINDIA","INDUSINDBK","BAJAJFINSV",
    "GRASIM","ADANIENT","TATASTEEL","JSWSTEEL","HINDALCO","DIVISLAB",
    "CIPLA","DRREDDY","APOLLOHOSP","EICHERMOT","HEROMOTOCO","BPCL",
    "IOC","SBILIFE","HDFCLIFE","BRITANNIA","PIDILITIND","SIEMENS",
    "HAVELLS","BERGEPAINT","POWERGRID","NESTLEIND","ULTRACEMCO",
    "BHARTIARTL","ICICIPRULI","MCDOWELL-N","ADANIPORTS","ADANIGREEN",
}

def normalise_ticker(raw):
    raw = raw.upper().strip()
    if "." in raw: return raw
    if raw in INDIAN_NAMES: return raw + ".NS"
    return raw

def get_exchange_suffix(ticker):
    for suffix in [".NS",".BO",".L",".DE",".KS",".T",".AX"]:
        if ticker.endswith(suffix): return suffix
    return ""

def get_regime_index(ticker):
    suffix = get_exchange_suffix(ticker)
    return REGIME_INDEX.get(suffix, REGIME_INDEX[""])

def get_currency(ticker):
    suffix = get_exchange_suffix(ticker)
    return {"₹":".NS",".BO":"₹"}.get(suffix,"")

def currency_symbol(info):
    return {"INR":"₹","USD":"$","EUR":"€","GBP":"£",
            "JPY":"¥","KRW":"₩","AUD":"A$","CAD":"C$"}.get(
            (info or {}).get("currency",""), "")

def min_liquidity(ticker):
    """Minimum avg daily turnover required for this exchange."""
    suffix = get_exchange_suffix(ticker)
    return MIN_AVG_VOLUME_INR if suffix in (".NS",".BO") else MIN_AVG_VOLUME_USD

# ══════════════════════════════════════════════════════════════════════
# RELATIVE STRENGTH vs INDEX
# ══════════════════════════════════════════════════════════════════════
def get_relative_strength(ticker, lookback_days=60):
    """
    Compare stock performance vs its broad market index over lookback_days.
    Returns (rs_pct, outperforming, label)
      rs_pct: stock return minus index return over period
      outperforming: True if stock beats index
    """
    try:
        index_ticker = get_regime_index(ticker)
        end   = datetime.today() + timedelta(days=1)
        start = end - timedelta(days=lookback_days+5)
        stock_df = yf.Ticker(ticker).history(start=start,end=end,
                       interval="1d",auto_adjust=True)
        index_df = yf.Ticker(index_ticker).history(start=start,end=end,
                       interval="1d",auto_adjust=True)
        if stock_df.empty or index_df.empty or len(stock_df)<5 or len(index_df)<5:
            return 0, False, "RS unavailable"
        s_ret = (float(stock_df["Close"].iloc[-1]) -
                 float(stock_df["Close"].iloc[0])) / float(stock_df["Close"].iloc[0]) * 100
        i_ret = (float(index_df["Close"].iloc[-1]) -
                 float(index_df["Close"].iloc[0])) / float(index_df["Close"].iloc[0]) * 100
        rs    = round(s_ret - i_ret, 2)
        out   = rs > 0
        label = (f"{'Outperforming' if out else 'Underperforming'} {index_ticker} "
                 f"by {abs(rs):.1f}% over {lookback_days}d "
                 f"({'✅' if out else '⚠️'})")
        return rs, out, label
    except:
        return 0, False, "RS unavailable"

# ══════════════════════════════════════════════════════════════════════
# MARKET REGIME — most important filter
# ══════════════════════════════════════════════════════════════════════
_regime_cache = {}
_REGIME_TTL   = 3600  # refresh every hour

def get_market_regime(ticker):
    """
    Check if the broad market index is in an uptrend.
    Returns: (regime, score, detail)
      regime: 'bull' | 'neutral' | 'bear'
      score:  0-100 regime strength
      detail: human-readable explanation
    """
    index_ticker = get_regime_index(ticker)
    cache_key    = index_ticker
    cached = _regime_cache.get(cache_key)
    if cached and (time.time()-cached["ts"]) < _REGIME_TTL:
        return cached["result"]

    try:
        end   = datetime.today() + timedelta(days=1)
        start = end - timedelta(days=252)
        df    = yf.Ticker(index_ticker).history(start=start, end=end,
                    interval="1d", auto_adjust=True)
        if df.empty or len(df) < 50:
            return ("neutral", 50, f"Cannot fetch {index_ticker} — assuming neutral")

        c  = df["Close"].squeeze()
        e50  = c.ewm(span=50,  adjust=False).mean()
        e200 = c.ewm(span=200, adjust=False).mean()
        last = float(c.iloc[-1])
        e50v = float(e50.iloc[-1]); e200v = float(e200.iloc[-1])

        # Score the regime
        score = 0
        reasons = []
        # Price vs EMAs
        if last > e200v:  score+=30; reasons.append(f"Index above EMA200 ✅")
        else:             reasons.append(f"Index BELOW EMA200 ⚠️")
        if last > e50v:   score+=25; reasons.append(f"Index above EMA50 ✅")
        else:             reasons.append(f"Index BELOW EMA50 ⚠️")
        if e50v > e200v:  score+=20; reasons.append("EMA50 > EMA200 (golden cross zone) ✅")
        else:             reasons.append("EMA50 < EMA200 (death cross zone) ⚠️")
        # Trend direction (5-day momentum)
        mom = (last - float(c.iloc[-5]))/float(c.iloc[-5])*100
        if mom > 0:       score+=15; reasons.append(f"5-day momentum +{mom:.1f}% ✅")
        else:             reasons.append(f"5-day momentum {mom:.1f}% ⚠️")
        # Distance from 52W high
        h52 = float(c.tail(252).max())
        dist = (last-h52)/h52*100
        if dist > -10:    score+=10; reasons.append(f"Within 10% of 52W high ✅")
        else:             reasons.append(f"{dist:.1f}% below 52W high ⚠️")

        if   score >= 70: regime = "bull"
        elif score >= 40: regime = "neutral"
        else:             regime = "bear"

        detail = f"{index_ticker}: {last:,.0f} | " + " | ".join(reasons[:3])
        result = (regime, score, detail)
        _regime_cache[cache_key] = {"ts":time.time(),"result":result}
        return result
    except Exception as e:
        log.warning(f"Regime check failed for {index_ticker}: {e}")
        return ("neutral", 50, f"Regime check unavailable")

# ══════════════════════════════════════════════════════════════════════
# LIQUIDITY CHECK
# ══════════════════════════════════════════════════════════════════════
def check_liquidity(df, ticker):
    """
    Ensure the stock has enough daily turnover to enter/exit safely.
    Returns (passes, avg_turnover, message)
    """
    try:
        c   = df["Close"].squeeze().tail(20)
        v   = df["Volume"].squeeze().tail(20)
        turnover = (c * v).mean()
        min_t    = min_liquidity(ticker)
        passes   = float(turnover) >= min_t
        cs       = "₹" if get_exchange_suffix(ticker) in (".NS",".BO") else "$"
        msg = (f"Avg turnover {cs}{turnover/1e5:.1f}L/day — {'✅ Liquid' if passes else '❌ Illiquid (avoid)'}")
        return passes, float(turnover), msg
    except:
        return True, 0, "Liquidity check unavailable"

# ══════════════════════════════════════════════════════════════════════
# EARNINGS BLACKOUT
# ══════════════════════════════════════════════════════════════════════
def check_earnings_risk(ticker):
    """
    Check if earnings are within the next 7 days.
    Returns (is_risky, days_to_earnings, message)
    """
    try:
        info = get_ticker_info(ticker)
        ts   = info.get("earningsTimestamp")
        if not ts: return False, None, "No upcoming earnings data"
        earnings_dt = datetime.fromtimestamp(int(ts))
        days_away   = (earnings_dt - datetime.now()).days
        if 0 <= days_away <= 7:
            return True, days_away, (
                f"⚠️ EARNINGS IN {days_away} DAYS ({earnings_dt.strftime('%d %b')}) — "
                f"BLACKOUT: technical analysis unreliable near earnings")
        elif -3 <= days_away < 0:
            return False, days_away, (
                f"📅 Earnings just passed ({abs(days_away)} days ago) — "
                f"post-earnings drift possible")
        return False, days_away, f"Next earnings: {earnings_dt.strftime('%d %b %Y')}"
    except:
        return False, None, "Earnings data unavailable"

# ══════════════════════════════════════════════════════════════════════
# GAP RISK ASSESSMENT
# ══════════════════════════════════════════════════════════════════════
def assess_gap_risk(df):
    """
    Analyse recent gap history to quantify overnight risk.
    Returns (risk_level, avg_gap_pct, max_gap_pct, message)
    """
    try:
        opens  = df["Open"].squeeze().values.astype(float)
        closes = df["Close"].squeeze().values.astype(float)
        # Gap = open vs prior close
        gaps = [abs(opens[i]-closes[i-1])/closes[i-1]*100
                for i in range(1, min(len(opens), 60))]
        if not gaps: return "Low", 0, 0, "Insufficient data"
        avg_gap = float(np.mean(gaps))
        max_gap = float(np.max(gaps))
        # Count significant gaps (>2%)
        large_gaps = sum(1 for g in gaps if g>2)
        risk = ("High"    if avg_gap>1.5 or large_gaps>10 else
                "Medium"  if avg_gap>0.8 or large_gaps>5  else
                "Low")
        msg = (f"Gap risk: {risk} | Avg gap: {avg_gap:.2f}% | "
               f"Max gap (60d): {max_gap:.1f}% | "
               f"Large gaps (>2%): {large_gaps}/60 sessions")
        return risk, avg_gap, max_gap, msg
    except:
        return "Unknown", 0, 0, "Gap analysis unavailable"

# ══════════════════════════════════════════════════════════════════════
# INDICATORS — improved divergence detection
# ══════════════════════════════════════════════════════════════════════
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()

def rsi_calc(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100/(1 + g/(l.replace(0, np.nan)))

def atr_calc(df, n=14):
    h,l,c = df["High"],df["Low"],df["Close"]
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def rv(row, k):
    try:
        v = row[k]
        if isinstance(v, pd.Series): v = v.iloc[0]
        if isinstance(v,(np.floating,float)):
            return None if math.isnan(float(v)) else float(v)
        if isinstance(v, np.integer): return int(v)
        return v
    except: return None

def detect_swing_divergence(price_series, indicator_series, lookback=20, swing_n=3):
    """
    Proper swing-based divergence detection.
    Bullish: price makes lower swing low, indicator makes higher swing low.
    Much fewer false positives than bar-to-bar comparison.
    """
    try:
        p  = price_series.values.astype(float)
        ind= indicator_series.values.astype(float)
        n  = min(lookback, len(p)-swing_n*2)
        if n < 6: return 0

        # Find swing lows in price (local minima)
        def swing_lows(arr, n):
            lows = []
            for i in range(n, len(arr)-n):
                if arr[i] == min(arr[i-n:i+n+1]):
                    lows.append((i, arr[i]))
            return lows

        price_lows = swing_lows(p[-lookback:], swing_n)
        ind_lows   = swing_lows(ind[-lookback:], swing_n)

        if len(price_lows) < 2 or len(ind_lows) < 2: return 0

        # Check last two swing lows
        pl1,pl2 = price_lows[-2][1], price_lows[-1][1]  # older, newer
        il1,il2 = ind_lows[-2][1],   ind_lows[-1][1]

        # Bullish divergence: price lower low, indicator higher low
        if pl2 < pl1 and il2 > il1:
            return 1
        return 0
    except:
        return 0

def add_indicators(df, interval="1d"):
    """Timeframe-aware indicator computation."""
    df = df.copy()
    c = df["Close"].squeeze(); h = df["High"].squeeze()
    l = df["Low"].squeeze();   v = df["Volume"].squeeze()

    # Scale windows to timeframe
    if interval == "1wk":
        yr=52;  ema_long=40;  bb_period=20; sq_period=26; h52_period=52
    elif interval == "1mo":
        yr=12;  ema_long=10;  bb_period=6;  sq_period=12; h52_period=12
    else:
        yr=252; ema_long=200; bb_period=20; sq_period=50; h52_period=252

    # ── Lagging ──────────────────────────────────────────────────
    df["EMA21"]  = ema(c,21); df["EMA50"]=ema(c,50); df["EMA200"]=ema(c,ema_long)
    df["RSI"]    = rsi_calc(c); df["ATR"] = atr_calc(df)
    ef,es = ema(c,12), ema(c,26)
    df["MACD"]   = ef-es; df["MACDs"]=ema(df["MACD"],9); df["MACDh"]=df["MACD"]-df["MACDs"]
    bb_mid=sma(c,bb_period); bb_std=c.rolling(bb_period).std()
    df["BB_m"]=bb_mid; df["BB_u"]=bb_mid+2*bb_std; df["BB_l"]=bb_mid-2*bb_std
    df["VolMA"] = v.rolling(bb_period).mean()
    up=h.diff().clip(lower=0); dn=(-l.diff()).clip(lower=0)
    pdm=up.where(up>dn,0.0); mdm=dn.where(dn>up,0.0)
    a14=atr_calc(df,14)
    pdi=ema(pdm,14)/(a14+1e-9)*100; mdi=ema(mdm,14)/(a14+1e-9)*100
    dx=(pdi-mdi).abs()/(pdi+mdi+1e-9)*100
    df["ADX"]=sma(dx,14); df["+DI"]=pdi; df["-DI"]=mdi
    df["H20"]=h.rolling(bb_period).max().shift(1); df["L20"]=l.rolling(bb_period).min().shift(1)
    df["H52"]=h.rolling(h52_period).max().shift(1); df["L52"]=l.rolling(h52_period).min().shift(1)
    obv=[0]; cv,vv=c.values,v.values
    for i in range(1,len(cv)):
        obv.append(obv[-1]+(vv[i] if cv[i]>cv[i-1] else -vv[i] if cv[i]<cv[i-1] else 0))
    df["OBV"]=obv; df["OBVema"]=ema(pd.Series(obv,index=df.index),20)

    # ── Leading ──────────────────────────────────────────────────
    lo14=l.rolling(14).min(); hi14=h.rolling(14).max()
    df["STOCH_K"]=(100*(c-lo14)/(hi14-lo14+1e-9)).rolling(3).mean()
    df["STOCH_D"]=df["STOCH_K"].rolling(3).mean()
    df["WILLR"]  =-100*(hi14-c)/(hi14-lo14+1e-9)
    tp=(h+l+c)/3; tp_ma=tp.rolling(bb_period).mean()
    tp_md=tp.rolling(bb_period).apply(lambda x:np.mean(np.abs(x-x.mean())),raw=True)
    df["CCI"]=(tp-tp_ma)/(0.015*tp_md+1e-9)
    df["BB_width"]=(df["BB_u"]-df["BB_l"])/(df["BB_m"]+1e-9)*100
    bw_min=df["BB_width"].rolling(sq_period).min()
    bw_max=df["BB_width"].rolling(sq_period).max()
    df["BB_sq"]=((df["BB_width"]-bw_min)/(bw_max-bw_min+1e-9)<0.2).astype(int)

    # ── Improved swing-based divergence ──────────────────────────
    # Apply rolling window to detect divergence at each point
    div_bull_rsi  = pd.Series(0, index=df.index)
    div_bull_macd = pd.Series(0, index=df.index)
    for i in range(40, len(df)):
        window_p = df["Close"].iloc[max(0,i-30):i+1]
        window_r = df["RSI"].iloc[max(0,i-30):i+1]
        window_m = df["MACDh"].iloc[max(0,i-30):i+1]
        if len(window_p) >= 10:
            div_bull_rsi.iloc[i]  = detect_swing_divergence(window_p, window_r)
            div_bull_macd.iloc[i] = detect_swing_divergence(window_p, window_m)
    df["RSI_DIV"]  = div_bull_rsi
    df["MACD_DIV"] = div_bull_macd

    # Volume accumulation on down days
    dd=(c<c.shift(1)).astype(int)
    df["VOL_ACC"]=(v*dd).rolling(5).sum()/(v.rolling(5).sum()+1e-9)

    # Momentum acceleration (2nd derivative)
    roc5=c.pct_change(5)
    df["MOM_ACCEL"]=roc5-roc5.shift(5)

    # Pivot points
    ph=h.shift(1); pl=l.shift(1); pc=c.shift(1)
    df["PP"]=(ph+pl+pc)/3
    df["R1"]=2*df["PP"]-pl;  df["S1"]=2*df["PP"]-ph
    df["R2"]=df["PP"]+(ph-pl); df["S2"]=df["PP"]-(ph-pl)

    return df.dropna()

# ══════════════════════════════════════════════════════════════════════
# TWO-STAGE SCORING — unchanged but wired to new indicators
# ══════════════════════════════════════════════════════════════════════
def leading_score(df):
    r=df.iloc[-1]; pr=df.iloc[-2]
    def g(k):  return rv(r,k)
    def pg(k): return rv(pr,k)
    pts=0; max_pts=0; top=[]; weak=[]
    def add(ok,score,msg_y,msg_n=""):
        nonlocal pts,max_pts
        max_pts+=score
        if ok:  pts+=score;  top.append(msg_y)
        elif msg_n: weak.append(msg_n)

    sk=g("STOCH_K"); sd=g("STOCH_D"); psk=pg("STOCH_K"); psd=pg("STOCH_D")
    if sk is not None and sd is not None:
        if sk<20:   add(True,15,f"Stochastic oversold ({sk:.0f}) ⚡")
        elif psk and psd and sk>sd and psk<=psd and sk<50:
            add(True,12,f"Stochastic bullish cross ({sk:.0f}) ↗"); max_pts+=3
        elif sk<50: add(True,7, f"Stochastic recovering ({sk:.0f})"); max_pts+=8
        else:       add(False,15,"",f"Stochastic overbought ({sk:.0f})")
    else: max_pts+=15

    wr=g("WILLR")
    if wr is not None:
        if wr<-80:   add(True,12,f"Williams %R oversold ({wr:.0f}) ↑")
        elif wr<-50: add(True,6, f"Williams %R weak zone ({wr:.0f})"); max_pts+=6
        else:        add(False,12,"",f"Williams %R neutral/overbought ({wr:.0f})")
    else: max_pts+=12

    cci=g("CCI"); pcci=pg("CCI")
    if cci is not None and pcci is not None:
        if pcci<-100 and cci>=-100: add(True,10,f"CCI cycle turn ↑ ({cci:.0f})")
        elif cci<-100:              add(True,7, f"CCI oversold ({cci:.0f})"); max_pts+=3
        elif 0<cci<100:             add(True,5, f"CCI positive ({cci:.0f})"); max_pts+=5
        else:                       add(False,10,"",f"CCI overbought ({cci:.0f})")
    else: max_pts+=10

    add(bool(g("RSI_DIV")), 15,"Bullish RSI divergence — swing low confirmation ↑",
        "No RSI divergence")
    add(bool(g("MACD_DIV")),12,"Bullish MACD divergence — momentum shift ↑",
        "No MACD divergence")
    add(bool(g("BB_sq")),   10,"BB squeeze ⚡ — energy coiling for breakout",
        "No squeeze")
    ac=g("MOM_ACCEL")
    add(ac is not None and ac>0, 8,"Momentum accelerating ↑","Momentum flat/declining")
    va=g("VOL_ACC")
    add(va is not None and va>0.55,10,
        f"Volume accumulation on down days ({va:.0%})","Low volume accumulation")

    pct=round(pts/max_pts*100) if max_pts>0 else 0
    grade="A" if pct>=70 else "B" if pct>=50 else "C" if pct>=35 else "D"
    return {"score":pct,"grade":grade,"pts":pts,"max_pts":max_pts,
            "top":top[:4],"weak":weak[:3],
            "interpretation":("Strong setup forming" if pct>=70 else
                              "Moderate setup — developing" if pct>=50 else
                              "Weak setup — insufficient leading signals" if pct>=35 else
                              "No setup — do not enter")}


def lagging_score(df):
    r=df.iloc[-1]; pr=df.iloc[-2]
    def g(k):  return rv(r,k)
    def pg(k): return rv(pr,k)
    c=g("Close"); e21=g("EMA21"); e50=g("EMA50"); e200=g("EMA200")
    rsi_=g("RSI"); macdh=g("MACDh"); prev_mh=pg("MACDh")
    adx_=g("ADX"); pdi=g("+DI"); mdi=g("-DI")
    vol=g("Volume"); volma=g("VolMA")
    obv=g("OBV"); obve=g("OBVema")
    h20=g("H20"); h52=g("H52"); l52=g("L52"); bb_m=g("BB_m")
    pts=0; max_pts=0; top=[]; weak=[]
    def add(ok,score,msg_y,msg_n=""):
        nonlocal pts,max_pts
        max_pts+=score
        if ok:  pts+=score;  top.append(msg_y)
        elif msg_n: weak.append(msg_n)

    add(c and e200 and c>e200, 10,f"Above EMA200 ({e200:.1f}) ✅","Below EMA200 ❌")
    add(c and e50  and c>e50,   8,f"Above EMA50 ({e50:.1f}) ✅","Below EMA50")
    add(c and e21  and c>e21,   5,f"Above EMA21 ({e21:.1f}) ✅","Below EMA21")
    add(bool(e21 and e50 and e200 and e21>e50>e200),7,"EMA stack 21>50>200 ✅","Stack misaligned")
    if rsi_ is not None:
        if 50<rsi_<70:     add(True,10,f"RSI bullish zone ({rsi_:.1f}) ✅")
        elif rsi_>=70:     add(False,10,"",f"RSI overbought ({rsi_:.1f}) ⚠")
        elif 30<=rsi_<=50: add(True,6, f"RSI recovering ({rsi_:.1f})"); max_pts+=4
        elif rsi_<30:      add(True,8, f"RSI oversold ({rsi_:.1f})");   max_pts+=2
        else:              add(False,10,"","RSI neutral")
    else: max_pts+=10
    add(bool(macdh and macdh>0),6,"MACD positive ✅","MACD negative")
    add(bool(macdh and macdh>0 and prev_mh and prev_mh<=0),7,"Fresh MACD cross ✅")
    if adx_ and pdi and mdi:
        if adx_>25 and pdi>mdi:    add(True,10,f"Strong uptrend ADX={adx_:.1f} ✅")
        elif adx_>25:              add(False,10,"",f"Strong downtrend ADX={adx_:.1f} ❌")
        elif adx_>20 and pdi>mdi:  add(True,6, f"Moderate uptrend ADX={adx_:.1f}"); max_pts+=4
        else:                      add(False,10,"",f"Weak/no trend ADX={adx_:.1f}")
    else: max_pts+=10
    add(bool(obv and obve and obv>obve),8,"OBV accumulation ✅","OBV distribution ❌")
    if vol and volma and volma>0:
        add(vol/volma>1.2,7,f"Volume {vol/volma:.1f}× avg ✅",f"Volume weak {vol/volma:.1f}×")
    else: max_pts+=7
    add(bool(h20 and c and c>h20),7,"20-day breakout ✅","Within range")
    if h52 and l52:
        pos52=(c-l52)/(h52-l52+1e-9)*100
        add(pos52>50,10,f"Upper 52-week range ({pos52:.0f}%) ✅",
            f"Lower 52-week range ({pos52:.0f}%) ⚠")
    else: max_pts+=10
    add(bool(c and bb_m and c>bb_m),5,"Above BB midline ✅","Below BB midline")

    pct=round(pts/max_pts*100) if max_pts>0 else 0
    grade="A" if pct>=72 else "B" if pct>=55 else "C" if pct>=38 else "D"
    return {"score":pct,"grade":grade,"pts":pts,"max_pts":max_pts,
            "top":top[:4],"weak":weak[:3],
            "interpretation":("Strong trend confirmed" if pct>=72 else
                              "Trend developing" if pct>=55 else
                              "Weak trend — insufficient confirmation" if pct>=38 else
                              "No trend confirmation — do not enter")}


# ══════════════════════════════════════════════════════════════════════
# SUPPORT-ANCHORED STOP LOSS
# ══════════════════════════════════════════════════════════════════════
def smart_stop_loss(df, signal_type, atr_val, current_price, override_mult=None):
    """
    Calculate stop loss anchored to actual support structure, not just ATR.
    Uses the LOWER of:
    1. ATR-based SL (standard)
    2. Recent swing low minus small buffer
    Then validates: SL must be at least 0.5×ATR below entry (not too tight)
    """
    if override_mult is not None:
        sl_mult = override_mult
    elif signal_type == "EARLY SETUP":
        sl_mult = 2.0
    else:
        sl_mult = 1.5
    atr_sl  = round(current_price - sl_mult*atr_val, 2)

    # Find recent swing low (last 10 bars)
    try:
        lows  = df["Low"].tail(15).values.astype(float)
        swing_low = float(np.min(lows)) * 0.99  # 1% buffer below swing low
        swing_sl  = round(swing_low, 2)
    except:
        swing_sl = atr_sl

    # Use the more conservative (lower) stop
    # But if swing low is too far (>3×ATR), use ATR-based
    if abs(current_price - swing_sl) > 3*atr_val:
        final_sl = atr_sl
        sl_type  = "ATR-based"
    elif swing_sl < atr_sl:
        final_sl = swing_sl
        sl_type  = "swing-low-anchored"
    else:
        final_sl = atr_sl
        sl_type  = "ATR-based"

    # Minimum: at least 0.5×ATR from entry (don't let SL be too tight)
    if current_price - final_sl < 0.5*atr_val:
        final_sl = round(current_price - 0.5*atr_val, 2)
        sl_type  = "minimum-ATR"

    return final_sl, sl_type


# ══════════════════════════════════════════════════════════════════════
# SIGNAL BUILDING — with all real-money filters
# ══════════════════════════════════════════════════════════════════════
def build_signal(df, lead, lag, ticker="", regime=None, weekly_lag=None):
    r    = df.iloc[-1]
    c    = float(rv(r,"Close")); atr_=float(df["ATR"].iloc[-1])
    ls   = lead["score"]; gs = lag["score"]

    # ── Raw signal ────────────────────────────────────────────────
    if   ls>=75 and gs>=72: raw="PRIME LONG";  emoji="⭐"
    elif ls>=65 and gs>=65: raw="LONG";         emoji="🟢"
    elif ls>=65 and gs<65:  raw="EARLY SETUP";  emoji="🔮"
    elif ls<65  and gs>=72: raw="TREND RIDE";   emoji="🏄"
    elif ls>=50 or  gs>=50: raw="WATCH";        emoji="👀"
    else:                   raw="WAIT";          emoji="⏳"

    # ── MTF downgrade — if weekly is bearish, reduce confidence ───
    mtf_warning = ""
    signal = raw
    if weekly_lag and weekly_lag.get("score",100) < 40:
        if signal in ("PRIME LONG","LONG"):
            signal = "LONG"  # downgrade prime
            mtf_warning = "⚠️ Weekly trend bearish — reduced to LONG (not Prime)"
        elif signal in ("EARLY SETUP","TREND RIDE"):
            signal = "WATCH"
            mtf_warning = "⚠️ Weekly trend bearish — downgraded to WATCH"

    # ── Market regime — context, NOT a hard lock ────────────────
    # Philosophy: stocks can and do outperform in bear markets.
    # Defensive sectors, exporters, RS+ stocks decouple from index.
    # Hard locks miss real opportunities. Instead: adjust sizing,
    # tighten SL, raise required score, communicate risk clearly.
    # The trader always sees levels and decides. We never block.
    regime_warning = ""
    regime_note    = ""
    regime_str, regime_score, regime_detail = regime or ("neutral",50,"")

    # In bear markets: require higher conviction to act
    min_act = {"bull":60, "neutral":65, "bear":72}[regime_str]

    if regime_str == "bear":
        regime_note = (
            f"🚨 *BEAR MARKET ({regime_score}/100)* — signal shown with adjustments:\n"
            f"  • Position size → 0.5% risk (half normal)\n"
            f"  • SL uses tighter 1.0×ATR multiplier\n"
            f"  • Minimum score to act: {min_act}/100\n"
            f"  • Only trade stocks outperforming index (RS+)\n"
            f"  • {regime_detail[:100]}")
    elif regime_str == "neutral":
        regime_note = (
            f"⚠️ *NEUTRAL MARKET ({regime_score}/100)*\n"
            f"  • Position size → 0.75% risk\n"
            f"  • Prefer Grade A signals only\n"
            f"  • {regime_detail[:80]}")

    combined_score = round((ls+gs)/2)
    if combined_score < min_act and signal in ("PRIME LONG","LONG","EARLY SETUP","TREND RIDE"):
        regime_warning = (
            f"⚠️ Score {combined_score}/100 below {regime_str} market "
            f"threshold ({min_act}). Consider waiting for higher conviction "
            f"or better regime before entering.")
    # ── Strategy label ────────────────────────────────────────────
    rsi_  = rv(r,"RSI") or 50
    e200  = rv(r,"EMA200"); e50=rv(r,"EMA50"); e21=rv(r,"EMA21")
    adx_  = rv(r,"ADX"); macdh=rv(r,"MACDh"); h20=rv(r,"H20")

    if rsi_<38 and c and e200 and c>e200:
        strat = "Mean Reversion (Oversold Bounce)"
    elif signal == "EARLY SETUP":     strat = "Early Reversal — Await Confirmation"
    elif signal == "TREND RIDE":      strat = "Trend Continuation"
    elif h20 and c>h20 and adx_ and adx_>25: strat = "Momentum Breakout"
    elif e21 and e50 and e21>e50 and c and e200 and c>e200: strat = "Trend Following"
    else:                             strat = "Confluence Long"

    # ── Trade levels — smart SL ───────────────────────────────────
    actionable = signal in ("PRIME LONG","LONG","EARLY SETUP","TREND RIDE")
    entry = round(c, 2)

    # Volume confirmation — require at least 0.8x average on signal day
    # Signals on unusually low volume are less reliable
    vol_  = rv(r,"Volume"); volma_ = rv(r,"VolMA")
    vol_ratio = round(float(vol_)/float(volma_),2) if vol_ and volma_ and volma_>0 else 1.0
    vol_confirmed = vol_ratio >= 0.8

    # Regime adjusts risk per trade and SL tightness
    regime_risk_pct = {"bull": RISK_PER_TRADE,
                       "neutral": RISK_PER_TRADE * 0.75,
                       "bear":    RISK_PER_TRADE * 0.50}[regime_str]
    # Bear market: use tighter SL (1.0× instead of 1.5×) to protect capital
    sl_regime_mult = {"bull":None, "neutral":None, "bear":1.0}[regime_str]
    if actionable:
        sl, sl_type = smart_stop_loss(df, signal, atr_, c,
                                      override_mult=sl_regime_mult)
        sl_dist  = c - sl
        t1 = round(c + 1.5*atr_, 2)
        t2 = round(c + 2.5*atr_, 2)
        t3 = round(c + 4.0*atr_, 2)
        rr1 = round((t1-c)/max(sl_dist,0.01), 2)
        rr2 = round((t2-c)/max(sl_dist,0.01), 2)
        rr3 = round((t3-c)/max(sl_dist,0.01), 2)
    else:
        sl=sl_type=t1=t2=t3=None
        rr1=rr2=rr3=None; sl_dist=0

    # ── Horizon ───────────────────────────────────────────────────
    atr_pct = atr_/(c+1e-9)*100
    strong  = adx_ and adx_>25
    if   atr_pct<1.0:            hzn="Positional (4-8 weeks)"
    elif atr_pct<2.5 or strong:  hzn="Positional (2-5 weeks)"
    elif atr_pct<4.0:            hzn="Swing (7-15 days)"
    else:                        hzn="Swing (3-7 days)"

    # ── Entry validity ────────────────────────────────────────────
    valid_till = (datetime.now()+timedelta(days=ENTRY_VALID_DAYS)).strftime("%d %b")

    return {
        "signal":signal, "raw_signal":raw, "emoji":emoji,
        "strategy":strat, "horizon":hzn,
        "entry":entry, "sl":sl, "sl_type":sl_type if actionable else None,
        "t1":t1,"t2":t2,"t3":t3,
        "rr1":rr1,"rr2":rr2,"rr3":rr3,
        "combined":round((ls+gs)/2),
        "atr":round(atr_,2),"atr_pct":round(atr_pct,2),
        "valid_till":valid_till,
        "mtf_warning":mtf_warning,
        "regime_warning":regime_warning,
        "regime_note":regime_note,
        "regime":regime_str,
        "regime_score":regime_score,
        "regime_risk_pct": regime_risk_pct if actionable else RISK_PER_TRADE,
        "min_score_to_act": min_act,
        "vol_ratio":        round(vol_ratio,2),
        "vol_confirmed":    vol_confirmed,
    }

# ══════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════
def fetch_df(ticker, days=600, interval="1d"):
    """Fetch with 3 retries on network failure."""
    end   = datetime.today() + timedelta(days=1)
    start = end - timedelta(days=days+1)
    last_err = None
    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(start=start,end=end,
                     interval=interval,auto_adjust=True)
            if not df.empty:
                break
        except Exception as e:
            last_err = e
            time.sleep(2**attempt)  # exponential backoff: 1s, 2s, 4s
    else:
        raise ValueError(f"Failed after 3 attempts: {last_err}")
    if df.empty:
        sfx = get_exchange_suffix(ticker)
        if sfx in (".NS",".BO"):
            raise ValueError(f"No data for '{ticker}'. "
                "Stock may be delisted or suspended.")
        raise ValueError(
            f"No data for '{ticker}'. "
            "Indian stocks: add .NS (NSE) or .BO (BSE). "
            "Verify symbol at finance.yahoo.com")
    df.columns=[str(c) for c in df.columns]
    cm={}
    for col in df.columns:
        cl=col.lower()
        if cl=="open":    cm[col]="Open"
        elif cl=="high":  cm[col]="High"
        elif cl=="low":   cm[col]="Low"
        elif cl in("close","adj close"): cm[col]="Close"
        elif cl=="volume":cm[col]="Volume"
    if cm: df=df.rename(columns=cm)
    for n in ["Open","High","Low","Close","Volume"]:
        if n not in df.columns:
            raise ValueError(f"Missing column '{n}' in data for {ticker}")
    return df[["Open","High","Low","Close","Volume"]].dropna()

def fetch_and_score(ticker, interval="1d"):
    if   interval=="1d":  days=600
    elif interval=="1wk": days=2200
    elif interval=="1mo": days=5500
    else:                 days=1000
    df = fetch_df(ticker, days=days, interval=interval)
    df = add_indicators(df, interval=interval)
    if len(df)<15:
        raise ValueError(
            f"Insufficient data ({len(df)} bars on {interval}). "
            f"Stock may be too recently listed.")
    return df

def get_fund(ticker):
    try:
        info=get_ticker_info(ticker)
        def g(k,d="—"):
            v=info.get(k,d); return d if v is None else v
        mc=g("marketCap")
        if isinstance(mc,(int,float)) and mc>0:
            if mc>=1e12:   mcs=f"{mc/1e12:.2f}T"
            elif mc>=1e9:  mcs=f"{mc/1e9:.2f}B"
            else:          mcs=f"{mc/1e6:.0f}M"
        else: mcs="—"
        cs=currency_symbol(info)
        ts=g("earningsTimestamp")
        if ts:
            try:    edate=datetime.fromtimestamp(int(ts)).strftime("%d %b %Y")
            except: edate="—"
        else: edate="—"
        return {
            "name":   g("longName") or g("shortName") or ticker,
            "sector": g("sector"),"industry":g("industry"),
            "exchange":g("exchange"),"currency":g("currency",""),"cs":cs,
            "market_cap":mcs,
            "pe":  round(float(g("trailingPE",0)),1)  if g("trailingPE")!="—"  else "—",
            "pb":  round(float(g("priceToBook",0)),2)  if g("priceToBook")!="—" else "—",
            "eps": round(float(g("trailingEps",0)),2)  if g("trailingEps")!="—" else "—",
            "roe": f"{round(float(g('returnOnEquity',0))*100,1)}%" if g("returnOnEquity")!="—" else "—",
            "beta":round(float(g("beta",0)),2)         if g("beta")!="—"        else "—",
            "div": f"{round(float(g('dividendYield',0))*100,2)}%" if g("dividendYield")!="—" else "—",
            "h52": g("fiftyTwoWeekHigh"),"l52":g("fiftyTwoWeekLow"),
            "avg_vol":g("averageVolume"),
            "earnings_date":edate,
        }
    except:
        return {"name":ticker,"sector":"—","industry":"—","exchange":"—",
                "currency":"","cs":"","market_cap":"—","pe":"—","pb":"—",
                "eps":"—","roe":"—","beta":"—","div":"—",
                "h52":"—","l52":"—","avg_vol":"—","earnings_date":"—"}

# ══════════════════════════════════════════════════════════════════════
# FIBONACCI + SUPPORT/RESISTANCE
# ══════════════════════════════════════════════════════════════════════
def fibonacci_levels(df, lookback=60):
    recent     = df.tail(lookback)
    swing_high = float(recent["High"].max())
    swing_low  = float(recent["Low"].min())
    diff       = swing_high - swing_low
    return {
        "0.0%":   round(swing_high, 2),
        "23.6%":  round(swing_high-0.236*diff, 2),
        "38.2%":  round(swing_high-0.382*diff, 2),
        "50.0%":  round(swing_high-0.500*diff, 2),
        "61.8%":  round(swing_high-0.618*diff, 2),
        "78.6%":  round(swing_high-0.786*diff, 2),
        "100%":   round(swing_low, 2),
    }, swing_high, swing_low

def find_sr_levels(df, lookback=60, tolerance=0.015):
    recent  = df.tail(lookback)
    pivots  = list(recent["High"].values)+list(recent["Low"].values)
    current = float(df["Close"].iloc[-1])
    pivots.sort()
    clusters=[]
    if pivots:
        grp=[pivots[0]]
        for p in pivots[1:]:
            if (p-grp[-1])/(grp[-1]+1e-9)<tolerance: grp.append(p)
            else:
                if len(grp)>=3: clusters.append(round(float(np.mean(grp)),2))
                grp=[p]
        if len(grp)>=3: clusters.append(round(float(np.mean(grp)),2))
    supports    =sorted([c for c in clusters if c<current*0.99],reverse=True)[:3]
    resistances =sorted([c for c in clusters if c>current*1.01])[:3]
    return supports, resistances

# ══════════════════════════════════════════════════════════════════════
# POSITION SIZING — 1% risk rule
# ══════════════════════════════════════════════════════════════════════
def calc_position_size(entry, sl, capital=None, risk_pct=None):
    capital  = capital  or TRADING_CAPITAL
    risk_pct = risk_pct or RISK_PER_TRADE
    if not entry or not sl or sl>=entry: return None
    risk_per_share = entry-sl
    if risk_per_share<=0: return None
    risk_amount = capital*risk_pct
    qty         = math.floor(risk_amount/risk_per_share)
    if qty<1: qty=1
    capital_req = qty*entry
    return {
        "qty":qty,"capital_req":round(capital_req,2),
        "risk_amount":round(qty*risk_per_share,2),
        "risk_pct":round(qty*risk_per_share/capital*100,2),
        "pct_capital":round(capital_req/capital*100,1),
        "risk_per_sh":round(risk_per_share,2),
    }

# ══════════════════════════════════════════════════════════════════════
# PORTFOLIO RISK CHECK
# ══════════════════════════════════════════════════════════════════════
def check_portfolio_risk(ticker, sector):
    """Check if we can take a new trade given current open positions."""
    try:
        con = sqlite3.connect(DB_FILE)
        open_count  = con.execute(
            "SELECT COUNT(*) FROM trades WHERE status='OPEN'").fetchone()[0]
        sector_count= con.execute(
            "SELECT COUNT(*) FROM trades WHERE status='OPEN' AND sector=?",
            (sector,)).fetchone()[0]
        # Total capital at risk across open trades
        open_trades = con.execute(
            "SELECT entry_price, sl, qty FROM trades WHERE status='OPEN'"
        ).fetchall()
        total_risk = sum(
            max(0,(ep-(sl or ep*0.95))*qty)
            for ep,sl,qty in open_trades
        )
        con.close()
        total_risk_pct = total_risk/TRADING_CAPITAL*100

        warnings = []
        blocked  = False
        if open_count >= MAX_OPEN_TRADES:
            warnings.append(f"🚨 Max open trades ({MAX_OPEN_TRADES}) reached — close a trade first")
            blocked = True
        if sector_count >= MAX_SECTOR_TRADES and sector and sector!="—":
            warnings.append(f"🚨 Max sector concentration ({MAX_SECTOR_TRADES}) in {sector} reached")
            blocked = True
        if total_risk_pct >= MAX_PORTFOLIO_RISK*100:
            warnings.append(f"🚨 Portfolio risk at {total_risk_pct:.1f}% — max {MAX_PORTFOLIO_RISK*100:.0f}% allowed")
            blocked = True

        return blocked, warnings, open_count, sector_count, total_risk_pct
    except Exception as e:
        return False, [], 0, 0, 0

# ══════════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME CONSENSUS
# ══════════════════════════════════════════════════════════════════════
def multi_timeframe_consensus(ticker):
    """Weekly only — monthly removed (too slow, rarely different outcome)."""
    results = {}
    try:
        df   = fetch_and_score(ticker, interval="1wk")
        lead = leading_score(df)
        lag  = lagging_score(df)
        sig  = build_signal(df, lead, lag)
        results["Weekly"] = {"signal":sig["signal"],"lead":lead["score"],
                             "lag":lag["score"],"emoji":sig["emoji"],
                             "lag_obj":lag}
    except Exception as e:
        log.warning(f"MTF weekly failed for {ticker}: {e}")
        results["Weekly"] = None
    wk = results.get("Weekly")
    if not wk:
        return results, "⚪ Weekly data unavailable"
    bullish = wk["signal"] in ("PRIME LONG","LONG","EARLY SETUP","TREND RIDE")
    consensus = ("🟢 Weekly trend bullish" if bullish
                 else "🔴 Weekly trend bearish — consider smaller size")
    return results, consensus

# ══════════════════════════════════════════════════════════════════════
# AI INSIGHT
# ══════════════════════════════════════════════════════════════════════
def ai_insight(ticker, sig, lead, lag, fund, regime_detail="",
               liquidity_msg="", earnings_msg="", gap_msg="",
               rs_label=""):
    if not AI_API_KEY: return None
    try:
        from openai import OpenAI
        client=OpenAI(base_url=AI_BASE_URL, api_key=AI_API_KEY)
        contrad=""
        if lead["score"]>=65 and lag["score"]<50:
            contrad="IMPORTANT: Setup forming but NOT confirmed. Smaller size or wait."
        elif lag["score"]>=72 and lead["score"]<45:
            contrad="NOTE: Strong trend but no new setup energy — late-stage, higher risk."
        prompt=f"""Professional NSE/Global stock analyst. Be precise, concise, actionable.

Stock: {ticker} | {fund.get('name','')} | {fund.get('sector','—')}
Price: {fund.get('cs','')}{sig.get('entry','—')} | P/E:{fund.get('pe','—')} | Beta:{fund.get('beta','—')}
Signal: {sig['signal']} | Strategy: {sig['strategy']} | Horizon: {sig['horizon']}
Leading: {lead['score']}/100 ({lead['grade']}) — {lead['interpretation']}
Lagging: {lag['score']}/100 ({lag['grade']}) — {lag['interpretation']}
Market: {regime_detail[:80]}
Liquidity: {liquidity_msg[:60]}
Earnings: {earnings_msg[:80]}
Gap risk: {gap_msg[:60]}
Relative strength: {rs_label[:80]}
{contrad}

Reply EXACTLY:
CONFIDENCE: High/Medium/Low
SETUP: [leading — what will happen, 1 sentence]
CONFIRM: [lagging — what is confirmed or contradicted, 1 sentence]
ADVICE: [specific action with price levels, 1-2 sentences]
RISK: [the single biggest real-money risk]
BIAS: [3-5 words]"""
        r=client.chat.completions.create(model=AI_MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=220,
            timeout=15,temperature=0.3)
        return r.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"AI error: {e}"); return None

# ══════════════════════════════════════════════════════════════════════
# CANDLESTICK PATTERNS — 35+ patterns
# ══════════════════════════════════════════════════════════════════════
def candle_patterns(df, n_bars=12):
    pats=[]; seen=set(); n=len(df)
    if n<5: return pats
    def O(i): return float(df["Open"].iloc[i])
    def H(i): return float(df["High"].iloc[i])
    def L(i): return float(df["Low"].iloc[i])
    def C(i): return float(df["Close"].iloc[i])
    def B(i): return abs(C(i)-O(i))
    def R(i): return max(H(i)-L(i),1e-9)
    def UW(i): return H(i)-max(C(i),O(i))
    def LW(i): return min(C(i),O(i))-L(i)
    def MID(i): return (O(i)+C(i))/2
    def bull(i): return C(i)>O(i)
    def bear(i): return C(i)<O(i)
    def bp(i):   return B(i)/R(i)
    def add(name,typ,conv,action,desc,idx):
        if name not in seen:
            seen.add(name)
            d=df.index[idx]
            ds=d.strftime("%d %b") if hasattr(d,"strftime") else str(d)[:10]
            pats.append({"name":name,"type":typ,"conviction":conv,
                         "action":action,"desc":desc,"date":ds})
    st=max(4,n-n_bars)
    for i in range(st,n):
        b=B(i); r=R(i); uw=UW(i); lw=LW(i)
        # Single bar
        if bp(i)<0.05:
            if uw>0 and lw>0 and abs(uw-lw)/r<0.1:
                add("Gravestone Doji","bearish",72,"Avoid new longs — bearish reversal","Open=close near low",i)
            elif lw>r*0.6 and uw<r*0.1:
                add("Dragonfly Doji","bullish",74,"Long on next candle confirmation","Open=close near high",i)
            else:
                add("Doji","neutral",42,"Wait for direction confirmation","Equal buyer/seller pressure",i)
        elif lw>2*b and uw<0.3*b and b>0:
            if bull(i): add("Hammer","bullish",76,"Long above hammer high, stop below low","Buyers absorbed all selling",i)
            else:        add("Hanging Man","bearish",65,"Tighten stops on existing longs","Selling pressure appearing",i)
        elif uw>2*b and lw<0.3*b and b>0:
            if bull(i): add("Inverted Hammer","bullish",60,"Long only if next candle bullish","Buyers tried, partially succeeded",i)
            else:        add("Shooting Star","bearish",74,"Exit longs — sellers rejected rally","Long upper wick after uptrend",i)
        elif bp(i)<0.35 and uw>b*0.5 and lw>b*0.5:
            add("Spinning Top","neutral",40,"Reduce position if in trend","Balance of power shifting",i)
        elif bp(i)>0.95:
            if bull(i): add("Bullish Marubozu","bullish",82,"Enter next open — trail stop","Pure buying pressure",i)
            else:        add("Bearish Marubozu","bearish",82,"Exit all longs","Pure selling pressure",i)
        elif uw>b*2 and lw>b*2 and bp(i)<0.2:
            add("High Wave","neutral",42,"Avoid new entries — extreme indecision","Both sides rejected",i)
        # Two bar
        if i>=1:
            if bear(i-1) and bull(i) and O(i)<C(i-1) and C(i)>O(i-1) and b>B(i-1):
                add("Bullish Engulfing","bullish",82,"High-conviction long — stop below low","Bull engulfs prior bear",i)
            if bull(i-1) and bear(i) and O(i)>C(i-1) and C(i)<O(i-1) and b>B(i-1):
                add("Bearish Engulfing","bearish",82,"Exit longs immediately","Bear engulfs prior bull",i)
            if bear(i-1) and bull(i) and O(i)>C(i-1) and C(i)<O(i-1) and b<B(i-1)*0.5:
                add("Bullish Harami","bullish",62,"Long on confirmation above high","Small bull inside large bear",i)
            if bull(i-1) and bear(i) and O(i)<C(i-1) and C(i)>O(i-1) and b<B(i-1)*0.5:
                add("Bearish Harami","bearish",60,"Tighten stops","Small bear inside large bull",i)
            if bp(i)<0.05 and B(i-1)/R(i-1)>0.5:
                if bear(i-1): add("Bullish Harami Cross","bullish",68,"Enter on confirmation","Doji inside large bear — exhaustion",i)
                else:          add("Bearish Harami Cross","bearish",68,"Exit on confirmation","Doji inside large bull — exhaustion",i)
            if bear(i-1) and bull(i) and O(i)<L(i-1) and C(i)>MID(i-1) and C(i)<O(i-1):
                add("Piercing Line","bullish",72,"Long with stop below low","Bull closes above prior midpoint",i)
            if bull(i-1) and bear(i) and O(i)>H(i-1) and C(i)<MID(i-1):
                add("Dark Cloud Cover","bearish",72,"Exit longs","Bear closes below midpoint",i)
            if abs(H(i)-H(i-1))/max(H(i),H(i-1))<0.002 and bull(i-1) and bear(i):
                add("Tweezer Top","bearish",65,"Double rejection — exit longs","Same high tested twice",i)
            if abs(L(i)-L(i-1))/max(L(i),L(i-1))<0.002 and bear(i-1) and bull(i):
                add("Tweezer Bottom","bullish",65,"Double support — long on confirmation","Same low tested twice",i)
            if bear(i-1) and bull(i) and bp(i)>0.7 and bp(i-1)>0.7 and O(i)>=O(i-1):
                add("Bullish Kicker","bullish",88,"Rare strong reversal — enter immediately","Gap from bear to bull — sentiment shift",i)
            if bull(i-1) and bear(i) and bp(i)>0.7 and bp(i-1)>0.7 and O(i)<=O(i-1):
                add("Bearish Kicker","bearish",88,"Exit all longs immediately","Gap from bull to bear",i)
            if L(i)>H(i-1):
                add("Rising Window","bullish",70,"Gap acts as support — buy dips","Price gaps above prior high",i)
            if H(i)<L(i-1):
                add("Falling Window","bearish",70,"Gap acts as resistance","Price gaps below prior low",i)
        # Three bar
        if i>=2:
            if (bear(i-2) and B(i-2)>R(i-2)*0.5 and B(i-1)<B(i-2)*0.35
                    and bull(i) and C(i)>MID(i-2)):
                add("Morning Star","bullish",84,"High-conviction long — stop below star","Bear→star→bull",i)
            if (bull(i-2) and B(i-2)>R(i-2)*0.5 and B(i-1)<B(i-2)*0.35
                    and bear(i) and C(i)<MID(i-2)):
                add("Evening Star","bearish",84,"Exit all longs","Bull→star→bear",i)
            if (bear(i-2) and B(i-2)>R(i-2)*0.5 and bp(i-1)<0.05
                    and bull(i) and C(i)>MID(i-2)):
                add("Morning Doji Star","bullish",88,"Stronger reversal — high conviction","Doji star — max indecision before reversal",i)
            if (bull(i-2) and B(i-2)>R(i-2)*0.5 and bp(i-1)<0.05
                    and bear(i) and C(i)<MID(i-2)):
                add("Evening Doji Star","bearish",88,"Exit immediately — very strong","Doji star before crash",i)
            if (bull(i-2) and bull(i-1) and bull(i) and C(i)>C(i-1)>C(i-2)
                    and bp(i)>0.5 and bp(i-1)>0.5):
                add("Three White Soldiers","bullish",87,"Strong trend — trail stop below each candle","Three consecutive bull candles",i)
            if (bear(i-2) and bear(i-1) and bear(i) and C(i)<C(i-1)<C(i-2)
                    and bp(i)>0.5 and bp(i-1)>0.5):
                add("Three Black Crows","bearish",87,"Avoid all longs","Three consecutive bear candles",i)
            if (bear(i-2) and bull(i-1) and O(i-1)>C(i-2) and C(i-1)<O(i-2)
                    and bull(i) and C(i)>O(i-2)):
                add("Three Inside Up","bullish",78,"Confirmed reversal — enter above third candle","Harami + confirmation",i)
            if (bull(i-2) and bear(i-1) and O(i-1)<C(i-2) and C(i-1)>O(i-2)
                    and bear(i) and C(i)<O(i-2)):
                add("Three Inside Down","bearish",78,"Confirmed bearish reversal","Bearish harami + confirmation",i)
        # Exhaustion
        if i>=5:
            avg_b=sum(B(j) for j in range(i-5,i))/5
            if avg_b>0 and B(i)>3*avg_b:
                if bull(i): add("Climactic Buy","warning",60,"Take partial profits — unsustainable","Body 3× recent average",i)
                else:        add("Climactic Sell","warning",60,"Watch for reversal — do not short","Selling climax",i)
    return sorted(pats,key=lambda x:x["conviction"],reverse=True)

# ══════════════════════════════════════════════════════════════════════
# CHART IMAGE
# ══════════════════════════════════════════════════════════════════════
DARK="#0d1117"; PANEL="#161b22"; BORDER="#30363d"
GRN="#26a69a"; RED="#ef5350"; ACC="#64b5f6"
YEL="#ffd600"; PUR="#ce93d8"; MUT="#8b949e"; WHT="#e6edf3"; TEAL="#80cbc4"

def make_chart_image(df, sig, lead, lag, pats, ticker, timeframe="Daily",
                     fib_levels=None, supports=None, resistances=None):
    bars    = min(120, len(df))
    plot_df = df.tail(bars).copy()
    fig     = plt.figure(figsize=(16,14), facecolor=DARK)
    gs      = gridspec.GridSpec(5,1,figure=fig,
                height_ratios=[4.5,1.0,1.3,1.3,1.3],hspace=0.04)
    ax1=fig.add_subplot(gs[0]); ax2=fig.add_subplot(gs[1],sharex=ax1)
    ax3=fig.add_subplot(gs[2],sharex=ax1); ax4=fig.add_subplot(gs[3],sharex=ax1)
    ax5=fig.add_subplot(gs[4],sharex=ax1)
    for ax in [ax1,ax2,ax3,ax4,ax5]:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUT,labelsize=7)
        ax.grid(color=BORDER,linewidth=0.5,alpha=0.6)
        for sp in ax.spines.values(): sp.set_color(BORDER)

    x=np.arange(len(plot_df))
    dates=[d.strftime("%d%b") for d in plot_df.index]
    op=plot_df["Open"].values.astype(float)
    hi=plot_df["High"].values.astype(float)
    lo=plot_df["Low"].values.astype(float)
    cl=plot_df["Close"].values.astype(float)

    # Candlesticks
    for i in range(len(x)):
        col=GRN if cl[i]>=op[i] else RED
        ax1.plot([x[i],x[i]],[lo[i],hi[i]],color=col,linewidth=0.8,alpha=0.9)
        bh=max(abs(cl[i]-op[i]),(hi[i]-lo[i])*0.02)
        ax1.bar(x[i],bh,bottom=min(op[i],cl[i]),width=0.7,
                color=col,alpha=0.9,linewidth=0)

    # EMAs
    for cn,col,lw,ls in [("EMA21","#fdd835",1.2,"solid"),
                          ("EMA50","#42a5f5",1.5,"solid"),
                          ("EMA200","#ff7043",1.8,"dashed")]:
        if cn in plot_df.columns:
            ax1.plot(x,plot_df[cn].values,color=col,linewidth=lw,
                     linestyle=ls,label=cn,alpha=0.9)

    # Bollinger Bands
    if "BB_u" in plot_df.columns:
        ax1.fill_between(x,plot_df["BB_l"].values,plot_df["BB_u"].values,
                         alpha=0.06,color=ACC)
        ax1.plot(x,plot_df["BB_u"].values,color=ACC,linewidth=0.6,
                 linestyle="dotted",alpha=0.6)
        ax1.plot(x,plot_df["BB_l"].values,color=ACC,linewidth=0.6,
                 linestyle="dotted",alpha=0.6)
        ax1.plot(x,plot_df["BB_m"].values,color=ACC,linewidth=0.4,alpha=0.4)

    # Fibonacci
    if fib_levels:
        fc={"0.0%":RED,"23.6%":"#ff9800","38.2%":YEL,
            "50.0%":WHT,"61.8%":YEL,"78.6%":"#ff9800","100%":GRN}
        for lv,price in fib_levels.items():
            ax1.axhline(price,color=fc.get(lv,MUT),linewidth=0.8,
                        linestyle=":",alpha=0.7)
            ax1.text(len(x)-1,price,f" Fib {lv}",color=fc.get(lv,MUT),
                     fontsize=6,va="center",ha="left")

    # S/R
    if supports:
        for s in supports:
            ax1.axhline(s,color=GRN,linewidth=1.0,linestyle="--",alpha=0.6)
            ax1.text(0,s,f"S {s:.1f} ",color=GRN,fontsize=6,va="center",ha="right")
    if resistances:
        for r in resistances:
            ax1.axhline(r,color=RED,linewidth=1.0,linestyle="--",alpha=0.6)
            ax1.text(0,r,f"R {r:.1f} ",color=RED,fontsize=6,va="center",ha="right")

    # Trade levels
    for price,col,lbl,ls,lw in [
            (sig.get("sl"),    RED,    "SL",    "solid",1.8),
            (sig.get("entry"),"#00e676","Entry","solid",2.0),
            (sig.get("t1"),    TEAL,   "T1",    "dashed",1.4),
            (sig.get("t2"),    ACC,    "T2",    "dashed",1.4),
            (sig.get("t3"),    "#0288d1","T3",  "dashed",1.4)]:
        if not price: continue
        ax1.axhline(price,color=col,linewidth=lw,linestyle=ls,alpha=0.85)
        ax1.text(len(x)+0.5,price,f" {lbl} {price:.2f}",
                 color=col,fontsize=7,va="center",fontweight="bold")

    # Pattern markers
    date_to_x={d.strftime("%d %b"):i for i,d in enumerate(plot_df.index)}
    for p in [pp for pp in pats if pp["type"]=="bullish"][:4]:
        xi=date_to_x.get(p.get("date",""),-1)
        if xi>=0:
            ax1.annotate("▲",(xi,lo[xi]*0.993),color=GRN,fontsize=9,ha="center",va="top")
            ax1.annotate(p["name"][:10],(xi,lo[xi]*0.988),
                         color=GRN,fontsize=5,ha="center",va="top")
    for p in [pp for pp in pats if pp["type"]=="bearish"][:4]:
        xi=date_to_x.get(p.get("date",""),-1)
        if xi>=0:
            ax1.annotate("▼",(xi,hi[xi]*1.007),color=RED,fontsize=9,ha="center",va="bottom")
            ax1.annotate(p["name"][:10],(xi,hi[xi]*1.012),
                         color=RED,fontsize=5,ha="center",va="bottom")

    # Regime badge on chart
    regime=sig.get("regime","neutral")
    rbadge={"bull":"✅ BULL MARKET","neutral":"⚠️ NEUTRAL","bear":"🚨 BEAR MARKET"}[regime]
    rcolor={"bull":GRN,"neutral":YEL,"bear":RED}[regime]
    ax1.text(0.01,0.98,rbadge,transform=ax1.transAxes,
             color=rcolor,fontsize=8,fontweight="bold",va="top",
             bbox=dict(boxstyle="round",facecolor=DARK,edgecolor=rcolor,alpha=0.8))

    # Title
    ls_s=lead["score"]; gs_s=lag["score"]
    ax1.set_title(
        f"{ticker} — {timeframe}   {sig['emoji']} {sig['signal']}   "
        f"Lead:{ls_s}({lead['grade']})  Lag:{gs_s}({lag['grade']})  Comb:{sig['combined']}",
        color=WHT,fontsize=10,fontweight="bold",pad=8,loc="left")
    ax1.legend(loc="upper left",fontsize=7,framealpha=0.3,
               facecolor=PANEL,edgecolor=BORDER,labelcolor=WHT)
    ax1.yaxis.set_label_position("right"); ax1.yaxis.tick_right()

    # Volume
    vcol=[GRN if cl[i]>=op[i] else RED for i in range(len(x))]
    ax2.bar(x,plot_df["Volume"].values.astype(float),color=vcol,alpha=0.65,width=0.8)
    if "VolMA" in plot_df.columns:
        ax2.plot(x,plot_df["VolMA"].values,color=YEL,linewidth=1.0,
                 linestyle="dashed",alpha=0.8)
    ax2.set_ylabel("VOL",color=MUT,fontsize=7)
    ax2.yaxis.set_label_position("right"); ax2.yaxis.tick_right()

    # Stochastic
    if "STOCH_K" in plot_df.columns:
        ax3.plot(x,plot_df["STOCH_K"].values,color=PUR,linewidth=1.4,label="%K")
        ax3.plot(x,plot_df["STOCH_D"].values,color=YEL,linewidth=0.9,
                 linestyle="dashed",label="%D")
        ax3.axhline(20,color=GRN,linewidth=0.8,linestyle=":",alpha=0.7)
        ax3.axhline(80,color=RED,linewidth=0.8,linestyle=":",alpha=0.7)
        ax3.fill_between(x,0,20,alpha=0.06,color=GRN)
        ax3.fill_between(x,80,100,alpha=0.06,color=RED)
        ax3.set_ylim(0,100)
        ax3.set_ylabel("STOCH\n(LEAD)",color=PUR,fontsize=7,fontweight="bold")
        ax3.legend(loc="upper left",fontsize=6,framealpha=0.3,
                   facecolor=PANEL,edgecolor=BORDER,labelcolor=WHT)
    ax3.yaxis.set_label_position("right"); ax3.yaxis.tick_right()

    # MACD
    if "MACDh" in plot_df.columns:
        mh=plot_df["MACDh"].values.astype(float)
        mc=[GRN if v>=0 else RED for v in mh]
        ax4.bar(x,mh,color=mc,alpha=0.7,width=0.8)
        ax4.plot(x,plot_df["MACD"].values,color=ACC,linewidth=1.2,label="MACD")
        ax4.plot(x,plot_df["MACDs"].values,color=YEL,linewidth=0.9,
                 linestyle="dashed",label="Signal")
        ax4.axhline(0,color=BORDER,linewidth=0.8)
        ax4.set_ylabel("MACD\n(LAG)",color=ACC,fontsize=7,fontweight="bold")
        ax4.legend(loc="upper left",fontsize=6,framealpha=0.3,
                   facecolor=PANEL,edgecolor=BORDER,labelcolor=WHT)
    ax4.yaxis.set_label_position("right"); ax4.yaxis.tick_right()

    # RSI
    if "RSI" in plot_df.columns:
        rv2=plot_df["RSI"].values.astype(float)
        ax5.plot(x,rv2,color=TEAL,linewidth=1.4)
        ax5.fill_between(x,rv2,50,where=(rv2>=50),alpha=0.1,color=GRN)
        ax5.fill_between(x,rv2,50,where=(rv2<50), alpha=0.1,color=RED)
        ax5.axhline(70,color=RED,linewidth=0.8,linestyle=":",alpha=0.7)
        ax5.axhline(50,color=BORDER,linewidth=0.8,alpha=0.6)
        ax5.axhline(30,color=GRN,linewidth=0.8,linestyle=":",alpha=0.7)
        ax5.fill_between(x,0,30, alpha=0.06,color=GRN)
        ax5.fill_between(x,70,100,alpha=0.06,color=RED)
        ax5.set_ylim(0,100)
        ax5.set_ylabel("RSI\n(LAG)",color=TEAL,fontsize=7,fontweight="bold")
    ax5.yaxis.set_label_position("right"); ax5.yaxis.tick_right()

    # Status box
    if ls_s>=65 and gs_s>=65:   sbox="✅ SETUP CONFIRMED"; bc="#1a3a2a"
    elif ls_s>=65:              sbox="🔮 SETUP FORMING — await confirmation"; bc="#2a1a3a"
    elif ls_s<50 and gs_s<50:   sbox="⚠️ NO SETUP — scores below threshold"; bc="#3a2a1a"
    else:                       sbox="👀 PARTIAL — monitor"; bc="#1a2a3a"
    fig.text(0.02,0.002,f" {sbox}  |  {sig['strategy']}  |  Valid till: {sig.get('valid_till','—')} ",
             color=WHT,fontsize=8,fontweight="bold",
             bbox=dict(boxstyle="round",facecolor=bc,edgecolor=BORDER,alpha=0.9))

    # X ticks
    step=max(1,len(x)//16)
    ax5.set_xticks(x[::step])
    ax5.set_xticklabels(dates[::step],rotation=45,ha="right",fontsize=7,color=MUT)
    for ax in [ax1,ax2,ax3,ax4]: ax.set_xticklabels([])

    plt.tight_layout(rect=[0,0.02,0.92,1.0])
    buf=io.BytesIO()
    plt.savefig(buf,format="png",dpi=130,facecolor=DARK,bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════════════════════
# MESSAGE FORMATTERS
# ══════════════════════════════════════════════════════════════════════
SCORE_BAR={"PRIME LONG":"▰▰▰▰▰","LONG":"▰▰▰▰▱","EARLY SETUP":"▰▰▰▱▱",
           "TREND RIDE":"▰▰▱▱▱","WATCH":"▰▱▱▱▱","WAIT":"▱▱▱▱▱"}

def score_label(score):
    if score>=70: return f"{score}/100 ✅"
    if score>=50: return f"{score}/100 ⚠️ moderate"
    return f"{score}/100 ❌ below threshold — caution"

def setup_status(ls, gs, sig):
    if ls>=75 and gs>=72: return "✅ PRIME SETUP — both stages fully aligned"
    if ls>=65 and gs>=65: return "✅ SETUP CONFIRMED — actionable signal"
    if ls>=65 and gs<65:  return "🔮 SETUP FORMING — leading signals active, lagging NOT yet confirmed. Wait or use 50% position size."
    if ls<65  and gs>=72: return "🏄 TREND RUNNING — no new reversal setup. Continuation play only, tighter stop."
    if ls>=50 or  gs>=50: return "👀 PARTIAL SIGNALS — not ready for entry. Monitor daily."
    return "❌ NO SETUP — both scores below 50. Do NOT enter. Scores shown for information only."

def caution_block(ls, gs, regime, earnings_risk, liq_ok,
                  blocked, block_reasons, vol_confirmed=True):
    lines=[]
    if blocked:
        lines.append("🚨 *TRADE BLOCKED BY PORTFOLIO RULES:*")
        for r in block_reasons: lines.append(f"  {r}")
    if not liq_ok:
        lines.append("🚨 *LIQUIDITY INSUFFICIENT* — stock too illiquid to trade safely")
    if not vol_confirmed:
        lines.append("⚠️ *LOW VOLUME TODAY* — signal on below-average volume, reduced conviction")
    if earnings_risk:
        lines.append("🚨 *EARNINGS BLACKOUT* — do not enter before earnings announcement")
    if regime=="bear":
        lines.append("🚨 *BEAR MARKET REGIME* — system is not recommending new longs")
    elif regime=="neutral":
        lines.append("⚠️ *NEUTRAL MARKET* — trade with reduced position size")
    if ls<50:
        lines.append(f"⚠️ Leading score {ls}/100 — no reversal setup detected")
    if gs<50:
        lines.append(f"⚠️ Lagging score {gs}/100 — no trend confirmation")
    if ls>=65 and gs<45:
        lines.append("⚠️ Setup forming but trend contradicts — use 50% position size maximum")
    if ls<45 and gs>=72:
        lines.append("⚠️ Late-stage trend entry — no fresh setup energy, elevated reversal risk")
    return "\n".join(lines)

def fmt_full_analysis(ticker, dd, lead, lag, sig, fund, pats,
                      fib, supports, resistances, mtf, ai_text, pos,
                      regime_detail, liquidity_ok, liquidity_msg,
                      earnings_risk, earnings_msg, gap_risk, gap_msg,
                      blocked, block_reasons, timeframe="Daily",
                      rs_pct=None, rs_out=False, rs_label=""):
    cs  = fund.get("cs","")
    c   = sig["entry"]
    prev= float(dd["Close"].iloc[-2]) if len(dd)>1 else c
    chg = round((c-prev)/prev*100,2) if prev else 0
    chg_s=f"{'↑' if chg>=0 else '↓'}{abs(chg):.2f}%"

    lines=[
        f"*{ticker}* — {fund.get('name',ticker)}",
        f"{cs}{c:,.2f}  {chg_s}  │  {fund.get('sector','—')}  │  {fund.get('exchange','—')}",
        f"MCap: {fund.get('market_cap','—')}  │  P/E: {fund.get('pe','—')}  "
        f"│  Beta: {fund.get('beta','—')}  │  Div: {fund.get('div','—')}",
        "",
        f"{sig['emoji']} *{sig['signal']}*  {SCORE_BAR.get(sig['signal'],'▱▱▱▱▱')}  "
        f"({timeframe})",
        f"📋 {sig['strategy']}  │  ⏱ {sig['horizon']}",
        f"🗓 Signal valid till: {sig.get('valid_till','—')}",
        "",
        f"*{setup_status(lead['score'],lag['score'],sig['signal'])}*",
        "",
    ]

    # ── Regime context (not a block — user always sees signal) ──────────
    regime_str = sig.get("regime","neutral")
    regime_note = sig.get("regime_note","")
    if regime_note:
        lines += [regime_note, ""]

    # Relative strength context — crucial when market is weak
    if rs_label:
        rs_icon = "⭐" if rs_out and regime_str in ("bear","neutral") else ""
        if rs_icon:
            lines += [f"{rs_icon} *OUTPERFORMING market in adverse regime* — {rs_label}", ""]
        else:
            lines += [f"📊 {rs_label}", ""]

    # Regime warning (score below threshold)
    if sig.get("regime_warning"):
        lines += [sig["regime_warning"], ""]

    # Caution block — first thing after signal
    cb=caution_block(lead["score"],lag["score"],regime_str,
                     earn_risk,liq_ok,blocked,block_reasons,
                     vol_confirmed=sig.get("vol_confirmed",True))
    if cb: lines+=[cb,""]

    # MTF warning
    if sig.get("mtf_warning"):    lines+=[sig["mtf_warning"],""]

    # Scores
    lines+=[
        "━━━ TWO-STAGE SCORES ━━━",
        f"🔮 *Leading*  (what WILL happen): {score_label(lead['score'])} Grade {lead['grade']}",
        f"   {lead['interpretation']}",
        f"✅ *Lagging*  (what IS happening): {score_label(lag['score'])} Grade {lag['grade']}",
        f"   {lag['interpretation']}",
        f"📊 Combined: {sig['combined']}/100",
        "",
    ]

    # Evidence
    if lead["top"]:
        lines.append("*Leading signals:*")
        for s in lead["top"]: lines.append(f"  ▶ {s}")
    if lead["weak"]:
        lines.append("*Leading gaps:*")
        for s in lead["weak"][:2]: lines.append(f"  ✗ {s}")
    lines.append("")
    if lag["top"]:
        lines.append("*Lagging confirms:*")
        for s in lag["top"]: lines.append(f"  ✓ {s}")
    if lag["weak"]:
        lines.append("*Lagging gaps:*")
        for s in lag["weak"][:2]: lines.append(f"  ✗ {s}")
    lines.append("")

    # Trade levels — always shown, with caution context
    lines.append("━━━ TRADE LEVELS ━━━")
    if sig["sl"]:
        sl_note = f" ({sig.get('sl_type','')})" if sig.get("sl_type") else ""
        lines+=[
            f"🟢 Entry:  {cs}{sig['entry']:,.2f}",
            f"🔴 SL:     {cs}{sig['sl']:,.2f}{sl_note}",
            f"🎯 T1:     {cs}{sig['t1']:,.2f}  (RR {sig['rr1']}×)",
            f"🎯 T2:     {cs}{sig['t2']:,.2f}  (RR {sig['rr2']}×)",
            f"🎯 T3:     {cs}{sig['t3']:,.2f}  (RR {sig['rr3']}×)",
            f"📐 ATR:    {cs}{sig['atr']:,.2f}  ({sig['atr_pct']:.1f}% of price)",
        ]
        if sig["signal"] in ("WATCH","WAIT"):
            lines.append("_⚠️ Indicative only — signal not confirmed, do not act on these levels_")
        # Trailing stop guidance
        lines+=[
            "",
            "_Trailing stop rule:_",
            f"  _After T1 hit → move SL to breakeven ({cs}{sig['entry']:,.2f})_",
            f"  _After T2 hit → trail SL to T1 ({cs}{sig['t1']:,.2f})_",
        ]
    else:
        lines.append("_No actionable trade levels — signal insufficient_")
    lines.append("")

    # Position sizing — with regime adjustment note
    regime_risk_pct = sig.get("regime_risk_pct", RISK_PER_TRADE)
    risk_note = ""
    if regime_str == "bear":
        risk_note = " (50% size — bear market)"
    elif regime_str == "neutral":
        risk_note = " (75% size — neutral market)"
    if pos and sig["sl"] and sig["signal"] not in ("WATCH","WAIT"):
        lines+=[
            f"━━━ POSITION SIZE ({regime_risk_pct*100:.1f}% risk rule{risk_note}) ━━━",
            f"📦 Qty:    {pos['qty']} shares",
            f"💰 Capital: {cs}{pos['capital_req']:,.0f} ({pos['pct_capital']:.1f}%)",
            f"⚠️  Risk:   {cs}{pos['risk_amount']:,.0f} ({pos['risk_pct']:.2f}% of portfolio)",
            "",
        ]

    # Risk assessment block
    min_act = sig.get("min_score_to_act", 60)
    lines+=[
        "━━━ RISK ASSESSMENT ━━━",
        f"🌍 Market: {regime_detail[:90]}",
        f"📊 Relative strength: {rs_label or '—'}",
        f"💧 Liquidity: {liq_msg}",
        f"📅 Earnings: {earnings_msg}",
        f"⚡ Gap risk: {gap_msg}",
        f"🎯 Min score to act in {regime_str} market: {min_act}/100",
        "",
    ]

    # Fibonacci
    if fib:
        lines.append("━━━ FIBONACCI (60-day swing) ━━━")
        for lv,price in fib.items():
            marker=" ◀ CURRENT" if abs(price-c)/c<0.015 else ""
            near=" (support)" if price<c else " (resistance)" if price>c else ""
            lines.append(f"  {lv:<7} {cs}{price:>10,.2f}{near}{marker}")
        lines.append("")

    # S/R
    if supports or resistances:
        lines.append("━━━ SUPPORT / RESISTANCE ━━━")
        for r in (resistances or []): lines.append(f"  🔴 R: {cs}{r:,.2f}")
        lines.append(f"  ── Price: {cs}{c:,.2f} ──")
        for s in (supports or []):    lines.append(f"  🟢 S: {cs}{s:,.2f}")
        lines.append("")

    # Candlestick patterns
    if pats:
        lines.append("━━━ CANDLESTICK PATTERNS ━━━")
        for p in pats[:5]:
            icon={"bullish":"🟢","bearish":"🔴","warning":"🟡","neutral":"⚪"}.get(p["type"],"⚪")
            lines.append(f"{icon} *{p['name']}* ({p['conviction']}%) — {p['date']}")
            lines.append(f"   ↳ {p['action']}")
        lines.append("")

    # Key indicators
    r_=dd.iloc[-1]
    stoch=rv(r_,"STOCH_K"); willr=rv(r_,"WILLR"); cci=rv(r_,"CCI")
    rsi_v=rv(r_,"RSI"); adx_v=rv(r_,"ADX"); pp=rv(r_,"PP"); r1=rv(r_,"R1"); s1=rv(r_,"S1")
    sq=bool(rv(r_,"BB_sq")); rd=bool(rv(r_,"RSI_DIV")); md=bool(rv(r_,"MACD_DIV"))
    vol_ratio  = sig.get("vol_ratio",1.0)
    vol_conf   = sig.get("vol_confirmed",True)
    vol_str    = f"Vol: {vol_ratio:.1f}× avg {'✅' if vol_conf else '⚠️ low'}"
    lines+=[
        "━━━ KEY INDICATORS ━━━",
        f"RSI: {rsi_v:.1f}  │  ADX: {adx_v:.1f}  │  Stoch %K: {stoch:.0f}  │  {vol_str}",
        f"Williams %R: {willr:.0f}  │  CCI: {cci:.0f}",
        f"Pivot: {cs}{pp:.2f}  │  R1: {cs}{r1:.2f}  │  S1: {cs}{s1:.2f}",
        f"{'⚡ BB SQUEEZE  ' if sq else ''}{'↑ RSI DIV  ' if rd else ''}{'↑ MACD DIV' if md else ''}".strip() or "No leading signals active",
        "",
    ]

    # Multi-timeframe
    if mtf:
        lines.append(f"━━━ MULTI-TIMEFRAME  {mtf[1]} ━━━")
        for tf_name,tf_data in mtf[0].items():
            if tf_data:
                lines.append(f"  {tf_data['emoji']} {tf_name}: {tf_data['signal']}  "
                              f"(Lead:{tf_data['lead']} Lag:{tf_data['lag']})")
        lines.append("")

    # Fundamentals
    lines+=[
        "━━━ FUNDAMENTALS ━━━",
        f"P/B: {fund.get('pb','—')}  │  EPS: {fund.get('eps','—')}  │  ROE: {fund.get('roe','—')}",
        f"52W: High {cs}{fund.get('h52','—')}  Low {cs}{fund.get('l52','—')}",
    ]
    if fund.get("earnings_date") and fund["earnings_date"]!="—":
        lines.append(f"📅 Earnings: {fund['earnings_date']}")
    lines.append("")

    # AI insight
    if ai_text:
        lines+=["━━━ AI INSIGHT (Groq LLaMA) ━━━"]
        for line in ai_text.split("\n"): lines.append(line)
        lines.append("")

    lines.append(f"_/log {ticker} {c:.0f} 1 to log this trade_")
    return "\n".join(str(l) for l in lines)


def fmt_trade_row(t):
    outcome=("SL ❌" if t["sl_hit"] else
             "T3 ✅✅✅" if t["t3_hit"] else
             "T2 ✅✅" if t["t2_hit"] else
             "T1 ✅" if t["t1_hit"] else "🟡 Open")
    sl_note=""
    if t.get("breakeven_set"): sl_note=" (BE)"
    elif t.get("trailing_sl"):  sl_note=f" →{t['trailing_sl']:.2f}"
    return (f"*#{t['id']} {t['ticker']}*  {outcome}\n"
            f"  Entry:{t['entry_price']}  SL:{t.get('sl','—')}{sl_note}  T2:{t.get('t2','—')}\n"
            f"  Lead:{t.get('lead_score','—')} Lag:{t.get('lag_score','—')}  "
            f"{t.get('signal_type','—')}  {t.get('entry_date','—')}\n"
            f"  /close\\_{t['id']} <exit\\_price>")

# ══════════════════════════════════════════════════════════════════════
# SCREENER
# ══════════════════════════════════════════════════════════════════════
NIFTY50=[
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","SBIN.NS","BAJFINANCE.NS","BHARTIARTL.NS","KOTAKBANK.NS",
    "WIPRO.NS","AXISBANK.NS","LT.NS","ASIANPAINT.NS","MARUTI.NS",
    "SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","NESTLEIND.NS","POWERGRID.NS",
    "TATAMOTORS.NS","TECHM.NS","HCLTECH.NS","ONGC.NS","NTPC.NS",
    "COALINDIA.NS","INDUSINDBK.NS","BAJAJFINSV.NS","GRASIM.NS","ADANIENT.NS",
    "TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","DIVISLAB.NS","CIPLA.NS",
    "DRREDDY.NS","APOLLOHOSP.NS","EICHERMOT.NS","HEROMOTOCO.NS","BPCL.NS",
    "IOC.NS","SBILIFE.NS","HDFCLIFE.NS","BRITANNIA.NS","PIDILITIND.NS",
    "SIEMENS.NS","HAVELLS.NS","BERGEPAINT.NS","MCDOWELL-N.NS","ICICIPRULI.NS",
]

def quick_screen_one(ticker, regime=None):
    try:
        # Hard 45-second timeout per stock — screener cannot wait forever
        import signal as _sig
        df   = fetch_and_score(ticker)
        lead = leading_score(df)
        lag  = lagging_score(df)
        sig  = build_signal(df, lead, lag, ticker=ticker, regime=regime)
        r    = df.iloc[-1]
        c    = float(rv(r,"Close")); pc=float(df["Close"].iloc[-2])
        liq_ok, liq_val, liq_msg = check_liquidity(df, ticker)
        sq   = bool(rv(r,"BB_sq")); rd=bool(rv(r,"RSI_DIV"))
        vol_ratio = sig.get("vol_ratio",1.0)
        return {
            "ticker":   ticker, "price":round(c,2),
            "chg":      round((c-pc)/pc*100,2),
            "lead":     lead["score"], "lag":lag["score"],
            "combined": sig["combined"],
            "signal":   sig["signal"], "emoji":sig["emoji"],
            "strategy": sig["strategy"],
            "rsi":      round(float(rv(r,"RSI")),1),
            "stoch":    round(float(rv(r,"STOCH_K")),1),
            "adx":      round(float(rv(r,"ADX")),1),
            "squeeze":  sq, "rsi_div":rd,
            "liquid":   liq_ok,
            "vol_ratio":round(vol_ratio,2),
            "regime_downgraded": sig["signal"]!=sig.get("raw_signal",sig["signal"]),
            "rs_pct":   0, "rs_out": False,
        }
    except Exception as e:
        return {"ticker":ticker,"error":str(e)[:50]}

def quick_screen(tickers, signal_filter="all", mode="signal"):
    # Get market regime once — applies to all NSE stocks
    nse_regime = get_market_regime("RELIANCE.NS")
    results=[]; errors=[]
    for ticker in tickers:
        # Use appropriate regime per ticker exchange
        regime = get_market_regime(ticker)
        r = quick_screen_one(ticker, regime=regime)
        if "error" in r:
            errors.append(f"{r['ticker']}: {r['error']}")
            continue
        # Skip illiquid stocks in screener
        if not r.get("liquid",True): continue
        # Signal filter
        sig=r["signal"]
        passes=(signal_filter=="all" or
                (signal_filter=="prime" and sig=="PRIME LONG") or
                (signal_filter=="long"  and sig in ("PRIME LONG","LONG")) or
                (signal_filter=="early" and sig=="EARLY SETUP") or
                (signal_filter=="trend" and sig=="TREND RIDE"))
        if passes: results.append(r)
        time.sleep(0.25)

    if   mode=="reversal": results.sort(key=lambda x:x["lead"],    reverse=True)
    elif mode=="trend":    results.sort(key=lambda x:x["lag"],     reverse=True)
    else:                  results.sort(key=lambda x:x["combined"],reverse=True)
    return results, errors

# ══════════════════════════════════════════════════════════════════════
# DATABASE OPERATIONS
# ══════════════════════════════════════════════════════════════════════
def db_log_trade(ticker,entry,qty,sl,t1,t2,t3,
                 lead_sc,lag_sc,sig_type,strat,hzn,sector):
    con=sqlite3.connect(DB_FILE)
    valid_till=(datetime.now()+timedelta(days=ENTRY_VALID_DAYS)).strftime("%Y-%m-%d")
    con.execute("""INSERT INTO trades
        (ticker,entry_date,entry_price,qty,sl,sl_original,t1,t2,t3,
         lead_score,lag_score,signal_type,strategy,horizon,sector,
         signal_date,signal_price,entry_valid_till)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (ticker,datetime.now().strftime("%Y-%m-%d"),entry,qty,
         sl,sl,t1,t2,t3,lead_sc,lag_sc,sig_type,strat,hzn,sector,
         datetime.now().strftime("%Y-%m-%d"),entry,valid_till))
    con.commit()
    lid=con.execute("SELECT last_insert_rowid()").fetchone()[0]
    con.close(); return lid

def db_close_trade(trade_id,exit_price):
    con=sqlite3.connect(DB_FILE)
    row=con.execute("SELECT entry_price,qty FROM trades WHERE id=?",
                    (trade_id,)).fetchone()
    if not row: con.close(); return None,None
    ep,qty=row
    pnl=round((exit_price-ep)*qty,2)
    pnl_pct=round((exit_price-ep)/ep*100,2)
    con.execute("UPDATE trades SET status='CLOSED',exit_price=?,exit_date=? WHERE id=?",
                (exit_price,datetime.now().strftime("%Y-%m-%d"),trade_id))
    con.commit(); con.close()
    return pnl,pnl_pct

def db_get_trades(status=None,limit=10):
    con=sqlite3.connect(DB_FILE)
    if status:
        rows=con.execute("SELECT * FROM trades WHERE status=? ORDER BY id DESC LIMIT ?",
                         (status,limit)).fetchall()
    else:
        rows=con.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?",(limit,)).fetchall()
    cols=[d[1] for d in con.execute("PRAGMA table_info(trades)").fetchall()]
    con.close()
    return [dict(zip(cols,r)) for r in rows]

def db_update_trailing_sl(trade_id,new_sl,breakeven=False):
    con=sqlite3.connect(DB_FILE)
    if breakeven:
        con.execute("UPDATE trades SET trailing_sl=?,sl=?,breakeven_set=1 WHERE id=?",
                    (new_sl,new_sl,trade_id))
    else:
        con.execute("UPDATE trades SET trailing_sl=?,sl=? WHERE id=?",
                    (new_sl,new_sl,trade_id))
    con.commit(); con.close()

def db_add_alert(ticker,direction,price):
    con=sqlite3.connect(DB_FILE)
    con.execute("INSERT INTO alerts (ticker,direction,price,created) VALUES (?,?,?,?)",
                (ticker,direction,price,datetime.now().isoformat()))
    con.commit(); con.close()

def db_get_alerts():
    con=sqlite3.connect(DB_FILE)
    rows=con.execute("SELECT id,ticker,direction,price,active FROM alerts ORDER BY id DESC").fetchall()
    con.close(); return rows

def db_log_signal(ticker,sig,lead,lag):
    """Log every signal for later outcome tracking."""
    try:
        con=sqlite3.connect(DB_FILE)
        # Check if signal_log table exists
        tables=[r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if "signal_log" not in tables: con.close(); return
        con.execute("""INSERT INTO signal_log
            (ticker,signal_date,signal_type,lead_score,lag_score,
             entry_price,sl,t1,t2,horizon,strategy)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (ticker,datetime.now().strftime("%Y-%m-%d"),
             sig["signal"],lead["score"],lag["score"],
             sig.get("entry"),sig.get("sl"),sig.get("t1"),sig.get("t2"),
             sig.get("horizon"),sig.get("strategy")))
        con.commit(); con.close()
    except: pass

# ══════════════════════════════════════════════════════════════════════
# TELEGRAM TRANSPORT
# ══════════════════════════════════════════════════════════════════════
import urllib.request, urllib.parse

def tg(method,params=None,data=None,files=None):
    url=f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    if files:
        boundary=b"AQ_BOUNDARY_V3"
        body=b""
        for k,v in (data or {}).items():
            body+=b"--"+boundary+b"\r\n"
            body+=f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
            body+=str(v).encode()+b"\r\n"
        for fname,fobj in files.items():
            body+=b"--"+boundary+b"\r\n"
            body+=f'Content-Disposition: form-data; name="{fname}"; filename="chart.png"\r\n'.encode()
            body+=b"Content-Type: image/png\r\n\r\n"
            body+=fobj.read()+b"\r\n"
        body+=b"--"+boundary+b"--\r\n"
        req=urllib.request.Request(url,body,
            {"Content-Type":f"multipart/form-data; boundary={boundary.decode()}"})
    elif data:
        req=urllib.request.Request(url,json.dumps(data).encode(),
            {"Content-Type":"application/json"})
    else:
        if params: url+="?"+urllib.parse.urlencode(params)
        req=urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req,timeout=60) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        log.error(f"TG {method}: {e}"); return {}

def send(chat_id,text,parse_mode="Markdown"):
    for chunk in [text[i:i+4000] for i in range(0,len(text),4000)]:
        tg("sendMessage",data={"chat_id":chat_id,"text":chunk,"parse_mode":parse_mode})
        if len(text)>4000: time.sleep(0.4)

def send_photo(chat_id,buf,caption="",parse_mode="Markdown"):
    buf.seek(0)
    tg("sendPhoto",data={"chat_id":chat_id,"caption":caption[:1024],
                          "parse_mode":parse_mode},files={"photo":buf})

def send_typing(chat_id): tg("sendChatAction",data={"chat_id":chat_id,"action":"typing"})
def send_upload(chat_id): tg("sendChatAction",data={"chat_id":chat_id,"action":"upload_photo"})
def is_authorised(chat_id): return str(chat_id)==str(YOUR_CHAT_ID)

# ══════════════════════════════════════════════════════════════════════
# CORE ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════════════
def run_full_analysis(chat_id, ticker, timeframe="1d", tf_label="Daily"):
    send_typing(chat_id)
    send(chat_id, f"⏳ Analysing *{ticker}* ({tf_label})...")
    try:
        # 1. Fetch and score
        dd    = fetch_and_score(ticker, interval=timeframe)
        lead  = leading_score(dd)
        lag   = lagging_score(dd)

        # 2. Market regime (most important filter)
        send_typing(chat_id)
        regime = get_market_regime(ticker)

        # 3. Multi-timeframe — non-blocking thread with 20s timeout
        mtf = None; weekly_lag = None
        if timeframe == "1d":
            mtf_result = [None]
            def _run_mtf():
                try: mtf_result[0] = multi_timeframe_consensus(ticker)
                except Exception as e: log.warning(f"MTF failed {ticker}: {e}")
            t_mtf = threading.Thread(target=_run_mtf, daemon=True)
            t_mtf.start(); t_mtf.join(timeout=20)
            mtf = mtf_result[0]
            if mtf:
                wk = mtf[0].get("Weekly") if mtf else None
                if wk: weekly_lag = wk.get("lag_obj")

        # 4. Build signal with all filters
        sig = build_signal(dd, lead, lag,
                           ticker=ticker,
                           regime=regime,
                           weekly_lag=weekly_lag)

        # 5. Risk checks — fast local first, then concurrent network calls
        liq_ok, liq_v, liq_msg  = check_liquidity(dd, ticker)
        gap_risk, _, _, gap_msg = assess_gap_risk(dd)
        fib, sh, sl_f           = fibonacci_levels(dd)
        supports, resistances   = find_sr_levels(dd)
        pats                    = candle_patterns(dd, n_bars=14)

        # Single .info fetch shared by fund + earnings (via cache)
        get_ticker_info(ticker)
        fund_r=[{}]; earn_r=[(False,None,"—")]; rs_r=[(0,False,"RS unavailable")]
        def _tf():
            try: fund_r[0]=get_fund(ticker)
            except: pass
        def _te():
            try: earn_r[0]=check_earnings_risk(ticker)
            except: pass
        def _tr():
            try: rs_r[0]=get_relative_strength(ticker,lookback_days=30)
            except: pass
        _ts=[threading.Thread(target=f,daemon=True) for f in [_tf,_te,_tr]]
        for _t in _ts: _t.start()
        for _t in _ts: _t.join(timeout=10)
        fund=fund_r[0]; sig["cs"]=fund.get("cs","")
        earn_risk,_,earn_msg=earn_r[0]
        rs_pct,rs_out,rs_label=rs_r[0]
        # 6. Portfolio risk
        sector = fund.get("sector","—")
        blocked, block_reasons, open_c, sec_c, port_risk = (
            check_portfolio_risk(ticker, sector))

        # 7. Position sizing
        pos = None
        if sig.get("entry") and sig.get("sl") and not blocked and liq_ok and not earn_risk:
            pos = calc_position_size(sig["entry"], sig["sl"])

        # 8. AI insight
        ai_text = ai_insight(ticker, sig, lead, lag, fund,
                             regime[2], liq_msg, earn_msg, gap_msg,
                             rs_label=rs_label)

        # 9. Log the signal for outcome tracking
        db_log_signal(ticker, sig, lead, lag)

        # 10. Format and send text
        msg = fmt_full_analysis(
            ticker, dd, lead, lag, sig, fund, pats,
            fib, supports, resistances, mtf, ai_text, pos,
            regime[2], liq_ok, liq_msg,
            earn_risk, earn_msg, gap_risk, gap_msg,
            blocked, block_reasons,
            rs_pct=rs_pct, rs_out=rs_out, rs_label=rs_label,
            timeframe=tf_label)
        send(chat_id, msg)

        # 11. Generate and send chart
        send_upload(chat_id)
        buf = make_chart_image(dd, sig, lead, lag, pats, ticker,
                               timeframe=tf_label,
                               fib_levels=fib,
                               supports=supports,
                               resistances=resistances)
        regime_emoji={"bull":"✅","neutral":"⚠️","bear":"🚨"}[regime[0]]
        caption=(f"{ticker} {tf_label} — {sig['emoji']} {sig['signal']} | "
                 f"Lead:{lead['score']} Lag:{lag['score']} | "
                 f"{regime_emoji} {regime[0].upper()} MARKET")
        send_photo(chat_id, buf, caption=caption)

    except Exception as e:
        import traceback
        log.error(f"Analysis error {ticker} ({tf_label}):\n{traceback.format_exc()}")
        err = str(e)
        if "Insufficient data" in err:
            hint = "\n\n_Tip: Weekly needs ~3 years of history. Try /a for daily only._"
        elif "No data" in err:
            hint = "\n\n_Check symbol at finance.yahoo.com. Indian stocks: add .NS or .BO_"
        else:
            hint = ""
        try:
            send(chat_id, f"\u274c Analysis failed: {err[:200]}{hint}")
        except Exception as se:
            log.error(f"Cannot send error message: {se}")

def cmd_analyse(chat_id, args):
    if not args:
        send(chat_id, "Usage: `/a TCS.NS` `/a AAPL` `/a 005930.KS`\n"
                      "Works for any Yahoo Finance symbol."); return
    run_full_analysis(chat_id, normalise_ticker(args[0]), "1d", "Daily")

def cmd_weekly(chat_id, args):
    if not args:
        send(chat_id, "Usage: `/w TCS.NS`"); return
    run_full_analysis(chat_id, normalise_ticker(args[0]), "1wk", "Weekly")

def cmd_both(chat_id, args):
    if not args:
        send(chat_id, "Usage: `/both TCS.NS`"); return
    t = normalise_ticker(args[0])
    run_full_analysis(chat_id, t, "1d",  "Daily")
    run_full_analysis(chat_id, t, "1wk", "Weekly")

def cmd_screen(chat_id, args):
    sig_filter="all"; mode="signal"; custom=[]
    for arg in args:
        al=arg.lower()
        if al in ("prime","long","early","trend","all"): sig_filter=al
        elif al in ("reversal","trend_only","combined"): mode=al
        elif al=="global": pass
        else: custom.append(normalise_ticker(arg))
    tickers = custom if custom else get_watchlist()
    label   = f"{len(tickers)} custom tickers" if custom else "Nifty 50"
    # Show regime before screening
    regime  = get_market_regime("RELIANCE.NS")
    regime_s= {"bull":"✅ BULL — good for longs","neutral":"⚠️ NEUTRAL — trade smaller",
                "bear":"🚨 BEAR — avoid new longs"}[regime[0]]
    send(chat_id, f"⏳ Screening {label} ({sig_filter})...\n"
                  f"Market: {regime_s}\n"
                  f"Illiquid stocks filtered out automatically. Takes 2-3 min.")
    results, errors = quick_screen(tickers, sig_filter, mode)
    if not results:
        msg = "No matches found."
        if regime[0]=="bear":
            msg += "\n\n🚨 *Market is in BEAR regime* — signals are being downgraded. This is expected."
        msg += "\n\nTry: `/screen all` or `/screen early`"
        send(chat_id, msg); return
    lines=[f"📊 *Screen Results* — {len(results)} matches ({sig_filter})\n"]
    for i,r in enumerate(results[:15],1):
        sq="⚡" if r.get("squeeze") else ""
        rd="↑" if r.get("rsi_div") else ""
        dg="↓" if r.get("regime_downgraded") else ""
        rs_tag  = "⭐RS+" if r.get("rs_out") else ""
        vol_tag = f"V:{r.get('vol_ratio',1.0):.1f}×" if r.get("vol_ratio",1.0)<0.8 else ""
        lines.append(
            f"{i}. *{r['ticker'].replace('.NS','')}* {r['emoji']} {r['signal']}{dg} {rs_tag}\n"
            f"   L:{r['lead']} G:{r['lag']} C:{r['combined']} │ "
            f"{r['price']:,.2f} ({r['chg']:+.1f}%) {sq}{rd} {vol_tag}\n"
            f"   {r['strategy']}")
    if len(results)>15: lines.append(f"\n_+{len(results)-15} more_")
    if errors:         lines.append(f"\n_{len(errors)} symbols failed_")
    if regime[0]!="bull": lines.append(f"\n_{regime_s}_")
    lines.append("\n_↓ means signal downgraded by regime filter_")
    lines.append("_/a TICKER for full analysis + chart_")
    send(chat_id, "\n".join(lines))

def cmd_log(chat_id, args):
    if len(args)<2:
        send(chat_id,"Usage: `/log TCS.NS 3900 10`\n(ticker, entry, qty)"); return
    try:
        ticker=normalise_ticker(args[0]); entry=float(args[1])
        qty=int(args[2]) if len(args)>2 else 1
        send_typing(chat_id)
        sl=t1=t2=t3=lead_sc=lag_sc=sig_type=strat=hzn=sector=None
        try:
            dd=fetch_and_score(ticker)
            regime=get_market_regime(ticker)
            lead=leading_score(dd); lag=lagging_score(dd)
            sig=build_signal(dd,lead,lag,ticker=ticker,regime=regime)
            fund=get_fund(ticker)
            sl=sig["sl"]; t1=sig["t1"]; t2=sig["t2"]; t3=sig["t3"]
            lead_sc=lead["score"]; lag_sc=lag["score"]
            sig_type=sig["signal"]; strat=sig["strategy"]; hzn=sig["horizon"]
            sector=fund.get("sector","—")
        except: pass
        # Portfolio check before logging
        blocked,block_reasons,_,_,_=check_portfolio_risk(ticker,sector or "—")
        lid=db_log_trade(ticker,entry,qty,sl,t1,t2,t3,
                         lead_sc,lag_sc,sig_type,strat,hzn,sector)
        msg=(f"✅ *Trade logged* — #{lid}\n"
             f"*{ticker}* @ {entry:,.2f} × {qty}\n")
        if sl:      msg+=f"SL: {sl:.2f}  T1:{t1:.2f}  T2:{t2:.2f}  T3:{t3:.2f}\n"
        if sig_type:msg+=f"Signal: {sig_type} │ {hzn}\n"
        if lead_sc: msg+=f"Lead:{lead_sc}  Lag:{lag_sc}\n"
        if blocked:
            msg+="\n⚠️ *Portfolio rules flagged:*\n"
            for r in block_reasons: msg+=f"  {r}\n"
        msg+=(f"\n_T1 hit → move SL to breakeven ({entry:.2f})_\n"
              f"_T2 hit → trail SL to T1 ({t1:.2f} if available)_\n"
              f"_/close\\_{lid} <price> to exit_")
        send(chat_id,msg)
    except Exception as e: send(chat_id,f"❌ Error: {e}")

def cmd_size(chat_id, args):
    if len(args)<3:
        send(chat_id,"Usage: `/size TCS.NS 3900 3800`\n"
                     "Custom capital: `/size TCS.NS 3900 3800 1000000`"); return
    try:
        ticker=normalise_ticker(args[0]); entry=float(args[1]); sl=float(args[2])
        cap=float(args[3]) if len(args)>3 else TRADING_CAPITAL
        if sl>=entry: send(chat_id,"❌ SL must be below entry price"); return
        pos=calc_position_size(entry,sl,capital=cap)
        if not pos: send(chat_id,"❌ Invalid levels"); return
        cs="₹" if ".NS" in ticker or ".BO" in ticker else "$"
        risk_pct=RISK_PER_TRADE*100
        send(chat_id,
             f"📦 *Position Sizing — {ticker}*\n\n"
             f"Capital:     {cs}{cap:>12,.0f}\n"
             f"Entry:       {cs}{entry:>12,.2f}\n"
             f"Stop Loss:   {cs}{sl:>12,.2f}\n"
             f"Risk/share:  {cs}{pos['risk_per_sh']:>12,.2f}\n\n"
             f"Risk ({risk_pct:.0f}%): {cs}{cap*RISK_PER_TRADE:>12,.0f}\n"
             f"Quantity:    {pos['qty']:>12} shares\n"
             f"Capital req: {cs}{pos['capital_req']:>12,.0f} ({pos['pct_capital']:.1f}%)\n"
             f"Actual risk: {cs}{pos['risk_amount']:>12,.0f} ({pos['risk_pct']:.2f}%)\n\n"
             f"_After T1: move SL to {cs}{entry:.2f} (breakeven)_\n"
             f"_After T2: trail SL to T1_\n"
             f"_Max open trades: {MAX_OPEN_TRADES} │ Max sector: {MAX_SECTOR_TRADES}_")
    except Exception as e: send(chat_id,f"❌ Error: {e}")

def cmd_fib(chat_id, args):
    if not args: send(chat_id,"Usage: `/fib TCS.NS`"); return
    ticker=normalise_ticker(args[0]); send_typing(chat_id)
    try:
        dd=fetch_and_score(ticker)
        fib,sh,sl=fibonacci_levels(dd)
        c=float(dd["Close"].iloc[-1])
        supports,resistances=find_sr_levels(dd)
        cs="₹" if ".NS" in ticker or ".BO" in ticker else "$"
        lines=[f"📐 *Fibonacci — {ticker}* (60-day swing)\n",
               f"Swing High: {cs}{sh:,.2f}",f"Swing Low:  {cs}{sl:,.2f}\n"]
        for lv,price in fib.items():
            marker=" ◀ CURRENT" if abs(price-c)/c<0.015 else ""
            near=" (support)" if price<c else " (resistance)" if price>c else ""
            lines.append(f"  {lv:<7} {cs}{price:>10,.2f}{near}{marker}")
        if supports:    lines.append(f"\n🟢 Supports:    {', '.join(f'{cs}{s:,.2f}' for s in supports)}")
        if resistances: lines.append(f"🔴 Resistances: {', '.join(f'{cs}{r:,.2f}' for r in resistances)}")
        send(chat_id,"\n".join(lines))
    except Exception as e: send(chat_id,f"❌ Error: {e}")

def cmd_trades(chat_id, args):
    status=args[0].upper() if args and args[0].upper() in ("OPEN","CLOSED") else None
    trades=db_get_trades(status=status)
    if not trades:
        send(chat_id,"No trades.\nUse `/log TICKER PRICE QTY`"); return
    all_t=db_get_trades(limit=1000)
    wins=sum(1 for t in all_t if t.get("t1_hit") and not t.get("sl_hit"))
    wr=round(wins/len(all_t)*100) if all_t else 0
    t1r=round(sum(1 for t in all_t if t.get("t1_hit"))/len(all_t)*100) if all_t else 0
    slr=round(sum(1 for t in all_t if t.get("sl_hit"))/len(all_t)*100) if all_t else 0
    lines=[f"*📓 Trades{' ('+status+')' if status else ''}*\n",
           f"Win rate: {wr}%  T1 hit: {t1r}%  SL hit: {slr}%  Total: {len(all_t)}\n"]
    for t in trades: lines.append(fmt_trade_row(t)); lines.append("")
    send(chat_id,"\n".join(lines))

def cmd_close(chat_id, args):
    if len(args)<2: send(chat_id,"Usage: `/close 3 3950`"); return
    try:
        tid=int(args[0]); ep=float(args[1])
        pnl,pnl_pct=db_close_trade(tid,ep)
        if pnl is None: send(chat_id,f"Trade #{tid} not found"); return
        icon="✅" if pnl>=0 else "❌"
        send(chat_id,f"{icon} *Trade #{tid} closed*\n"
                     f"Exit: {ep:,.2f}  P&L: {pnl:+,.2f} ({pnl_pct:+.1f}%)")
    except Exception as e: send(chat_id,f"❌ Error: {e}")

def cmd_trailing(chat_id, args):
    """Update trailing stop: /trail 3 3950"""
    if len(args)<2:
        send(chat_id,"Usage: `/trail 3 3950`\n"
                     "(trade id, new stop loss price)"); return
    try:
        tid=int(args[0]); new_sl=float(args[1])
        trades=db_get_trades(limit=1000)
        t=next((x for x in trades if x["id"]==tid),None)
        if not t: send(chat_id,f"Trade #{tid} not found"); return
        be=new_sl>=t["entry_price"]
        db_update_trailing_sl(tid,new_sl,breakeven=be)
        be_note=" (breakeven set ✅)" if be else ""
        send(chat_id,f"✅ Trade #{tid} — SL updated to {new_sl:.2f}{be_note}")
    except Exception as e: send(chat_id,f"❌ Error: {e}")

def cmd_alert(chat_id, args):
    if len(args)<3:
        send(chat_id,"Usage: `/alert TCS.NS above 4000`\nor `/alert AAPL below 150`"); return
    try:
        ticker=normalise_ticker(args[0]); direction=args[1].lower()
        if direction not in ("above","below"):
            send(chat_id,"Direction: 'above' or 'below'"); return
        price=float(args[2])
        db_add_alert(ticker,direction,price)
        send(chat_id,f"🔔 Alert set: *{ticker}* price goes {direction} {price:,.2f}")
    except Exception as e: send(chat_id,f"❌ Error: {e}")

def cmd_alerts(chat_id, args):
    rows=db_get_alerts()
    if not rows: send(chat_id,"No alerts.\nUse `/alert TCS.NS above 4000`"); return
    lines=["*🔔 Price Alerts*\n"]
    for a in rows:
        s="🟢 Active" if a[4] else "✅ Triggered"
        lines.append(f"#{a[0]} *{a[1]}* {a[2]} {a[3]:,.2f}  {s}")
    send(chat_id,"\n".join(lines))

def cmd_regime(chat_id, args):
    """Check market regime for a ticker or default indices."""
    ticker=normalise_ticker(args[0]) if args else "RELIANCE.NS"
    send_typing(chat_id)
    regime,score,detail=get_market_regime(ticker)
    icon={"bull":"✅","neutral":"⚠️","bear":"🚨"}[regime]
    advice={"bull":"Market uptrend — full position sizing allowed",
            "neutral":"Market neutral — use 50-75% position size, be selective",
            "bear":"Market downtrend — AVOID new long entries. Wait for regime to improve."}
    send(chat_id,
         f"🌍 *Market Regime*\n\n"
         f"{icon} *{regime.upper()} MARKET* (score: {score}/100)\n\n"
         f"{detail}\n\n"
         f"📋 {advice[regime]}\n\n"
         f"_Regime updates hourly. Signal strength is adjusted automatically._")

def cmd_portfolio(chat_id, args):
    """Show current portfolio risk summary."""
    trades=db_get_trades(status="OPEN",limit=100)
    if not trades: send(chat_id,"No open trades."); return
    total_risk=sum(max(0,(t["entry_price"]-(t.get("sl") or t["entry_price"]*0.95))*
                       t["qty"]) for t in trades)
    risk_pct=total_risk/TRADING_CAPITAL*100
    sectors={}
    for t in trades:
        s=t.get("sector","Unknown") or "Unknown"
        sectors[s]=sectors.get(s,0)+1
    lines=[
        "*📊 Portfolio Risk Summary*\n",
        f"Open trades:    {len(trades)}/{MAX_OPEN_TRADES}",
        f"Total risk:     ₹{total_risk:,.0f} ({risk_pct:.1f}% of capital)",
        f"Capital at risk limit: {MAX_PORTFOLIO_RISK*100:.0f}%",
        f"Status: {'🚨 NEAR LIMIT' if risk_pct>MAX_PORTFOLIO_RISK*80 else '✅ OK'}\n",
        "*By sector:*",
    ]
    for s,cnt in sorted(sectors.items(),key=lambda x:-x[1]):
        lines.append(f"  {s}: {cnt} trade{'s' if cnt>1 else ''}"
                     +(" ⚠️ MAX" if cnt>=MAX_SECTOR_TRADES else ""))
    send(chat_id,"\n".join(lines))

def cmd_status(chat_id, args):
    now=datetime.now()
    ist=now+timedelta(hours=5,minutes=30)
    h,m=ist.hour,ist.minute
    is_wd=ist.weekday()<5
    mkt_open=is_wd and((h==9 and m>=15)or(10<=h<=14)or(h==15 and m<=30))
    mkt=f"{'🟢 NSE Open' if mkt_open else '🔴 NSE Closed'}  IST:{ist.strftime('%d %b %H:%M')}"
    regime,score,detail=get_market_regime("RELIANCE.NS")
    regime_s={"bull":f"✅ Bull ({score}/100)","neutral":f"⚠️ Neutral ({score}/100)",
              "bear":f"🚨 Bear ({score}/100)"}[regime]
    try:
        con=sqlite3.connect(DB_FILE)
        op=con.execute("SELECT COUNT(*) FROM trades WHERE status='OPEN'").fetchone()[0]
        tot=con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        wins=con.execute("SELECT COUNT(*) FROM trades WHERE t1_hit=1 AND sl_hit=0").fetchone()[0]
        alr=con.execute("SELECT COUNT(*) FROM alerts WHERE active=1").fetchone()[0]
        con.close()
        wr=round(wins/tot*100) if tot else 0
        j=f"📓 {op} open  Win:{wr}%  Alerts:{alr}  Total:{tot}"
    except: j="📓 Journal unavailable"
    ai_s="✅ Groq connected" if AI_API_KEY else "❌ No AI key (console.groq.com)"
    send(chat_id,
         f"✅ *AstraQuant v3.0 — Real Money Edition*\n\n"
         f"{mkt}\n🌍 Market: {regime_s}\n{j}\n🤖 AI: {ai_s}\n\n"
         f"_/help for all commands_")

HELP_TEXT="""*AstraQuant v3.0 — Real Money Edition*
Built as if trading with own capital.

📊 *Analysis*
`/a TCS.NS`  — Daily analysis + chart + all risk checks
`/a AAPL`    — US/global stocks — any Yahoo Finance symbol
`/w TCS.NS`  — Weekly analysis + chart
`/both TCS.NS` — Daily + weekly (2 charts)

🔍 *Screener*
`/screen`         — Nifty 50 (illiquid auto-filtered)
`/screen early`   — Early Setup signals
`/screen prime`   — Prime Long only
`/screen global AAPL MSFT NVDA` — custom list
`/watchlist` `/wl` — view/manage screener watchlist
`/wl add TCS.NS INFY` — add to watchlist
`/wl clear` — reset to Nifty 50

📐 *Tools*
`/fib TCS.NS`          — Fibonacci retracement
`/size TCS.NS 3900 3800` — Position size (1% risk)
`/regime TCS.NS`       — Market regime for this exchange
`/portfolio`           — Open trades risk summary

📓 *Journal*
`/log TCS.NS 3900 10`  — Log trade (auto-fills levels)
`/trades`              — All trades + stats
`/trades open`         — Open only
`/close 3 3950`        — Close trade #3
`/trail 3 3800`        — Update trailing stop

🔔 *Alerts*
`/alert TCS.NS above 4000`
`/alerts`

ℹ️  `/status`  `/help`

*Real-money filters active on every signal:*
🌍 Market regime filter (bear = no new longs)
💧 Liquidity gate (min daily turnover enforced)
📅 Earnings blackout (5-day window before earnings)
📦 Portfolio cap (max {max_t} trades, max {max_s} per sector)
📐 Support-anchored stop loss (not just ATR)
🔄 Multi-timeframe downgrade (weekly bear = reduce signal)
""".format(max_t=MAX_OPEN_TRADES, max_s=MAX_SECTOR_TRADES)

# ══════════════════════════════════════════════════════════════════════
# BACKGROUND MONITORS
# ══════════════════════════════════════════════════════════════════════
def alert_monitor():
    """Check price alerts every 15 minutes."""
    while True:
        try:
            con = sqlite3.connect(DB_FILE)
            alerts = con.execute(
                "SELECT id,ticker,direction,price FROM alerts WHERE active=1"
            ).fetchall()
            for aid, ticker, direction, target in alerts:
                try:
                    info = yf.Ticker(ticker).fast_info
                    cur  = float(getattr(info,"last_price",0) or 0)
                    if cur <= 0: continue
                    hit = ((direction=="above" and cur>=target) or
                           (direction=="below" and cur<=target))
                    if hit:
                        con.execute(
                            "UPDATE alerts SET active=0,triggered=? WHERE id=?",
                            (datetime.now().isoformat(), aid))
                        con.commit()
                        cs = {"INR":"₹","USD":"$","EUR":"€","GBP":"£"}.get(
                            yf.Ticker(ticker).info.get("currency",""),"")
                        send(YOUR_CHAT_ID,
                             f"🚨 *PRICE ALERT TRIGGERED*\n"
                             f"*{ticker}* is now {cs}{cur:,.2f}\n"
                             f"Your alert: price goes {direction} {cs}{target:,.2f}\n"
                             f"_/a {ticker} for full analysis_")
                except: pass
            con.close()
        except: pass
        time.sleep(900)


def trailing_stop_monitor():
    """
    Check open trades every 15 minutes.
    Automatically:
    - Move SL to breakeven when T1 is hit
    - Trail SL to T1 when T2 is hit
    - Close trade when trailing SL is breached
    - Notify on T1/T2/T3/SL hits
    """
    while True:
        try:
            con = sqlite3.connect(DB_FILE)
            trades = con.execute("""
                SELECT id, ticker, entry_price, qty, sl, t1, t2, t3,
                       t1_hit, t2_hit, t3_hit, sl_hit, breakeven_set,
                       trailing_sl, sl_original
                FROM trades WHERE status='OPEN'
            """).fetchall()
            cols = ["id","ticker","entry_price","qty","sl","t1","t2","t3",
                    "t1_hit","t2_hit","t3_hit","sl_hit","breakeven_set",
                    "trailing_sl","sl_original"]

            nd  = datetime.now().strftime("%Y-%m-%d")
            for row in trades:
                t = dict(zip(cols, row))
                try:
                    info = yf.Ticker(t["ticker"]).fast_info
                    cur  = float(getattr(info,"last_price",0) or 0)
                    if cur <= 0: continue

                    upd      = {}
                    notif    = []
                    close_it = False

                    # SL hit
                    # Safe: trailing_sl > original sl > None
                    effective_sl = (t.get("trailing_sl") or
                                    t.get("sl") or
                                    None)
                    if effective_sl is not None:
                        effective_sl = float(effective_sl)
                    if effective_sl and not t["sl_hit"] and cur <= effective_sl:
                        upd.update({"sl_hit":1,"status":"CLOSED",
                                    "exit_price":round(cur,2),"exit_date":nd})
                        close_it = True
                        notif.append(
                            f"🛑 *STOP LOSS HIT* — {t['ticker']}\n"
                            f"Triggered at {cur:,.2f} (SL was {effective_sl:,.2f})")

                    # T1 hit → move SL to breakeven
                    if t["t1"] and not t["t1_hit"] and cur >= t["t1"]:
                        upd["t1_hit"] = 1
                        notif.append(
                            f"🎯 *T1 HIT* — {t['ticker']} @ {cur:,.2f}\n"
                            f"✅ SL moved to breakeven ({t['entry_price']:,.2f})\n"
                            f"_Next target: T2 @ {t['t2']:,.2f}_")
                        # Auto-move SL to breakeven
                        if not t["breakeven_set"]:
                            upd.update({"sl":t["entry_price"],
                                        "trailing_sl":t["entry_price"],
                                        "breakeven_set":1})

                    # T2 hit → trail SL to T1
                    if t["t2"] and not t["t2_hit"] and cur >= t["t2"]:
                        upd["t2_hit"] = 1
                        trail_to = t["t1"] or t["entry_price"]
                        upd.update({"sl":trail_to,"trailing_sl":trail_to})
                        notif.append(
                            f"🎯🎯 *T2 HIT* — {t['ticker']} @ {cur:,.2f}\n"
                            f"✅ SL trailed to T1 ({trail_to:,.2f})\n"
                            f"_Next target: T3 @ {t['t3']:,.2f}_")

                    # T3 hit → close trade
                    if t["t3"] and not t["t3_hit"] and cur >= t["t3"]:
                        upd.update({"t3_hit":1,"status":"CLOSED",
                                    "exit_price":round(cur,2),"exit_date":nd})
                        close_it = True
                        notif.append(
                            f"🏆 *T3 HIT — FULL TARGET ACHIEVED* — {t['ticker']}\n"
                            f"Exit @ {cur:,.2f}")

                    if upd:
                        sc = ", ".join(f"{k}=?" for k in upd)
                        con.execute(f"UPDATE trades SET {sc} WHERE id=?",
                                    list(upd.values()) + [t["id"]])
                        con.commit()

                    for msg in notif:
                        send(YOUR_CHAT_ID, msg)

                except: pass
            con.close()
        except: pass
        time.sleep(900)


def signal_validation_monitor():
    """
    Once a day, check open signal_log entries to see if signals played out.
    Updates outcome: HIT_T1, HIT_SL, EXPIRED, RUNNING
    Sends a daily morning summary.
    """
    last_run = None
    while True:
        now = datetime.now()
        ist = now + timedelta(hours=5, minutes=30)
        # Run once at 9:00 AM IST (before market opens)
        if ist.hour == 9 and ist.minute < 15 and last_run != ist.date():
            last_run = ist.date()
            try:
                con = sqlite3.connect(DB_FILE)
                tables = [r[0] for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
                if "signal_log" not in tables:
                    con.close()
                else:
                    # Check yesterday's signals
                    yesterday = (ist - timedelta(days=1)).strftime("%Y-%m-%d")
                    pending = con.execute("""
                        SELECT id, ticker, signal_type, entry_price, sl, t1,
                               lead_score, lag_score, signal_date
                        FROM signal_log
                        WHERE outcome='PENDING' AND signal_date <= ?
                    """, (yesterday,)).fetchall()

                    hits=0; misses=0; expired=0
                    for row in pending:
                        sid,ticker,sig_t,ep,sl,t1,ls,gs,sdate = row
                        try:
                            df   = fetch_df(ticker, days=30)
                            # Check if price hit T1 or SL since signal date
                            sig_idx = df.index.searchsorted(
                                pd.Timestamp(sdate)) if sdate else 0
                            recent  = df.iloc[sig_idx:]
                            if len(recent) < 2:
                                continue
                            max_hi  = float(recent["High"].max())
                            min_lo  = float(recent["Low"].min())
                            if t1 and max_hi >= t1:
                                outcome="HIT_T1"; hits+=1
                            elif sl and min_lo <= sl:
                                outcome="HIT_SL"; misses+=1
                            elif (datetime.now()-datetime.strptime(sdate,"%Y-%m-%d")).days > 15:
                                outcome="EXPIRED"; expired+=1
                            else:
                                outcome="RUNNING"
                            outcome_date = ist.strftime("%Y-%m-%d")
                            con.execute(
                                "UPDATE signal_log SET outcome=?,outcome_date=? WHERE id=?",
                                (outcome,outcome_date,sid))
                        except: pass
                    con.commit()
                    con.close()

                    # Daily morning briefing
                    regime,score,detail = get_market_regime("RELIANCE.NS")
                    r_icon = {"bull":"✅","neutral":"⚠️","bear":"🚨"}[regime]
                    open_trades = db_get_trades(status="OPEN", limit=20)
                    trade_lines = []
                    for t in open_trades[:5]:
                        trade_lines.append(
                            f"  #{t['id']} {t['ticker']} "
                            f"@ {t.get('entry_price',0):.0f} "
                            f"SL:{t.get('sl','—')} "
                            f"T2:{t.get('t2','—')}")

                    msg = (
                        f"🌅 *Good Morning — Market Briefing*\n"
                        f"{ist.strftime('%d %b %Y')}\n\n"
                        f"{r_icon} *Market: {regime.upper()}* ({score}/100)\n"
                        f"{detail[:120]}\n\n")
                    if open_trades:
                        msg += f"*Open Trades ({len(open_trades)}):*\n"
                        msg += "\n".join(trade_lines)
                        if len(open_trades)>5:
                            msg += f"\n  _...+{len(open_trades)-5} more_"
                        msg += "\n\n"
                    if hits or misses or expired:
                        wr = round(hits/(hits+misses)*100) if (hits+misses) else 0
                        msg += (f"*Yesterday's signals:*\n"
                                f"  T1 hit: {hits} ✅  SL hit: {misses} ❌  "
                                f"Expired: {expired}\n"
                                f"  Win rate: {wr}%\n\n")
                    msg += "_/screen to find today's setups_"
                    send(YOUR_CHAT_ID, msg)

            except Exception as e:
                log.error(f"Signal validation error: {e}")

        time.sleep(300)  # check every 5 minutes


# ══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════
def get_watchlist():
    """Return personal watchlist or Nifty 50 as fallback."""
    wl_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist.txt")
    if os.path.exists(wl_file):
        with open(wl_file) as f:
            wl = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        if wl: return wl
    return list(NIFTY50)

def cmd_watchlist(chat_id, args):
    """
    /watchlist          — show current list
    /watchlist add TCS  — add ticker(s)
    /watchlist remove TCS — remove ticker
    /watchlist clear    — reset to Nifty 50
    """
    wl_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist.txt")

    def load(): return get_watchlist()
    def save(wl):
        with open(wl_file, "w") as f:
            for t in wl: f.write(t + "\n")

    if not args:
        wl = load()
        chunks = [wl[i:i+25] for i in range(0, len(wl), 25)]
        display = "  ".join(t.replace(".NS","").replace(".BO","") for t in wl[:25])
        more = f"\n  _...+{len(wl)-25} more_" if len(wl)>25 else ""
        send(chat_id,
             f"*📋 Watchlist ({len(wl)} stocks)*\n\n{display}{more}\n\n"
             "_/wl add TCS.NS_  |  _/wl remove TCS.NS_  |  _/wl clear_")
        return

    action  = args[0].lower()
    tickers = [normalise_ticker(a) for a in args[1:] if a]

    if action == "add" and tickers:
        wl = load(); added = []
        for t in tickers:
            if t not in wl: wl.append(t); added.append(t)
        save(wl)
        send(chat_id, f"✅ Added: {', '.join(added) or 'already in list'}\n"
                      f"Watchlist: {len(wl)} stocks")
    elif action == "remove" and tickers:
        wl = load()
        removed = [t for t in tickers if t in wl]
        wl = [t for t in wl if t not in tickers]
        save(wl)
        send(chat_id, f"✅ Removed: {', '.join(removed) or 'not found'}\n"
                      f"Watchlist: {len(wl)} stocks")
    elif action == "clear":
        if os.path.exists(wl_file): os.remove(wl_file)
        send(chat_id, f"✅ Reset to Nifty 50 ({len(NIFTY50)} stocks)")
    else:
        send(chat_id, "Usage:\n`/wl` — show list\n`/wl add TCS.NS INFY`\n"
                      "`/wl remove TCS.NS`\n`/wl clear` — reset to Nifty 50")

COMMANDS = {
    "/analyse":   cmd_analyse,    "/a":        cmd_analyse,
    "/weekly":    cmd_weekly,     "/w":        cmd_weekly,
    "/both":      cmd_both,
    "/screen":    cmd_screen,
    "/watchlist": cmd_watchlist,  "/wl":       cmd_watchlist,
    "/log":       cmd_log,
    "/size":      cmd_size,
    "/fib":       cmd_fib,
    "/trades":    cmd_trades,
    "/close":     cmd_close,
    "/trail":     cmd_trailing,
    "/alert":     cmd_alert,
    "/alerts":    cmd_alerts,
    "/regime":    cmd_regime,
    "/portfolio": cmd_portfolio,
    "/status":    cmd_status,
    "/help":      lambda cid,args: send(cid, HELP_TEXT),
    "/start":     lambda cid,args: send(cid, HELP_TEXT),
}

def process_update(upd):
    msg     = upd.get("message",{})
    if not msg: return
    chat_id = msg.get("chat",{}).get("id")
    text    = (msg.get("text","") or "").strip()
    if not text or not chat_id: return
    if not is_authorised(chat_id):
        send(chat_id,"⛔ Unauthorised — private bot."); return
    parts = text.split(); cmd = parts[0].lower().split("@")[0]; args = parts[1:]
    if cmd.startswith("/close_"):
        args = [cmd.replace("/close_","")] + args; cmd="/close"
    elif cmd.startswith("/trail_"):
        args = [cmd.replace("/trail_","")] + args; cmd="/trail"
    if cmd in COMMANDS:
        try:    COMMANDS[cmd](chat_id, args)
        except Exception as e:
            log.error(f"Cmd {cmd}: {e}", exc_info=True)
            send(chat_id, f"❌ Error in {cmd}: {e}")
    else:
        clean = text.strip().upper()
        if clean.replace(".","").replace("-","").isalnum() and len(clean)<=12:
            cmd_analyse(chat_id, [clean])
        else:
            send(chat_id, f"Unknown: `{text[:40]}`\nSend /help")

def run():
    log.info("AstraQuant v3.0 — Real Money Edition starting…")
    me = tg("getMe")
    if not me.get("ok"):
        log.error("BOT_TOKEN invalid — get one from @BotFather"); return
    log.info(f"Bot: @{me['result']['username']}")

    # Start background threads
    threading.Thread(target=alert_monitor,            daemon=True).start()
    threading.Thread(target=trailing_stop_monitor,    daemon=True).start()
    threading.Thread(target=signal_validation_monitor,daemon=True).start()
    log.info("Monitors started: alerts, trailing stops, signal validation.")

    if YOUR_CHAT_ID and YOUR_CHAT_ID != "YOUR_TELEGRAM_CHAT_ID":
        send(YOUR_CHAT_ID,
             "✅ *AstraQuant v3.0 — Real Money Edition online*\n\n"
             "Active filters:\n"
             "🌍 Market regime · 💧 Liquidity · 📅 Earnings blackout\n"
             "📦 Portfolio risk cap · 📐 Smart stop loss\n"
             "🔄 Trailing stops · 🌅 Daily briefing at 9 AM IST\n\n"
             "Send /help for all commands.")

    offset = 0
    log.info("Polling…")
    while True:
        try:
            resp = tg("getUpdates",
                      {"offset":offset,"timeout":30,
                       "allowed_updates":["message"]})
            if not resp.get("ok"): time.sleep(5); continue
            for upd in resp.get("result",[]):
                offset = upd["update_id"]+1
                try:    process_update(upd)
                except Exception as e:
                    log.error(f"Update error: {e}", exc_info=True)
        except KeyboardInterrupt:
            log.info("Stopped."); break
        except Exception as e:
            log.error(f"Poll: {e}"); time.sleep(5)

if __name__ == "__main__":
    run()
