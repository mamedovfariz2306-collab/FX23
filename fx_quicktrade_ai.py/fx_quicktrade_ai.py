# FX QuickTrade AI — Twelve Data (Stable) | Candlestick | Education Only
# Live FX (Twelve Data) + Demo, EMA/RSI/ATR indikatorları və AI məsləhət

import os, json, time, requests, math
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
TD_API_KEY = os.getenv("TWELVEDATA_API_KEY")

# ─────────────────────────────────────────────────────────────────────────────
# Köməkçi indikatorlar
# ─────────────────────────────────────────────────────────────────────────────
def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False).mean()

def rsi(s: pd.Series, period: int = 14):
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)

def atr(df: pd.DataFrame, period: int = 14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ─────────────────────────────────────────────────────────────────────────────
# Twelve Data API
# ─────────────────────────────────────────────────────────────────────────────
def _td_interval(tf: str) -> str:
    return {"M1": "1min", "M5": "5min", "M15": "15min", "H1": "1h"}.get(tf, "5min")

def get_twelvedata_fx(symbol: str, timeframe: str = "M5",
                      output_size: int = 300, max_retries: int = 2) -> tuple[pd.DataFrame|None, dict]:
    """Twelve Data time_series. Retry + diaqnostika qaytarır."""
    diag = {"status":"init", "detail":""}

    if not TD_API_KEY:
        diag["status"]="error"; diag["detail"]="TWELVEDATA_API_KEY tapılmadı (.env)."
        return None, diag

    interval = _td_interval(timeframe)
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,               # EUR/USD, GBP/USD...
        "interval": interval,           # 5min, 1min, 1h ...
        "outputsize": output_size,
        "order": "asc",
        "apikey": TD_API_KEY,
    }

    for attempt in range(max_retries+1):
        try:
            r = requests.get(url, params=params, timeout=20)
        except requests.RequestException as e:
            diag["status"]="error"; diag["detail"]=f"Bağlantı xətası: {e}"
            return None, diag

        if r.status_code != 200:
            diag["status"]="error"; diag["detail"]=f"HTTP {r.status_code}: {r.text[:200]}"
            return None, diag

        data = r.json()

        if "status" in data and data["status"] == "error":
            msg = data.get("message","unknown")
            diag["status"]="error"; diag["detail"]=f"TwelveData error: {msg}"
            return None, diag

        if "values" not in data:
            diag["status"]="warn"; diag["detail"]=f"Gözlənilməyən cavab: {list(data.keys())[:3]}"
            if attempt < max_retries:
                time.sleep(3)
                continue
            return None, diag

        vals = data["values"]
        if not vals:
            diag["status"]="warn"; diag["detail"]="Boş 'values' gəldi."
            return None, diag

        rows = []
        try:
            for v in vals:
                rows.append([
                    pd.to_datetime(v["datetime"]),
                    float(v["open"]),
                    float(v["high"]),
                    float(v["low"]),
                    float(v["close"]),
                    float(v.get("volume", 0.0)),
                ])
            df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
            diag["status"]="ok"; diag["detail"]=f"{len(df)} bar"
            return df.set_index("ts").sort_index(), diag
        except Exception as e:
            diag["status"]="error"; diag["detail"]=f"Parse xətası: {e}"
            return None, diag

# ─────────────────────────────────────────────────────────────────────────────
# AI məsləhət məntiqi (sadə qaydalar)
# ─────────────────────────────────────────────────────────────────────────────
def compute_features(df: pd.DataFrame):
    f = df.copy()
    f["ema_fast"] = ema(f["close"], 9)
    f["ema_slow"] = ema(f["close"], 21)
    f["rsi"] = rsi(f["close"], 14)
    f["atr"] = atr(f, 14)
    f["trend"] = np.where(f["ema_fast"] > f["ema_slow"], 1, -1)
    return f

def ai_advice(feats: pd.DataFrame, horizon_min: int = 5):
    latest = feats.iloc[-1]
    price = float(latest["close"])
    ema_fast = float(latest["ema_fast"])
    ema_slow = float(latest["ema_slow"])
    r = float(latest["rsi"])
    a = float(latest["atr"])

    evidence = {
        "ema_fast": round(ema_fast,6),
        "ema_slow": round(ema_slow,6),
        "rsi": round(r,2),
        "atr": round(a,6),
        "trend": int(latest["trend"])
    }

    action, side, conf = "WAIT", None, 0.5
    explanation = []

    if ema_fast > ema_slow:
        explanation.append("EMA(9) > EMA(21) → qısa-müddətli yüksəliş.")
        if 50 <= r <= 68:
            action, side, conf = "ENTER", "BUY", 0.7
            explanation.append("RSI 50–68 aralığında.")
        elif r < 35:
            action, side, conf = "ENTER", "BUY", 0.6
            explanation.append("RSI oversold (<35).")
        else:
            explanation.append("RSI uyğun deyil → gözlə.")
    elif ema_fast < ema_slow:
        explanation.append("EMA(9) < EMA(21) → qısa-müddətli eniş.")
        if 32 <= r <= 50:
            action, side, conf = "ENTER", "SELL", 0.7
            explanation.append("RSI 32–50 aralığında.")
        elif r > 65:
            action, side, conf = "ENTER", "SELL", 0.6
            explanation.append("RSI overbought (>65).")
        else:
            explanation.append("RSI uyğun deyil → gözlə.")
    else:
        explanation.append("EMA-lar eynidir → qeyri-müəyyən.")

    sl = tp = None
    if action == "ENTER":
        atr_mult_sl = 1.2
        atr_mult_tp = 2.0
        if side == "BUY":
            sl = price - atr_mult_sl * a
            tp = price + atr_mult_tp * a
        elif side == "SELL":
            sl = price + atr_mult_sl * a
            tp = price - atr_mult_tp * a
        explanation.append(f"SL≈{atr_mult_sl}×ATR, TP≈{atr_mult_tp}×ATR.")
    else:
        explanation.append("Siqnal zəifdir — giriş tövsiyə edilmir.")

    return {
        "action": action,
        "side": side,
        "entry_price": price,
        "sl": sl,
        "tp": tp,
        "confidence": round(conf, 2),
        "evidence": evidence,
        "explanation": " ".join(explanation)
    }

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FX QuickTrade AI — TwelveData", layout="wide")
st.title("💹 FX QuickTrade AI — Twelve Data (Education Only)")
st.caption("Live FX (Twelve Data) + EMA/RSI/ATR əsasında qısa-müddətli məsləhət. Not financial advice.")
st.sidebar.caption(f"TwelveData key: {'OK' if TD_API_KEY else 'MISSING'}")

st.sidebar.header("Settings")
market = st.sidebar.selectbox("Market", ["Forex"])
symbol = st.sidebar.selectbox("Symbol", ["EUR/USD","GBP/USD","USD/JPY","XAU/USD"])
timeframe = st.sidebar.selectbox("Timeframe", ["M1","M5","M15","H1"], index=1)
horizon_min = st.sidebar.slider("Order horizon (min)", 1, 60, 5)

st.sidebar.header("Data source")
data_mode = st.sidebar.radio("Source", ["Live (Twelve Data)", "Demo (synthetic)"], index=0)
json_text = st.sidebar.text_area("Or paste OHLCV JSON [ts,open,high,low,close,volume]")
refresh = st.sidebar.button("🔄 Refresh")

# ─────────────────────────────────────────────────────────────────────────────
# Data yükləmə
# ─────────────────────────────────────────────────────────────────────────────
def load_candles() -> tuple[pd.DataFrame|None, dict]:
    # 1) JSON üstünlük
    if json_text.strip():
        try:
            arr = json.loads(json_text)
            df = pd.DataFrame(arr, columns=["ts","open","high","low","close","volume"])
            if np.issubdtype(pd.Series(df["ts"]).dtype, np.integer):
                df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            else:
                df["ts"] = pd.to_datetime(df["ts"])
            return df.set_index("ts").sort_index(), {"status":"ok","detail":"JSON pasted"}
        except Exception as e:
            return None, {"status":"error","detail":f"JSON oxunmadı: {e}"}

    # 2) Live
    if data_mode == "Live (Twelve Data)":
        st.info(f"Yüklənir: {symbol} ({timeframe}) — Twelve Data…")
        live, diag = get_twelvedata_fx(symbol, timeframe, output_size=300, max_retries=2)
        if live is not None and len(live) > 0:
            return live, {"status":"ok","detail":f"Live {len(live)} bar"}
        return None, {"status":"warn","detail":f"Live alınmadı → {diag.get('detail','n/a')}"}

    # 3) Demo
    np.random.seed(7)
    n = 300
    base = 1.0870
    close = base + np.cumsum(np.random.normal(0, 0.00005, n))
    high  = close + np.abs(np.random.normal(0, 0.00008, n))
    low   = close - np.abs(np.random.normal(0, 0.00008, n))
    open_ = np.r_[close[0], close[:-1]]
    vol   = np.random.randint(100, 500, size=n)
    ts    = pd.date_range("2025-01-01", periods=n, freq="1min")
    demo  = pd.DataFrame({"ts": ts, "open": open_, "high": high, "low": low, "close": close, "volume": vol})
    return demo.set_index("ts").sort_index(), {"status":"demo","detail":"synthetic"}

# ─────────────────────────────────────────────────────────────────────────────
# Çart + AI məsləhət
# ─────────────────────────────────────────────────────────────────────────────
raw_df, info = load_candles()

if info["status"] == "warn":
    st.warning(f"Twelve Data xəbərdarlığı: {info['detail']}")
elif info["status"] == "error":
    st.error(info["detail"])
elif info["status"] == "demo":
    st.info("Live alınmadı. Demo rejimində göstərilir.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"{symbol} — {timeframe} candlestick chart")
    if raw_df is None or len(raw_df) < 20:
        st.info("Data azdır və ya gəlmədi.")
    else:
        # ── Şam qrafiki (Plotly) ──
        fig = go.Figure(data=[go.Candlestick(
            x=raw_df.index,
            open=raw_df["open"],
            high=raw_df["high"],
            low=raw_df["low"],
            close=raw_df["close"],
            increasing_line_color="green",
            decreasing_line_color="red"
        )])
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_dark",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("ℹ️ Current settings")
    st.write({
        "market": market,
        "symbol": symbol,
        "timeframe": timeframe,
        "source": data_mode,
        "horizon_min": horizon_min
    })

st.markdown("---")

if raw_df is not None and len(raw_df) >= 50:
    feats = compute_features(raw_df)
    advise = ai_advice(feats, horizon_min=horizon_min)

    st.subheader("✅ AI Advice")
    st.write({
        "action": advise["action"],
        "side": advise["side"],
        "entry_price": round(advise["entry_price"], 6),
        "sl": round(advise["sl"], 6) if advise["sl"] else None,
        "tp": round(advise["tp"], 6) if advise["tp"] else None,
        "confidence": advise["confidence"]
    })

    st.markdown("**Why this advice?**")
    st.write(advise["explanation"])

    with st.expander("Evidence (indicators)"):
        st.json(advise["evidence"])
else:
    st.stop()
