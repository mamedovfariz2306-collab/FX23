# fx_quicktrade_ai.py
# FX QuickTrade AI — TradingView-like: Candle + Volume + RSI + MACD
# Sidebar + 3 dilli AI məsləhət (cədvəl) + Siqnal Jurnalı
# ✔ AI Auto-Analiz siqnalı (conf threshold) + 3 dəqiqəlik qiymətləndirmə
# Live: TwelveData (əgər API key var) / Demo
# Education only. Not financial advice.

import os, requests
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone

# ───────────── Page / Style ─────────────
st.set_page_config(page_title="FX QuickTrade AI", layout="wide", page_icon="💹")
st.markdown("""
<style>
.block-container{padding-top:.6rem;padding-bottom:.6rem;}
[data-testid="stSidebar"]{width:330px;}
.js-plotly-plot .plotly .cursor-crosshair {cursor: crosshair;}
</style>
""", unsafe_allow_html=True)

# ───────────── I18N (AZ/RU/EN) ─────────────
TXT = {
  "az": {
    "lang":"Azərbaycan", "settings":"Parametrlər", "market":"Bazar", "forex":"Forex",
    "symbol":"Alət (symbol)", "timeframe":"Zaman çərçivəsi", "source":"Mənbə",
    "live":"Live (Twelve Data)", "demo":"Demo (synthetic)",
    "risk":"Risk %", "balance":"Balans",
    "loading":"Yüklənir: {sym} ({tf}) — {src}...",
    "td_missing":"TWELVEDATA_API_KEY tapılmadı. Demo rejimə keçdim.",
    "td_error":"TwelveData xətası → {err}. Demo rejimə keçdim.",
    "no_data":"Data azdır və ya gəlmədi.",
    "chart_title":"{sym} — {tf} şam qrafiki",
    "ai_title":"🤖 AI məsləhət", "why":"Niyə bu məsləhət?", "evidence":"Sübutlar", "ind_last":"Göstəricilər",
    "buy":"Al (Long)", "sell":"Sat (Short)", "skip":"Keç (No trade)",
    "size":"ölçü (lot)", "sl":"Stop-Loss", "tp":"Take-Profit", "conf":"İnam",
    "journal":"Siqnal jurnalı (session)", "save_sig":"Siqnalı əlavə et",
    "clear":"Jurnalı təmizlə", "csv":"Jurnalı yüklə (CSV)",
    "auto_ai_title":"AI Auto-Analiz siqnalı",
    "enable_auto_ai":"AI özü siqnal yazsın",
    "ai_conf_th":"Minimum inam (0–1)", "ai_dur":"Müddət (dəqiqə)",
    "cooldown":"Cooldown (bar sayı)", "manual_refresh":"↻ Yenilə"
  },
  "ru": {
    "lang":"Русский", "settings":"Параметры", "market":"Рынок", "forex":"Форекс",
    "symbol":"Инструмент (symbol)", "timeframe":"Таймфрейм", "source":"Источник",
    "live":"Live (Twelve Data)", "demo":"Демо (synthetic)",
    "risk":"Риск %", "balance":"Баланс",
    "loading":"Загрузка: {sym} ({tf}) — {src}...",
    "td_missing":"TWELVEDATA_API_KEY не найден. Перехожу в демо.",
    "td_error":"Ошибка TwelveData → {err}. Перехожу в демо.",
    "no_data":"Недостаточно данных.",
    "chart_title":"{sym} — {tf} свечной график",
    "ai_title":"🤖 Рекомендация AI", "why":"Почему так?", "evidence":"Доказательства", "ind_last":"Индикаторы",
    "buy":"Покупка (Long)", "sell":"Продажа (Short)", "skip":"Пропуск",
    "size":"объем (лот)", "sl":"Стоп-лосс", "tp":"Тейк-профит", "conf":"Уверенность",
    "journal":"Журнал сигналов (сессия)", "save_sig":"Добавить сигнал",
    "clear":"Очистить журнал", "csv":"Скачать журнал (CSV)",
    "auto_ai_title":"AI Авто-анализ сигнал",
    "enable_auto_ai":"AI сам пишет сигнал",
    "ai_conf_th":"Мин. уверенность (0–1)", "ai_dur":"Длительность (мин)",
    "cooldown":"Кулдаун (свечей)", "manual_refresh":"↻ Обновить"
  },
  "en": {
    "lang":"English", "settings":"Settings", "market":"Market", "forex":"Forex",
    "symbol":"Symbol", "timeframe":"Timeframe", "source":"Source",
    "live":"Live (Twelve Data)", "demo":"Demo (synthetic)",
    "risk":"Risk %", "balance":"Balance",
    "loading":"Loading: {sym} ({tf}) — {src}...",
    "td_missing":"TWELVEDATA_API_KEY missing. Switching to demo.",
    "td_error":"TwelveData error → {err}. Switching to demo.",
    "no_data":"No or insufficient data.",
    "chart_title":"{sym} — {tf} candlestick chart",
    "ai_title":"🤖 AI Advice", "why":"Why this advice?", "evidence":"Evidence", "ind_last":"Indicators",
    "buy":"Buy (Long)", "sell":"Sell (Short)", "skip":"Skip",
    "size":"size (lot)", "sl":"Stop-Loss", "tp":"Take-Profit", "conf":"Confidence",
    "journal":"Signal journal (session)", "save_sig":"Add signal",
    "clear":"Clear journal", "csv":"Download journal (CSV)",
    "auto_ai_title":"AI Auto-Analysis signal",
    "enable_auto_ai":"Let AI auto-write signals",
    "ai_conf_th":"Min confidence (0–1)", "ai_dur":"Duration (min)",
    "cooldown":"Cooldown (bars)", "manual_refresh":"↻ Refresh"
  }
}

# ───────────── Sidebar ─────────────
lang = st.sidebar.selectbox("Language / Dil / Язык", ["az","ru","en"], format_func=lambda k: TXT[k]["lang"])
t = TXT[lang]

st.sidebar.header(t["settings"])
market = st.sidebar.selectbox(t["market"], [t["forex"]])
symbol = st.sidebar.selectbox(t["symbol"], ["EUR/USD","GBP/USD","USD/JPY","XAU/USD"])
tf = st.sidebar.selectbox(t["timeframe"], ["M1","M5","M15","M30","H1","H4"])
source = st.sidebar.radio(t["source"], [t["live"], t["demo"]], index=0)

st.sidebar.subheader("Risk")
risk_pct = st.sidebar.slider(t["risk"], 0.1, 3.0, 1.2, 0.05)
balance  = st.sidebar.number_input(t["balance"], min_value=10.0, value=1000.0, step=10.0)

# AI Auto-Analiz siqnal ayarları
st.sidebar.subheader(t["auto_ai_title"])
enable_auto_ai = st.sidebar.checkbox(t["enable_auto_ai"], value=True)
ai_conf_th     = st.sidebar.slider(t["ai_conf_th"], 0.0, 1.0, 0.55, 0.01)
ai_dur_min     = st.sidebar.number_input(t["ai_dur"], min_value=1, value=3, step=1)
cooldown_bars  = st.sidebar.number_input(t["cooldown"], min_value=1, value=3, step=1)

# Manual refresh düyməsi
if st.sidebar.button(t["manual_refresh"]):
    st.rerun()

# ───────────── TwelveData / Demo ─────────────
load_dotenv()
TD_API_KEY = os.getenv("TWELVEDATA_API_KEY")

def _interval(tf_):
    return {"M1":"1min","M5":"5min","M15":"15min","M30":"30min","H1":"1h","H4":"4h"}.get(tf_,"5min")

def _td_fetch(sym, interval, outsize="800"):
    base = "https://api.twelvedata.com/time_series"
    q1 = {"symbol":sym,"exchange":"forex","interval":interval,"outputsize":outsize,"format":"JSON","apikey":TD_API_KEY}
    j1 = requests.get(base, params=q1, timeout=15).json()
    if j1.get("status")!="error" and "values" in j1: return j1
    sym2 = sym.replace("/","")
    q2 = {"symbol":sym2,"interval":interval,"outputsize":outsize,"format":"JSON","apikey":TD_API_KEY}
    return requests.get(base, params=q2, timeout=15).json()

def get_live_df(sym, tf_):
    if not TD_API_KEY:
        st.warning(t["td_missing"]); return None
    try:
        j = _td_fetch(sym, _interval(tf_))
        if j.get("status")=="error": st.info(t["td_error"].format(err=j)); return None
        vals = j.get("values", [])
        if not vals: return None
        df = pd.DataFrame(vals)
        for c in ["open","high","low","close","volume"]:
            if c not in df.columns: df[c] = np.nan
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True).rename(columns={"datetime":"ts"})
        df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].apply(pd.to_numeric, errors="coerce")
        return df
    except Exception as e:
        st.info(t["td_error"].format(err=e)); return None

def demo_df(n=900, start=1.10, seed=7):
    np.random.seed(seed)
    steps = np.random.normal(0, 0.00085, n).cumsum()
    close = start + steps
    high  = close + np.abs(np.random.normal(0, 0.0006, n))
    low   = close - np.abs(np.random.normal(0, 0.0006, n))
    open_ = np.r_[close[0], close[:-1]]
    vol   = np.random.randint(100, 500, n)
    ts = pd.date_range(end=pd.Timestamp.utcnow().floor("min"), periods=n, freq="min")
    return pd.DataFrame({"ts":ts,"open":open_,"high":high,"low":low,"close":close,"volume":vol})

# ───────────── Indicators ─────────────
def ema(s, span): return pd.Series(s).ewm(span=span, adjust=False).mean()
def rsi(close, period=14):
    d = pd.Series(close).diff()
    up = d.clip(lower=0); down = -d.clip(upper=0)
    rs = up.rolling(period).mean() / (down.rolling(period).mean() + 1e-12)
    return 100 - (100/(1+rs))
def atr(df, period=14):
    tr = pd.concat([
        (df["high"]-df["low"]).abs(),
        (df["high"]-df["close"].shift()).abs(),
        (df["low"]-df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()
def macd(close, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig  = ema(line, signal)
    return line, sig, (line - sig)

def enrich(df):
    x = df.copy()
    x["ema21"] = ema(x["close"], 21)
    x["ema50"] = ema(x["close"], 50)
    x["ema200"] = ema(x["close"], 200)
    x["rsi14"] = rsi(x["close"], 14)
    x["atr14"] = atr(x, 14)
    x["macd"], x["macdsig"], x["macdhist"] = macd(x["close"])
    return x

# ───────────── AI logic ─────────────
def ai_decide(row: pd.Series):
    price=float(row["close"]); ema21=float(row["ema21"]); ema50=float(row["ema50"]); ema200=float(row["ema200"])
    rsi14=float(row["rsi14"]); atr14=float(row["atr14"]); m=float(row["macd"]); ms=float(row["macdsig"])
    score=0.0; reasons=[]
    if ema21>ema50>ema200: score+=0.9; reasons.append("Trend: EMA21>EMA50>EMA200 (bullish)")
    elif ema21<ema50<ema200: score-=0.9; reasons.append("Trend: EMA21<EMA50<EMA200 (bearish)")
    else: reasons.append("Trend: mixed EMAs")
    if rsi14>55: score+=0.4; reasons.append(f"RSI(14)={rsi14:.1f} > 55 (up momentum)")
    elif rsi14<45: score-=0.4; reasons.append(f"RSI(14)={rsi14:.1f} < 45 (down momentum)")
    else: reasons.append(f"RSI(14)={rsi14:.1f} neutral")
    if m>ms: score+=0.3; reasons.append("MACD>Signal (bullish)")
    else: score-=0.3; reasons.append("MACD<Signal (bearish)")
    vol = (atr14/max(price,1e-9))
    if vol<0.0008: score*=0.8; reasons.append(f"Low volatility (ATR/Price={vol:.4f}) → confidence down")
    side="skip"
    if score>=0.4: side="buy"
    elif score<=-0.4: side="sell"
    sl=tp=None
    if side=="buy":  sl=price-1.2*atr14; tp=price+1.8*atr14
    if side=="sell": sl=price+1.2*atr14; tp=price-1.8*atr14
    conf=max(0.0,min(1.0,abs(score)/1.6))
    evidence={"price":price,"ema21":ema21,"ema50":ema50,"ema200":ema200,"rsi14":rsi14,"atr14":atr14,
              "macd":m,"macd_signal":ms,"score":round(score,3)}
    return side, sl, tp, conf, reasons, evidence

def side_label(side):
    return {"buy":t["buy"],"sell":t["sell"],"skip":t["skip"]}[side]

def translate_reason(line: str, lang_code: str) -> str:
    if lang_code=="az":
        return (line.replace("bullish","yuxarı")
                    .replace("bearish","aşağı")
                    .replace("mixed EMAs","qarışıq EMA-lar")
                    .replace("up momentum","yuxarı impuls")
                    .replace("down momentum","aşağı impuls")
                    .replace("neutral","neytral")
                    .replace("Low volatility","Aşağı volatillik")
                    .replace("confidence down","inam azaldıldı")
                    .replace("MACD>Signal","MACD>Siqnal")
                    .replace("MACD<Signal","MACD<Siqnal"))
    if lang_code=="ru":
        return (line.replace("bullish","бычий")
                    .replace("bearish","медвежий")
                    .replace("mixed EMAs","смешанные EMA")
                    .replace("up momentum","восходящий импульс")
                    .replace("down momentum","нисходящий импульс")
                    .replace("neutral","нейтрально")
                    .replace("Low volatility","Низкая волатильность")
                    .replace("confidence down","уверенность снижена"))
    return line

# ───────────── Load data ─────────────
src_txt = t["live"] if source==t["live"] else t["demo"]
st.info(t["loading"].format(sym=symbol, tf=tf, src=src_txt))

if source==t["live"]:
    df = get_live_df(symbol, tf)
    if df is None or len(df)<120: df = demo_df()
else:
    df = demo_df()

if df is None or len(df)==0:
    st.warning(t["no_data"]); st.stop()

feat = enrich(df)
last = feat.iloc[-1]
curr_price = float(last["close"])
last_ts = pd.to_datetime(last["ts"])

# prev price / last_bar_ts (duplikat siqnal olmaması üçün)
if "prev_price" not in st.session_state:
    st.session_state.prev_price = curr_price
if "last_ai_signal_ts" not in st.session_state:
    st.session_state.last_ai_signal_ts = {}  # key: (symbol, tf) -> ts

prev_price = st.session_state.prev_price
st.session_state.prev_price = curr_price

pair_key = (symbol, tf)
last_fired_ts = st.session_state.last_ai_signal_ts.get(pair_key)

# ───────────── Chart (TradingView-like) ─────────────
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
    row_heights=[0.55, 0.12, 0.16, 0.17]
)
fig.add_trace(go.Candlestick(x=feat["ts"], open=feat["open"], high=feat["high"], low=feat["low"], close=feat["close"],
                             name="Price", increasing_line_color="#2ecc71", decreasing_line_color="#e74c3c"), row=1, col=1)
fig.add_trace(go.Scatter(x=feat["ts"], y=feat["ema21"],  name="EMA21",  mode="lines", line=dict(width=1.2)), row=1, col=1)
fig.add_trace(go.Scatter(x=feat["ts"], y=feat["ema50"],  name="EMA50",  mode="lines", line=dict(width=1.2)), row=1, col=1)
fig.add_trace(go.Scatter(x=feat["ts"], y=feat["ema200"], name="EMA200", mode="lines", line=dict(width=1.2)), row=1, col=1)
vol_colors = np.where(feat["close"]>=feat["open"], "#2ecc71", "#e74c3c")
fig.add_trace(go.Bar(x=feat["ts"], y=feat["volume"], name="Volume", marker_color=vol_colors, opacity=0.6), row=2, col=1)
fig.add_trace(go.Scatter(x=feat["ts"], y=feat["rsi14"], name="RSI(14)", mode="lines"), row=3, col=1)
fig.add_hline(y=70, line_dash="dot", line_width=1, line_color="#e74c3c", row=3, col=1)
fig.add_hline(y=30, line_dash="dot", line_width=1, line_color="#2ecc71", row=3, col=1)
macd_colors = np.where(feat["macdhist"]>=0, "#16a34a", "#dc2626")
fig.add_trace(go.Bar(x=feat["ts"], y=feat["macdhist"], name="MACD hist", marker_color=macd_colors), row=4, col=1)
fig.update_layout(
    height=780, margin=dict(l=6, r=6, t=10, b=4),
    hovermode="x unified", xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot", spikecolor="#999", spikethickness=1),
    xaxis2=dict(showspikes=True), xaxis3=dict(showspikes=True), xaxis4=dict(showspikes=True)
)
st.subheader(t["chart_title"].format(sym=symbol, tf=tf))
st.plotly_chart(fig, use_container_width=True, theme=None)

# ───────────── AI Advice (table) ─────────────
side, sl, tp, conf, reasons, evidence = ai_decide(last)
price = curr_price
risk_dec = risk_pct/100.0
risk_amt = balance * risk_dec
size = round(risk_amt / (evidence["atr14"]*1.2 + 1e-9), 4)

st.markdown(f"### {t['ai_title']}")
adv_tbl = pd.DataFrame([
    ["Action", side_label(side)],
    ["Entry", round(price,6)],
    [t["size"], size],
    [t["sl"], round(sl,6) if sl else None],
    [t["tp"], round(tp,6) if tp else None],
    [t["conf"], round(conf,2)],
], columns=["Field", "Value"])
st.table(adv_tbl)

st.markdown(f"**{t['why']}**")
for r in reasons:
    st.markdown("- " + translate_reason(r, lang))

with st.expander(t["evidence"]):
    st.json({k:(round(v,6) if isinstance(v,(int,float)) else v) for k,v in evidence.items()})

with st.expander(t["ind_last"]):
    st.json({
      "EMA21": float(last["ema21"]), "EMA50": float(last["ema50"]), "EMA200": float(last["ema200"]),
      "RSI(14)": float(last["rsi14"]), "ATR(14)": float(last["atr14"]),
      "MACD": float(last["macd"]), "MACD_signal": float(last["macdsig"])
    })

# ───────────── Journal state ─────────────
st.markdown(f"### {t['journal']}")
if "journal" not in st.session_state:
    st.session_state.journal = []
if "signals" not in st.session_state:
    st.session_state.signals = []  # AI auto + manual signals

# Manual “Siqnalı əlavə et”
btn_cols = st.columns([1,1,6,1])
if btn_cols[0].button(t["save_sig"]):
    st.session_state.journal.append({
        "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol, "tf": tf, "action": side_label(side),
        "price": round(price,6), "size": size,
        "sl": round(sl,6) if sl else None, "tp": round(tp,6) if tp else None,
        "conf": round(conf,2), "result": ""
    })
if btn_cols[-1].button(t["clear"]):
    st.session_state.journal = []
    st.session_state.signals = []
    st.session_state.last_ai_signal_ts = {}

def _eng(side_str):
    m = {"buy":"buy","sell":"sell","skip":"skip",
         TXT["az"]["buy"]:"buy", TXT["az"]["sell"]:"sell", TXT["az"]["skip"]:"skip",
         TXT["ru"]["buy"]:"buy", TXT["ru"]["sell"]:"sell", TXT["ru"]["skip"]:"skip",
         TXT["en"]["buy"]:"buy", TXT["en"]["sell"]:"sell", TXT["en"]["skip"]:"skip"}
    return m.get(side_str, "buy")

def _maybe_fire_ai_auto(sig_side, sig_conf, ts_bar, dur_min):
    """Yeni bar üçün: conf>=threshold və cooldown keçibsə → siqnal yaz."""
    if sig_side=="skip": return
    if sig_conf < ai_conf_th: return
    # cooldown: eyni (symbol,tf) üçün son siqnal barından sonra X bar keçməsə, yazma
    if last_fired_ts is not None:
        bars_passed = (ts_bar - pd.to_datetime(last_fired_ts)).total_seconds() / 60.0
        if bars_passed < cooldown_bars:  # M1 üçün dəqiqə=bar
            return
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=dur_min)
    entry = float(price)
    side_eng = _eng(sig_side)
    sig = {
        "id": f"{now.timestamp()}",
        "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol, "tf": tf, "side": side_eng,
        "entry": entry, "duration_min": int(dur_min),
        "expires_at": expires.strftime("%Y-%m-%d %H:%M:%S"),
        "status":"ACTIVE", "result":""
    }
    st.session_state.signals.append(sig)
    st.session_state.journal.append({
        "time_utc": sig["created_at"], "symbol": symbol, "tf": tf,
        "action": side_label(side_eng), "price": round(entry,6),
        "size": None, "sl": None, "tp": None, "conf": round(sig_conf,2), "result": "⏳"
    })
    # bu pair üçün son fired bar timestamp
    st.session_state.last_ai_signal_ts[pair_key] = ts_bar.isoformat()

def _settle_signals(curr_px):
    now = datetime.now(timezone.utc)
    for sig in st.session_state.signals:
        if sig["status"] != "ACTIVE":
            continue
        exp = datetime.strptime(sig["expires_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if now >= exp:
            ok = (curr_px > sig["entry"]) if sig["side"] == "buy" else (curr_px < sig["entry"])
            sig["status"] = "SETTLED"
            sig["result"] = "✅" if ok else "❌"
            # journal-da update
            for j in reversed(st.session_state.journal):
                if j.get("time_utc") == sig["created_at"] and j.get("price") == round(sig["entry"], 6):
                    j["result"] = sig["result"]
                    break

# AI Auto-Analiz siqnalı: yeni bar üçün şərtlər tutursa, yaz
if enable_auto_ai:
    _maybe_fire_ai_auto(side, conf, last_ts, ai_dur_min)

# Vaxtı bitən siqnalları qiymətləndir
_settle_signals(curr_price)

# Jurnalı göstər
if len(st.session_state.journal):
    ...

