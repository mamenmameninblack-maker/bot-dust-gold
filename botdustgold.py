import os
import time
import random
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
import numpy as np
import yfinance as yf # type: ignore
import pytz

# ============================================================
# CONFIG GÉNÉRALE
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Yahoo Finance
GOLD_SYMBOL = "GC=F"
DUST_SYMBOL = "DUST"
YF_INTERVAL = "5m"

# Intervalles
TRADING_INTERVAL_SEC = 45 * 60   # 45 minutes
NEWS_HOUR = 22                   # 22h00 heure de Paris

# Fuseau horaire
TZ_PARIS = pytz.timezone("Europe/Paris")

# Mémoire interne
last_news_date = None
last_trade_alert = 0


# ============================================================
# HORAIRES DE FONCTIONNEMENT (OPTION C)
# ============================================================

def is_trading_time():
    now = datetime.now(TZ_PARIS)
    wd = now.weekday()
    h = now.hour
    m = now.minute

    if wd >= 5:
        return False

    if (h == 3 and m >= 30) or (h == 4) or (h == 5 and m == 0):
        return False

    return True


# ============================================================
# OUTILS TELEGRAM
# ============================================================

def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ TELEGRAM_TOKEN ou CHAT_ID manquant.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("Erreur Telegram:", e)


# ============================================================
# FETCH YAHOO FINANCE (PATCH 1D)
# ============================================================

def fetch_intraday(symbol: str, interval: str = YF_INTERVAL, lookback: int = 300):
    try:
        # Choose period based on interval
        if interval.endswith("m"):
            period = "5d"
        else:
            period = "1mo"

        # Download data
        df = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False
        )

        if df is None or df.empty:
            print(f"⚠️ No Yahoo Finance data for {symbol}.")
            return None

        # Standardize column names
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })

        df = df.reset_index()

        # Normalize datetime column
        if "Datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["Datetime"])
        elif "Date" in df.columns:
            df["datetime"] = pd.to_datetime(df["Date"])
        else:
            df["datetime"] = pd.to_datetime(df.index)

        df = df[["datetime", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("datetime").reset_index(drop=True)

        # PATCH : force OHLCV columns to 1D
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].apply(
                lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x
            )

        # Limit to lookback
        if len(df) > lookback:
            df = df.iloc[-lookback:]

        return df

    except Exception as e:
        print(f"Yahoo Finance error for {symbol} :", e)
        return None


# ============================================================
# INDICATEURS TECHNIQUES
# ============================================================

def ema(series, period):
    series = pd.Series(series).astype(float)
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    series = pd.Series(series).astype(float)
    delta = series.diff()

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain = pd.Series(gain).rolling(period).mean()
    loss = pd.Series(loss).rolling(period).mean()

    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series, fast, slow, signal):
    series = pd.Series(series).astype(float)
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series, period=20, std_mult=2.0):
    series = pd.Series(series).astype(float)
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ma, ma + std_mult * std, ma - std_mult * std

def stochastic(high, low, close, k=5, d=3, smooth=3):
    high = pd.Series(high).astype(float)
    low = pd.Series(low).astype(float)
    close = pd.Series(close).astype(float)

    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()

    k_raw = 100 * (close - lowest) / (highest - lowest + 1e-9)
    k_smooth = k_raw.rolling(smooth).mean()
    d_line = k_smooth.rolling(d).mean()

    return k_smooth, d_line

def vwap(df):
    close = df["close"].squeeze().astype(float)
    volume = df["volume"].squeeze().astype(float)
    pv = close * volume
    return pv.cumsum() / (volume.cumsum() + 1e-9)


# ============================================================
# OUTILS LECTURE INSTITUTIONNELLE
# ============================================================

def detect_market_structure(close, lookback=20):
    close = pd.Series(close).astype(float)
    recent = close.tail(lookback)

    if len(recent) < 4:
        return "Structure insuffisante."

    if recent.is_monotonic_increasing:
        return "Structure haussière (HH/HL)."
    if recent.is_monotonic_decreasing:
        return "Structure baissière (LH/LL)."

    return "Structure mixte."


def detect_liquidity_sweep(high, low, lookback=10):
    high = pd.Series(high).astype(float)
    low = pd.Series(low).astype(float)

    if len(high) < lookback + 1:
        return None

    recent_high = high.tail(lookback + 1)
    recent_low = low.tail(lookback + 1)

    notes = []
    if recent_high.iloc[-1] > recent_high.iloc[:-1].max():
        notes.append("Balayage de liquidité haute.")
    if recent_low.iloc[-1] < recent_low.iloc[:-1].min():
        notes.append("Balayage de liquidité basse.")
    return notes


def detect_premium_discount(last_close, ref_ema, tol=0.002):
    if ref_ema <= 0:
        return None
    ratio = last_close / ref_ema
    if ratio > 1 + tol:
        return "Zone premium institutionnelle."
    if ratio < 1 - tol:
        return "Zone discount institutionnelle."
    return "Zone d'équilibre."


def detect_fvg(high, low):
    high = pd.Series(high).astype(float)
    low = pd.Series(low).astype(float)

    if len(high) < 3:
        return None

    h1, h2, h3 = high.iloc[-3], high.iloc[-2], high.iloc[-1]
    l1, l2, l3 = low.iloc[-3], low.iloc[-2], low.iloc[-1]

    notes = []
    if l3 > h1:
        notes.append("FVG haussier.")
    if h3 < l1:
        notes.append("FVG baissier.")
    return notes


# ============================================================
# OUTILS SUPPORT / RÉSISTANCE + RANGE
# ============================================================

def find_support_resistance(close, lookback=50):
    close = pd.Series(close).astype(float)
    recent = close.tail(lookback)

    # Sécurité : si pas assez de données
    if len(recent) < 5:
        return float(recent.min()), float(recent.max())

    supports = []
    resistances = []

    for i in range(2, len(recent) - 2):
        if recent.iloc[i] < recent.iloc[i-1] and recent.iloc[i] < recent.iloc[i+1]:
            supports.append(recent.iloc[i])
        if recent.iloc[i] > recent.iloc[i-1] and recent.iloc[i] > recent.iloc[i+1]:
            resistances.append(recent.iloc[i])

    support = max(supports) if supports else float(recent.min())
    resistance = min(resistances) if resistances else float(recent.max())

    return round(support, 2), round(resistance, 2)


def detect_range(close, support, resistance):
    close = pd.Series(close).astype(float)
    last = close.iloc[-1]
    return support < last < resistance


# ============================================================
# ANALYSE OR (GC=F)
# ============================================================

def analyze_gold(df):
    # Sécurisation des colonnes (évite les erreurs 2D)
    close = df["close"].squeeze().astype(float)
    high = df["high"].squeeze().astype(float)
    low = df["low"].squeeze().astype(float)
    volume = df["volume"].squeeze().astype(float)

    # Indicateurs
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)

    rsi9 = rsi(close, 9)
    macd_line, signal_line, hist = macd(close, 8, 21, 5)
    bb_mid, bb_up, bb_low = bollinger(close, 20, 2.0)
    stoch_k, stoch_d = stochastic(high, low, close, 5, 3, 3)
    vwap_j = vwap(df)

    # Extraction sécurisée des dernières valeurs
    last_close = float(close.iloc[-1])
    last_ema200 = float(ema200.iloc[-1])
    last_rsi9 = float(rsi9.iloc[-1])
    last_hist = float(hist.iloc[-1])
    last_vol = float(volume.iloc[-1])
    avg_vol = float(volume.rolling(20).mean().iloc[-1])
    last_vwap = float(vwap_j.iloc[-1])
    last_bb_mid = float(bb_mid.iloc[-1])


    if last_close > last_ema200:
        trend = "haussière"
    elif last_close < last_ema200:
        trend = "baissière"
    else:
        trend = "neutre"

    support, resistance = find_support_resistance(close)
    is_range = detect_range(close, support, resistance)

    psycho_notes = []
    inst_notes = []

    if abs(last_close - last_vwap) / (last_vwap + 1e-9) < 0.001:
        psycho_notes.append("Prix collé au VWAP → zone d'équilibre.")
        inst_notes.append("Prix au VWAP → zone institutionnelle.")

    if last_vol > 1.5 * avg_vol:
        psycho_notes.append("Volume institutionnel élevé.")
        inst_notes.append("Flux institutionnel détecté.")

    if last_hist > 0 and last_close > last_bb_mid:
        psycho_notes.append("Impulsion haussière.")
    elif last_hist < 0 and last_close < last_bb_mid:
        psycho_notes.append("Impulsion baissière.")

    inst_notes.append(detect_market_structure(close))

    liq = detect_liquidity_sweep(high, low)
    if liq:
        inst_notes.extend(liq)

    pd_text = detect_premium_discount(last_close, last_ema200)
    if pd_text:
        inst_notes.append(pd_text)

    fvg = detect_fvg(high, low)
    if fvg:
        inst_notes.extend(fvg)

    signal = "NEUTRE"
    bias_text = ""
    entry_price = round(last_close, 2)

    if trend == "haussière":
        bias_text = "Tendance haussière → achats privilégiés."
        if 40 <= last_rsi9 <= 50 and last_close > ema20.iloc[-1]:
            signal = "ACHAT"
            entry_price = round(last_close * 0.998, 2)

    elif trend == "baissière":
        bias_text = "Tendance baissière → ventes privilégiées."
        if 50 <= last_rsi9 <= 60 and last_close < ema20.iloc[-1]:
            signal = "VENTE"
            entry_price = round(last_close * 1.002, 2)

    else:
        bias_text = "Tendance neutre → prudence."

    if signal == "NEUTRE" and is_range:
        signal = "RANGE"
        bias_text = (
            f"Pas de direction claire — phase de range.\n"
            f"Cassure haussière > {resistance}\n"
            f"Cassure baissière < {support}"
        )

    return {
        "asset": "OR (GC=F)",
        "signal": signal,
        "entry_price": entry_price,
        "trend": trend,
        "bias_text": bias_text,
        "psycho_notes": psycho_notes,
        "inst_notes": inst_notes,
        "last_price": round(last_close, 2),
        "support": support,
        "resistance": resistance
    }


# ============================================================
# ANALYSE DUST
# ============================================================

def analyze_dust(df):
    close = df["close"].squeeze().astype(float)
    high = df["high"].squeeze().astype(float)
    low = df["low"].squeeze().astype(float)
    volume = df["volume"].squeeze().astype(float)

    ema9 = ema(close, 9)
    ema21 = ema(close, 21)
    ema100 = ema(close, 100)

    rsi7 = rsi(close, 7)
    macd_line, signal_line, hist = macd(close, 6, 19, 4)
    bb_mid, bb_up, bb_low = bollinger(close, 20, 1.5)
    stoch_k, stoch_d = stochastic(high, low, close, 5, 3, 3)

    last_close = close.iloc[-1]
    last_ema100 = ema100.iloc[-1]
    last_rsi7 = rsi7.iloc[-1]
    last_hist = hist.iloc[-1]
    last_vol = volume.iloc[-1]
    avg_vol = volume.rolling(20).mean().iloc[-1]

    if last_close > last_ema100:
        trend = "haussière"
    elif last_close < last_ema100:
        trend = "baissière"
    else:
        trend = "neutre"

    support, resistance = find_support_resistance(close)
    is_range = detect_range(close, support, resistance)

    psycho_notes = []
    inst_notes = []

    if last_vol > 1.8 * avg_vol:
        psycho_notes.append("Volume explosif sur DUST.")
        inst_notes.append("Flux spéculatif agressif.")

    if abs(last_hist) > 0.05:
        psycho_notes.append("Momentum MACD violent.")

    inst_notes.append(detect_market_structure(close))

    liq = detect_liquidity_sweep(high, low)
    if liq:
        inst_notes.extend(liq)

    pd_text = detect_premium_discount(last_close, last_ema100)
    if pd_text:
        inst_notes.append(pd_text)

    fvg = detect_fvg(high, low)
    if fvg:
        inst_notes.extend(fvg)

    signal = "NEUTRE"
    bias_text = ""
    entry_price = round(last_close, 2)

    if trend == "haussière":
        bias_text = "Tendance haussière → achats momentum."
        if last_rsi7 > 55 and last_close > ema9.iloc[-1] and last_hist > 0:
            signal = "ACHAT"
            entry_price = round(last_close * 0.997, 2)

    elif trend == "baissière":
        bias_text = "Tendance baissière → ventes."
        if last_rsi7 < 45 and last_close < ema9.iloc[-1] and last_hist < 0:
            signal = "VENTE"
            entry_price = round(last_close * 1.003, 2)

    else:
        bias_text = "Tendance neutre → prudence."

    if signal == "NEUTRE" and is_range:
        signal = "RANGE"
        bias_text = (
            f"Pas de direction claire — phase de range.\n"
            f"Cassure haussière > {resistance}\n"
            f"Cassure baissière < {support}"
        )

    return {
        "asset": "DUST",
        "signal": signal,
        "entry_price": entry_price,
        "trend": trend,
        "bias_text": bias_text,
        "psycho_notes": psycho_notes,
        "inst_notes": inst_notes,
        "last_price": round(last_close, 2),
        "support": support,
        "resistance": resistance
    }


# ============================================================
# MESSAGE TRADING
# ============================================================

def build_trading_message(gold, dust):
    header = random.choice(["📊", "📈", "⚡️", "💹"]) + " <b>Analyse OR (GC=F) & DUST</b>\n\n"
    parts = []

    parts.append(
        f"🟡 <b>OR (GC=F)</b>\n"
        f"Signal : <b>{gold['signal']}</b>\n"
        f"Prix : <b>{gold['last_price']}</b>\n"
        f"Entrée : <b>{gold['entry_price']}</b>\n"
        f"Tendance : {gold['trend']}\n"
        f"{gold['bias_text']}\n"
        f"Support : {gold['support']} — Résistance : {gold['resistance']}\n"
    )

    if gold["psycho_notes"]:
        parts.append("🧠 <b>Psycho OR :</b>\n" + "\n".join(f"• {n}" for n in gold["psycho_notes"]))

    parts.append("\n")

    # DUST
    parts.append(
        f"🟢 <b>DUST</b>\n"
        f"Signal : <b>{dust['signal']}</b>\n"
        f"Prix : <b>{dust['last_price']}</b>\n"
        f"Entrée : <b>{dust['entry_price']}</b>\n"
        f"Tendance : {dust['trend']}\n"
        f"{dust['bias_text']}\n"
        f"Support : {dust['support']} — Résistance : {dust['resistance']}\n"
    )

    if dust["psycho_notes"]:
        parts.append("🔥 <b>Psycho DUST :</b>\n" + "\n".join(f"• {n}" for n in dust["psycho_notes"]))

    # Institutionnel
    inst = []
    inst.append("• <b>OR :</b>")
    inst.extend(f"  - {n}" for n in gold["inst_notes"])
    inst.append("• <b>DUST :</b>")
    inst.extend(f"  - {n}" for n in dust["inst_notes"])

    parts.append("\n📐 <b>Lecture institutionnelle :</b>\n" + "\n".join(inst))

    return header + "\n".join(parts)


# ============================================================
# NEWS — UNE SEULE FOIS PAR JOUR À 22H
# ============================================================

def fetch_daily_news():
    if not NEWS_API_KEY:
        return None

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "gold OR GC=F OR miners OR gold mining OR DUST ETF",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": NEWS_API_KEY
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if data.get("status") != "ok":
            print("Erreur NewsAPI:", data)
            return None

        articles = data.get("articles", [])
        if not articles:
            return None

        msg = "📰 <b>Récap News OR & DUST — Journée</b>\n\n"
        for a in articles[:5]:
            title = a.get("title", "Titre inconnu")
            source = a.get("source", {}).get("name", "Source inconnue")
            msg += f"• <b>{title}</b> ({source})\n"

        return msg

    except Exception as e:
        print("Erreur fetch_daily_news:", e)
        return None


def check_news_alert():
    global last_news_date

    now = datetime.now(TZ_PARIS)
    target = now.replace(hour=NEWS_HOUR, minute=0, second=0, microsecond=0)

    if now >= target and (last_news_date is None or last_news_date != now.date()):
        news = fetch_daily_news()
        if news:
            send_telegram(news)
        last_news_date = now.date()


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

def main_loop():
    global last_trade_alert

    send_telegram("🚀 Bot OR / DUST démarré (Yahoo Finance).")

    while True:
        now = datetime.now(TZ_PARIS)
        now_ts = time.time()

        if not is_trading_time():
            print(f"[{now}] Hors horaires → pause.")
            time.sleep(600)
            continue

        if now_ts - last_trade_alert >= TRADING_INTERVAL_SEC:
            last_trade_alert = now_ts

            gold_df = fetch_intraday(GOLD_SYMBOL)
            dust_df = fetch_intraday(DUST_SYMBOL)

            if gold_df is not None and dust_df is not None:
                gold_sig = analyze_gold(gold_df)
                dust_sig = analyze_dust(dust_df)

                msg = build_trading_message(gold_sig, dust_sig)
                send_telegram(msg)

        check_news_alert()

        time.sleep(10)


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    main_loop()
