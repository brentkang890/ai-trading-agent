"""
AI Trading Agent (Full)
- Endpoints:
  - POST /analyze_csv   : upload CSV/XLSX containing OHLC (works for crypto & forex)
  - POST /analyze_chart : upload chart image (attempt OCR + digitize for crypto/forex)
  - GET  /analyze_live  : fetch live OHLC (Binance only — crypto). For forex, upload CSV.
Notes:
- For best results provide CSV export for forex data or crypto if possible.
- This is a best-effort digitizer for images; accuracy depends on image quality.
"""

import io, re, math, os, time
from typing import Optional, Tuple, Dict, List
from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import PlainTextResponse, FileResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np, pandas as pd, requests, cv2
import pytesseract

app = FastAPI(title="AI Trading Agent - Full")

# ------------------ Utilities (indicators & helpers) ------------------
def sma(series: pd.Series, length:int=20):
    return series.rolling(length, min_periods=1).mean()

def rsi(series: pd.Series, length:int=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(span=length, adjust=False).mean()
    ma_down = down.ewm(span=length, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0,1e-9))
    return 100 - (100 / (1 + rs))

def detect_support_resistance(series: pd.Series, window:int=20) -> Tuple[float,float]:
    # simple SR: recent highest high and lowest low in window
    high = series.rolling(window, min_periods=1).max().iloc[-1]
    low = series.rolling(window, min_periods=1).min().iloc[-1]
    return float(high), float(low)

def fmt(x:float):
    if abs(x) >= 1000: return f"{x:,.0f}"
    if abs(x) >= 1: return f"{x:,.2f}"
    return f"{x:.6f}"

# ------------------ CSV analysis endpoint ------------------
@app.post("/analyze_csv", response_class=PlainTextResponse)
async def analyze_csv(file: UploadFile = File(...), pair: Optional[str] = Form(None), timeframe: Optional[str] = Form(None)):
    """
    Accepts CSV or XLSX with OHLC columns. Works for crypto & forex.
    """
    try:
        contents = await file.read()
        ext = (file.filename or "").lower().split('.')[-1]
        if ext == 'csv':
            df = pd.read_csv(io.BytesIO(contents))
        elif ext in ('xls','xlsx'):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return PlainTextResponse("Unsupported file type. Use .csv or .xlsx")
        # normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        def find_col(k):
            for c in df.columns:
                if k in c: return c
            return None
        o,h,l,c = find_col('open'), find_col('high'), find_col('low'), find_col('close')
        if not all([o,h,l,c]):
            return PlainTextResponse("File must contain columns with open, high, low, close headers")
        df = df[[o,h,l,c]].rename(columns={o:'open',h:'high',l:'low',c:'close'}).dropna().reset_index(drop=True)
        # ensure numeric
        for col in ['open','high','low','close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().reset_index(drop=True)
        if df.shape[0] < 6:
            return PlainTextResponse("Not enough candles (need at least 6 rows)")
        text = analyze_ohlc(df, pair=pair, timeframe=timeframe)
        return PlainTextResponse(text)
    except Exception as e:
        return PlainTextResponse(f"Error analyze_csv: {e}")

# ------------------ Live data (Binance) ------------------
BINANCE_SPOT="https://api.binance.com/api/v3/klines"
@app.get("/analyze_live", response_class=PlainTextResponse)
async def analyze_live(exchange: str = Query(...), symbol: str = Query(...), interval: str = Query("1h"), limit: int = Query(200)):
    """
    Live fetch for crypto from Binance only.
    For forex, upload CSV via /analyze_csv.
    """
    try:
        exchange = exchange.lower()
        if exchange == 'binance':
            r = requests.get(BINANCE_SPOT, params={'symbol': symbol.upper(), 'interval': interval, 'limit': limit}, timeout=15)
            r.raise_for_status()
            data = r.json()
            df = pd.DataFrame(data, columns=['open_time','open','high','low','close','vol','close_time','qav','trades','tb','tq','ig'])
            for col in ['open','high','low','close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df[['open','high','low','close']].dropna().reset_index(drop=True)
            return PlainTextResponse(analyze_ohlc(df, pair=symbol, timeframe=interval))
        else:
            return PlainTextResponse("Live fetch only supported for Binance (crypto). For forex, upload CSV via /analyze_csv.")
    except Exception as e:
        return PlainTextResponse(f"Error analyze_live: {e}")

# ------------------ Image chart analyzer (best-effort OCR digitizer) ------------------
def read_y_ticks(img_cv: np.ndarray) -> Dict[int,float]:
    h,w = img_cv.shape[:2]
    left_w = max(70, int(w*0.14))
    crop = img_cv[:, :left_w].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,-'
    data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
    ticks = {}
    for i,t in enumerate(data['text']):
        if not t or re.search(r'[A-Za-z]', t): continue
        m = re.search(r'[-+]?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?', t)
        if m:
            raw = m.group(0).replace(',', '.')
            try:
                val = float(raw)
            except: continue
            top = data['top'][i]; hgt = data['height'][i]; cy = int(top + hgt/2)
            ticks[cy] = val
    return ticks

def detect_candles_simple(img_cv: np.ndarray) -> List[Dict]:
    # rough approach: find colored or dark vertical bodies in plotting area
    h,w = img_cv.shape[:2]
    lm = int(w*0.12); rm = int(w*0.02); tm = int(h*0.06); bm = int(h*0.04)
    plot = img_cv[tm:h-bm, lm:w-rm].copy()
    ph,pw = plot.shape[:2]
    hsv = cv2.cvtColor(plot, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    col_mean = v.mean(axis=0)
    thr = np.percentile(col_mean,60)
    peaks = np.where(col_mean < thr)[0]
    groups=[]; i=0
    while i < len(peaks):
        j=i
        while j+1<len(peaks) and peaks[j+1]==peaks[j]+1: j+=1
        groups.append((peaks[i], peaks[j])); i=j+1
    candles=[]
    for g in groups:
        mid = (g[0]+g[1])//2
        col = plot[:, mid]
        gray = cv2.cvtColor(col.reshape(-1,1,3), cv2.COLOR_BGR2GRAY).flatten()
        dark = np.where(gray < np.percentile(gray,60))[0]
        if len(dark) < 3: continue
        top = int(dark.min()); bot = int(dark.max())
        candles.append({'cx': lm+mid, 'body_top': tm+top, 'body_bottom': tm+bot})
    return candles

def build_ohlc_from_image(img_cv: np.ndarray, candles: List[Dict], ticks_map: Optional[Tuple[np.ndarray,np.ndarray]]):
    if not candles or not ticks_map: return None
    ys, prices = ticks_map
    h,w = img_cv.shape[:2]
    lm = int(w*0.12); tm = int(h*0.06); rm = int(w*0.02); bm = int(h*0.04)
    plot = img_cv[tm:h-bm, lm:w-rm].copy(); ph,pw=plot.shape[:2]
    rows=[]
    for c in candles:
        cx_local = c['cx'] - lm
        if cx_local < 0 or cx_local >= pw: continue
        body_top_local = c['body_top'] - tm; body_bot_local = c['body_bottom'] - tm
        # find wick extents by scanning column brightness
        col = plot[:, max(0,min(pw-1,cx_local))]; gray = cv2.cvtColor(col.reshape(-1,1,3), cv2.COLOR_BGR2GRAY).flatten()
        thr = np.percentile(gray,70)
        wt=0; wb=ph-1
        for y in range(body_top_local, -1, -1):
            if gray[y] > thr: wt = y+1; break
            wt = y
        for y in range(body_bot_local, ph):
            if gray[y] > thr: wb = y-1; break
            wb = y
        # map pixels to price
        high_px = tm + wt; low_px = tm + wb
        open_px = tm + body_top_local; close_px = tm + body_bot_local
        high_p = float(np.interp(high_px, ys, prices))
        low_p = float(np.interp(low_px, ys, prices))
        open_p = float(np.interp(open_px, ys, prices))
        close_p = float(np.interp(close_px, ys, prices))
        rows.append({'open':open_p,'high':high_p,'low':low_p,'close':close_p,'cx':c['cx']})
    if not rows: return None
    df = pd.DataFrame(rows).sort_values('cx').reset_index(drop=True)[['open','high','low','close']]
    return df

@app.post("/analyze_chart", response_class=PlainTextResponse)
async def analyze_chart(file: UploadFile = File(...), pair: Optional[str] = Form(None), timeframe: Optional[str] = Form(None)):
    """
    Best-effort: attempt to read price ticks on left and detect candle bodies.
    If OCR cannot read ticks, returns helpful message.
    """
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return PlainTextResponse(f"Cannot read image: {e}")
    ticks = read_y_ticks(img_cv)
    if not ticks or len(ticks) < 2:
        return PlainTextResponse("OCR failed to detect Y-axis price ticks. Please provide high-res chart with visible price axis, or upload CSV for accurate analysis.")
    ys = np.array(sorted(ticks.keys())); prices = np.array([ticks[y] for y in ys])
    candles = detect_candles_simple(img_cv)
    if not candles:
        return PlainTextResponse("No candle bodies detected. Ensure chart has visible candlesticks and not heatmap/area chart.")
    df = build_ohlc_from_image(img_cv, candles, (ys, prices))
    if df is None or df.shape[0] < 6:
        return PlainTextResponse("Failed to build OHLC from image reliably. Recommend upload CSV export for best accuracy.")
    return PlainTextResponse(analyze_ohlc(df, pair=pair, timeframe=timeframe))

# ------------------ Core analysis logic ------------------
def analyze_ohlc(df: pd.DataFrame, pair: Optional[str]=None, timeframe: Optional[str]=None) -> str:
    df = df.copy().dropna().reset_index(drop=True)
    df['sma20'] = sma(df['close'],20)
    df['rsi14'] = rsi(df['close'],14)
    last = df.iloc[-1]
    trend = "up" if last['close'] > last['sma20'] else "down"
    last_rsi = float(last['rsi14']) if not math.isnan(last['rsi14']) else None
    recent_high, recent_low = detect_support_resistance(df['high'], window=10)[0], detect_support_resistance(df['low'], window=10)[1] if False else (df['high'].rolling(10,min_periods=1).max().iloc[-1], df['low'].rolling(10,min_periods=1).min().iloc[-1])
    # bias rules
    if last['close'] > last['sma20'] and (last_rsi is None or last_rsi < 75):
        bias = "bullish"
        entry = last['close']; sl = recent_low * 0.995; rr = entry - sl; tp1 = entry + rr*1.5; tp2 = entry + rr*2.5
        reason = f"Harga di atas SMA20 ({fmt(last['sma20'])}); RSI {round(last_rsi,1) if last_rsi else 'n/a'}"
    elif last['close'] < last['sma20'] and (last_rsi is None or last_rsi > 25):
        bias = "bearish"
        entry = last['close']; sl = recent_high * 1.005; rr = sl - entry; tp1 = entry - rr*1.5; tp2 = entry - rr*2.5
        reason = f"Harga di bawah SMA20 ({fmt(last['sma20'])}); RSI {round(last_rsi,1) if last_rsi else 'n/a'}"
    else:
        bias = "neutral"; entry = last['close']; sl = recent_low*0.995; tp1 = entry + (entry-sl)*1.2; tp2 = entry + (entry-sl)*2.0
        reason = "Trend kurang jelas; gunakan konfirmasi tambahan (multi-timeframe / volume)"
    # support/resistance quick
    res = df['high'].tail(30).max(); sup = df['low'].tail(30).min()
    txt = []
    header = f"Analisa {pair or ''} {timeframe or ''}".strip()
    txt.append(header); txt.append("")
    txt.append(reason)
    txt.append(f"Bias: {bias}")
    txt.append(f"Entry: {fmt(entry)}")
    txt.append(f"Stop Loss: {fmt(sl)}")
    txt.append(f"Take Profit 1: {fmt(tp1)}")
    txt.append(f"Take Profit 2: {fmt(tp2)}")
    txt.append("")
    txt.append(f"Recent Resistance (30c): {fmt(res)}  |  Recent Support (30c): {fmt(sup)}")
    txt.append(f"SMA20: {fmt(last['sma20'])}  |  RSI14: {round(last_rsi,1) if last_rsi else 'n/a'}")
    txt.append("")
    txt.append("⚠️ Catatan: Ini sinyal otomatis. Verifikasi manual sebelum open posisi. Untuk forex (live), unggah CSV OHLC karena live fetch hanya untuk Binance crypto.")
    return "\\n".join(txt)

# ------------------ Health check ------------------
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "AI Trading Agent - OK"
