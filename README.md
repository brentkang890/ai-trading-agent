# AI Trading Agent (Full)

## Features
- Analyze OHLC from CSV/XLSX (works for crypto & forex)
- Analyze candlestick chart images (best-effort OCR digitization)
- Fetch live OHLC from Binance (crypto) via /analyze_live
- Returns natural-language analysis: Entry / SL / TP + reasons

## Endpoints
- `POST /analyze_csv` (multipart/form-data) -> file=@data.csv
- `POST /analyze_chart` (multipart/form-data) -> file=@chart.png
- `GET /analyze_live?exchange=binance&symbol=BTCUSDT&interval=1h&limit=200`

## Deployment (Railway)
1. Fork repo or upload these files to your GitHub repo.
2. Connect repo to Railway and deploy.
3. Set up ChatGPT Custom GPT action with the deployed URL.

## Notes
- Live fetch currently supports Binance (crypto). For forex live data, upload CSV.
- Image digitizer is best-effort. For reliable signals, use CSV export of OHLC.
