import yfinance as yf
from datetime import datetime
from typing import Optional

TICKER_MAP = {
    "GOLD":       "GC=F",
    "SILVER":     "SLV",
    "COPPER":     "COPX",
    "RARE_EARTH": "REMX",
    "OIL":        "USO",
    "BTC":        "BTC-USD",
    "GOOGL":      "GOOGL",
    "MSFT":       "MSFT",
    "NVDA":       "NVDA",
    "AAPL":       "AAPL",
    "META":       "META",
    "AMZN":       "AMZN",
}

_cache: dict = {}
_cache_ts: dict = {}
CACHE_TTL = 300  # seconds


def get_current_price(asset: str) -> Optional[float]:
    ticker = TICKER_MAP.get(asset, asset)
    now = datetime.now()
    cached_at = _cache_ts.get(asset)
    if cached_at and (now - cached_at).seconds < CACHE_TTL and asset in _cache:
        return _cache[asset]
    try:
        data = yf.download(ticker, period="2d", interval="1d", progress=False, auto_adjust=True)
        if not data.empty:
            price = float(data["Close"].iloc[-1])
            _cache[asset] = price
            _cache_ts[asset] = now
            return price
    except Exception:
        pass
    return None


def get_macro_prices() -> dict:
    tickers = {"VIX": "^VIX", "DXY": "DX-Y.NYB", "TNX": "^TNX"}
    result = {}
    for key, ticker in tickers.items():
        try:
            data = yf.download(ticker, period="2d", interval="1d", progress=False, auto_adjust=True)
            if not data.empty:
                result[key] = round(float(data["Close"].iloc[-1]), 3)
            else:
                result[key] = None
        except Exception:
            result[key] = None
    return result
