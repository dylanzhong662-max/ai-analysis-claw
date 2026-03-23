"""
BTC 战略级信号生成脚本

持仓周期：6个月 ~ 3年
决策频率：每日最多一次（日线收盘后）
核心逻辑：减半周期定位 + 宏观流动性 + 多时间框架技术 + 链上/衍生品情绪 + 7项利空风险评估

自动获取的数据：
  - BTC-USD 日/周/月线（yfinance）
  - ETH/SPX/NDX/DXY/TNX/VIX/Gold（yfinance）
  - 恐惧与贪婪指数（alternative.me 免费 API）
  - BTC 永续合约资金费率（Binance 公开 API）

需手动提供（Glassnode / CryptoQuant 付费接口）：
  - MVRV Ratio
  - 交易所净流量 (Exchange Net Flow)
  - 矿工头寸指数 (MPI)
  - 稳定币供应比率 (SSR)
  - M2 供应量（FRED）

依赖安装：
    pip install yfinance pandas numpy anthropic openai curl_cffi urllib3 httpx
"""

import yfinance as yf
import pandas as pd
import numpy as np
import urllib3
import os
import argparse
import httpx
from anthropic import Anthropic
from openai import OpenAI
from curl_cffi import requests as curl_requests
from datetime import datetime, date

# ─────────────────────────────────────────────
# API 配置
# ─────────────────────────────────────────────
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.openai-proxy.org/anthropic")
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY",  "sk-6BV9Xfa9AJ09pkt0AHFPQtZUtlM28pCOnon6ArdIJW1fVyDP")
ANTHROPIC_MODEL    = "claude-sonnet-4-6"

DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "sk-9574b3366dfd41178a5493d0f6af33c0")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

CLAUDE_MODELS   = {"claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"}
DEEPSEEK_MODELS = {"deepseek-reasoner", "deepseek-chat"}

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─────────────────────────────────────────────
# BTC 减半历史
# ─────────────────────────────────────────────
HALVING_DATES = [
    date(2012, 11, 28),
    date(2016, 7,  9),
    date(2020, 5,  11),
    date(2024, 4,  19),
]
NEXT_HALVING_ESTIMATE = date(2028, 4, 1)


# ─────────────────────────────────────────────
# 技术指标计算
# ─────────────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast    = calc_ema(series, fast)
    ema_slow    = calc_ema(series, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_rsi(series: pd.Series, period: int) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low   = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close  = (df['Low']  - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(com=period - 1, adjust=False).mean()


def calc_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid       = series.rolling(window=period).mean()
    std       = series.rolling(window=period).std()
    upper     = mid + std_dev * std
    lower     = mid - std_dev * std
    pct_b     = (series - lower) / (upper - lower)
    bandwidth = (upper - lower) / mid * 100
    return upper, mid, lower, pct_b, bandwidth


def calc_adx(df: pd.DataFrame, period: int = 14):
    high  = df['High'].squeeze()
    low   = df['Low'].squeeze()
    close = df['Close'].squeeze()
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    up_move   = high - high.shift()
    down_move = low.shift() - low
    plus_dm  = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    atr_s    = tr.ewm(com=period - 1, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(com=period - 1, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(com=period - 1, adjust=False).mean() / atr_s
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx      = dx.ewm(com=period - 1, adjust=False).mean()
    return adx, plus_di, minus_di


def calc_obv(df: pd.DataFrame) -> pd.Series:
    close     = df['Close'].squeeze()
    volume    = df['Volume'].squeeze()
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def calc_roc(series: pd.Series, period: int = 10) -> pd.Series:
    return (series / series.shift(period) - 1) * 100


def fmt_series(series: pd.Series, decimals: int = 0, n: int = 12) -> str:
    values = series.dropna().tail(n).round(decimals).tolist()
    return ", ".join(str(int(v) if decimals == 0 else v) for v in values)


# ─────────────────────────────────────────────
# 数据获取
# ─────────────────────────────────────────────

def _make_session() -> curl_requests.Session:
    session = curl_requests.Session(impersonate="chrome", verify=False)
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    return session


def fetch_btc_data():
    """获取 BTC-USD 日线 / 周线 / 月线数据"""
    session = _make_session()
    ticker  = "BTC-USD"
    print(f"正在获取 {ticker} 数据...")

    daily   = yf.download(ticker, period="6mo", interval="1d",
                          auto_adjust=True, progress=False, session=session)
    weekly  = yf.download(ticker, period="4y",  interval="1wk",
                          auto_adjust=True, progress=False, session=session)
    monthly = yf.download(ticker, period="5y",  interval="1mo",
                          auto_adjust=True, progress=False, session=session)

    if daily.empty or weekly.empty:
        raise ValueError(f"{ticker} 数据获取失败，请检查网络")

    print(f"  日线: {len(daily)} 条  |  周线: {len(weekly)} 条  |  月线: {len(monthly)} 条")
    return daily, weekly, monthly


def fetch_context_data() -> dict:
    """
    获取宏观 + 加密生态背景数据：
      ETH-USD  → ETH/BTC 比率
      ^GSPC    → S&P 500
      ^NDX     → 纳斯达克 100（科技股代理，BTC 相关性更高）
      DX-Y.NYB → 美元指数
      ^TNX     → 10Y 国债收益率
      ^VIX     → 恐慌指数
      GC=F     → 黄金
    """
    session = _make_session()
    tickers = {
        "eth":  "ETH-USD",
        "spx":  "^GSPC",
        "ndx":  "^NDX",
        "dxy":  "DX-Y.NYB",
        "tnx":  "^TNX",
        "vix":  "^VIX",
        "gold": "GC=F",
    }
    ctx = {}
    for key, sym in tickers.items():
        try:
            df = yf.download(sym, period="6mo", interval="1d",
                             auto_adjust=True, progress=False, session=session)
            ctx[key] = df if not df.empty else pd.DataFrame()
            status   = f"{len(df)} 条" if not df.empty else "失败"
            print(f"  {sym:12s}: {status}")
        except Exception as e:
            ctx[key] = pd.DataFrame()
            print(f"  {sym:12s}: 失败 ({e})")
    return ctx


def fetch_fear_greed() -> dict:
    """
    获取加密货币恐惧与贪婪指数（alternative.me 免费 API）
    返回最新值 + 近7日历史
    """
    try:
        session = _make_session()
        resp    = session.get("https://api.alternative.me/fng/?limit=7", timeout=10)
        data    = resp.json()
        entries = data.get("data", [])
        if not entries:
            return {}
        latest  = entries[0]
        history = [
            {"value": int(e["value"]), "classification": e["value_classification"]}
            for e in entries
        ]
        result = {
            "value":          int(latest["value"]),
            "classification": latest["value_classification"],
            "history_7d":     history,
        }
        print(f"  Fear&Greed: {result['value']} ({result['classification']})")
        return result
    except Exception as e:
        print(f"  Fear&Greed 获取失败: {e}")
        return {}


def fetch_btc_funding_rate() -> dict:
    """
    获取 BTC/USDT 永续合约资金费率（Binance 公开接口，无需 API Key）
    包含当前费率 + 近16期（约5日）历史
    """
    try:
        session = _make_session()

        # 当前预测资金费率
        snap     = session.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT", timeout=10
        )
        snap_d   = snap.json()
        cur_rate = round(float(snap_d.get("lastFundingRate", 0)) * 100, 4)

        # 近16期历史（每8小时结算一次）
        hist     = session.get(
            "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=16", timeout=10
        )
        rates    = [round(float(r["fundingRate"]) * 100, 4) for r in hist.json()[-16:]]
        avg_rate = round(sum(rates) / len(rates), 4) if rates else 0

        result = {
            "current_rate":     cur_rate,
            "recent_rates_16":  rates,
            "avg_rate":         avg_rate,
        }
        print(f"  Funding Rate: {cur_rate:.4f}% (16期均值: {avg_rate:.4f}%)")
        return result
    except Exception as e:
        print(f"  Funding Rate 获取失败: {e}")
        return {}


# ─────────────────────────────────────────────
# 减半周期计算
# ─────────────────────────────────────────────

def compute_halving_metrics(today: date | None = None) -> dict:
    if today is None:
        today = date.today()

    past      = [h for h in HALVING_DATES if h <= today]
    last      = max(past) if past else HALVING_DATES[0]
    days_sin  = (today - last).days
    days_nxt  = max((NEXT_HALVING_ESTIMATE - today).days, 0)
    cycle_pct = min(days_sin / 1461 * 100, 100)

    if cycle_pct < 25:
        phase    = "减半后早期（历史牛市蜜月期，主升浪前半段）"
        phase_en = "Early-Bull"
    elif cycle_pct < 50:
        phase    = "减半后中期（历史牛市高峰区，泡沫加速/顶部分配期）"
        phase_en = "Mid-Late-Bull"
    elif cycle_pct < 75:
        phase    = "减半后晚期（历史熊市主跌段）"
        phase_en = "Bear-Decline"
    else:
        phase    = "下次减半前积累期（历史底部/价值建仓区）"
        phase_en = "Accumulation"

    return {
        "last_halving":         last.strftime("%Y-%m-%d"),
        "days_since_halving":   days_sin,
        "months_since_halving": round(days_sin / 30.44, 1),
        "days_to_next_halving": days_nxt,
        "months_to_next":       round(days_nxt / 30.44, 1),
        "cycle_pct":            round(cycle_pct, 1),
        "phase":                phase,
        "phase_en":             phase_en,
    }


# ─────────────────────────────────────────────
# 长期价位结构
# ─────────────────────────────────────────────

def compute_long_term_levels(weekly: pd.DataFrame, monthly: pd.DataFrame) -> dict:
    wc      = weekly['Close'].squeeze().dropna()
    mc      = monthly['Close'].squeeze().dropna()
    current = float(wc.iloc[-1])
    result  = {}

    ma200w = wc.rolling(200).mean().dropna()
    result["ma200w"] = round(float(ma200w.iloc[-1]), 0) if len(ma200w) > 0 else None

    ma50w  = wc.rolling(50).mean().dropna()
    result["ma50w"]  = round(float(ma50w.iloc[-1]),  0) if len(ma50w)  > 0 else None

    if result["ma200w"]:
        result["pct_above_200w"] = round((current - result["ma200w"]) / result["ma200w"] * 100, 1)
    else:
        result["pct_above_200w"] = None

    result["ath"]          = round(float(wc.max()), 0)
    result["pct_from_ath"] = round((current - result["ath"]) / result["ath"] * 100, 1)

    cycle_low = round(float(wc.tail(208).min()), 0)
    result["cycle_low_4y"]        = cycle_low
    result["pct_above_cycle_low"] = round((current - cycle_low) / cycle_low * 100, 1)

    m_rsi = calc_rsi(mc, 14).dropna()
    result["monthly_rsi14"] = round(float(m_rsi.iloc[-1]), 1) if len(m_rsi) > 0 else None

    w_rsi = calc_rsi(wc, 14).dropna()
    result["weekly_rsi14"] = round(float(w_rsi.iloc[-1]), 1) if len(w_rsi) > 0 else None

    return result


# ─────────────────────────────────────────────
# 宏观背景摘要
# ─────────────────────────────────────────────

def _corr_30d(series_a: pd.Series, series_b: pd.Series) -> float | None:
    """计算两个日收益率序列的30日相关系数"""
    common = series_a.index.intersection(series_b.index)
    if len(common) < 20:
        return None
    ra = series_a[common].pct_change().dropna().tail(30)
    rb = series_b[common].pct_change().dropna().tail(30)
    n  = min(len(ra), len(rb))
    if n < 10:
        return None
    return round(float(ra.iloc[-n:].corr(rb.iloc[-n:])), 3)


def summarize_context(ctx: dict, btc_close: pd.Series) -> dict:
    result = {}

    def _last_n(df, col, n=10):
        if df.empty or col not in df.columns:
            return []
        return df[col].squeeze().dropna().tail(n).round(2).tolist()

    def _trend(vals, n=10):
        v = vals[-n:] if len(vals) >= n else vals
        if len(v) < 2:
            return "N/A"
        chg = (v[-1] - v[0]) / abs(v[0]) * 100 if v[0] != 0 else 0
        return f"{'↑' if chg > 0 else '↓'}{abs(chg):.1f}%({len(v)}日)"

    # ── ETH / ETH·BTC 比率 ──
    eth_df     = ctx.get("eth", pd.DataFrame())
    eth_closes = _last_n(eth_df, "Close")
    result["eth_last"]  = eth_closes[-1] if eth_closes else None
    result["eth_trend"] = _trend(eth_closes)

    if not eth_df.empty and len(btc_close) > 0:
        eth_c  = eth_df['Close'].squeeze().dropna()
        btc_c  = btc_close.reindex(eth_c.index, method='ffill').dropna()
        common = eth_c.index.intersection(btc_c.index)
        if len(common) > 0:
            ratio = (eth_c[common] / btc_c[common]).dropna().tail(10)
            result["eth_btc_ratio"]  = round(float(ratio.iloc[-1]), 5)
            result["eth_btc_trend"]  = _trend(ratio.tolist())
            result["eth_btc_series"] = [round(x, 5) for x in ratio.tolist()]
        else:
            result["eth_btc_ratio"]  = None
            result["eth_btc_series"] = []
    else:
        result["eth_btc_ratio"]  = None
        result["eth_btc_series"] = []

    # ── S&P 500 ──
    spx_df     = ctx.get("spx", pd.DataFrame())
    spx_closes = _last_n(spx_df, "Close")
    result["spx_last"]           = spx_closes[-1] if spx_closes else None
    result["spx_trend"]          = _trend(spx_closes)
    result["spx_series"]         = spx_closes
    result["btc_spx_corr_30d"]   = None
    if not spx_df.empty and len(btc_close) >= 20:
        spx_c = spx_df['Close'].squeeze().dropna()
        btc_c = btc_close.reindex(spx_c.index, method='ffill').dropna()
        result["btc_spx_corr_30d"] = _corr_30d(btc_c, spx_c)

    # ── 纳斯达克 100 ──
    ndx_df     = ctx.get("ndx", pd.DataFrame())
    ndx_closes = _last_n(ndx_df, "Close")
    result["ndx_last"]          = ndx_closes[-1] if ndx_closes else None
    result["ndx_trend"]         = _trend(ndx_closes)
    result["ndx_series"]        = ndx_closes
    result["btc_ndx_corr_30d"]  = None
    if not ndx_df.empty and len(btc_close) >= 20:
        ndx_c = ndx_df['Close'].squeeze().dropna()
        btc_c = btc_close.reindex(ndx_c.index, method='ffill').dropna()
        result["btc_ndx_corr_30d"] = _corr_30d(btc_c, ndx_c)

    # ── DXY ──
    dxy_df     = ctx.get("dxy", pd.DataFrame())
    dxy_closes = _last_n(dxy_df, "Close")
    result["dxy_last"]   = dxy_closes[-1] if dxy_closes else None
    result["dxy_trend"]  = _trend(dxy_closes)
    result["dxy_series"] = dxy_closes
    if not dxy_df.empty and 'Close' in dxy_df.columns:
        result["dxy_ema20"] = round(float(calc_ema(dxy_df['Close'].squeeze(), 20).dropna().iloc[-1]), 2)
    else:
        result["dxy_ema20"] = None

    # ── 10Y Yield ──
    tnx_df     = ctx.get("tnx", pd.DataFrame())
    tnx_closes = _last_n(tnx_df, "Close")
    result["tnx_last"]   = tnx_closes[-1] if tnx_closes else None
    result["tnx_trend"]  = _trend(tnx_closes)
    result["tnx_series"] = tnx_closes

    # ── VIX ──
    vix_df     = ctx.get("vix", pd.DataFrame())
    vix_closes = _last_n(vix_df, "Close")
    result["vix_last"]   = vix_closes[-1] if vix_closes else None
    result["vix_trend"]  = _trend(vix_closes)
    if result["vix_last"]:
        v = result["vix_last"]
        result["vix_regime"] = (
            "系统性恐慌(>40)" if v > 40 else
            "极度恐慌(>30)"   if v > 30 else
            "高波动(>20)"     if v > 20 else
            "中性(15-20)"     if v > 15 else
            "低波动/乐观(<15)"
        )
    else:
        result["vix_regime"] = "N/A"

    # ── Gold ──
    gold_df     = ctx.get("gold", pd.DataFrame())
    gold_closes = _last_n(gold_df, "Close")
    result["gold_last"]  = gold_closes[-1] if gold_closes else None
    result["gold_trend"] = _trend(gold_closes)

    if result["gold_last"] and len(btc_close) > 0:
        result["btc_gold_ratio"] = round(float(btc_close.dropna().iloc[-1]) / result["gold_last"], 2)
    else:
        result["btc_gold_ratio"] = None

    return result


# ─────────────────────────────────────────────
# 衍生品情绪摘要
# ─────────────────────────────────────────────

def _funding_rate_comment(rate: float) -> str:
    if rate > 0.05:
        return "极高正费率(>0.05%) → 多头极度拥挤，谨防'多杀多'踩踏回调"
    elif rate > 0.02:
        return "偏高正费率 → 市场偏多，多头付出较大持仓成本"
    elif rate > 0.0:
        return "正常正费率 → 市场轻微看多，情绪中性偏多"
    elif rate > -0.02:
        return "略负费率 → 市场偏空或中性，空头轻微主导"
    else:
        return "极低负费率 → 市场极度悲观，注意'空头挤压'短暂反弹风险"


def _fg_comment(value: int) -> str:
    if value >= 75:
        return "极度贪婪(≥75) → 历史反向指标：市场过热，短期回调风险高"
    elif value >= 55:
        return "贪婪(55-74) → 情绪偏多，需警惕高位风险"
    elif value >= 45:
        return "中性(45-54) → 市场情绪平衡"
    elif value >= 25:
        return "恐惧(25-44) → 市场悲观，历史上往往是中线买点"
    else:
        return "极度恐惧(<25) → 历史经典抄底区域，恐慌性抛售末期信号"


# ─────────────────────────────────────────────
# 提示词构建
# ─────────────────────────────────────────────

def build_prompt(daily: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame,
                 ctx: dict | None = None,
                 fear_greed: dict | None = None,
                 funding_rate: dict | None = None) -> str:

    # ── 日线指标 ──
    close_d              = daily['Close'].squeeze()
    d_ema20, d_ema50     = calc_ema(close_d, 20), calc_ema(close_d, 50)
    d_ema200             = calc_ema(close_d, 200)
    d_rsi14, d_rsi7      = calc_rsi(close_d, 14), calc_rsi(close_d, 7)
    d_atr14              = calc_atr(daily, 14)
    d_macd, _, _         = calc_macd(close_d)
    d_adx, _, _          = calc_adx(daily)
    _, _, _, d_pctb, _   = calc_bollinger_bands(close_d)
    d_obv                = calc_obv(daily)
    d_roc20              = calc_roc(close_d, 20)

    # ── 周线指标 ──
    close_w              = weekly['Close'].squeeze()
    w_ema20, w_ema50     = calc_ema(close_w, 20), calc_ema(close_w, 50)
    w_rsi14              = calc_rsi(close_w, 14)
    w_atr14              = calc_atr(weekly, 14)
    w_macd, _, _         = calc_macd(close_w)
    w_adx, _, _          = calc_adx(weekly)

    # ── 月线指标 ──
    close_m              = monthly['Close'].squeeze()
    m_rsi14              = calc_rsi(close_m, 14)
    m_macd, _, _         = calc_macd(close_m)

    def _last(s):
        return s.dropna().iloc[-1]

    current_price = int(_last(close_d))
    today_str     = datetime.now().strftime("%Y-%m-%d")

    d_rsi14_v  = round(float(_last(d_rsi14)),  1)
    d_rsi7_v   = round(float(_last(d_rsi7)),   1)
    d_macd_v   = int(_last(d_macd))
    d_ema20_v  = int(_last(d_ema20))
    d_ema50_v  = int(_last(d_ema50))
    d_ema200_v = int(_last(d_ema200))
    d_atr14_v  = int(_last(d_atr14))
    d_adx_v    = round(float(_last(d_adx)), 1)
    d_pctb_v   = round(float(_last(d_pctb)), 3)

    w_rsi14_v  = round(float(_last(w_rsi14)),  1)
    w_macd_v   = int(_last(w_macd))
    w_atr14_v  = int(_last(w_atr14))
    w_adx_v    = round(float(_last(w_adx)), 1)
    w_ema20_v  = int(_last(w_ema20))
    w_ema50_v  = int(_last(w_ema50))

    m_rsi14_v  = round(float(_last(m_rsi14)), 1) if len(m_rsi14.dropna()) > 0 else "N/A"
    m_macd_v   = int(_last(m_macd))             if len(m_macd.dropna())  > 0 else "N/A"

    obv_tail   = d_obv.dropna().tail(10).tolist()
    obv_trend  = "上升（量价配合）" if obv_tail[-1] > obv_tail[0] else "下降（量价背离）"

    lt      = compute_long_term_levels(weekly, monthly)
    halving = compute_halving_metrics()
    ms      = summarize_context(ctx or {}, close_d)
    fg      = fear_greed  or {}
    fr      = funding_rate or {}

    # ── 序列字符串 ──
    n_d, n_w, n_m = 20, 12, 12

    d_closes_s = fmt_series(close_d,  0, n_d)
    d_ema20_s  = fmt_series(d_ema20,  0, n_d)
    d_macd_s   = fmt_series(d_macd,   0, n_d)
    d_rsi14_s  = fmt_series(d_rsi14,  1, n_d)
    d_rsi7_s   = fmt_series(d_rsi7,   1, n_d)
    d_pctb_s   = fmt_series(d_pctb,   3, n_d)
    d_roc20_s  = fmt_series(d_roc20,  1, n_d)

    w_closes_s = fmt_series(close_w,  0, n_w)
    w_rsi14_s  = fmt_series(w_rsi14,  1, n_w)
    w_macd_s   = fmt_series(w_macd,   0, n_w)
    w_adx_s    = fmt_series(w_adx,    1, n_w)

    m_closes_s = fmt_series(close_m,  0, n_m)
    m_rsi14_s  = fmt_series(m_rsi14,  1, n_m)
    m_macd_s   = fmt_series(m_macd,   0, n_m)

    # ── 结构性目标价位（做空优先参考，非机械 ATR）──
    ma200w_val   = lt.get("ma200w")
    cycle_low_4y = lt.get("cycle_low_4y")

    # 做空目标：优先 200WMA，若已破则参考周期低点
    if ma200w_val and current_price > ma200w_val:
        short_target_primary   = int(ma200w_val)
        short_target_label_p   = f"200WMA = {short_target_primary:,}"
        short_target_secondary = int(ma200w_val * 0.82)   # 200WMA 再下 ~18%
        short_target_label_s   = f"200WMA × 0.82 ≈ {short_target_secondary:,}"
    else:
        short_target_primary   = int(cycle_low_4y * 0.9) if cycle_low_4y else current_price - int(2.5 * w_atr14_v)
        short_target_label_p   = f"周期低点折扣 ≈ {short_target_primary:,}"
        short_target_secondary = int(cycle_low_4y * 0.7) if cycle_low_4y else current_price - int(4.0 * w_atr14_v)
        short_target_label_s   = f"深度熊市目标 ≈ {short_target_secondary:,}"

    # 做多目标：前高或机械 ATR
    ath_val       = lt.get("ath", current_price + int(2.5 * w_atr14_v))
    long_target_p = int(ath_val * 0.95) if ath_val > current_price else current_price + int(2.5 * w_atr14_v)

    long_entry_lo  = current_price - int(0.3 * w_atr14_v)
    long_entry_hi  = current_price + int(0.3 * w_atr14_v)
    long_stop      = current_price - int(2.0 * w_atr14_v)
    short_entry_lo = current_price - int(0.3 * w_atr14_v)
    short_entry_hi = current_price + int(0.3 * w_atr14_v)
    short_stop     = current_price + int(2.0 * w_atr14_v)

    # ── 辅助格式化 ──
    def _v(val, unit=""):
        return f"{val}{unit}" if val is not None else "N/A"

    def _bull_bear_200w():
        m = lt.get("ma200w")
        if m is None:
            return "N/A"
        if current_price >= m:
            return f"✅ 高于200WMA ({m:,}) +{lt.get('pct_above_200w',0):.1f}% → 历史牛市区"
        else:
            return f"⚠️ 低于200WMA ({m:,}) {lt.get('pct_above_200w',0):.1f}% → 历史熊市区"

    def _monthly_rsi_comment():
        v = m_rsi14_v
        if not isinstance(v, (int, float)):
            return "N/A"
        if v > 85:
            return f"{v} — ⛔ 历史极度超买，每次>85均为周期顶部"
        elif v > 75:
            return f"{v} — ⚠️ 超买区，周期顶部风险上升"
        elif v > 60:
            return f"{v} — 牛市健康区间"
        elif v > 40:
            return f"{v} — 中性区间"
        elif v > 30:
            return f"{v} — 偏弱/熊市区间"
        else:
            return f"{v} — ✅ 历史超卖，接近周期底部"

    # ── 衍生品情绪区块 ──
    fg_section = ""
    if fg:
        fg_hist_str = " → ".join(
            f"{e['value']}({e['classification'][:4]})" for e in reversed(fg.get("history_7d", []))
        )
        fg_section = f"""
### 恐惧与贪婪指数（Crypto Fear & Greed Index）
- **当前值**: **{fg.get('value', 'N/A')} — {fg.get('classification', 'N/A')}**
- 近7日趋势: {fg_hist_str}
- **解读**: {_fg_comment(fg.get('value', 50))}
- 反向指标用法：极度贪婪时减仓/做空，极度恐惧时做多布局"""
    else:
        fg_section = "\n### 恐惧与贪婪指数：获取失败，请手动参考 alternative.me"

    fr_section = ""
    if fr:
        fr_rates_str = ", ".join(str(r) for r in fr.get("recent_rates_16", []))
        fr_section = f"""
### 资金费率（BTC 永续合约 — Binance）
- **当前费率**: **{fr.get('current_rate', 'N/A')}%**（每8小时结算）
- 近16期均值: {fr.get('avg_rate', 'N/A')}%
- 近16期序列: [{fr_rates_str}]
- **解读**: {_funding_rate_comment(fr.get('current_rate', 0))}"""
    else:
        fr_section = "\n### 资金费率：获取失败，请手动参考 Binance/Coinglass"

    # ── 相关性解读 ──
    spx_corr = ms.get("btc_spx_corr_30d")
    ndx_corr = ms.get("btc_ndx_corr_30d")

    def _corr_comment(corr, label):
        if corr is None:
            return "N/A"
        if corr > 0.7:
            return f"{corr} — 高相关：BTC紧跟{label}，{label}大跌时BTC同步抛售风险极高"
        elif corr > 0.5:
            return f"{corr} — 中高相关：宏观驱动明显，{label}下跌时BTC受牵连"
        elif corr > 0.3:
            return f"{corr} — 中等相关：{label}有影响但加密市场保有独立性"
        else:
            return f"{corr} — 低相关：BTC走独立行情"

    # ═══════════════════════════════════════════
    # 提示词正文
    # ═══════════════════════════════════════════
    prompt = f"""
# BTC 战略级交易信号分析

**分析日期**: {today_str}
**持仓周期**: 中长期（6个月 ~ 3年）
**决策频率**: 每日最多一次，以日线收盘数据为准
**分析框架**: 减半周期 × 宏观流动性 × 多周期技术 × 衍生品情绪 × 链上结构 × 利空风险控制

---

## 一、当前价格快照

| 指标 | 日线 | 周线 |
|------|------|------|
| 当前价格 | **${current_price:,}** | — |
| EMA-20 | {d_ema20_v:,} | {w_ema20_v:,} |
| EMA-50 | {d_ema50_v:,} | {w_ema50_v:,} |
| EMA-200（日线）| {d_ema200_v:,} | — |
| RSI-14 | {d_rsi14_v} | {w_rsi14_v} |
| RSI-7 | {d_rsi7_v} | — |
| **月线 RSI-14** | — | **{_monthly_rsi_comment()}** |
| MACD | {d_macd_v:,} | {w_macd_v:,} |
| ADX | {d_adx_v} | {w_adx_v} |
| ATR-14 | {d_atr14_v:,} | {w_atr14_v:,} |
| BB %B（日线）| {d_pctb_v:.3f} | — |
| OBV 趋势（近10日）| {obv_trend} | — |

---

## 二、BTC 减半周期位置

| 指标 | 数值 |
|------|------|
| 上次减半日期 | {halving['last_halving']} |
| 距上次减半 | **{halving['days_since_halving']} 天 / {halving['months_since_halving']} 个月** |
| 距下次减半（预估） | {halving['days_to_next_halving']} 天（约 {halving['months_to_next']:.0f} 个月后）|
| 本轮周期进度 | **{halving['cycle_pct']}%**（4年/1461天为满周期）|
| 历史类比阶段 | **{halving['phase']}** |

**减半周期历史规律（4次减半统计，不保证重复）：**
- 0–25%（0–12月）: 牛市蜜月，主升浪前期，做多胜率历史最高
- 25–50%（12–24月）: 牛市后期/泡沫区，历史最高点多出现于此阶段末
- 50–75%（24–36月）: 熊市主跌段，做多成功率低
- 75–100%（36–48月）: 底部积累，机构悄然建仓

---

## 三、BTC 长期价格结构

| 指标 | 数值 | 含义 |
|------|------|------|
| **200 周均线 (200WMA)** | {_v(lt.get('ma200w'))} | {_bull_bear_200w()} |
| 50 周均线 (50WMA) | {_v(lt.get('ma50w'))} | {'价格 > 50WMA ✅' if lt.get('ma50w') and current_price > lt.get('ma50w') else '价格 < 50WMA ⚠️'} |
| 距200WMA 偏离度 | {_v(lt.get('pct_above_200w'), '%')} | {'极度泡沫>150%' if lt.get('pct_above_200w') and lt.get('pct_above_200w') > 150 else ('偏高>80%' if lt.get('pct_above_200w') and lt.get('pct_above_200w') > 80 else '正常范围')} |
| 历史 ATH | {_v(lt.get('ath'))} | 距ATH: {_v(lt.get('pct_from_ath'), '%')} |
| 近4年周期低点 | {_v(lt.get('cycle_low_4y'))} | 距低点: +{_v(lt.get('pct_above_cycle_low'), '%')} |

---

## 四、多时间框架序列数据（⚠️ 从旧到新，最后一个 = 最新）

### 月线（近 {n_m} 个月）
```
收盘价: [{m_closes_s}]
RSI-14: [{m_rsi14_s}]
MACD:   [{m_macd_s}]
```

### 周线（近 {n_w} 周）
```
收盘价: [{w_closes_s}]
RSI-14: [{w_rsi14_s}]
MACD:   [{w_macd_s}]
ADX:    [{w_adx_s}]
```

### 日线（近 {n_d} 日）
```
收盘价: [{d_closes_s}]
EMA-20: [{d_ema20_s}]
MACD:   [{d_macd_s}]
RSI-14: [{d_rsi14_s}]
RSI-7:  [{d_rsi7_s}]
BB %B:  [{d_pctb_s}]
ROC-20: [{d_roc20_s}]
```

---

## 五、宏观与跨资产背景

### 美元指数 (DXY)
- 当前: **{_v(ms.get('dxy_last'))}**  |  EMA-20: {_v(ms.get('dxy_ema20'))}  |  趋势: {ms.get('dxy_trend','N/A')}
- 近10日: {ms.get('dxy_series',[])}
- 解读: DXY {'> EMA20 → 美元强势，BTC历史负相关，风险资产承压；做多BTC降低 bias_score 0.05–0.10' if ms.get('dxy_last') and ms.get('dxy_ema20') and ms.get('dxy_last') > ms.get('dxy_ema20') else '< EMA20 → 美元偏弱，有利BTC上涨'}

### 美国 10Y 国债收益率
- 当前: **{_v(ms.get('tnx_last'), '%')}**  |  趋势: {ms.get('tnx_trend','N/A')}
- 近10日: {ms.get('tnx_series',[])}
- 解读: {'实际利率上行 → 持有BTC机会成本增加 → 估值压力' if ms.get('tnx_trend','').startswith('↑') else '实际利率下行 → 无风险收益率降低 → 风险资产吸引力上升'}

### VIX 恐慌指数
- 当前: **{_v(ms.get('vix_last'))}**  |  状态: **{ms.get('vix_regime','N/A')}**  |  趋势: {ms.get('vix_trend','N/A')}
- 解读: {'VIX>30 → 系统性恐慌，BTC可能跟跌但也是历史底部机会区' if ms.get('vix_last') and ms.get('vix_last') > 30 else ('VIX>20 → 风险偏好下降，注意与SPX同步抛售' if ms.get('vix_last') and ms.get('vix_last') > 20 else 'VIX≤20 → 市场平静，不构成额外压力')}

### S&P 500 vs 纳斯达克 100（科技股代理）
| 指数 | 当前 | 趋势 | BTC/指数 30日相关系数 |
|------|------|------|----------------------|
| S&P 500 (^GSPC) | {_v(ms.get('spx_last'))} | {ms.get('spx_trend','N/A')} | {_corr_comment(spx_corr, 'SPX')} |
| **纳斯达克 100 (^NDX)** | **{_v(ms.get('ndx_last'))}** | **{ms.get('ndx_trend','N/A')}** | **{_corr_comment(ndx_corr, 'NDX')}** |

> ⚠️ **BTC 与纳斯达克相关性通常高于 S&P 500**（科技股/成长股属性），当纳指遭遇流动性挤兑时，BTC 往往率先下跌。

近10日 NDX 序列: {ms.get('ndx_series',[])}

### ETH/BTC 比率（加密风险偏好代理）
- 当前: **{_v(ms.get('eth_btc_ratio'))}**  |  趋势: {ms.get('eth_btc_trend','N/A')}
- 近10日: {ms.get('eth_btc_series',[])}
- 解读: {'ETH/BTC上升 → 山寨季，资金从BTC外溢，BTC主导性下降' if ms.get('eth_btc_trend','').startswith('↑') else 'ETH/BTC下降 → 资金回流BTC，BTC主导性上升；牛市早期或熊市防御特征'}

### 黄金 (GC=F)
- 当前: **{_v(ms.get('gold_last'))}**  |  趋势: {ms.get('gold_trend','N/A')}
- BTC/黄金比率: **{_v(ms.get('btc_gold_ratio'))} 盎司**

---

## 六、衍生品情绪指标（自动获取）
{fg_section}
{fr_section}

---

## 七、链上指标（需手动输入 — Glassnode / CryptoQuant）

> 以下字段如有数据请在分析时引用；若无数据则跳过该项。

| 指标 | 当前数值（手动填入） | 判读阈值 |
|------|---------------------|---------|
| **MVRV Ratio** | _____ | >3.5=顶部极度过热；1-2=合理；<1=极度低估/历史底部 |
| **交易所净流量 (Exchange Net Flow)** | _____ BTC | 大量流入=抛售压力；净流出=囤币信号 |
| **矿工头寸指数 (MPI)** | _____ | >2=矿工大量抛售；<-1=矿工囤币 |
| **稳定币供应比率 (SSR)** | _____ | SSR低=稳定币多、做多弹药充足；SSR高=稳定币少、资金已入场 |
| **全球 M2 供应量趋势** | _____ | M2同比上升=流动性宽松=BTC长期利好；下降=收紧周期 |

---

## 八、预计算入场锚点（基于**周线 ATR-14 = {w_atr14_v:,}**）

> 止损为战略级（2×周ATR），不被日线噪音击出。
> **做空目标必须基于结构支撑位（200WMA优先），而非机械ATR倍数。**

| 方向 | 入场区间 | 战略止损（2×周ATR）| 第一目标（结构位）| 第二目标（深度目标）|
|------|---------|-------------------|-----------------|-------------------|
| 做多 | {long_entry_lo:,} – {long_entry_hi:,} | {long_stop:,} | {long_target_p:,}（前ATH附近） | 视持仓情况滚动 |
| **做空** | {short_entry_lo:,} – {short_entry_hi:,} | **{short_stop:,}** | **{short_target_label_p}** | {short_target_label_s} |

**⚠️ 做空目标设置原则（防止过度激进）：**
- **第一目标**：200WMA（{_v(lt.get('ma200w'))}）— 历史最重要支撑，是做空的合理首要目标
- **第二目标**：仅在 200WMA 已有效跌破（周线收盘价低于200WMA）后才启用
- **禁止**：在 200WMA 未跌破的情况下，将目标直接设在 $40,000 以下（过于激进，R:R失真）

---

## 九、分析任务

你是一位专注 BTC 的宏观加密货币基金经理，管理 6个月~3年 持仓周期的仓位。

### 步骤一：判断当前 BTC 周期阶段
`Early-Bull` | `Mid-Bull` | `Late-Bull` | `Distribution` | `Bear-Decline` | `Accumulation`

### 步骤二：7项关键利空风险逐条评估（**必须全部评估**）

| # | 触发器 | 评估重点 |
|---|--------|---------|
| 1 | **200WMA 跌破** | 当前高于200WMA {_v(lt.get('pct_above_200w'),'%')}，跌破即确认熊市；跌破概率？ |
| 2 | **月线RSI超买** | RSI={m_rsi14_v}，>80为历史顶部信号；当前风险级别？ |
| 3 | **宏观流动性收紧** | 美联储政策 + 10Y收益率（当前{_v(ms.get('tnx_last'),'%')}）+ M2趋势？ |
| 4 | **美元强势** | DXY={_v(ms.get('dxy_last'))} vs EMA20={_v(ms.get('dxy_ema20'))}；强势持续性？ |
| 5 | **纳指/科技股崩盘传导** | NDX趋势={ms.get('ndx_trend','N/A')}；BTC/NDX相关系数={_v(ndx_corr)}；同步下跌风险？ |
| 6 | **监管/黑天鹅** | 当前监管环境、ETF政策稳定性评估 |
| 7 | **减半周期见顶** | 周期进度{halving['cycle_pct']}%，月RSI={m_rsi14_v}；是否进入顶部分配区？ |

### 步骤三：衍生品情绪综合判断
结合恐惧贪婪指数（{fg.get('value','N/A')} — {fg.get('classification','N/A')}）和资金费率（{fr.get('current_rate','N/A')}%），判断当前市场情绪是否过热/过冷。

### 步骤四：输出交易信号（严格 JSON 格式）

```json
{{
  "period": "Strategic",
  "analysis_date": "{today_str}",
  "btc_cycle_regime": "Early-Bull | Mid-Bull | Late-Bull | Distribution | Bear-Decline | Accumulation",
  "macro_environment": "Risk-On | Risk-Off | Neutral | Transitioning",
  "downside_risk_level": "High | Medium | Low",
  "sentiment_summary": {{
    "fear_greed_index": {fg.get('value','null')},
    "fear_greed_classification": "{fg.get('classification','N/A')}",
    "funding_rate_current": {fr.get('current_rate','null')},
    "funding_rate_signal": "<过热/中性/极度悲观>",
    "ndx_correlation": {ndx_corr if ndx_corr is not None else 'null'},
    "correlation_risk": "<高/中/低>"
  }},
  "asset_analysis": [
    {{
      "asset": "BTC",
      "holding_timeframe": "6mo-3yr",
      "regime": "Trending | Mean-Reverting | Choppy",
      "action": "long | short | no_trade",
      "bias_score": <0.0–1.0>,
      "position_sizing": "full | half | quarter | scale_in",
      "entry_zone": "<价格区间>",
      "profit_target": <数字（做空时必须基于200WMA或明确结构支撑位）或 null>,
      "stop_loss": <数字 或 null>,
      "risk_reward_ratio": <数字 或 null>,
      "key_bearish_risks": [
        {{"risk_id": 1, "risk_type": "200WMA跌破", "severity": "High|Medium|Low", "trigger_level": "<价位>", "current_status": "<一句话>"}},
        {{"risk_id": 2, "risk_type": "月线RSI超买", "severity": "High|Medium|Low", "trigger_level": "RSI>80", "current_status": "<当前RSI={m_rsi14_v}>"}},
        {{"risk_id": 3, "risk_type": "宏观流动性收紧", "severity": "High|Medium|Low", "trigger_level": "<条件>", "current_status": "<一句话>"}},
        {{"risk_id": 4, "risk_type": "美元强势", "severity": "High|Medium|Low", "trigger_level": "<DXY水平>", "current_status": "<一句话>"}},
        {{"risk_id": 5, "risk_type": "纳指崩盘传导", "severity": "High|Medium|Low", "trigger_level": "<NDX跌幅>", "current_status": "<一句话>"}},
        {{"risk_id": 6, "risk_type": "监管/黑天鹅", "severity": "High|Medium|Low", "trigger_level": "重大监管事件", "current_status": "<判断>"}},
        {{"risk_id": 7, "risk_type": "减半周期见顶", "severity": "High|Medium|Low", "trigger_level": "周期>45%且月RSI>80", "current_status": "<当前位置>"}}
      ],
      "cycle_context": "<减半周期 + 200WMA + 月RSI综合>",
      "macro_catalyst": "<核心宏观逻辑>",
      "technical_setup": "<月/周/日三级信号综合>",
      "sentiment_assessment": "<恐惧贪婪指数 + 资金费率综合解读>",
      "invalidation_condition": "<失效条件，需含具体价位>",
      "justification": "<≤500字，必须涵盖：①减半周期 ②宏观流动性 ③纳指相关性 ④技术结构 ⑤情绪指标>"
    }}
  ]
}}
```

---

## 硬性约束（违反任意一条强制 no_trade）

1. 止损不得小于 **1.5×周ATR = {int(1.5 * w_atr14_v):,}**
2. **R:R ≥ 2.0**
3. 价格低于200WMA（{_v(lt.get('ma200w'))}）时，做多 bias_score ≤ 0.50
4. 月线 RSI > 85 时，做多 bias_score ≤ 0.50
5. bias_score < 0.50 → 强制 no_trade
6. downside_risk_level = High → position_sizing 最多 "quarter"

## 信号质量过滤

- 周线 MACD ({w_macd_v:,}) < 0 且 ADX ({w_adx_v}) > 20 → 趋势性下跌，做多 bias_score ≤ 0.55
- 月线 RSI ({m_rsi14_v}) > 75 → 做多 bias_score ≤ 0.60
- 减半周期 ({halving['cycle_pct']}%) > 45% 且月RSI > 70 → 顶部预警，做多 bias_score 额外 -0.10
- 200WMA 偏离 > 150% → 做多 bias_score ≤ 0.55
- DXY > EMA20 且趋势向上 → 做多 bias_score -0.05–0.10
- 恐惧贪婪指数 > 75（极度贪婪）→ 做多 bias_score -0.10；做空 bias_score +0.05
- 恐惧贪婪指数 < 25（极度恐惧）→ 做空 bias_score -0.10；做多 bias_score +0.05（反向信号）
- 资金费率 > 0.05% → 做多 bias_score -0.10（多头拥挤警告）
- 资金费率 < -0.03% → 做空 bias_score -0.10（空头挤压风险）
- NDX/BTC 相关系数 > 0.7 且 NDX 下跌 → 做多 bias_score -0.05
- ETH/BTC 比率下降 → BTC 主导性强，做多 bias_score +0.05
- **做空 profit_target 设置规则**：
  - 200WMA 未跌破：第一目标 = 200WMA（{_v(lt.get('ma200w'))}），不得设在200WMA以下
  - 200WMA 已跌破（周线收盘确认）：可设第二目标（~{short_target_secondary:,}）
  - 任何情况下禁止将首次做空目标设在当前价格 50% 以下（过于激进，R:R失真）
""".strip()

    return prompt


# ─────────────────────────────────────────────
# API 调用
# ─────────────────────────────────────────────

def call_claude_api(prompt: str) -> str:
    print(f"\n正在调用 Claude API（模型: {ANTHROPIC_MODEL}）...")
    client = Anthropic(
        base_url=ANTHROPIC_BASE_URL,
        api_key=ANTHROPIC_API_KEY,
        http_client=httpx.Client(verify=False),
    )
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    result = ""
    for block in message.content:
        if hasattr(block, "text"):
            result += block.text
    return result


def call_deepseek_api(prompt: str, model: str) -> str:
    import re, time
    print(f"\n正在调用 DeepSeek API（模型: {model}）...")
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content or ""
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            return raw
        except Exception as e:
            print(f"  第 {attempt + 1} 次调用失败: {e}")
            if attempt < 2:
                time.sleep(5)
    return ""


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BTC 战略级信号生成脚本（6个月~3年持仓）")
    parser.add_argument("--api",   action="store_true",
                        help="直接调用 API 获取分析结果（默认只生成提示词文件）")
    parser.add_argument("--model", default=ANTHROPIC_MODEL,
                        help=(
                            f"模型 ID（需配合 --api）。"
                            f"Claude: {', '.join(sorted(CLAUDE_MODELS))}；"
                            f"DeepSeek: {', '.join(sorted(DEEPSEEK_MODELS))}。"
                            f"默认: {ANTHROPIC_MODEL}"
                        ))
    args = parser.parse_args()

    # 获取 BTC 数据
    daily, weekly, monthly = fetch_btc_data()

    # 获取宏观背景
    print("\n正在获取宏观背景数据（ETH / SPX / NDX / DXY / TNX / VIX / Gold）...")
    ctx = fetch_context_data()

    # 获取衍生品情绪
    print("\n正在获取衍生品情绪数据...")
    fg = fetch_fear_greed()
    fr = fetch_btc_funding_rate()

    # 生成提示词
    prompt = build_prompt(daily, weekly, monthly, ctx=ctx, fear_greed=fg, funding_rate=fr)

    # 保存提示词
    output_path = "btc_prompt_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"\n提示词已保存: {output_path}")

    if args.api:
        model = args.model
        if model in DEEPSEEK_MODELS:
            analysis = call_deepseek_api(prompt, model)
        else:
            analysis = call_claude_api(prompt)

        api_output_path = "btc_api_output.txt"
        with open(api_output_path, "w", encoding="utf-8") as f:
            f.write(analysis)
        print("\n" + "=" * 60)
        print(analysis)
        print("=" * 60)
        print(f"\n分析结果已保存: {api_output_path}")
    else:
        print("\n" + "=" * 60)
        print(prompt)
        print("=" * 60)
        print("\n将上方内容粘贴到 Claude.ai，或使用 --api 直接调用。")
        print("示例：")
        print("  python3 btc_analysis.py --api")
        print("  python3 btc_analysis.py --api --model deepseek-reasoner")
        print("  python3 btc_analysis.py --api --model deepseek-chat")


if __name__ == "__main__":
    main()
