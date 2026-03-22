"""
纳斯达克科技股中长线分析脚本
支持 GOOGL / MSFT / NVDA / AAPL / META / AMZN 等科技股
持仓周期：1–6 个月（中长线摆动交易）

运行方式：
    python tech_stock_analysis.py --ticker GOOGL          # 生成提示词文件
    python tech_stock_analysis.py --ticker GOOGL --api    # 直接调用 Claude API
    python tech_stock_analysis.py --ticker NVDA --api --model deepseek-reasoner

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
from datetime import datetime


# ─────────────────────────────────────────────
# API 配置
# ─────────────────────────────────────────────
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.openai-proxy.org/anthropic")
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY",  "ANTHROPIC_API_KEY_REMOVED")
ANTHROPIC_MODEL    = "claude-sonnet-4-6"

DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_REMOVED")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

CLAUDE_MODELS   = {"claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"}
DEEPSEEK_MODELS = {"deepseek-reasoner", "deepseek-chat"}

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ─────────────────────────────────────────────
# 技术指标计算
# ─────────────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low   = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close  = (df['Low']  - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(com=period - 1, adjust=False).mean()


def calc_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid   = series.rolling(window=period).mean()
    std   = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    pct_b     = (series - lower) / (upper - lower)
    bandwidth = (upper - lower) / mid * 100
    return upper, mid, lower, pct_b, bandwidth


def calc_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    low_min  = df['Low'].squeeze().rolling(window=k_period).min()
    high_max = df['High'].squeeze().rolling(window=k_period).max()
    k = 100 * (df['Close'].squeeze() - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    return k, d


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


def calc_roc(series: pd.Series, period: int = 20) -> pd.Series:
    return (series / series.shift(period) - 1) * 100


def fmt_series(series: pd.Series, decimals: int = 2, n: int = 12) -> str:
    values = series.dropna().tail(n).round(decimals).tolist()
    return ", ".join(str(v) for v in values)


# ─────────────────────────────────────────────
# 数据获取
# ─────────────────────────────────────────────

def _make_session() -> curl_requests.Session:
    session = curl_requests.Session(impersonate="chrome", verify=False)
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    return session


def fetch_stock_data(ticker: str):
    """
    获取三个时间维度的股价数据：
      - 日线 3 个月：用于精确入场定时
      - 周线 2 年：主分析时间框架，识别中线信号
      - 月线 5 年：长期趋势背景
    """
    session = _make_session()
    print(f"正在获取 {ticker} 股价数据...")

    daily = yf.download(
        ticker, period="3mo", interval="1d",
        auto_adjust=True, progress=False, session=session
    )
    weekly = yf.download(
        ticker, period="2y", interval="1wk",
        auto_adjust=True, progress=False, session=session
    )
    monthly = yf.download(
        ticker, period="5y", interval="1mo",
        auto_adjust=True, progress=False, session=session
    )

    if daily.empty or weekly.empty:
        raise ValueError(f"无法获取 {ticker} 数据，请检查 ticker 或网络连接")

    print(f"  日线: {len(daily)} 条 | 周线: {len(weekly)} 条 | 月线: {len(monthly)} 条")
    return daily, weekly, monthly


def fetch_macro_data() -> dict:
    """
    获取宏观基准数据：
      - QQQ  : 纳斯达克100 ETF（大盘方向锚点）
      - XLK  : 科技板块 ETF（板块轮动信号）
      - SPY  : 标普500（风险偏好参照）
      - ^TNX : 美国10年期国债收益率（成长股估值敏感因子）
      - ^VIX : 恐慌指数（市场情绪）
      - DX-Y.NYB : 美元指数（海外营收汇率影响）
    """
    session = _make_session()
    tickers_map = {
        "qqq": "QQQ",
        "xlk": "XLK",
        "spy": "SPY",
        "tnx": "^TNX",
        "vix": "^VIX",
        "dxy": "DX-Y.NYB",
    }
    macro = {}
    print("正在获取宏观基准数据...")
    for key, t in tickers_map.items():
        try:
            df = yf.download(
                t, period="1y", interval="1wk",
                auto_adjust=True, progress=False, session=session
            )
            macro[key] = df if not df.empty else pd.DataFrame()
            print(f"  {t:12s}: {'%d 条' % len(df) if not df.empty else '失败'}")
        except Exception as e:
            macro[key] = pd.DataFrame()
            print(f"  {t:12s}: 获取失败 ({e})")
    return macro


def fetch_intelligence_data(ticker: str) -> dict:
    """
    获取基本面情报数据（通过 yfinance Ticker 对象）：
      - 下次财报日期 / 距今天数
      - EPS / 营收预估（当季、下季、全年）
      - 分析师评级与目标价
      - 估值指标：Forward P/E、PEG、P/B、营收增速、利润率等
      - 机构持股比例、Beta、空头回补天数
    """
    print(f"正在获取 {ticker} 情报数据...")

    result = {
        # 财报
        "earnings_date": None,
        "earnings_days_away": None,
        # EPS/营收预估
        "eps_estimate_current_q": None,
        "eps_estimate_next_q": None,
        "eps_estimate_current_y": None,
        "eps_growth_estimate": None,
        "revenue_estimate_current_y": None,
        "revenue_growth_estimate": None,
        # 分析师
        "analyst_target_mean": None,
        "analyst_target_high": None,
        "analyst_target_low": None,
        "analyst_recommendation": None,
        "analyst_strong_buy": None,
        "analyst_buy": None,
        "analyst_hold": None,
        "analyst_sell": None,
        # 估值
        "forward_pe": None,
        "trailing_pe": None,
        "peg_ratio": None,
        "price_to_book": None,
        # 成长
        "revenue_growth": None,
        "earnings_growth": None,
        # 盈利质量
        "gross_margins": None,
        "operating_margins": None,
        "profit_margins": None,
        "return_on_equity": None,
        "free_cashflow": None,
        # 风险
        "debt_to_equity": None,
        "beta": None,
        "short_ratio": None,
        "institutional_ownership": None,
        # 价格结构
        "52w_high": None,
        "52w_low": None,
        "market_cap": None,
    }

    try:
        t = yf.Ticker(ticker, session=_make_session())

        # ── 财报日期 ──
        try:
            cal = t.calendar
            ed = None
            if cal is not None:
                if isinstance(cal, dict) and "Earnings Date" in cal:
                    ed = cal["Earnings Date"][0] if cal["Earnings Date"] else None
                elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.columns and not cal.empty:
                    ed = cal["Earnings Date"].iloc[0]
            if ed is not None:
                if hasattr(ed, 'date'):
                    ed = ed.date()
                result["earnings_date"] = str(ed)
                result["earnings_days_away"] = int((pd.Timestamp(ed) - pd.Timestamp.now()).days)
        except Exception as e:
            print(f"  财报日期: 获取失败 ({e})")

        # ── EPS 预估 ──
        try:
            ee = t.earnings_estimate
            if ee is not None and not ee.empty:
                for idx, key in [("0q", "eps_estimate_current_q"), ("+1q", "eps_estimate_next_q"),
                                  ("0y", "eps_estimate_current_y")]:
                    if idx in ee.index and "avg" in ee.columns:
                        result[key] = round(float(ee.loc[idx, "avg"]), 3)
                if "0y" in ee.index and "growth" in ee.columns:
                    result["eps_growth_estimate"] = round(float(ee.loc["0y", "growth"]) * 100, 1)
        except Exception as e:
            print(f"  EPS预估: 获取失败 ({e})")

        # ── 营收预估 ──
        try:
            re_ = t.revenue_estimate
            if re_ is not None and not re_.empty and "0y" in re_.index:
                if "avg" in re_.columns:
                    result["revenue_estimate_current_y"] = round(float(re_.loc["0y", "avg"]) / 1e9, 2)
                if "growth" in re_.columns:
                    result["revenue_growth_estimate"] = round(float(re_.loc["0y", "growth"]) * 100, 1)
        except Exception as e:
            print(f"  营收预估: 获取失败 ({e})")

        # ── 分析师目标价 ──
        try:
            apt = t.analyst_price_targets
            if apt and isinstance(apt, dict):
                result["analyst_target_mean"] = round(float(apt["mean"]), 2) if apt.get("mean") else None
                result["analyst_target_high"] = round(float(apt["high"]), 2) if apt.get("high") else None
                result["analyst_target_low"]  = round(float(apt["low"]),  2) if apt.get("low")  else None
        except Exception as e:
            print(f"  分析师目标价: 获取失败 ({e})")

        # ── 分析师评级分布 ──
        try:
            rec = t.recommendations_summary
            if rec is not None and not rec.empty:
                row = rec.iloc[0]
                sb = int(row.get("strongBuy", 0))
                b  = int(row.get("buy", 0))
                h  = int(row.get("hold", 0))
                s  = int(row.get("sell", 0))
                result["analyst_strong_buy"] = sb
                result["analyst_buy"]        = b
                result["analyst_hold"]       = h
                result["analyst_sell"]       = s
                total = sb + b + h + s
                if total > 0:
                    bull_pct = (sb + b) / total * 100
                    result["analyst_recommendation"] = f"看多{sb+b}/{total} ({bull_pct:.0f}%)"
        except Exception as e:
            print(f"  分析师评级: 获取失败 ({e})")

        # ── 估值与基本面指标 ──
        try:
            info = t.info
            pct_fields = {
                "revenue_growth":       "revenueGrowth",
                "earnings_growth":      "earningsGrowth",
                "gross_margins":        "grossMargins",
                "operating_margins":    "operatingMargins",
                "profit_margins":       "profitMargins",
                "return_on_equity":     "returnOnEquity",
                "institutional_ownership": "heldPercentInstitutions",
            }
            for rk, ik in pct_fields.items():
                v = info.get(ik)
                if v is not None:
                    result[rk] = f"{v * 100:.1f}%"

            float_fields = {
                "forward_pe":   ("forwardPE", 2),
                "trailing_pe":  ("trailingPE", 2),
                "peg_ratio":    ("pegRatio", 2),
                "price_to_book":("priceToBook", 2),
                "debt_to_equity":("debtToEquity", 2),
                "beta":         ("beta", 2),
                "short_ratio":  ("shortRatio", 2),
                "52w_high":     ("fiftyTwoWeekHigh", 2),
                "52w_low":      ("fiftyTwoWeekLow", 2),
            }
            for rk, (ik, dec) in float_fields.items():
                v = info.get(ik)
                if v is not None:
                    result[rk] = round(float(v), dec)

            mc = info.get("marketCap")
            if mc:
                result["market_cap"] = f"${mc / 1e9:.1f}B"

            fcf = info.get("freeCashflow")
            if fcf:
                result["free_cashflow"] = f"${fcf / 1e9:.1f}B"

        except Exception as e:
            print(f"  基本面指标: 获取失败 ({e})")

        # ── 盈利惊喜历史（最近4季） ──
        try:
            eh = t.earnings_history
            if eh is not None and not eh.empty:
                surprises = []
                for _, row in eh.tail(4).iterrows():
                    eps_est = row.get("epsEstimate")
                    eps_act = row.get("epsActual")
                    if eps_est is not None and eps_act is not None and eps_est != 0:
                        surprise_pct = (eps_act - eps_est) / abs(eps_est) * 100
                        surprises.append(round(surprise_pct, 1))
                result["eps_surprise_history"] = surprises
                result["eps_beat_count"] = sum(1 for s in surprises if s > 0)
        except Exception as e:
            print(f"  盈利惊喜历史: 获取失败 ({e})")

        # ── 季度营收趋势 ──
        try:
            qf = t.quarterly_financials
            if qf is not None and not qf.empty:
                rev_row = None
                for label in ["Total Revenue", "TotalRevenue"]:
                    if label in qf.index:
                        rev_row = qf.loc[label]
                        break
                if rev_row is not None:
                    rev_sorted = rev_row.sort_index().dropna().tail(5)
                    rev_list = [round(float(v) / 1e9, 1) for v in rev_sorted.values]
                    result["quarterly_revenue_trend"] = rev_list
                    if len(rev_list) >= 3:
                        g1 = (rev_list[-1] - rev_list[-2]) / abs(rev_list[-2]) * 100
                        g2 = (rev_list[-2] - rev_list[-3]) / abs(rev_list[-3]) * 100
                        result["revenue_acceleration"] = round(g1 - g2, 1)
        except Exception as e:
            print(f"  季度营收趋势: 获取失败 ({e})")

    except Exception as e:
        print(f"  情报数据整体获取失败: {e}")

    valid_count = sum(1 for v in result.values() if v is not None)
    print(f"  情报数据完成，有效字段: {valid_count}/{len(result)}")
    return result


# ─────────────────────────────────────────────
# 同业对标数据
# ─────────────────────────────────────────────

# 每只股票的关键同业对标（广告/云/芯片/消费科技等维度）
_PEER_MAP = {
    "GOOGL": ["META", "MSFT"],   # 数字广告 + 云计算
    "GOOG":  ["META", "MSFT"],
    "META":  ["GOOGL", "SNAP"],  # 数字广告 + 社交
    "MSFT":  ["GOOGL", "AMZN"],  # 云计算 + AI SaaS
    "NVDA":  ["AMD", "AVGO"],    # AI 芯片
    "AMD":   ["NVDA", "INTC"],   # 芯片
    "AMZN":  ["MSFT", "GOOGL"],  # 云计算 + 电商
    "AAPL":  ["MSFT", "GOOGL"],  # 大市值科技
    "TSLA":  ["RIVN", "F"],      # 电动车
    "NFLX":  ["DIS", "WBD"],     # 流媒体
}


def fetch_peer_data(ticker: str) -> dict:
    """
    获取同业对标股票近8周相对 QQQ 的超额表现。
    用于判断个股在板块内的相对强弱。
    """
    peers = _PEER_MAP.get(ticker.upper(), [])
    if not peers:
        return {}

    session = _make_session()
    result = {}
    print(f"正在获取 {ticker} 同业对标数据 ({', '.join(peers)})...")

    try:
        qqq_df = yf.download("QQQ", period="2mo", interval="1wk",
                              auto_adjust=True, progress=False, session=session)
        if qqq_df.empty:
            return {}
        qqq_ret = float((qqq_df["Close"].squeeze().iloc[-1] /
                         qqq_df["Close"].squeeze().iloc[0] - 1) * 100)

        for peer in peers:
            try:
                peer_df = yf.download(peer, period="2mo", interval="1wk",
                                      auto_adjust=True, progress=False, session=session)
                if peer_df.empty:
                    continue
                peer_ret = float((peer_df["Close"].squeeze().iloc[-1] /
                                  peer_df["Close"].squeeze().iloc[0] - 1) * 100)
                result[peer] = {
                    "8w_return_pct":  round(peer_ret, 1),
                    "vs_qqq_8w_pct":  round(peer_ret - qqq_ret, 1),
                    "current_price":  round(float(peer_df["Close"].squeeze().iloc[-1]), 2),
                }
                print(f"  {peer}: {peer_ret:+.1f}%  vs QQQ: {peer_ret - qqq_ret:+.1f}%")
            except Exception:
                pass
    except Exception as e:
        print(f"  同业对标数据获取失败: {e}")

    return result


# ─────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> dict:
    close = df['Close'].squeeze()
    ind = {}
    ind['ema20']     = calc_ema(close, 20)
    ind['ema50']     = calc_ema(close, 50)
    ind['ema200']    = calc_ema(close, 200)
    macd, sig, hist  = calc_macd(close)
    ind['macd']      = macd
    ind['macd_sig']  = sig
    ind['macd_hist'] = hist
    ind['rsi14']     = calc_rsi(close, 14)
    ind['rsi7']      = calc_rsi(close, 7)
    stk, stk_d       = calc_stochastic(df)
    ind['stoch_k']   = stk
    ind['stoch_d']   = stk_d
    ind['atr14']     = calc_atr(df, 14)
    bb_up, bb_mid, bb_lo, pct_b, bw = calc_bollinger_bands(close)
    ind['bb_upper']  = bb_up
    ind['bb_lower']  = bb_lo
    ind['bb_pct_b']  = pct_b
    ind['bb_bw']     = bw
    adx, pdi, mdi    = calc_adx(df)
    ind['adx']       = adx
    ind['plus_di']   = pdi
    ind['minus_di']  = mdi
    ind['obv']       = calc_obv(df)
    ind['roc20']     = calc_roc(close, 20)
    return ind


def compute_relative_strength(stock_df: pd.DataFrame, bench_df: pd.DataFrame, period: int = 20) -> dict:
    """计算个股相对 QQQ 的滚动相对强弱（period 周）"""
    try:
        sc = stock_df['Close'].squeeze().dropna()
        bc = bench_df['Close'].squeeze().dropna()
        common = sc.index.intersection(bc.index)
        sc = sc.reindex(common)
        bc = bc.reindex(common)
        rs = (sc / sc.shift(period)) / (bc / bc.shift(period)) - 1
        rs_clean = rs.dropna()
        rs_last = round(float(rs_clean.iloc[-1]) * 100, 2)
        rs_trend = [round(x * 100, 2) for x in rs_clean.tail(6).tolist()]
        return {
            "rs_vs_qqq_pct": rs_last,
            "rs_trend_6w":   rs_trend,
            "signal":        "跑赢QQQ" if rs_last > 0 else "跑输QQQ",
        }
    except Exception:
        return {"rs_vs_qqq_pct": None, "rs_trend_6w": [], "signal": "N/A"}


# ─────────────────────────────────────────────
# 宏观摘要
# ─────────────────────────────────────────────

def summarize_macro_equity(macro: dict) -> dict:
    result = {}

    def _last_n(df, col, n=8):
        if df.empty or col not in df.columns:
            return []
        return df[col].squeeze().dropna().tail(n).round(3).tolist()

    def _trend(vals):
        if len(vals) < 2:
            return "N/A"
        chg = (vals[-1] - vals[0]) / abs(vals[0]) * 100 if vals[0] != 0 else 0
        return f"{'↑' if chg > 0 else '↓'} {abs(chg):.1f}% (近8周)"

    def _ema_cross_status(df, fast=20, slow=50):
        if df.empty or 'Close' not in df.columns:
            return "N/A"
        close = df['Close'].squeeze().dropna()
        if len(close) < slow:
            return "N/A"
        ema_f = calc_ema(close, fast).iloc[-1]
        ema_s = calc_ema(close, slow).iloc[-1]
        return "金叉(多头)" if ema_f > ema_s else "死叉(空头)"

    # QQQ
    qqq = macro.get("qqq", pd.DataFrame())
    qqq_closes = _last_n(qqq, "Close")
    result["qqq_last"]   = round(qqq_closes[-1], 2) if qqq_closes else None
    result["qqq_trend"]  = _trend(qqq_closes)
    result["qqq_cross"]  = _ema_cross_status(qqq)
    result["qqq_series"] = qqq_closes

    # XLK vs QQQ 板块轮动
    xlk = macro.get("xlk", pd.DataFrame())
    xlk_closes = _last_n(xlk, "Close")
    result["xlk_last"]  = round(xlk_closes[-1], 2) if xlk_closes else None
    result["xlk_trend"] = _trend(xlk_closes)
    if qqq_closes and xlk_closes and len(qqq_closes) > 1 and len(xlk_closes) > 1:
        n = min(len(qqq_closes), len(xlk_closes))
        xlk_ret = (xlk_closes[-1] / xlk_closes[-n] - 1) * 100
        qqq_ret = (qqq_closes[-1] / qqq_closes[-n] - 1) * 100
        result["xlk_vs_qqq_pct"] = round(xlk_ret - qqq_ret, 2)
        result["sector_rotation"] = "科技板块领涨" if xlk_ret > qqq_ret else "科技板块落后"
    else:
        result["xlk_vs_qqq_pct"] = None
        result["sector_rotation"] = "N/A"

    # SPY
    spy = macro.get("spy", pd.DataFrame())
    spy_closes = _last_n(spy, "Close")
    result["spy_last"]  = round(spy_closes[-1], 2) if spy_closes else None
    result["spy_trend"] = _trend(spy_closes)

    # 10Y 收益率
    tnx = macro.get("tnx", pd.DataFrame())
    tnx_vals = _last_n(tnx, "Close")
    result["tnx_last"]   = round(tnx_vals[-1], 3) if tnx_vals else None
    result["tnx_trend"]  = _trend(tnx_vals)
    result["tnx_series"] = tnx_vals

    # VIX
    vix = macro.get("vix", pd.DataFrame())
    vix_vals = _last_n(vix, "Close")
    result["vix_last"]  = round(vix_vals[-1], 2) if vix_vals else None
    result["vix_trend"] = _trend(vix_vals)
    if result["vix_last"]:
        v = result["vix_last"]
        result["vix_regime"] = (
            "极度恐慌(>35)" if v > 35 else
            ("恐慌(>25)"    if v > 25 else
            ("高波动(>20)"  if v > 20 else
            ("中性(>15)"    if v > 15 else "低波动/乐观(<15)")))
        )

    # DXY
    dxy = macro.get("dxy", pd.DataFrame())
    dxy_vals = _last_n(dxy, "Close")
    result["dxy_last"]  = round(dxy_vals[-1], 2) if dxy_vals else None
    result["dxy_trend"] = _trend(dxy_vals)

    return result


# ─────────────────────────────────────────────
# 情报分析区块格式化
# ─────────────────────────────────────────────

def format_intelligence_section(intel: dict, current_price: float) -> str:
    def _v(key, default="N/A"):
        val = intel.get(key)
        return str(val) if val is not None else default

    # 分析师目标价上行空间
    pt_upside_str = ""
    if intel.get("analyst_target_mean") and current_price:
        upside = (intel["analyst_target_mean"] - current_price) / current_price * 100
        pt_upside_str = f"  (**较当前价上行空间: {upside:+.1f}%**)"

    # 财报风险提示
    days = intel.get("earnings_days_away")
    if days is not None and days >= 0:
        if days <= 5:
            earnings_alert = f"\n> **⚠️ 财报风险警告：距下次财报仅 {days} 天，强制 no_trade**"
        elif days <= 15:
            earnings_alert = f"\n> **注意：距下次财报 {days} 天，建议降低仓位**"
        else:
            earnings_alert = f"\n> 距下次财报：约 {days} 天（{_v('earnings_date')}）"
    else:
        earnings_alert = ""

    # Forward P/E 解读
    fpe = intel.get("forward_pe")
    fpe_interp = "N/A"
    if isinstance(fpe, (int, float)):
        fpe_interp = "偏高，估值有压力" if fpe > 35 else ("合理偏高" if fpe > 25 else "合理区间")

    # PEG 解读
    peg = intel.get("peg_ratio")
    peg_interp = "N/A"
    if isinstance(peg, (int, float)):
        peg_interp = "估值偏贵 (>2)" if peg > 2 else ("合理 (1–2)" if peg > 1 else "低估 (<1)")

    # Beta 解读
    beta = intel.get("beta")
    beta_interp = "N/A"
    if isinstance(beta, (int, float)):
        beta_interp = "高波动性，止损需宽" if beta > 1.3 else ("中等波动" if beta > 0.8 else "低波动")

    # 空头回补解读
    sr = intel.get("short_ratio")
    sr_interp = "N/A"
    if isinstance(sr, (int, float)):
        sr_interp = "空头拥挤，潜在轧空行情" if sr > 5 else ("空头中等" if sr > 2 else "空头较少")

    section = f"""
---

## 三、情报分析 (Intelligence Analysis)

### 3.1 财报与盈利预估
{earnings_alert}

| 指标 | 数值 |
|------|------|
| 下次财报日期 | {_v('earnings_date')} |
| 距财报天数 | {_v('earnings_days_away')} 天 |
| 本季度 EPS 预估 (均值) | ${_v('eps_estimate_current_q')} |
| 下季度 EPS 预估 (均值) | ${_v('eps_estimate_next_q')} |
| 全年 EPS 预估 (均值) | ${_v('eps_estimate_current_y')} |
| 全年 EPS 预期增速 | {_v('eps_growth_estimate')}% |
| 全年营收预估 | ${_v('revenue_estimate_current_y')}B |
| 营收预期增速 | {_v('revenue_growth_estimate')}% |

### 3.2 分析师共识

| 指标 | 数值 |
|------|------|
| 综合评级 | {_v('analyst_recommendation')} |
| 平均目标价 | ${_v('analyst_target_mean')}{pt_upside_str} |
| 目标价区间 | ${_v('analyst_target_low')} – ${_v('analyst_target_high')} |
| 强烈买入 / 买入 / 持有 / 卖出 | {_v('analyst_strong_buy')} / {_v('analyst_buy')} / {_v('analyst_hold')} / {_v('analyst_sell')} |

### 3.3 估值与财务质量

| 指标 | 数值 | 解读 |
|------|------|------|
| 市值 | {_v('market_cap')} | — |
| 远期市盈率 (Forward P/E) | {_v('forward_pe')} | {fpe_interp} |
| PEG 比率 | {_v('peg_ratio')} | {peg_interp} |
| 市净率 (P/B) | {_v('price_to_book')} | — |
| 营收增长 (YoY) | {_v('revenue_growth')} | — |
| 盈利增长 (YoY) | {_v('earnings_growth')} | — |
| 毛利率 | {_v('gross_margins')} | — |
| 运营利润率 | {_v('operating_margins')} | — |
| 净利润率 | {_v('profit_margins')} | — |
| 股本回报率 (ROE) | {_v('return_on_equity')} | — |
| 自由现金流 | {_v('free_cashflow')} | — |
| 债务权益比 | {_v('debt_to_equity')} | — |
| Beta | {_v('beta')} | {beta_interp} |
| 空头回补天数 | {_v('short_ratio')} 天 | {sr_interp} |
| 机构持股比例 | {_v('institutional_ownership')} | — |
| 52周高点 | ${_v('52w_high')} | — |
| 52周低点 | ${_v('52w_low')} | — |

### 3.4 盈利惊喜历史（最近4季 EPS 实际 vs 预估）
"""

    # 盈利惊喜历史
    surprises = intel.get("eps_surprise_history", [])
    beat_count = intel.get("eps_beat_count")
    if surprises:
        surprise_cells = " | ".join(
            f"{'✅' if s > 0 else '❌'} {s:+.1f}%" for s in surprises
        )
        beat_label = f"{beat_count}/{len(surprises)} 季超预期" if beat_count is not None else "N/A"
        surprise_trend = (
            "连续超预期，盈利质量高" if beat_count == len(surprises)
            else ("多数超预期" if beat_count and beat_count >= len(surprises) * 0.75
            else ("多数不及预期，注意盈利压力" if beat_count is not None and beat_count < len(surprises) * 0.5
            else "超预期表现一般"))
        )
        section += f"""
| 指标 | Q-3 | Q-2 | Q-1 | 最近季 | 综合 |
|------|-----|-----|-----|--------|------|
| EPS 惊喜% | {surprise_cells} | {beat_label} |

> **解读**: {surprise_trend}
"""
    else:
        section += "\n> 盈利惊喜数据暂不可用\n"

    # 季度营收趋势
    rev_trend = intel.get("quarterly_revenue_trend", [])
    rev_accel = intel.get("revenue_acceleration")
    section += "\n### 3.5 季度营收趋势（最近5季，单位 $B）\n"
    if rev_trend:
        rev_str = " → ".join(f"${v}B" for v in rev_trend)
        accel_str = ""
        if rev_accel is not None:
            if rev_accel > 1:
                accel_str = f"  ✅ **营收加速** (环比增速提升 {rev_accel:+.1f}pp)"
            elif rev_accel < -1:
                accel_str = f"  ⚠️ **营收减速** (环比增速下降 {rev_accel:.1f}pp)"
            else:
                accel_str = f"  营收增速平稳 ({rev_accel:+.1f}pp)"
        section += f"\n{rev_str}{accel_str}\n"
    else:
        section += "\n> 季度营收数据暂不可用\n"

    return section.strip()


# ─────────────────────────────────────────────
# 构建提示词
# ─────────────────────────────────────────────

def build_prompt_equity(
    ticker: str,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    monthly: pd.DataFrame,
    macro: dict,
    intel: dict,
    perf_metrics: dict | None = None,
    peer_data: dict | None = None,
) -> str:

    # ── 指标计算 ──
    w_ind = compute_indicators(weekly)
    m_ind = compute_indicators(monthly)
    close_w = weekly['Close'].squeeze()
    close_m = monthly['Close'].squeeze()
    close_d = daily['Close'].squeeze()

    current_price = round(float(close_d.iloc[-1]), 2)

    # 周线关键指标快照
    def _safe_last(series):
        s = series.dropna()
        return round(float(s.iloc[-1]), 2) if len(s) > 0 else None

    w_ema20   = _safe_last(w_ind['ema20'])
    w_ema50   = _safe_last(w_ind['ema50'])
    w_ema200  = _safe_last(w_ind['ema200'])
    w_macd    = _safe_last(w_ind['macd'])
    w_rsi14   = _safe_last(w_ind['rsi14'])
    w_rsi7    = _safe_last(w_ind['rsi7'])
    w_atr14   = _safe_last(w_ind['atr14'])
    w_adx     = round(float(w_ind['adx'].dropna().iloc[-1]), 1) if w_ind['adx'].dropna().shape[0] > 0 else None
    w_pdi     = round(float(w_ind['plus_di'].dropna().iloc[-1]), 1) if w_ind['plus_di'].dropna().shape[0] > 0 else None
    w_mdi     = round(float(w_ind['minus_di'].dropna().iloc[-1]), 1) if w_ind['minus_di'].dropna().shape[0] > 0 else None
    w_bb_pctb = round(float(w_ind['bb_pct_b'].dropna().iloc[-1]), 3) if w_ind['bb_pct_b'].dropna().shape[0] > 0 else None
    w_bb_bw   = round(float(w_ind['bb_bw'].dropna().iloc[-1]), 2) if w_ind['bb_bw'].dropna().shape[0] > 0 else None
    stoch_k   = round(float(w_ind['stoch_k'].dropna().iloc[-1]), 1) if w_ind['stoch_k'].dropna().shape[0] > 0 else None
    stoch_d   = round(float(w_ind['stoch_d'].dropna().iloc[-1]), 1) if w_ind['stoch_d'].dropna().shape[0] > 0 else None
    w_roc20   = round(float(w_ind['roc20'].dropna().iloc[-1]), 2) if w_ind['roc20'].dropna().shape[0] > 0 else None

    # 月线关键指标
    m_ema20  = _safe_last(m_ind['ema20'])
    m_macd   = _safe_last(m_ind['macd'])
    m_rsi14  = _safe_last(m_ind['rsi14'])

    # OBV 趋势（周线，近6期）
    obv_s = w_ind['obv'].dropna().tail(6).tolist()
    obv_trend = "上升" if len(obv_s) >= 2 and obv_s[-1] > obv_s[0] else "下降"

    # 相对强弱 vs QQQ
    rs_data = compute_relative_strength(weekly, macro.get("qqq", pd.DataFrame()), period=20)

    # 宏观摘要
    ms = summarize_macro_equity(macro)

    # 52周高低位（基于日线最近252个交易日）
    close_d_full = close_d.dropna()
    high_52w = round(float(close_d_full.tail(252).max()), 2)
    low_52w  = round(float(close_d_full.tail(252).min()), 2)
    pct_from_high = round((current_price - high_52w) / high_52w * 100, 1)
    pct_from_low  = round((current_price - low_52w)  / low_52w  * 100, 1)

    # 序列数据
    n_w, n_m = 12, 12
    weekly_closes  = fmt_series(close_w, 2, n_w)
    weekly_macd    = fmt_series(w_ind['macd'], 2, n_w)
    weekly_rsi14   = fmt_series(w_ind['rsi14'], 2, n_w)
    weekly_ema20   = fmt_series(w_ind['ema20'], 2, n_w)
    weekly_ema50   = fmt_series(w_ind['ema50'], 2, n_w)
    weekly_adx     = fmt_series(w_ind['adx'], 1, n_w)
    weekly_stoch_k = fmt_series(w_ind['stoch_k'], 1, n_w)
    weekly_bb_pctb = fmt_series(w_ind['bb_pct_b'], 3, n_w)
    monthly_closes = fmt_series(close_m, 2, n_m)
    monthly_macd   = fmt_series(m_ind['macd'], 2, n_m)
    monthly_rsi14  = fmt_series(m_ind['rsi14'], 2, n_m)

    # 预计算入场锚点（基于周线 ATR，中长线适用）
    atr = w_atr14 or 1.0
    long_entry_lo  = round(current_price - 0.5 * atr, 2)
    long_entry_hi  = round(current_price + 0.5 * atr, 2)
    long_stop      = round(current_price - 2.0 * atr, 2)
    long_target    = round(current_price + 4.0 * atr, 2)
    short_entry_lo = round(current_price - 0.5 * atr, 2)
    short_entry_hi = round(current_price + 0.5 * atr, 2)
    short_stop     = round(current_price + 2.0 * atr, 2)
    short_target   = round(current_price - 4.0 * atr, 2)

    # 情报分析区块
    intel_section = format_intelligence_section(intel, current_price)

    # 同业对标区块
    peer_section = ""
    if peer_data:
        peer_rows = []
        for peer, data in peer_data.items():
            vs = data.get("vs_qqq_8w_pct")
            ret = data.get("8w_return_pct")
            price = data.get("current_price")
            vs_str = f"{vs:+.1f}%" if vs is not None else "N/A"
            ret_str = f"{ret:+.1f}%" if ret is not None else "N/A"
            signal = "领先" if vs and vs > 0 else "落后"
            peer_rows.append(f"| {peer} | ${price} | {ret_str} | {vs_str} | {signal} |")
        peer_table = "\n".join(peer_rows)
        peer_section = f"""
---

## 四、同业对标分析（近8周表现）

| 对标股 | 当前价 | 8周涨跌 | vs QQQ 超额 | 信号 |
|--------|--------|---------|------------|------|
{peer_table}

> **解读方向**：若同业整体跑输 QQQ，说明该细分赛道（广告/云/AI）面临结构性压力；若 {ticker} 跑输同业，说明个股层面存在相对弱势，需谨慎。
"""

    # 股票专属行业背景（静态 prompt engineering，给模型提供行业分析框架）
    _INDUSTRY_CONTEXT = {
        "GOOGL": """
### 行业特有分析维度（GOOGL / Alphabet）

请在 `competitive_intelligence` 字段中，额外关注以下 GOOGL 专属因素：

1. **搜索广告 vs AI 替代威胁**：AI Overviews / ChatGPT 搜索对传统搜索广告 CPM 的影响迹象（目前技术面上反映为何种走势？）
2. **Google Cloud 增速**：云业务是否保持 >25% YoY 增速？是否在抢夺 AWS/Azure 份额？估值重新定价的关键催化剂。
3. **YouTube 广告**：与 Meta Reels / TikTok 的用户时长竞争态势，是否存在广告主预算流失？
4. **监管风险**：DOJ 反垄断诉讼（搜索/广告技术）是否有新进展，是否构成中期估值压制？
5. **Gemini AI 货币化**：AI Studio、Workspace AI 的付费转化节奏，与 OpenAI / MSFT Copilot 的竞争差距。
""",
        "META": """
### 行业特有分析维度（Meta Platforms）

请在 `competitive_intelligence` 字段中，额外关注以下 Meta 专属因素：

1. **广告 CPM 趋势**：季度广告 ARPU 是否加速？与 GOOGL/TikTok 的广告预算份额对比。
2. **Reels 货币化**：短视频 Reels 的广告负载率是否接近 Feed，是否仍是超额增长来源？
3. **AI 基础设施投入**：CapEx 是否超预期抬升？对 FCF 和估值的影响。
4. **Reality Labs 亏损**：元宇宙部门亏损是否收窄，或成为持续负担？
5. **中国广告主依赖**：Temu / Shein 等中国广告主占比，关税政策影响敞口。
""",
        "NVDA": """
### 行业特有分析维度（NVIDIA）

请在 `competitive_intelligence` 字段中，额外关注以下 NVDA 专属因素：

1. **Blackwell 出货节奏**：新一代 GPU 是否如期量产，数据中心客户订单可见度。
2. **AI 资本开支超级周期**：微软/谷歌/Meta/亚马逊 CapEx 指引，是否支撑持续需求。
3. **竞争威胁**：AMD MI300X 市场份额是否扩大？谷歌 TPU、亚马逊 Trainium 自研芯片渗透率。
4. **出口管制风险**：H20/L20 对华出口限制是否收紧，中国收入占比与替代预案。
5. **估值锚点**：以 CY26 EPS 为基础的 Forward P/E，当前定价隐含的增速预期是否现实。
""",
        "MSFT": """
### 行业特有分析维度（Microsoft）

请在 `competitive_intelligence` 字段中，额外关注以下 MSFT 专属因素：

1. **Azure 增速**：云业务是否保持 >25% YoY？是否在抢占 AWS 份额？AI 相关云需求占比。
2. **Copilot 货币化**：Microsoft 365 Copilot 的付费席位增速，是否带动 ARPU 提升。
3. **OpenAI 押注**：与 OpenAI 的合作是否加深，或面临竞争（GPT-5 vs Copilot 定位冲突）。
4. **Activision 整合**：游戏业务是否产生协同，对营收多元化的贡献。
5. **企业 IT 预算周期**：大型企业 IT 支出是否重启，对 Office 365 / Azure 续约率的影响。
""",
        "AMZN": """
### 行业特有分析维度（Amazon）

请在 `competitive_intelligence` 字段中，额外关注以下 AMZN 专属因素：

1. **AWS 增速**：云业务是否保持 >20% YoY？AI 服务（Bedrock）的增量贡献。
2. **零售利润率扩张**：电商部门是否持续实现运营杠杆，物流成本压缩进度。
3. **广告业务**：第三方广告是否维持高增速，逐步成为高利润率收入来源。
4. **Prime 续费率**：订阅经济黏性，是否有提价空间。
5. **Anthropic 投资**：AI 模型布局对 AWS 竞争力的加分，与微软/谷歌的 AI 云差距是否收窄。
""",
    }
    industry_section = _INDUSTRY_CONTEXT.get(ticker.upper(), "")

    # 性能反馈区块
    perf_section = ""
    if perf_metrics:
        wrf = perf_metrics.get("win_rate_float")
        cl  = perf_metrics.get("consecutive_losses", 0)
        warns = []
        if wrf is not None and wrf < 0.40:
            warns.append("⚠️ 胜率低于40%：bias_score 门槛强制提升至 ≥0.65")
        if cl >= 2:
            warns.append(f"⚠️ 已连续亏损 {cl} 次：需 bias_score ≥0.75 才入场")
        warn_str = "\n".join(warns) if warns else "当前表现正常，维持标准决策流程。"
        perf_section = f"""
---

## 近期回测表现反馈

| 指标 | 数值 |
|------|------|
| 胜率 | {perf_metrics.get('win_rate', 'N/A')} |
| 平均盈利 | {perf_metrics.get('avg_win', 'N/A')} |
| 平均亏损 | {perf_metrics.get('avg_loss', 'N/A')} |
| 盈利因子 | {perf_metrics.get('profit_factor', 'N/A')} |
| 总收益 | {perf_metrics.get('total_return', 'N/A')} |
| 最近连续亏损 | {cl} 次 |

{warn_str}
"""

    def _v(val, suffix=""):
        return f"{val}{suffix}" if val is not None else "N/A"

    ema_cross_status = (
        "金叉 (EMA50 > EMA200，多头结构)" if w_ema50 and w_ema200 and w_ema50 > w_ema200
        else "死叉 (EMA50 < EMA200，空头结构)" if w_ema50 and w_ema200
        else "N/A"
    )
    ema20_bias = (
        "价格高于EMA20，短期偏多" if w_ema20 and current_price > w_ema20
        else "价格低于EMA20，短期偏空" if w_ema20
        else "N/A"
    )
    ema50_bias = (
        "价格高于EMA50，中期偏多" if w_ema50 and current_price > w_ema50
        else "价格低于EMA50，中期偏空" if w_ema50
        else "N/A"
    )
    ema200_bias = (
        "价格高于EMA200，长期牛市" if w_ema200 and current_price > w_ema200
        else "价格低于EMA200，长期熊市" if w_ema200
        else "N/A"
    )

    today_str = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
# {ticker} 纳斯达克科技股中长线分析请求

**分析日期**: {today_str}
**持仓目标周期**: 1–6 个月（中长线，减少交易频率，持仓让利润奔跑）
**主分析时间框架**: 周线 (W1) + 月线 (M1) 趋势，日线辅助入场精确定时
{perf_section}
---

## 一、行情分析 (Price Action Analysis)

### 1.1 价格概要

- **当前价格**: ${current_price}
- **52周高点**: ${high_52w}  |  距高点: {pct_from_high:+.1f}%
- **52周低点**: ${low_52w}   |  距低点: {pct_from_low:+.1f}%

### 1.2 周线关键指标快照

| 指标 | 当前值 | 信号解读 |
|------|--------|----------|
| EMA-20 (周) | {_v(w_ema20)} | {ema20_bias} |
| EMA-50 (周) | {_v(w_ema50)} | {ema50_bias} |
| EMA-200 (周) | {_v(w_ema200)} | {ema200_bias} |
| EMA 金/死叉 (50/200周) | {ema_cross_status} | 长期趋势方向判断 |
| MACD (周) | {_v(w_macd)} | {'正值，多头动能' if w_macd and w_macd > 0 else '负值，空头动能'} |
| RSI-14 (周) | {_v(w_rsi14)} | {'超买区 >70，谨慎追多' if w_rsi14 and w_rsi14 > 70 else ('超卖区 <30，关注反弹' if w_rsi14 and w_rsi14 < 30 else '中性区间 30–70')} |
| RSI-7 (周) | {_v(w_rsi7)} | {'极度超买 >80' if w_rsi7 and w_rsi7 > 80 else ('极度超卖 <20' if w_rsi7 and w_rsi7 < 20 else '正常范围')} |
| ADX (周) | {_v(w_adx)} | {'强趋势 >25' if w_adx and w_adx > 25 else ('弱趋势/振荡 <20' if w_adx and w_adx < 20 else '趋势形成中 20–25')} |
| +DI / -DI | {_v(w_pdi)} / {_v(w_mdi)} | {'+DI>-DI 多头主导' if w_pdi and w_mdi and w_pdi > w_mdi else '-DI>+DI 空头主导'} |
| Stochastic %K / %D | {_v(stoch_k)} / {_v(stoch_d)} | {'超买死叉，谨慎做多' if stoch_k and stoch_d and stoch_k > 80 and stoch_k < stoch_d else ('超卖金叉，关注做多' if stoch_k and stoch_d and stoch_k < 20 and stoch_k > stoch_d else '中性区间')} |
| BB %B (周) | {_v(w_bb_pctb)} | {'突破上轨 >1' if w_bb_pctb and w_bb_pctb > 1 else ('跌破下轨 <0' if w_bb_pctb and w_bb_pctb < 0 else '布林带内运行')} |
| BB 带宽 % | {_v(w_bb_bw)} | {'带宽扩张，趋势加速' if w_bb_bw and w_bb_bw > 10 else '带宽收缩，蓄势'} |
| ROC-20 (周) | {_v(w_roc20, '%')} | {'正动量' if w_roc20 and w_roc20 > 0 else '负动量'} |
| OBV 趋势 (近6周) | {obv_trend} | {'量价配合，机构积累' if obv_trend == '上升' else '量价背离，机构派发'} |
| 相对强弱 vs QQQ (20周) | {_v(rs_data.get('rs_vs_qqq_pct'), '%')} | {rs_data.get('signal', 'N/A')} |

### 1.3 月线长期趋势背景

| 指标 | 当前值 | 解读 |
|------|--------|------|
| EMA-20 (月) | {_v(m_ema20)} | {'价格高于月线EMA20，长期趋势向上' if m_ema20 and current_price > m_ema20 else '价格低于月线EMA20，长期趋势向下'} |
| MACD (月) | {_v(m_macd)} | {'月线多头动能' if m_macd and m_macd > 0 else '月线空头动能'} |
| RSI-14 (月) | {_v(m_rsi14)} | {'月线超买，长期注意顶部风险' if m_rsi14 and m_rsi14 > 70 else ('月线超卖，长期关注底部机会' if m_rsi14 and m_rsi14 < 30 else '月线中性')} |

### 1.4 序列数据（从旧到新排列，最后一个值 = 最新）

**周线数据（近12周）**：
收盘价:     [{weekly_closes}]
EMA-20:     [{weekly_ema20}]
EMA-50:     [{weekly_ema50}]
MACD:       [{weekly_macd}]
RSI-14:     [{weekly_rsi14}]
ADX:        [{weekly_adx}]
Stoch %K:   [{weekly_stoch_k}]
BB %B:      [{weekly_bb_pctb}]

**月线数据（近12个月）**：
收盘价:     [{monthly_closes}]
MACD:       [{monthly_macd}]
RSI-14:     [{monthly_rsi14}]

---

## 二、结构化分析 (Structured Analysis)

### 2.1 五层市场层级评估

| 层级 | 维度 | 当前状态 | 方向 |
|------|------|----------|------|
| L1 宏观 | Fed政策 + 10Y收益率 | {_v(ms.get('tnx_last'), '%')}  趋势: {ms.get('tnx_trend', 'N/A')} | 收益率{'上升→成长股承压' if ms.get('tnx_trend','').startswith('↑') else '下降→成长股受益'} |
| L2 指数 | QQQ 趋势 | {_v(ms.get('qqq_last'))}  趋势: {ms.get('qqq_trend', 'N/A')}  EMA: {ms.get('qqq_cross', 'N/A')} | {'多头' if ms.get('qqq_cross','').startswith('金') else '空头'} |
| L3 板块 | XLK vs QQQ | XLK超额: {_v(ms.get('xlk_vs_qqq_pct'), '%')}  {ms.get('sector_rotation', 'N/A')} | {'有利' if ms.get('xlk_vs_qqq_pct') and ms['xlk_vs_qqq_pct'] > 0 else '不利'} |
| L4 个股 | {ticker} vs QQQ (20周) | 超额收益: {_v(rs_data.get('rs_vs_qqq_pct'), '%')}  {rs_data.get('signal', 'N/A')} | {'有利' if rs_data.get('rs_vs_qqq_pct') and rs_data['rs_vs_qqq_pct'] > 0 else '不利'} |
| L5 技术 | 入场时机 | 基于上方指标综合评判 | 待模型判断 |

### 2.2 基准走势（近8周）

QQQ 周收盘序列: {ms.get('qqq_series', [])}
SPY 趋势: {ms.get('spy_trend', 'N/A')}  |  QQQ趋势: {ms.get('qqq_trend', 'N/A')}

### 2.3 市场情绪

VIX: {_v(ms.get('vix_last'))}  |  状态: {ms.get('vix_regime', 'N/A')}  |  趋势: {ms.get('vix_trend', 'N/A')}
DXY: {_v(ms.get('dxy_last'))}  |  趋势: {ms.get('dxy_trend', 'N/A')}

{intel_section}
{peer_section}
{industry_section}

---

## 五、预计算入场锚点（基于周线 ATR-14 = {_v(w_atr14)}）

> 中长线持仓止损和目标均基于**周线 ATR**，给价格充分的波动空间（目标持仓 4–26 周）。
> entry_zone 必须在此范围内；可在 ±0.5×wATR 范围内根据支撑/阻力微调。

| 方向 | entry_zone | stop_loss (2×wATR) | profit_target (4×wATR, RR=2.0) | 预计持仓 |
|------|-----------|--------------------|---------------------------------|---------|
| 做多 | {long_entry_lo} – {long_entry_hi} | {long_stop} | {long_target} | 4–26 周 |
| 做空 | {short_entry_lo} – {short_entry_hi} | {short_stop} | {short_target} | 4–26 周 |

---

## 六、分析任务

请基于以上三个维度的数据，按照**纳斯达克科技股中长线分析框架**完成分析，严格按以下 JSON 格式输出：

```json
{{
  "period": "Weekly",
  "stock_ticker": "{ticker}",
  "overall_market_sentiment": "Risk-On | Risk-Off | Neutral",
  "qqq_assessment": "<QQQ趋势及其对{ticker}的方向性影响>",
  "sector_assessment": "<XLK板块轮动信号，利好还是利空科技股>",
  "macro_rate_environment": "<10Y收益率趋势和Fed政策偏向对成长股估值的影响>",
  "earnings_risk_flag": true | false,
  "earnings_days_away": <整数 或 null>,
  "asset_analysis": [
    {{
      "asset": "{ticker}",
      "regime": "Trending-Up | Trending-Down | Post-Earnings Trending | Mean-Reverting | Pre-Earnings Choppy | Consolidation",
      "action": "long | short | no_trade",
      "bias_score": <0.0–1.0>,
      "entry_zone": "<价格区间，基于预计算锚点>",
      "profit_target": <数字 或 null>,
      "stop_loss": <数字 或 null>,
      "risk_reward_ratio": <数字 或 null>,
      "invalidation_condition": "<使该观点失效的具体市场信号>",
      "estimated_holding_weeks": <预计持仓周数 4–26>,

      "price_action_analysis": {{
        "trend_structure": "<EMA对齐情况，金叉/死叉状态，月线MACD方向>",
        "momentum_signals": "<MACD、RSI、Stochastic综合描述>",
        "volatility_context": "<ATR、BB%B、带宽>",
        "volume_obv": "<OBV趋势，量价配合还是背离>",
        "relative_strength_vs_qqq": "<跑赢或跑输QQQ，幅度，近6周趋势>"
      }},

      "structured_analysis": {{
        "market_hierarchy_alignment": "<L1–L5五层对齐情况：全部对齐/X层冲突>",
        "earnings_cycle_phase": "<财报周期阶段，距下次财报天数>",
        "sector_rotation_signal": "<XLK vs QQQ，对本股利好还是利空>",
        "regime_justification": "<判断当前制度的核心依据>"
      }},

      "intelligence_analysis": {{
        "earnings_estimate_trend": "<EPS预估修正方向（上调/下调/持平），对bias_score影响>",
        "analyst_consensus": "<综合评级、目标价上行空间、近期有无重大评级变化>",
        "valuation_context": "<Forward P/E是否合理、PEG是否偏高、营收增速是否加速/放缓>",
        "macro_catalyst": "<当前驱动该股中长线行情的核心宏观/行业逻辑>",
        "competitive_intelligence": "<AI周期机会、市场份额变化、监管风险、中国敞口等>"
      }},

      "justification": "<不超过300字的三维综合判断>"
    }}
  ]
}}
```

**硬性约束（违反任意一条 → 强制输出 no_trade）**：
- `earnings_days_away ≤ 5` → 财报二元风险，强制 no_trade
- QQQ 处于死叉（EMA50 < EMA200）→ 禁止做多
- `risk_reward_ratio < 2.0` → 盈亏比不足
- 止损距离 entry < 0.8×周ATR（{_v(w_atr14)}）= {round(0.8*atr, 2) if w_atr14 else 'N/A'} → 止损太紧

**信号质量过滤规则（全部适用）**：

*技术面规则*
- L1–L5 五层全部对齐同方向 → bias_score 允许 > 0.65
- 有 1 层冲突 → bias_score 上限 0.60
- 有 2 层及以上冲突 → bias_score 上限 0.50（建议 no_trade）
- 周线 MACD ({_v(w_macd)}) < 0 且 Trending 制度 → 禁止做多，评估做空
- 周线 RSI-7 ({_v(w_rsi7)}) > 75 → 做多 bias_score 上限 0.55
- 价格偏离周线 EMA-20 ({_v(w_ema20)}) 超过 5% → bias_score 上限 0.55
- ADX ({_v(w_adx)}) < 20 → 制度降级为 Consolidation，bias_score 上限 0.45
- OBV 与价格方向背离 → bias_score 降低 0.10
- VIX > 25 → 所有做多 bias_score -0.05；VIX > 35 → 默认 no_trade

*基本面规则*
- EPS 预估上调 ≥ 2%（`eps_growth_estimate` 提升）→ 做多 bias_score +0.05
- EPS 连续4季超预期（beat_count = 4/4）→ 做多 bias_score +0.05（高质量盈利）
- EPS 连续2季以上不及预期 → 做多 bias_score -0.10（盈利质量恶化）
- 季度营收加速（`revenue_acceleration` > 1pp）→ 做多 bias_score +0.05
- 季度营收减速（`revenue_acceleration` < -2pp）→ 做多 bias_score -0.08
- 分析师平均目标价上行空间 < 5% → 做多 bias_score 上限 0.55（没有上行催化剂）
- 分析师平均目标价上行空间 > 20% → 做多 bias_score 可额外 +0.03

*财报窗口规则*
- `earnings_days_away ≤ 5` → 财报二元风险，强制 no_trade（已在硬性约束中）
- `earnings_days_away` 6–15 天 → bias_score 上限 0.55，降低仓位（财报前不确定性高）
- `earnings_days_away` 16–30 天 → bias_score 上限 0.65（中等财报风险窗口）
- `bias_score < 0.50` → 一律 no_trade

**中长线持仓特别提示**：
- 止损设置在 2×周ATR 以外，给价格足够呼吸空间，避免被短期波动洗出
- 目标在 4×周ATR（即 RR=2.0），但若基本面持续改善可考虑跟踪止损以延长持仓
- 财报周期内个股可能大幅波动，但中长线视角下单次财报不是决策核心，除非出现趋势反转信号
- 每次分析频率建议每 2–4 周一次，避免过度频繁操作
"""
    return prompt.strip()


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
        max_tokens=8096,
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
    parser = argparse.ArgumentParser(
        description="纳斯达克科技股中长线分析脚本（持仓周期 1–6 个月）"
    )
    parser.add_argument(
        "--ticker",
        default="GOOGL",
        help="股票代码，例如 GOOGL / MSFT / NVDA / AAPL / META / AMZN（默认: GOOGL）",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="直接调用 API 获取分析结果（默认：只生成提示词文件）",
    )
    parser.add_argument(
        "--model",
        default=ANTHROPIC_MODEL,
        help=(
            f"指定调用的模型（需配合 --api）。"
            f"Claude: {', '.join(sorted(CLAUDE_MODELS))}；"
            f"DeepSeek: {', '.join(sorted(DEEPSEEK_MODELS))}。"
            f"默认: {ANTHROPIC_MODEL}"
        ),
    )
    args = parser.parse_args()
    ticker = args.ticker.upper()

    # ── 数据获取 ──
    daily, weekly, monthly = fetch_stock_data(ticker)

    print()
    macro = fetch_macro_data()

    print()
    intel = fetch_intelligence_data(ticker)

    print()
    peer_data = fetch_peer_data(ticker)

    # ── 构建提示词 ──
    print("\n正在构建分析提示词...")
    prompt = build_prompt_equity(
        ticker=ticker,
        daily=daily,
        weekly=weekly,
        monthly=monthly,
        macro=macro,
        intel=intel,
        perf_metrics=None,
        peer_data=peer_data,
    )

    # ── 保存提示词 ──
    output_path = f"{ticker.lower()}_prompt_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"提示词已保存到: {output_path}")

    if args.api:
        model = args.model
        if model in DEEPSEEK_MODELS:
            analysis = call_deepseek_api(prompt, model)
        else:
            analysis = call_claude_api(prompt)

        api_output_path = f"{ticker.lower()}_api_output.txt"
        with open(api_output_path, "w", encoding="utf-8") as f:
            f.write(analysis)
        print("\n" + "=" * 60)
        print(analysis)
        print("=" * 60)
        print(f"\nAPI 分析结果已保存到: {api_output_path}")
    else:
        print("\n" + "=" * 60)
        print(prompt)
        print("=" * 60)
        print(f"\n请将上方内容复制粘贴到 Claude.ai，或使用 --api 直接调用。")
        print(f"\n使用示例：")
        print(f"  python tech_stock_analysis.py --ticker {ticker} --api")
        print(f"  python tech_stock_analysis.py --ticker MSFT --api")
        print(f"  python tech_stock_analysis.py --ticker NVDA --api --model deepseek-reasoner")
        print(f"\n支持的股票代码：GOOGL  MSFT  NVDA  AAPL  META  AMZN  TSM  TSLA  NFLX  AMD")


if __name__ == "__main__":
    main()
