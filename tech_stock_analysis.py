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

try:
    from news_signal_bridge import fetch_news_signals, format_news_signals_section
    _NEWS_BRIDGE_AVAILABLE = True
except ImportError:
    _NEWS_BRIDGE_AVAILABLE = False


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
OPENAI_MODELS   = {"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o1", "o3-mini"}
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai-proxy.org/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", ANTHROPIC_API_KEY)

# 聚合平台每日调用限额（Claude + GPT 共享，DeepSeek R1 不受限）
_PLATFORM_USAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_usage.json")
PLATFORM_DAILY_LIMIT = int(os.getenv("PLATFORM_DAILY_LIMIT", "10"))

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _check_platform_quota() -> bool:
    """检查聚合平台今日调用次数是否已达上限（Claude + GPT 共享，跨脚本共用同一 api_usage.json）"""
    import json
    from datetime import date
    today = str(date.today())
    usage = {"date": today, "count": 0}
    if os.path.exists(_PLATFORM_USAGE_FILE):
        try:
            with open(_PLATFORM_USAGE_FILE) as f:
                data = json.load(f)
            if data.get("date") == today:
                usage = data
        except Exception:
            pass
    if usage["count"] >= PLATFORM_DAILY_LIMIT:
        print(f"  [配额] 今日聚合平台已调用 {usage['count']}/{PLATFORM_DAILY_LIMIT} 次，已达上限，跳过")
        return False
    usage["count"] += 1
    try:
        with open(_PLATFORM_USAGE_FILE, "w") as f:
            json.dump(usage, f)
    except Exception:
        pass
    print(f"  [配额] 今日聚合平台调用: {usage['count']}/{PLATFORM_DAILY_LIMIT}")
    return True


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


def describe_macd(ind: dict, tf_label: str = "周线") -> str:
    """将 MACD 指标转为文本形态描述，替代原始数字序列"""
    macd = ind['macd'].dropna()
    sig  = ind['macd_sig'].dropna()
    hist = ind['macd_hist'].dropna()
    if len(macd) < 2 or len(sig) < 2:
        return f"{tf_label}MACD: N/A"
    m, s   = float(macd.iloc[-1]), float(sig.iloc[-1])
    m_p, s_p = float(macd.iloc[-2]), float(sig.iloc[-2])
    if   m > s and m_p <= s_p: cross = "Newly Golden Cross"
    elif m < s and m_p >= s_p: cross = "Newly Death Cross"
    elif m > s:                cross = "Golden Cross ongoing"
    else:                      cross = "Death Cross ongoing"
    h_vals = hist.iloc[-3:].tolist() if len(hist) >= 3 else []
    if len(h_vals) == 3:
        if   h_vals[2] > h_vals[1] > h_vals[0]: hist_t = "histogram expanding (momentum strengthening)"
        elif h_vals[2] < h_vals[1] < h_vals[0]: hist_t = "histogram contracting (momentum weakening)"
        elif h_vals[2] > 0:                      hist_t = f"histogram positive ({h_vals[2]:.2f})"
        else:                                    hist_t = f"histogram negative ({h_vals[2]:.2f})"
    else:
        hist_t = ""
    return f"{tf_label} MACD {cross} (MACD={m:.2f}, Signal={s:.2f}); {hist_t}"


def describe_rsi(ind: dict, tf_label: str = "周线") -> str:
    """将 RSI 指标转为文本形态描述"""
    rsi14 = ind['rsi14'].dropna()
    rsi7  = ind['rsi7'].dropna()
    if rsi14.empty:
        return f"{tf_label} RSI: N/A"
    r = float(rsi14.iloc[-1])
    if   r > 70: level = f"overbought ({r:.0f})"
    elif r < 30: level = f"oversold ({r:.0f})"
    elif r > 55: level = f"firm ({r:.0f})"
    elif r < 45: level = f"weak ({r:.0f})"
    else:        level = f"neutral ({r:.0f})"
    trend = ""
    if len(rsi14) >= 4:
        r_p = float(rsi14.iloc[-4])
        if   r > 50 and r_p < 50: trend = ", crossed 50 midline ↑"
        elif r < 50 and r_p > 50: trend = ", broke below 50 midline ↓"
        elif r > r_p + 8:         trend = ", rapid recovery"
        elif r < r_p - 8:         trend = ", rapid decline"
    r7_note = ""
    if not rsi7.empty:
        r7 = float(rsi7.iloc[-1])
        if   r7 > 80: r7_note = f"; RSI-7={r7:.0f} extremely overbought (chasing-high risk)"
        elif r7 < 20: r7_note = f"; RSI-7={r7:.0f} extremely oversold (bounce momentum building)"
    return f"{tf_label} RSI-14={level}{trend}{r7_note}"


def describe_bb(ind: dict, tf_label: str = "周线") -> str:
    """将布林带指标转为文本形态描述"""
    pctb_s = ind['bb_pct_b'].dropna()
    bw_s   = ind['bb_bw'].dropna()
    if pctb_s.empty:
        return f"{tf_label}BB: N/A"
    pctb = float(pctb_s.iloc[-1])
    if   pctb > 1.0: pos = "above upper band (strong, watch for pullback)"
    elif pctb > 0.7: pos = "near upper band (bullish bias)"
    elif pctb > 0.4: pos = "above midline (neutral bullish)"
    elif pctb > 0.1: pos = "below midline (neutral bearish)"
    elif pctb >= 0:  pos = "near lower band (weak)"
    else:            pos = "below lower band (oversold — watch for bounce)"
    bw_note = ""
    if len(bw_s) >= 26:
        history = bw_s.tail(52).tolist() if len(bw_s) >= 52 else bw_s.tolist()
        curr_bw = float(bw_s.iloc[-1])
        pct = sum(1 for x in history if x <= curr_bw) / len(history) * 100
        if   pct < 20: bw_note = f"; bandwidth at {pct:.0f}th percentile (highly compressed — breakout setup)"
        elif pct > 80: bw_note = f"; bandwidth at {pct:.0f}th percentile (highly expanded — strong trend)"
    return f"{tf_label} BB %B={pctb:.2f}, {pos}{bw_note}"


def describe_price_vs_ema(close: pd.Series, ind: dict, tf_label: str = "周线") -> str:
    """价格相对各均线位置的文本描述"""
    if close.empty:
        return f"{tf_label}价格结构: N/A"
    price = float(close.iloc[-1])
    parts = []
    for period, key in [(20, 'ema20'), (50, 'ema50'), (200, 'ema200')]:
        ema_s = ind[key].dropna()
        if not ema_s.empty:
            v   = float(ema_s.iloc[-1])
            pct = (price - v) / v * 100
            parts.append(f"EMA{period}({'↑' if price > v else '↓'}{abs(pct):.1f}%)")
    return (f"{tf_label} price={price:.2f}; " + ", ".join(parts)) if parts else f"{tf_label} N/A"


# ─────────────────────────────────────────────
# 数据获取
# ─────────────────────────────────────────────

def _make_session() -> curl_requests.Session:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    session = curl_requests.Session(impersonate="chrome", verify=False, proxy=proxy)
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

            # yfinance 的 forwardPE 使用「下一财年」EPS，与市场惯例（当前财年）不符。
            # 用 earnings_estimate 里的 0y（当前财年）EPS 重算，与 Bloomberg/FactSet 对齐。
            curr_price = info.get("currentPrice") or info.get("regularMarketPrice")
            curr_eps   = result.get("eps_estimate_current_y")
            if curr_price and curr_eps and float(curr_eps) > 0:
                result["forward_pe"] = round(float(curr_price) / float(curr_eps), 2)
                print(f"  Forward P/E 重算: {curr_price:.2f} / {curr_eps:.3f} = {result['forward_pe']}")

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
# 新闻情绪获取（实盘用，回测不可用）
# ─────────────────────────────────────────────

def fetch_recent_news(ticker: str, max_items: int = 8) -> list[dict]:
    """
    获取近期新闻标题（yfinance .news），用于注入实盘分析提示词。

    注意：yfinance.news 返回的是当前最新新闻，仅适合实盘分析。
    回测中禁止使用（会造成未来信息泄漏）。

    返回列表，每项含：
      - title: 标题
      - publisher: 来源
      - link: URL
      - published: 发布时间（Unix 时间戳转 ISO 字符串）
    """
    try:
        t = yf.Ticker(ticker, session=_make_session())
        raw = t.news
        if not raw:
            print(f"  [新闻] {ticker}: 未获取到新闻")
            return []

        items = []
        for n in raw[:max_items]:
            # yfinance news 结构：{"title", "publisher", "link", "providerPublishTime", ...}
            published_ts = n.get("providerPublishTime", 0)
            try:
                published_str = pd.Timestamp(published_ts, unit="s").strftime("%Y-%m-%d %H:%M")
            except Exception:
                published_str = "unknown"
            items.append({
                "title":     n.get("title", ""),
                "publisher": n.get("publisher", ""),
                "link":      n.get("link", ""),
                "published": published_str,
            })
        print(f"  [新闻] {ticker}: 获取 {len(items)} 条近期新闻")
        return items
    except Exception as e:
        print(f"  [新闻] {ticker} 获取失败: {e}")
        return []


def _format_news_section(news_items: list[dict]) -> str:
    """将新闻列表格式化为 Markdown，注入到提示词中。"""
    if not news_items:
        return ""
    lines = [
        "",
        "## 近期新闻与市场情绪（实时，仅作背景参考）",
        "",
        "以下为近期主要新闻标题，供 LLM 判断市场情绪叙事。",
        "**注意：这些是已公开的新闻事件，不代表未来走势。**",
        "",
    ]
    for i, n in enumerate(news_items, 1):
        lines.append(f"{i}. **[{n['published']}]** {n['title']}  _(来源: {n['publisher']})_")
    lines.append("")
    lines.append("> **情绪评估提示**: 请根据上述新闻判断当前市场叙事是否支持或对冲你的技术信号。"
                 "若新闻与技术信号方向一致，可小幅提升 bias_score（≤+0.05）；"
                 "若重大负面事件（监管/诉讼/业绩预警），须降低 bias_score 或输出 no_trade。")
    lines.append("")
    return "\n".join(lines)


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
            "signal":        "Outperforming QQQ" if rs_last > 0 else "Underperforming QQQ",
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
        return f"{'↑' if chg > 0 else '↓'} {abs(chg):.1f}% (8W)"

    def _ema_cross_status(df, fast=20, slow=50):
        if df.empty or 'Close' not in df.columns:
            return "N/A"
        close = df['Close'].squeeze().dropna()
        if len(close) < slow:
            return "N/A"
        ema_f = calc_ema(close, fast).iloc[-1]
        ema_s = calc_ema(close, slow).iloc[-1]
        return "Golden Cross (bullish)" if ema_f > ema_s else "Death Cross (bearish)"

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
        result["sector_rotation"] = "Tech Sector Leading" if xlk_ret > qqq_ret else "Tech Sector Lagging"
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
            "Extreme Fear (>35)" if v > 35 else
            ("Fear (>25)"        if v > 25 else
            ("High Vol (>20)"    if v > 20 else
            ("Neutral (>15)"     if v > 15 else "Low Vol / Risk-On (<15)")))
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

    # Analyst price target upside
    pt_upside_str = ""
    if intel.get("analyst_target_mean") and current_price:
        upside = (intel["analyst_target_mean"] - current_price) / current_price * 100
        pt_upside_str = f"  (**{upside:+.1f}% upside to current price**)"

    # Earnings risk alert
    days = intel.get("earnings_days_away")
    if days is not None and days >= 0:
        if days <= 5:
            earnings_alert = f"\n> **⚠️ EARNINGS RISK: {days} days to earnings ({_v('earnings_date')}) — force no_trade**"
        elif days <= 15:
            earnings_alert = f"\n> **NOTE: {days} days to earnings — reduce position size**"
        else:
            earnings_alert = f"\n> Next earnings: ~{days} days ({_v('earnings_date')})"
    else:
        earnings_alert = ""

    # Forward P/E interpretation
    fpe = intel.get("forward_pe")
    fpe_interp = "N/A"
    if isinstance(fpe, (int, float)):
        fpe_interp = "Elevated, valuation pressure" if fpe > 35 else ("Slightly elevated" if fpe > 25 else "Reasonable")

    # PEG interpretation
    peg = intel.get("peg_ratio")
    peg_interp = "N/A"
    if isinstance(peg, (int, float)):
        peg_interp = "Expensive (>2)" if peg > 2 else ("Fair (1–2)" if peg > 1 else "Undervalued (<1)")

    # Beta interpretation
    beta = intel.get("beta")
    beta_interp = "N/A"
    if isinstance(beta, (int, float)):
        beta_interp = "High volatility — wider stops needed" if beta > 1.3 else ("Moderate volatility" if beta > 0.8 else "Low volatility")

    # Short ratio interpretation
    sr = intel.get("short_ratio")
    sr_interp = "N/A"
    if isinstance(sr, (int, float)):
        sr_interp = "Short-heavy — potential squeeze" if sr > 5 else ("Moderate short interest" if sr > 2 else "Low short interest")

    section = f"""
---

## III. Intelligence Analysis

### 3.1 Earnings & Estimates
{earnings_alert}

| Metric | Value |
|--------|-------|
| Next Earnings Date | {_v('earnings_date')} |
| Days to Earnings | {_v('earnings_days_away')} |
| Current Q EPS Estimate | ${_v('eps_estimate_current_q')} |
| Next Q EPS Estimate | ${_v('eps_estimate_next_q')} |
| Full Year EPS Estimate | ${_v('eps_estimate_current_y')} |
| Full Year EPS Growth | {_v('eps_growth_estimate')}% |
| Full Year Revenue Estimate | ${_v('revenue_estimate_current_y')}B |
| Revenue Growth Estimate | {_v('revenue_growth_estimate')}% |

### 3.2 Analyst Consensus

| Metric | Value |
|--------|-------|
| Consensus Rating | {_v('analyst_recommendation')} |
| Mean Price Target | ${_v('analyst_target_mean')}{pt_upside_str} |
| PT Range | ${_v('analyst_target_low')} – ${_v('analyst_target_high')} |
| Strong Buy / Buy / Hold / Sell | {_v('analyst_strong_buy')} / {_v('analyst_buy')} / {_v('analyst_hold')} / {_v('analyst_sell')} |

### 3.3 Valuation & Financial Quality

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Market Cap | {_v('market_cap')} | — |
| Forward P/E | {_v('forward_pe')} | {fpe_interp} |
| PEG Ratio | {_v('peg_ratio')} | {peg_interp} |
| Price/Book | {_v('price_to_book')} | — |
| Revenue Growth (YoY) | {_v('revenue_growth')} | — |
| Earnings Growth (YoY) | {_v('earnings_growth')} | — |
| Gross Margin | {_v('gross_margins')} | — |
| Operating Margin | {_v('operating_margins')} | — |
| Net Margin | {_v('profit_margins')} | — |
| ROE | {_v('return_on_equity')} | — |
| Free Cash Flow | {_v('free_cashflow')} | — |
| Debt/Equity | {_v('debt_to_equity')} | — |
| Beta | {_v('beta')} | {beta_interp} |
| Short Ratio | {_v('short_ratio')} days | {sr_interp} |
| Institutional Ownership | {_v('institutional_ownership')} | — |
| 52W High | ${_v('52w_high')} | — |
| 52W Low | ${_v('52w_low')} | — |

### 3.4 Earnings Surprise History (Last 4 Quarters)
"""

    # Earnings surprise history
    surprises = intel.get("eps_surprise_history", [])
    beat_count = intel.get("eps_beat_count")
    if surprises:
        surprise_cells = " | ".join(
            f"{'✅' if s > 0 else '❌'} {s:+.1f}%" for s in surprises
        )
        beat_label = f"{beat_count}/{len(surprises)} beats" if beat_count is not None else "N/A"
        surprise_trend = (
            "Consistent beats — high earnings quality" if beat_count == len(surprises)
            else ("Majority beats" if beat_count and beat_count >= len(surprises) * 0.75
            else ("Mostly misses — watch for earnings pressure" if beat_count is not None and beat_count < len(surprises) * 0.5
            else "Mixed beat/miss history"))
        )
        section += f"""
| Metric | Q-3 | Q-2 | Q-1 | Latest Q | Summary |
|--------|-----|-----|-----|----------|---------|
| EPS Surprise% | {surprise_cells} | {beat_label} |

> **Interpretation**: {surprise_trend}
"""
    else:
        section += "\n> Earnings surprise data unavailable\n"

    # Quarterly revenue trend
    rev_trend = intel.get("quarterly_revenue_trend", [])
    rev_accel = intel.get("revenue_acceleration")
    section += "\n### 3.5 Quarterly Revenue Trend (Last 5 Quarters, $B)\n"
    if rev_trend:
        rev_str = " → ".join(f"${v}B" for v in rev_trend)
        accel_str = ""
        if rev_accel is not None:
            if rev_accel > 1:
                accel_str = f"  ✅ **Revenue accelerating** (QoQ growth up {rev_accel:+.1f}pp)"
            elif rev_accel < -1:
                accel_str = f"  ⚠️ **Revenue decelerating** (QoQ growth down {rev_accel:.1f}pp)"
            else:
                accel_str = f"  Revenue growth stable ({rev_accel:+.1f}pp)"
        section += f"\n{rev_str}{accel_str}\n"
    else:
        section += "\n> Quarterly revenue data unavailable\n"

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
    news_items: list | None = None,
    news_context: dict | None = None,
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

    # OBV trend (weekly, last 6 bars)
    obv_s = w_ind['obv'].dropna().tail(6).tolist()
    obv_trend = "Rising" if len(obv_s) >= 2 and obv_s[-1] > obv_s[0] else "Falling"

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

    # ── 形态描述（替代原始数字序列，LLM 对文本形态感知力远强于浮点数组）──
    w_price_desc    = describe_price_vs_ema(close_w, w_ind, "Weekly")
    w_macd_desc     = describe_macd(w_ind, "Weekly")
    w_rsi_desc      = describe_rsi(w_ind, "Weekly")
    w_bb_desc       = describe_bb(w_ind, "Weekly")
    m_macd_desc     = describe_macd(m_ind, "Monthly")
    m_rsi_desc      = describe_rsi(m_ind, "Monthly")

    # 预计算入场锚点（基于周线 ATR，中长线适用）
    atr = w_atr14 or 1.0
    long_entry_lo  = round(current_price - 0.5 * atr, 2)
    long_entry_hi  = round(current_price + 0.5 * atr, 2)
    long_stop      = round(current_price - 2.5 * atr, 2)   # 2.0→2.5×ATR 中长线需要更宽止损
    long_target    = round(current_price + 5.0 * atr, 2)   # 4.0→5.0×ATR 保持 RR=2.0
    short_entry_lo = round(current_price - 0.5 * atr, 2)
    short_entry_hi = round(current_price + 0.5 * atr, 2)
    short_stop     = round(current_price + 2.5 * atr, 2)
    short_target   = round(current_price - 5.0 * atr, 2)

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
            signal = "Leading" if vs and vs > 0 else "Lagging"
            peer_rows.append(f"| {peer} | ${price} | {ret_str} | {vs_str} | {signal} |")
        peer_table = "\n".join(peer_rows)
        peer_section = f"""
---

## IV. Peer Comparison (Last 8 Weeks)

| Peer | Current Price | 8W Return | vs QQQ Excess | Signal |
|------|--------------|-----------|---------------|--------|
{peer_table}

> **Interpretation**: If peers broadly underperform QQQ, the sub-sector (ads/cloud/AI) faces structural headwinds. If {ticker} underperforms peers, there is stock-level relative weakness — be cautious.
"""

    # 股票专属行业背景（静态 prompt engineering，给模型提供行业分析框架）
    _INDUSTRY_CONTEXT = {
        "GOOGL": """
### 行业特有分析维度（GOOGL / Alphabet）— 更新于 2026Q1

请在 `competitive_intelligence` 字段中，额外关注以下 GOOGL 专属因素：

1. **AI Overviews 货币化验证**：AI 概览功能上线后搜索广告 CPM 是否出现结构性下滑？Q1 2026 广告收入增速是关键验证节点，市场已将此定价为最大不确定性。
2. **Gemini 2.0/2.5 vs 竞品**：Gemini Ultra/Flash 与 GPT-4.5、Claude 3.7、Llama 4 的能力对比，是否在开发者 API 使用量上重获份额？
3. **Google Cloud 增速**：是否维持 >28% YoY？TPU v6（Trillium）在推理侧对外部 GPU 的替代比例，Vertex AI 大客户拓展。
4. **DOJ 搜索垄断救济执行**：2025 年判决后的强制救济措施（默认搜索协议禁止/分拆预期），是否出现实质性落地影响估值。
5. **Waymo 商业化**：自动驾驶业务在旧金山/凤凰城的收入规模，是否接近独立融资或分拆节点，构成估值重估催化剂。
""",
        "META": """
### 行业特有分析维度（Meta Platforms）— 更新于 2026Q1

请在 `competitive_intelligence` 字段中，额外关注以下 Meta 专属因素：

1. **关税冲击下的中国广告主敞口**：Temu/Shein/速卖通等中国广告主贡献约 10% 营收，美中关税升级（2025年加至145%）是否导致这部分预算骤降？Q1 2026 广告收入指引是关键风险点。
2. **Llama 4 开源 AI 战略**：Llama 4 Scout/Maverick/Behemoth 发布后开发者生态扩张，开源策略是否帮助 Meta AI 助手实现 MAU 增长，间接强化广告平台护城河。
3. **WhatsApp 商业消息货币化**：B2C 消息服务在印度/巴西/东南亚的 ARPU 提升节奏，是否成长为新的高利润率收入线。
4. **CapEx 与 FCF 压力**：2025 全年 CapEx 指引 $600-650 亿，AI 数据中心投入是否进一步上调？FCF yield 压缩对估值倍数的影响。
5. **AI 眼镜 + AR 生态**：Ray-Ban Meta 眼镜销量（据报超百万），Orion AR 原型商业化时间表，是否构成下一个硬件增长故事。
""",
        "NVDA": """
### 行业特有分析维度（NVIDIA）— 更新于 2026Q1

请在 `competitive_intelligence` 字段中，额外关注以下 NVDA 专属因素：

1. **GB200 NVL 部署进展**：超大规模客户（微软/谷歌/Meta/亚马逊）GB200 机架交付节奏，NVLink 互联扩容情况。
2. **Rubin 架构时间表**：RX100（Rubin）预计 2026 量产，是否已有客户预购和产能规划披露，对 GB200 需求的替代/续接效应。
3. **自研芯片竞争**：AMD MI350/MI400 份额是否扩大；谷歌 Trillium TPUv6、亚马逊 Trainium3 在推理侧的替代渗透率趋势。
4. **出口管制动态**：对华/对中东 AI 芯片出口许可（H20/B20 后续政策），中国收入占比及替代预案（东南亚/中东数据中心绕路模式）。
5. **AI 推理需求结构**：训练算力向推理算力迁移比例，推理需求对 GPU 利用率的持续支撑力度，是否出现训练侧 CapEx 放缓迹象。
""",
        "MSFT": """
### 行业特有分析维度（Microsoft）— 更新于 2026Q1

请在 `competitive_intelligence` 字段中，额外关注以下 MSFT 专属因素：

1. **Azure AI 增速加速验证**：Azure 是否突破 35% YoY？AI 工作负载（Azure OpenAI Service、Copilot Studio）在 Azure 增速中的贡献比例，是否出现"AI 飞轮"效应（AI 带动云消耗增长）。
2. **Microsoft 365 Copilot 企业渗透**：付费 Copilot 席位是否超过 1000 万？企业续约率是否验证 ROI，ARPU 提升幅度（每用户 +$30/月）对营收的增量测算。
3. **OpenAI 关系重构**：GPT-4.5/5 发布后 OpenAI 独立商业化能力增强，与 Azure 绑定协议是否出现松动？Copilot 与 ChatGPT Enterprise 的企业客户竞争。
4. **GitHub Copilot 企业版**：代码 AI 助手在 Fortune 500 的渗透率，是否已成为开发者标配工具，构成新增长飞轮。
5. **Activision 游戏整合**：Xbox Game Pass 订阅数与 Activision 内容协同，Call of Duty 等 IP 对游戏收入的实际贡献是否达到收购预期。
""",
        "AMZN": """
### 行业特有分析维度（Amazon）— 更新于 2026Q1

请在 `competitive_intelligence` 字段中，额外关注以下 AMZN 专属因素：

1. **AWS AI 基础设施领先验证**：AWS 是否维持 >17% YoY？Amazon Nova 系列模型在 Bedrock 上的调用量，Trainium3/Inferentia 自研芯片是否降低 GPU 采购成本并提升 AI 云利润率。
2. **关税对零售业务的双向冲击**：美中关税上调对第三方中国卖家（占 GMV 约 30%）的冲击，同时跨境包裹豁免取消是否逼退 Temu/Shein 竞争，形成净正面效应还是负面效应？
3. **广告业务突破 $60B**：广告已成为第三大业务，CPM 定价能力是否持续强于 GOOGL/META？Sponsored Product 在 AI 推荐场景下的转化率提升。
4. **Project Kuiper**：低轨卫星互联网计划是否完成关键发射批次，商业服务是否启动，对 Starlink 的竞争时间表。
5. **Anthropic 深度整合**：Claude 3.7 在 AWS 客户中的企业 AI Agent 采用率，是否帮助 AWS 在 AI 云场景下追回被 Azure OpenAI 抢占的份额。
""",
        "AAPL": """
### 行业特有分析维度（Apple）— 更新于 2026Q1

请在 `competitive_intelligence` 字段中，额外关注以下 AAPL 专属因素：

1. **关税最大风险暴露**：约 90% iPhone 仍在中国组装，美中关税145%如正式覆盖电子产品，iPhone 售价需上涨 $200-300 方能维持利润率。印度/越南产能转移（目前约 15%）短期内无法完全对冲，是当前估值最大尾部风险。
2. **Apple Intelligence 换机催化效果**：iPhone 16 系列搭载 Apple Intelligence，换机周期是否被 AI 功能激活？中国市场 Apple Intelligence 延迟上线（需与百度/腾讯合作）是否造成中国区销量落后。
3. **服务业务高利润率持续性**：服务收入（App Store、Apple TV+、iCloud+）是否维持 >12% YoY，对整体毛利率（服务毛利 ~75% vs 硬件 ~36%）的支撑是否抵消关税冲击。
4. **Vision Pro 战略调整**：第一代 Vision Pro 销量远低于预期，是否宣布更低价格段产品（$1500-2000）？头显业务路线图是否仍构成下一代增长故事。
5. **DOJ / 欧盟 App Store 强制开放**：美国 Epic 判决执行、欧盟 DMA 合规落地，第三方支付渗透对 App Store 高利润率的实质冲击幅度。
""",
        # ─── 大宗商品 / ETF 专属上下文 ───
        "SLV": """
### 资产特有分析维度（白银 ETF - SLV）

请在 `competitive_intelligence` 字段中，额外关注以下白银专属因素：

1. **金银比（Gold/Silver Ratio）**：当前金银比是否处于历史高位（>80）？历史上高金银比后白银通常出现追涨修复行情。
2. **工业需求**：光伏（太阳能电池板）、电动汽车和半导体对白银的工业需求增量，是否形成价格支撑。
3. **美元指数（DXY）**：DXY 上涨对白银的压制作用（负相关），DXY 下跌时白银弹性通常大于黄金。
4. **实际利率**：美国实际利率（TIPS 收益率）下行是白银上涨的关键宏观因子。
5. **ETF 持仓变化**：SLV 基金的白银实物持仓吨数是否增加（资金净流入信号）。
""",
        "COPX": """
### 资产特有分析维度（铜矿 ETF - COPX）

请在 `competitive_intelligence` 字段中，额外关注以下铜矿专属因素：

1. **全球制造业 PMI**：铜被称为"博士铜"，是全球经济活动的领先指标。PMI >50 且上行时，铜价通常有支撑。
2. **能源转型需求**：电动汽车（每辆用铜约 80kg）、电网升级、风电/光伏扩产，是铜的长期结构性需求来源。
3. **中国需求**：中国消耗全球约 50% 的铜，中国地产和基建政策对铜价影响最大。
4. **供给侧**：智利/秘鲁矿区罢工或政策风险，以及老矿品位下降导致的供给收紧。
5. **库存水位**：LME + COMEX + SHFE 交易所铜库存水平，低库存叠加需求回升是强势信号。
""",
        "REMX": """
### 资产特有分析维度（稀土/战略金属 ETF - REMX，含钨敞口）

请在 `competitive_intelligence` 字段中，额外关注以下稀土/战略金属专属因素：

1. **中国稀土出口管制**：中国控制全球约 60% 稀土产量和 85% 加工能力。任何出口限制或配额削减都是价格强催化剂。
2. **钨供给**：中国控制全球约 80% 钨矿供给，美欧半导体/国防工业对钨的战略储备需求上升。
3. **电动汽车永磁体需求**：永磁电机（NdFeB）对镨、钕的需求，是稀土的最大增量来源。
4. **去中国化供应链**：美欧澳战略矿产投资政策（IRA 法案等），Lynas/MP Materials 等非中国产能扩张进度。
5. **国防采购**：F-35、导弹制导、雷达系统对稀土永磁体的国防需求，是非周期性支撑因子。
""",
        "USO": """
### 资产特有分析维度（原油 ETF - USO）

请在 `competitive_intelligence` 字段中，额外关注以下原油专属因素：

1. **OPEC+ 减产执行率**：沙特/俄罗斯等主要产油国是否遵守减产协议，以及是否有增产/减产声明变化。
2. **美国原油库存（EIA 周报）**：每周三发布的 EIA 原油库存变化，是短期价格的最强催化剂。
3. **全球需求前景**：IEA/EIA 的需求预测修正方向，中国航煤/汽油消费恢复进度。
4. **地缘政治风险溢价**：中东冲突升级（霍尔木兹海峡风险）、俄乌战争对俄罗斯出口的影响。
5. **美元与页岩油成本**：美国页岩油盈亏平衡价格（约 $55-65/桶），DXY 强势对油价的压制。
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
            warns.append("⚠️ Win rate below 40%: bias_score threshold raised to ≥0.65")
        if cl >= 2:
            warns.append(f"⚠️ {cl} consecutive losses: require bias_score ≥0.75 to enter")
        warn_str = "\n".join(warns) if warns else "Current performance normal — maintain standard decision process."
        perf_section = f"""
---

## Recent Backtest Performance Feedback

| Metric | Value |
|--------|-------|
| Win Rate | {perf_metrics.get('win_rate', 'N/A')} |
| Avg Win | {perf_metrics.get('avg_win', 'N/A')} |
| Avg Loss | {perf_metrics.get('avg_loss', 'N/A')} |
| Profit Factor | {perf_metrics.get('profit_factor', 'N/A')} |
| Total Return | {perf_metrics.get('total_return', 'N/A')} |
| Consecutive Losses | {cl} |

{warn_str}
"""

    def _v(val, suffix=""):
        return f"{val}{suffix}" if val is not None else "N/A"

    ema_cross_status = (
        "Golden Cross (EMA50 > EMA200, bullish structure)" if w_ema50 and w_ema200 and w_ema50 > w_ema200
        else "Death Cross (EMA50 < EMA200, bearish structure)" if w_ema50 and w_ema200
        else "N/A"
    )
    ema20_bias = (
        "Price above EMA20, short-term bullish" if w_ema20 and current_price > w_ema20
        else "Price below EMA20, short-term bearish" if w_ema20
        else "N/A"
    )
    ema50_bias = (
        "Price above EMA50, medium-term bullish" if w_ema50 and current_price > w_ema50
        else "Price below EMA50, medium-term bearish" if w_ema50
        else "N/A"
    )
    ema200_bias = (
        "Price above EMA200, long-term bull structure" if w_ema200 and current_price > w_ema200
        else "Price below EMA200, long-term bear structure" if w_ema200
        else "N/A"
    )

    today_str = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""# {ticker} — Weekly Analysis Request

**Analysis Date**: {today_str}
**Target Hold Period**: 4–26 weeks (medium-to-long term swing)
**Primary Timeframe**: Weekly (W1) + Monthly (M1) background; Daily for entry timing only
{perf_section}
---

## I. Price Action Analysis

### 1.1 Price Overview

- **Current Price**: ${current_price}
- **52-Week High**: ${high_52w}  |  Distance from high: {pct_from_high:+.1f}%
- **52-Week Low**: ${low_52w}   |  Distance from low: {pct_from_low:+.1f}%

### 1.2 Weekly Key Indicator Snapshot

| Indicator | Current | Signal |
|-----------|---------|--------|
| EMA-20 (W) | {_v(w_ema20)} | {ema20_bias} |
| EMA-50 (W) | {_v(w_ema50)} | {ema50_bias} |
| EMA-200 (W) | {_v(w_ema200)} | {ema200_bias} |
| EMA Cross (50/200 W) | {ema_cross_status} | Long-term trend direction |
| MACD (W) | {_v(w_macd)} | {'Positive — bullish momentum' if w_macd and w_macd > 0 else 'Negative — bearish momentum'} |
| RSI-14 (W) | {_v(w_rsi14)} | {'Overbought >70, caution on longs' if w_rsi14 and w_rsi14 > 70 else ('Oversold <30, watch for bounce' if w_rsi14 and w_rsi14 < 30 else 'Neutral 30–70')} |
| RSI-7 (W) | {_v(w_rsi7)} | {'Extremely overbought >80' if w_rsi7 and w_rsi7 > 80 else ('Extremely oversold <20' if w_rsi7 and w_rsi7 < 20 else 'Normal range')} |
| ADX (W) | {_v(w_adx)} | {'Strong trend >25' if w_adx and w_adx > 25 else ('Weak/choppy <20' if w_adx and w_adx < 20 else 'Trend forming 20–25')} |
| +DI / -DI | {_v(w_pdi)} / {_v(w_mdi)} | {'+DI > -DI: bulls in control' if w_pdi and w_mdi and w_pdi > w_mdi else '-DI > +DI: bears in control'} |
| Stochastic %K / %D | {_v(stoch_k)} / {_v(stoch_d)} | {'Overbought death cross — caution' if stoch_k and stoch_d and stoch_k > 80 and stoch_k < stoch_d else ('Oversold golden cross — watch long' if stoch_k and stoch_d and stoch_k < 20 and stoch_k > stoch_d else 'Neutral zone')} |
| BB %B (W) | {_v(w_bb_pctb)} | {'Above upper band >1' if w_bb_pctb and w_bb_pctb > 1 else ('Below lower band <0' if w_bb_pctb and w_bb_pctb < 0 else 'Inside Bollinger Bands')} |
| BB Bandwidth % | {_v(w_bb_bw)} | {'Expanding — trend accelerating' if w_bb_bw and w_bb_bw > 10 else 'Contracting — squeeze setup'} |
| ROC-20 (W) | {_v(w_roc20, '%')} | {'Positive momentum' if w_roc20 and w_roc20 > 0 else 'Negative momentum'} |
| OBV Trend (Last 6W) | {obv_trend} | {'Volume/price aligned — institutional accumulation' if obv_trend == 'Rising' else 'Volume/price divergence — institutional distribution'} |
| Relative Strength vs QQQ (20W) | {_v(rs_data.get('rs_vs_qqq_pct'), '%')} | {rs_data.get('signal', 'N/A')} |

### 1.3 Monthly Long-Term Background

| Indicator | Current | Interpretation |
|-----------|---------|----------------|
| EMA-20 (M) | {_v(m_ema20)} | {'Price above monthly EMA20 — long-term uptrend' if m_ema20 and current_price > m_ema20 else 'Price below monthly EMA20 — long-term downtrend'} |
| MACD (M) | {_v(m_macd)} | {'Monthly bullish momentum' if m_macd and m_macd > 0 else 'Monthly bearish momentum'} |
| RSI-14 (M) | {_v(m_rsi14)} | {'Monthly overbought — watch for long-term top' if m_rsi14 and m_rsi14 > 70 else ('Monthly oversold — watch for long-term bottom' if m_rsi14 and m_rsi14 < 30 else 'Monthly neutral')} |

### 1.4 Momentum Patterns (Pre-processed descriptions)

**Weekly Patterns**:
- {w_price_desc}
- {w_macd_desc}
- {w_rsi_desc}
- {w_bb_desc}

**Monthly Background (weight ≤ ±0.05)**:
- {m_macd_desc}
- {m_rsi_desc}

---

## II. Structured Analysis

### 2.1 Five-Level Market Hierarchy

| Level | Dimension | Current State | Direction |
|-------|-----------|---------------|-----------|
| L1 Macro | Fed Policy + 10Y Yield | {_v(ms.get('tnx_last'), '%')}  Trend: {ms.get('tnx_trend', 'N/A')} | Yield {'rising → growth compression' if ms.get('tnx_trend','').startswith('↑') else 'falling → growth tailwind'} |
| L2 Index | QQQ Trend | {_v(ms.get('qqq_last'))}  Trend: {ms.get('qqq_trend', 'N/A')}  EMA: {ms.get('qqq_cross', 'N/A')} | {'Bullish' if ms.get('qqq_cross','').startswith('Golden') else 'Bearish'} |
| L3 Sector | XLK vs QQQ | XLK excess: {_v(ms.get('xlk_vs_qqq_pct'), '%')}  {ms.get('sector_rotation', 'N/A')} | {'Favorable' if ms.get('xlk_vs_qqq_pct') and ms['xlk_vs_qqq_pct'] > 0 else 'Unfavorable'} |
| L4 Stock | {ticker} vs QQQ (20W) | Excess return: {_v(rs_data.get('rs_vs_qqq_pct'), '%')}  {rs_data.get('signal', 'N/A')} | {'Favorable' if rs_data.get('rs_vs_qqq_pct') and rs_data['rs_vs_qqq_pct'] > 0 else 'Unfavorable'} |
| L5 Technical | Entry Timing | See indicators above | Model to assess |

### 2.2 Benchmark Trend (Last 8 Weeks)

QQQ Weekly Close Series: {ms.get('qqq_series', [])}
SPY Trend: {ms.get('spy_trend', 'N/A')}  |  QQQ Trend: {ms.get('qqq_trend', 'N/A')}

### 2.3 Market Sentiment

VIX: {_v(ms.get('vix_last'))}  |  Regime: {ms.get('vix_regime', 'N/A')}  |  Trend: {ms.get('vix_trend', 'N/A')}
DXY: {_v(ms.get('dxy_last'))}  |  Trend: {ms.get('dxy_trend', 'N/A')}

{intel_section}
{peer_section}
{industry_section}
{format_news_signals_section(news_context) if news_context else (_format_news_section(news_items) if news_items else "")}
---

## Precomputed Entry Anchors (Weekly ATR-14 = {_v(w_atr14)})

> Stops at 2.5×wATR; target at 5×wATR (RR=2.0). Entry within ±0.5×wATR of current price.

| Direction | Entry Zone | Stop Loss (2.5×wATR) | Profit Target (5×wATR, RR=2.0) | Est. Hold |
|-----------|-----------|---------------------|-------------------------------|-----------|
| Long  | {long_entry_lo} – {long_entry_hi} | {long_stop} | {long_target} | 4–26W |
| Short | {short_entry_lo} – {short_entry_hi} | {short_stop} | {short_target} | 4–26W |

---

Analyze the market data above and return a valid JSON signal following the format defined in your system instructions.
"""
    return prompt.strip()


# ─────────────────────────────────────────────
# 信号解析工具
# ─────────────────────────────────────────────

def parse_signal(response: str) -> dict | None:
    """从 LLM 响应中提取交易信号 JSON"""
    import json, re
    try:
        return json.loads(response)
    except Exception:
        pass
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    start = response.find('{')
    if start != -1:
        depth, end = 0, -1
        for i, c in enumerate(response[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            try:
                return json.loads(response[start:end + 1])
            except Exception:
                pass
    return None


def extract_asset_signal(parsed: dict, asset: str) -> dict | None:
    """从解析后的 JSON 中提取指定资产的信号字典"""
    if not parsed:
        return None
    for item in parsed.get("asset_analysis", []):
        if item.get("asset", "").upper() == asset.upper():
            return item
    return None


# ─────────────────────────────────────────────
# System prompt loader
# ─────────────────────────────────────────────

def _load_system_prompt() -> str:
    """Load 纳斯达克科技股分析.markdown as the system prompt for equity analysis."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "纳斯达克科技股分析.markdown"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "纳斯达克科技股分析.md"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                print(f"  [System Prompt] Loaded from {os.path.basename(path)} ({len(content)} chars)")
                return content
            except Exception as e:
                print(f"  [System Prompt] Failed to load {path}: {e}")
    print("  [System Prompt] markdown file not found — system prompt will be empty")
    return ""


# ─────────────────────────────────────────────
# API 调用
# ─────────────────────────────────────────────

def call_claude_api(prompt: str, system_prompt: str = "") -> str:
    if not _check_platform_quota():
        return ""
    import time
    print(f"\n正在调用 Claude API（模型: {ANTHROPIC_MODEL}）...")
    client = Anthropic(
        base_url=ANTHROPIC_BASE_URL,
        api_key=ANTHROPIC_API_KEY,
        http_client=httpx.Client(verify=False, timeout=120.0),
    )
    for attempt in range(3):
        try:
            kwargs = dict(
                model=ANTHROPIC_MODEL,
                max_tokens=8096,
                messages=[{"role": "user", "content": prompt}],
            )
            if system_prompt:
                kwargs["system"] = system_prompt
            message = client.messages.create(**kwargs)
            result = ""
            for block in message.content:
                if hasattr(block, "text"):
                    result += block.text
            return result
        except Exception as e:
            if attempt < 2:
                wait = 15 * (attempt + 1)
                print(f"  [重试 {attempt + 1}/3] 错误: {e}，{wait}s 后重试...")
                time.sleep(wait)
            else:
                raise


def call_deepseek_api(prompt: str, model: str, system_prompt: str = "") -> str:
    import re, time
    print(f"\n正在调用 DeepSeek API（模型: {model}）...")
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    for attempt in range(3):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=model,
                max_tokens=8000,
                messages=messages,
            )
            raw = response.choices[0].message.content or ""
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            return raw
        except Exception as e:
            print(f"  第 {attempt + 1} 次调用失败: {e}")
            if attempt < 2:
                time.sleep(5)
    return ""


def call_openai_api(prompt: str, model: str, system_prompt: str = "") -> str:
    """通过聚合平台调用 GPT 系列模型（与 Claude 共用同一 API Key）"""
    if not _check_platform_quota():
        return ""
    import time
    print(f"\n正在调用 OpenAI API（模型: {model}，via 聚合平台）...")
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    for attempt in range(3):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=model,
                max_tokens=4000,
                messages=messages,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"  第 {attempt + 1} 次调用失败: {e}")
            if attempt < 2:
                time.sleep(5)
    return ""


def _call_any_model(prompt: str, model: str, system_prompt: str = "") -> str:
    """统一模型路由"""
    if model in DEEPSEEK_MODELS:
        return call_deepseek_api(prompt, model, system_prompt=system_prompt)
    elif model in OPENAI_MODELS:
        return call_openai_api(prompt, model, system_prompt=system_prompt)
    else:
        return call_claude_api(prompt, system_prompt=system_prompt)


def _force_no_trade(parsed: dict, raw_resp: str, asset_name: str, reason: str) -> str:
    import json as _json
    if parsed and "asset_analysis" in parsed:
        for item in parsed["asset_analysis"]:
            if item.get("asset", "").upper() == asset_name.upper():
                item["action"]            = "no_trade"
                item["bias_score"]        = 0.0
                item["profit_target"]     = None
                item["stop_loss"]         = None
                item["risk_reward_ratio"] = None
                item["position_size_pct"] = 0.0
                item["justification"]     = f"[{reason}]，强制 no_trade。"
        return _json.dumps(parsed, ensure_ascii=False, indent=2)
    return raw_resp


def call_dual_model_api(
    prompt: str,
    asset_name: str,
    screener_model: str = "deepseek-reasoner",
    confirm_model: str = None,
    bias_threshold: float = 0.55,
    system_prompt: str = "",
) -> str:
    """双模型交叉验证：初筛 + 确认，方向分歧时强制 no_trade"""
    import json as _json

    if confirm_model is None:
        confirm_model = ANTHROPIC_MODEL

    print(f"\n[双模型] Step 1 初筛 ({screener_model})...")
    screener_resp   = _call_any_model(prompt, screener_model, system_prompt=system_prompt)
    screener_parsed = parse_signal(screener_resp)
    screener_signal = extract_asset_signal(screener_parsed, asset_name)

    if not screener_signal:
        print("  [双模型] 初筛信号解析失败，直接返回初筛结果")
        return screener_resp

    screener_action = screener_signal.get("action", "no_trade")
    screener_bias   = float(screener_signal.get("bias_score", 0) or 0)

    if screener_action == "no_trade" or screener_bias < bias_threshold:
        print(f"  [双模型] 初筛: {screener_action} (bias={screener_bias:.2f}) → 低于阈值 {bias_threshold}，跳过确认模型")
        return screener_resp

    print(f"  [双模型] 初筛: {screener_action} (bias={screener_bias:.2f}) → 触发确认模型 ({confirm_model})...")
    confirm_resp   = _call_any_model(prompt, confirm_model, system_prompt=system_prompt)
    confirm_parsed = parse_signal(confirm_resp)
    confirm_signal = extract_asset_signal(confirm_parsed, asset_name)

    if not confirm_signal:
        print("  [双模型] 确认模型信号解析失败，使用初筛结果")
        return screener_resp

    confirm_action = confirm_signal.get("action", "no_trade")
    confirm_bias   = float(confirm_signal.get("bias_score", 0) or 0)

    if screener_action == confirm_action:
        print(f"  [双模型] ✓ 一致: {confirm_action} | 初筛 bias={screener_bias:.2f}, 确认 bias={confirm_bias:.2f}")
        note = (
            f"\n\n<!-- [双模型验证] 初筛({screener_model}): {screener_action} bias={screener_bias:.2f}"
            f" | 确认({confirm_model}): {confirm_action} bias={confirm_bias:.2f} | 结果: 一致，采用确认信号 -->"
        )
        return confirm_resp + note
    else:
        print(f"  [双模型] ✗ 分歧: 初筛={screener_action}(bias={screener_bias:.2f}), 确认={confirm_action}(bias={confirm_bias:.2f}) → 强制 no_trade")
        return _force_no_trade(confirm_parsed, confirm_resp, asset_name,
                               f"双模型分歧: 初筛({screener_model})={screener_action}, 确认({confirm_model})={confirm_action}")


def call_voting_model_api(
    prompt: str,
    asset_name: str,
    models: list = None,
    bias_threshold: float = 0.55,
    prefer_model: str = None,
    system_prompt: str = "",
) -> str:
    """多模型投票：多数决定最终信号，无共识时强制 no_trade"""
    from collections import Counter

    if models is None:
        models = ["deepseek-reasoner", ANTHROPIC_MODEL, "gpt-4o"]
    if prefer_model is None:
        prefer_model = ANTHROPIC_MODEL

    majority_threshold = len(models) // 2 + 1

    votes = []
    for model in models:
        print(f"\n[投票] 调用 {model}...")
        resp   = _call_any_model(prompt, model, system_prompt=system_prompt)
        parsed = parse_signal(resp)
        signal = extract_asset_signal(parsed, asset_name)
        if signal:
            action = signal.get("action", "no_trade")
            bias   = float(signal.get("bias_score", 0) or 0)
            if action != "no_trade" and bias < bias_threshold:
                print(f"  [{model}] bias={bias:.2f} 低于阈值 → 计为 no_trade")
                action = "no_trade"
        else:
            print(f"  [{model}] 信号解析失败 → 计为 no_trade")
            action, bias = "no_trade", 0.0
        print(f"  [{model}] 投票: {action} (bias={bias:.2f})")
        votes.append((model, action, bias, parsed, resp))

    action_counts = Counter(v[1] for v in votes)
    vote_summary  = " | ".join(f"{v[0]}:{v[1]}(bias={v[2]:.2f})" for v in votes)
    print(f"\n[投票] 汇总: {vote_summary}")
    print(f"[投票] 统计: {dict(action_counts)}")

    winning_action = None
    for action, count in action_counts.most_common():
        if count >= majority_threshold:
            winning_action = action
            break

    note_prefix = f"\n\n<!-- [多模型投票] {vote_summary}"

    if winning_action is None:
        print("[投票] 无多数共识 → 强制 no_trade")
        ref_vote = next((v for v in votes if v[0] == prefer_model), votes[0])
        return _force_no_trade(ref_vote[3], ref_vote[4], asset_name, "多模型投票无共识") \
               + note_prefix + " | 结果: 无共识，强制 no_trade -->"

    if winning_action == "no_trade":
        majority_votes = [v for v in votes if v[1] == "no_trade"]
        selected = next((v for v in majority_votes if v[0] == prefer_model), majority_votes[0])
        return selected[4] + note_prefix + " | 结果: 多数 no_trade -->"

    print(f"[投票] ✓ 多数共识: {winning_action}")
    majority_votes = [v for v in votes if v[1] == winning_action]
    selected = next((v for v in majority_votes if v[0] == prefer_model), majority_votes[0])
    return selected[4] + note_prefix + f" | 结果: 多数={winning_action}，采用 {selected[0]} 信号 -->"


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
# Python 硬规则前置检查
# ─────────────────────────────────────────────

def _python_pre_filter(
    ticker: str,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    intel: dict,
    macro: dict | None = None,
) -> str | None:
    """
    LLM 调用前的 Python 硬规则检查。
    返回 None  → 正常继续调用 LLM。
    返回字符串 → 跳过 LLM，直接输出 no_trade，字符串为原因。

    拦截规则（硬拦截，直接 no_trade）：
    1. 财报 ≤ 5 天：二元事件风险
    2. 个股周线死叉（EMA50 < EMA200）且价格低于 EMA200：确认下跌趋势

    注：QQQ 死叉改为软限制（bias 上限 0.55 + 仓位上限 0.2），不再硬拦截。
    """
    # 1. 财报 ≤ 5 天
    days = intel.get("earnings_days_away")
    if days is not None and 0 <= int(days) <= 5:
        return f"财报仅 {days} 天后（{intel.get('earnings_date', '')}），二元事件风险，强制 no_trade"

    # 2. 个股死叉 + 价格低于 EMA200
    if not weekly.empty and "Close" in weekly.columns and not daily.empty and "Close" in daily.columns:
        wc = weekly["Close"].squeeze().dropna()
        if len(wc) >= 200:
            w_e50  = float(calc_ema(wc, 50).dropna().iloc[-1])
            w_e200 = float(calc_ema(wc, 200).dropna().iloc[-1])
            curr   = float(daily["Close"].squeeze().dropna().iloc[-1])
            if w_e50 < w_e200 and curr < w_e200:
                return (
                    f"个股死叉（EMA50={w_e50:.1f} < EMA200={w_e200:.1f}）"
                    f"且价格={curr:.1f} < EMA200，确认下跌趋势，跳过 LLM"
                )

    return None


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
    parser.add_argument("--dual-model",         action="store_true",
                        help="启用双模型交叉验证（初筛+确认，分歧时强制 no_trade）")
    parser.add_argument("--screener-model",      default="deepseek-reasoner",
                        help="初筛/第一模型（默认: deepseek-reasoner）")
    parser.add_argument("--confirm-model",       default=ANTHROPIC_MODEL,
                        help=f"确认/第二模型（默认: {ANTHROPIC_MODEL}）")
    parser.add_argument("--third-model",         default=None,
                        help="第三模型，启用后切换为三模型投票制（例: gpt-4o）")
    parser.add_argument("--prefer-model",        default=ANTHROPIC_MODEL,
                        help=f"投票胜出时优先采用哪个模型的信号（默认: {ANTHROPIC_MODEL}）")
    parser.add_argument("--dual-bias-threshold", default=0.55, type=float,
                        help="触发确认模型/计票时的 bias 阈值（默认: 0.55）")
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

    print()
    # 优先使用 RAG 新闻桥接（语义检索 + LLM情绪标注），降级到 yfinance.news
    if _NEWS_BRIDGE_AVAILABLE:
        print("正在获取 RAG 新闻信号...")
        news_context = fetch_news_signals(ticker, lookback_hours=72)
        news_items = None   # bridge 接管，旧路径不再使用
    else:
        news_context = None
        news_items = fetch_recent_news(ticker)

    # ── 加载 system prompt ──
    system_prompt = _load_system_prompt()

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
        news_items=news_items,
        news_context=news_context,
    )

    # ── 保存提示词 ──
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", f"{ticker.lower()}_prompt_output.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"提示词已保存到: {output_path}")

    if args.api:
        # ── Python 硬规则前置检查（财报/死叉+价格低于EMA200）──────────────
        pre_reason = _python_pre_filter(ticker, daily, weekly, intel, macro)
        if pre_reason:
            import json as _json
            print(f"\n[前置过滤] {pre_reason}")
            no_trade_result = _json.dumps({
                "period": "Weekly", "stock_ticker": ticker,
                "overall_market_sentiment": "N/A",
                "qqq_assessment": "N/A", "sector_assessment": "N/A",
                "macro_rate_environment": "N/A",
                "earnings_risk_flag": intel.get("earnings_days_away") is not None and int(intel.get("earnings_days_away", 999)) <= 5,
                "earnings_days_away": intel.get("earnings_days_away"),
                "asset_analysis": [{
                    "asset": ticker, "regime": "pre_filtered",
                    "action": "no_trade", "bias_score": 0.0,
                    "entry_zone": "N/A", "profit_target": None, "stop_loss": None,
                    "risk_reward_ratio": None, "invalidation_condition": "N/A",
                    "estimated_holding_weeks": None, "position_size_pct": 0.0,
                    "price_action_analysis": {}, "structured_analysis": {}, "intelligence_analysis": {},
                    "justification": f"[Python前置过滤] {pre_reason}",
                }]
            }, ensure_ascii=False, indent=2)
            api_output_path = os.path.join("outputs", f"{ticker.lower()}_api_output.txt")
            with open(api_output_path, "w", encoding="utf-8") as f:
                f.write(no_trade_result)
            print(no_trade_result)
            print(f"\n分析结果已保存到: {api_output_path}")
            return

        if args.dual_model and args.third_model:
            analysis = call_voting_model_api(
                prompt,
                asset_name=ticker,
                models=[args.screener_model, args.confirm_model, args.third_model],
                bias_threshold=args.dual_bias_threshold,
                prefer_model=args.prefer_model,
                system_prompt=system_prompt,
            )
        elif args.dual_model:
            analysis = call_dual_model_api(
                prompt,
                asset_name=ticker,
                screener_model=args.screener_model,
                confirm_model=args.confirm_model,
                bias_threshold=args.dual_bias_threshold,
                system_prompt=system_prompt,
            )
        else:
            analysis = _call_any_model(prompt, args.model, system_prompt=system_prompt)

        api_output_path = os.path.join("outputs", f"{ticker.lower()}_api_output.txt")
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
