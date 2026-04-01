"""
黄金分析脚本
自动获取黄金价格数据、计算技术指标，生成可粘贴到 Claude.ai 的分析提示词

依赖安装：
    pip install yfinance pandas
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
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# Anthropic API 配置（方案二：直接调用 Claude API）
# 优先读取环境变量，未设置则使用下方默认值
# ─────────────────────────────────────────────
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.openai-proxy.org/anthropic")
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY",  "sk-6BV9Xfa9AJ09pkt0AHFPQtZUtlM28pCOnon6ArdIJW1fVyDP")
ANTHROPIC_MODEL    = "claude-sonnet-4-6"

# ─────────────────────────────────────────────
# DeepSeek API 配置
# ─────────────────────────────────────────────
DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "sk-9574b3366dfd41178a5493d0f6af33c0")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# ─────────────────────────────────────────────
# Binance API 配置（链上黄金交易）
# 必须通过环境变量设置，禁止硬编码
# ─────────────────────────────────────────────
BINANCE_API_KEY    = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
TRADING_SYMBOL     = "PAXGUSDT"
TRADE_LOG_PATH     = "trade_log.csv"
HALT_FILE          = "TRADING_HALT"
CIRCUIT_BREAKER_N  = 3   # 连续止损次数触发熔断

TRADE_LOG_COLUMNS = [
    "timestamp", "date", "signal_action", "symbol",
    "side", "quantity", "price", "stop_loss", "profit_target",
    "order_id", "status", "dry_run", "bias_score", "notes",
]

# 支持的模型列表（用于 --model 参数提示）
CLAUDE_MODELS   = {"claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"}
DEEPSEEK_MODELS = {"deepseek-reasoner", "deepseek-chat"}
# GPT / OpenAI 系列：通过聚合平台 openai-proxy.org，复用同一 API Key
OPENAI_MODELS   = {"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o1", "o3-mini"}
# 聚合平台的 OpenAI 兼容接口（与 ANTHROPIC_BASE_URL 同一平台，同一 key）
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai-proxy.org/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", ANTHROPIC_API_KEY)  # 默认复用聚合平台 key

# 聚合平台每日调用限额（Claude + GPT 共享，DeepSeek R1 不受限）
_PLATFORM_USAGE_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_usage.json")
PLATFORM_DAILY_LIMIT  = int(os.getenv("PLATFORM_DAILY_LIMIT", "10"))

# 禁用 SSL 警告（企业网络/VPN 自签名证书环境）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _check_platform_quota() -> bool:
    """
    检查聚合平台（Claude + GPT）今日调用次数是否已达上限。
    计数写入 api_usage.json，每日自动重置。跨脚本共享同一文件。
    返回 True 表示配额充足可以调用，False 表示已达上限。
    """
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
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(com=period - 1, adjust=False).mean()


def calc_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """返回 (upper, mid, lower, %B, bandwidth%)"""
    mid   = series.rolling(window=period).mean()
    std   = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    pct_b     = (series - lower) / (upper - lower)   # 0=下轨, 1=上轨
    bandwidth = (upper - lower) / mid * 100           # 带宽百分比
    return upper, mid, lower, pct_b, bandwidth


def calc_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """返回 (%K, %D)"""
    low_min  = df['Low'].squeeze().rolling(window=k_period).min()
    high_max = df['High'].squeeze().rolling(window=k_period).max()
    k = 100 * (df['Close'].squeeze() - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    return k, d


def calc_adx(df: pd.DataFrame, period: int = 14):
    """返回 (ADX, +DI, -DI)，衡量趋势强度"""
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
    """On-Balance Volume"""
    close  = df['Close'].squeeze()
    volume = df['Volume'].squeeze()
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def calc_roc(series: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change %"""
    return (series / series.shift(period) - 1) * 100


def fmt_series(series: pd.Series, decimals: int = 2, n: int = 10) -> str:
    """将 Series 最后 n 个值格式化为字符串列表"""
    values = series.dropna().tail(n).round(decimals).tolist()
    return ", ".join(str(v) for v in values)


# ─────────────────────────────────────────────
# 数据获取与指标计算
# ─────────────────────────────────────────────

def _make_session() -> curl_requests.Session:
    """创建禁用 SSL 验证的 curl_cffi session（yfinance 新版要求）
    支持通过 HTTPS_PROXY / HTTP_PROXY 环境变量配置代理，用于云服务器 IP 被限速的场景。
    """
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    session = curl_requests.Session(impersonate="chrome", verify=False, proxy=proxy)
    return session


def fetch_gold_data():
    ticker = "GC=F"
    print(f"正在获取黄金期货数据 ({ticker})...")

    session = _make_session()

    # 日线：3个月（仅用于近期动量参考及当前价格）
    daily = yf.download(
        ticker, period="3mo", interval="1d",
        auto_adjust=True, progress=False, session=session
    )
    # 周线：2年（主分析时间框架）
    weekly = yf.download(
        ticker, period="2y", interval="1wk",
        auto_adjust=True, progress=False, session=session
    )
    # 月线：5年（长期趋势背景）
    monthly = yf.download(
        ticker, period="5y", interval="1mo",
        auto_adjust=True, progress=False, session=session
    )

    if daily.empty or weekly.empty or monthly.empty:
        raise ValueError("数据获取失败，请检查网络或 ticker 是否正确")

    print(f"日线数据：{len(daily)} 条  |  周线数据：{len(weekly)} 条  |  月线数据：{len(monthly)} 条")
    return daily, weekly, monthly


def fetch_macro_data() -> dict:
    """
    获取宏观跨资产数据：
      - DX-Y.NYB : 美元指数现货 (DXY, ICE)
      - ^TNX  : 美国10年期国债收益率
      - ^VIX  : CBOE 恐慌指数
      - SI=F  : 白银期货（用于计算金银比）
    失败时静默跳过，返回空 DataFrame。
    """
    session = _make_session()
    tickers = {
        "dxy":    "DX-Y.NYB",
        "tnx":    "^TNX",
        "vix":    "^VIX",
        "silver": "SI=F",
    }
    macro = {}
    for key, ticker in tickers.items():
        try:
            df = yf.download(
                ticker, period="3mo", interval="1d",
                auto_adjust=True, progress=False, session=session
            )
            macro[key] = df if not df.empty else pd.DataFrame()
            status = f"{len(df)} 条" if not df.empty else "失败"
            print(f"  {ticker:8s}: {status}")
        except Exception as e:
            macro[key] = pd.DataFrame()
            print(f"  {ticker:8s}: 获取失败 ({e})")
    return macro


def fetch_paxg_price() -> dict:
    """
    从 CoinGecko 免费 API 获取 PAXG（链上代币化黄金）实时价格。
    PAXG = Paxos Gold，1 PAXG = 1 troy oz 实物黄金，运行在以太坊链上。
    免费接口无需 API Key，每分钟限速 ~30 次。
    """
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "pax-gold", "vs_currencies": "usd", "include_24hr_change": "true"}
    try:
        import requests as std_requests
        resp = std_requests.get(url, params=params, timeout=10)
        data = resp.json()
        paxg_usd = data.get("pax-gold", {}).get("usd")
        paxg_chg = data.get("pax-gold", {}).get("usd_24h_change")
        if paxg_usd:
            print(f"  PAXG (链上黄金): ${paxg_usd:.2f}  24h变化: {paxg_chg:+.2f}%" if paxg_chg else f"  PAXG: ${paxg_usd:.2f}")
        return {"price": paxg_usd, "change_24h": paxg_chg}
    except Exception as e:
        print(f"  PAXG 数据获取失败 ({e})")
        return {"price": None, "change_24h": None}


def compute_indicators(df: pd.DataFrame):
    close = df['Close'].squeeze()

    indicators = {}
    # ── 趋势 ──
    indicators['ema20']     = calc_ema(close, 20)
    indicators['ema50']     = calc_ema(close, 50)
    indicators['ema200']    = calc_ema(close, 200)
    macd, signal, hist      = calc_macd(close)
    indicators['macd']      = macd
    indicators['macd_sig']  = signal
    indicators['macd_hist'] = hist
    # ── 震荡 ──
    indicators['rsi14']     = calc_rsi(close, 14)
    indicators['rsi7']      = calc_rsi(close, 7)
    stoch_k, stoch_d        = calc_stochastic(df)
    indicators['stoch_k']   = stoch_k
    indicators['stoch_d']   = stoch_d
    # ── 波动 ──
    indicators['atr14']     = calc_atr(df, 14)
    indicators['atr3']      = calc_atr(df, 3)
    bb_up, bb_mid, bb_lo, pct_b, bw = calc_bollinger_bands(close)
    indicators['bb_upper']  = bb_up
    indicators['bb_mid']    = bb_mid
    indicators['bb_lower']  = bb_lo
    indicators['bb_pct_b']  = pct_b
    indicators['bb_bw']     = bw
    # ── 趋势强度 ──
    adx, plus_di, minus_di  = calc_adx(df)
    indicators['adx']       = adx
    indicators['plus_di']   = plus_di
    indicators['minus_di']  = minus_di
    # ── 量价 & 动量 ──
    indicators['obv']       = calc_obv(df)
    indicators['roc10']     = calc_roc(close, 10)
    indicators['roc20']     = calc_roc(close, 20)
    return indicators


def summarize_macro(macro: dict, gold_close: pd.Series) -> dict:
    """
    将宏观原始 DataFrame 转化为简洁的摘要字典，供 prompt 使用。
    """
    result = {}

    def _last_n(df, col, n=5):
        if df.empty or col not in df.columns:
            return []
        s = df[col].squeeze().dropna().tail(n)
        return s.round(3).tolist()

    def _trend(vals):
        if len(vals) < 2:
            return "N/A"
        chg = (vals[-1] - vals[0]) / abs(vals[0]) * 100 if vals[0] != 0 else 0
        return f"{'↑' if chg > 0 else '↓'} {abs(chg):.1f}% (5日)"

    # ── DXY ──
    dxy_df = macro.get("dxy", pd.DataFrame())
    dxy_closes = _last_n(dxy_df, "Close")
    result["dxy_last"]   = round(dxy_closes[-1], 2) if dxy_closes else None
    result["dxy_trend"]  = _trend(dxy_closes)
    result["dxy_ema20"]  = round(float(calc_ema(dxy_df['Close'].squeeze(), 20).dropna().iloc[-1]), 2) if not dxy_df.empty and 'Close' in dxy_df.columns else None
    result["dxy_series"] = dxy_closes

    # ── 10Y Yield ──
    tnx_df = macro.get("tnx", pd.DataFrame())
    tnx_closes = _last_n(tnx_df, "Close")
    result["tnx_last"]   = round(tnx_closes[-1], 3) if tnx_closes else None
    result["tnx_trend"]  = _trend(tnx_closes)
    result["tnx_series"] = tnx_closes

    # ── VIX ──
    vix_df = macro.get("vix", pd.DataFrame())
    vix_closes = _last_n(vix_df, "Close")
    result["vix_last"]   = round(vix_closes[-1], 2) if vix_closes else None
    result["vix_trend"]  = _trend(vix_closes)
    result["vix_series"] = vix_closes
    if result["vix_last"]:
        v = result["vix_last"]
        result["vix_regime"] = "恐慌" if v > 30 else ("高波动" if v > 20 else ("中性" if v > 15 else "低波动/乐观"))

    # ── 金银比 ──
    silver_df = macro.get("silver", pd.DataFrame())
    if not silver_df.empty:
        silver_close = silver_df['Close'].squeeze().dropna()
        gold_aligned  = gold_close.reindex(silver_close.index, method='ffill').dropna()
        silver_aligned = silver_close.reindex(gold_aligned.index, method='ffill').dropna()
        gs_ratio = (gold_aligned / silver_aligned).dropna()
        gs_last  = gs_ratio.tail(5).tolist()
        result["gs_ratio_last"]   = round(gs_last[-1], 1) if gs_last else None
        result["gs_ratio_series"] = [round(x, 1) for x in gs_last]
        result["gs_ratio_trend"]  = _trend(gs_last)
    else:
        result["gs_ratio_last"]   = None
        result["gs_ratio_series"] = []
        result["gs_ratio_trend"]  = "N/A"

    return result


# ─────────────────────────────────────────────
# 周涨跌幅计算
# ─────────────────────────────────────────────

def weekly_change(daily: pd.DataFrame):
    close = daily['Close'].squeeze()
    last_close  = close.iloc[-1]
    prev_close  = close.iloc[-2] if len(close) >= 2 else close.iloc[-1]

    # 过去5个交易日（约一周）
    close_5d_ago = close.iloc[-6] if len(close) >= 6 else close.iloc[0]

    day_chg     = (last_close - prev_close) / prev_close * 100
    week_chg    = (last_close - close_5d_ago) / close_5d_ago * 100
    return last_close, day_chg, week_chg, close_5d_ago


# ─────────────────────────────────────────────
# 性能指标读取
# ─────────────────────────────────────────────

def load_perf_metrics(perf_csv: str = "backtest_results/performance.csv") -> dict:
    """
    从 performance.csv 读取最新回测指标，用于向模型提供性能反馈。
    返回包含 win_rate / avg_win / avg_loss / consecutive_losses 的字典；
    若文件不存在则返回空字典。
    """
    if not os.path.exists(perf_csv):
        return {}

    df = pd.read_csv(perf_csv)
    if df.empty:
        return {}

    row = df.iloc[-1]

    # 计算最近信号的连续亏损次数
    consecutive_losses = 0
    signals_csv = perf_csv.replace("performance.csv", "signals.csv")
    if os.path.exists(signals_csv):
        sig = pd.read_csv(signals_csv)
        executed = sig[sig["exit_reason"].isin(["STOP_LOSS", "TAKE_PROFIT"])].copy()
        if not executed.empty:
            for _, r in executed.iloc[::-1].iterrows():
                if r["win"] == False or str(r["win"]).lower() == "false":
                    consecutive_losses += 1
                else:
                    break

    win_rate_str = str(row.get("win_rate", "N/A"))
    win_rate_float = float(win_rate_str.replace("%", "")) / 100 if "%" in win_rate_str else None

    return {
        "win_rate": win_rate_str,
        "win_rate_float": win_rate_float,
        "avg_win": str(row.get("avg_win_pct", "N/A")),
        "avg_loss": str(row.get("avg_loss_pct", "N/A")),
        "profit_factor": str(row.get("profit_factor", "N/A")),
        "total_return": str(row.get("total_return", "N/A")),
        "consecutive_losses": consecutive_losses,
    }


# ─────────────────────────────────────────────
# 生成提示词文本
# ─────────────────────────────────────────────

def build_prompt(daily: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame,
                 perf_metrics: dict | None = None,
                 macro: dict | None = None,
                 paxg: dict | None = None) -> str:
    # ── 周线为主分析框架（中长期：2周 ~ 3个月）──
    w_ind = compute_indicators(weekly)
    m_ind = compute_indicators(monthly)
    d_ind = compute_indicators(daily)   # 仅用于近期动量参考及当前价格

    close_w = weekly['Close'].squeeze()
    close_m = monthly['Close'].squeeze()
    close_d = daily['Close'].squeeze()

    # 当前价格（最新日线收盘）
    current_price = round(float(close_d.iloc[-1]), 2)

    # 周线核心指标快照（主框架）
    current_ema20_w  = round(float(w_ind['ema20'].dropna().iloc[-1]), 2)
    current_ema50_w  = round(float(w_ind['ema50'].dropna().iloc[-1]), 2)
    current_ema200_w = round(float(w_ind['ema200'].dropna().iloc[-1]), 2)
    current_macd_w   = round(float(w_ind['macd'].dropna().iloc[-1]), 2)
    current_rsi7_w   = round(float(w_ind['rsi7'].dropna().iloc[-1]), 2)
    current_rsi14_w  = round(float(w_ind['rsi14'].dropna().iloc[-1]), 2)

    # 月线核心指标快照（长期趋势背景）
    current_ema20_m  = round(float(m_ind['ema20'].dropna().iloc[-1]), 2)
    current_ema50_m  = round(float(m_ind['ema50'].dropna().iloc[-1]), 2)
    current_macd_m   = round(float(m_ind['macd'].dropna().iloc[-1]), 2)
    current_rsi14_m  = round(float(m_ind['rsi14'].dropna().iloc[-1]), 2)

    # 周线 ATR（中长期止损定位）
    atr14_weekly = round(float(w_ind['atr14'].dropna().iloc[-1]), 2)
    atr3_weekly  = round(float(w_ind['atr3'].dropna().iloc[-1]), 2)

    # 价格涨跌幅
    prev_week_close  = float(close_w.iloc[-2]) if len(close_w) >= 2 else current_price
    prev_month_close = float(close_m.iloc[-2]) if len(close_m) >= 2 else current_price
    price_3m_ago     = float(close_w.iloc[-13]) if len(close_w) >= 13 else float(close_w.iloc[0])
    week_chg         = (current_price - prev_week_close)  / prev_week_close  * 100
    month_chg        = (current_price - prev_month_close) / prev_month_close * 100
    three_month_chg  = (current_price - price_3m_ago)     / price_3m_ago     * 100

    # 成交量
    vol_current_d = int(daily['Volume'].squeeze().iloc[-1])
    vol_avg_d     = int(daily['Volume'].squeeze().tail(20).mean())

    # 周线序列（最近26周 ≈ 半年）
    n_w = 26
    series_weekly_close = fmt_series(close_w, 2, n_w)
    series_weekly_ema20 = fmt_series(w_ind['ema20'], 2, n_w)
    series_weekly_ema50 = fmt_series(w_ind['ema50'], 2, n_w)
    series_weekly_macd  = fmt_series(w_ind['macd'],  2, n_w)
    series_weekly_rsi7  = fmt_series(w_ind['rsi7'],  2, n_w)
    series_weekly_rsi14 = fmt_series(w_ind['rsi14'], 2, n_w)

    # 月线序列（最近24个月 = 2年）
    n_m = 24
    series_monthly_close = fmt_series(close_m, 2, n_m)
    series_monthly_ema20 = fmt_series(m_ind['ema20'], 2, n_m)
    series_monthly_macd  = fmt_series(m_ind['macd'],  2, n_m)
    series_monthly_rsi14 = fmt_series(m_ind['rsi14'], 2, n_m)

    # 52周价格结构（基于周线）
    high_52w = round(float(close_w.tail(52).max()), 1)
    low_52w  = round(float(close_w.tail(52).min()), 1)
    pct_from_high = round((current_price - high_52w) / high_52w * 100, 1)
    pct_from_low  = round((current_price - low_52w)  / low_52w  * 100, 1)

    # 周线高级指标快照
    stoch_k_w  = round(float(w_ind['stoch_k'].dropna().iloc[-1]), 1)
    stoch_d_w  = round(float(w_ind['stoch_d'].dropna().iloc[-1]), 1)
    adx_w      = round(float(w_ind['adx'].dropna().iloc[-1]), 1)
    plus_di_w  = round(float(w_ind['plus_di'].dropna().iloc[-1]), 1)
    minus_di_w = round(float(w_ind['minus_di'].dropna().iloc[-1]), 1)
    bb_pct_b_w = round(float(w_ind['bb_pct_b'].dropna().iloc[-1]), 3)
    bb_bw_w    = round(float(w_ind['bb_bw'].dropna().iloc[-1]), 2)
    bb_upper_w = round(float(w_ind['bb_upper'].dropna().iloc[-1]), 1)
    bb_lower_w = round(float(w_ind['bb_lower'].dropna().iloc[-1]), 1)
    roc10_w    = round(float(w_ind['roc10'].dropna().iloc[-1]), 2)
    roc20_w    = round(float(w_ind['roc20'].dropna().iloc[-1]), 2)

    obv_series_w = w_ind['obv'].dropna().tail(5).tolist()
    obv_trend_w  = "上升" if obv_series_w[-1] > obv_series_w[0] else "下降"

    # 近15周序列（高级指标）
    n_seq = 15
    series_stoch_k = fmt_series(w_ind['stoch_k'], 1, n_seq)
    series_stoch_d = fmt_series(w_ind['stoch_d'], 1, n_seq)
    series_adx     = fmt_series(w_ind['adx'],     1, n_seq)
    series_roc10   = fmt_series(w_ind['roc10'],   2, n_seq)
    series_bb_pctb = fmt_series(w_ind['bb_pct_b'], 3, n_seq)

    # ── 宏观跨资产摘要 ──
    ms = summarize_macro(macro or {}, close_d.dropna()) if macro is not None else summarize_macro({}, close_d.dropna())

    def _fmt_val(v, unit=""):
        return f"{v}{unit}" if v is not None else "N/A"

    macro_section = f"""
---

## 宏观跨资产信号

### 美元指数 (DXY)
- **当前**: {_fmt_val(ms.get('dxy_last'))}  |  **5日趋势**: {ms.get('dxy_trend', 'N/A')}  |  **EMA20**: {_fmt_val(ms.get('dxy_ema20'))}
- **近5日收盘**: {ms.get('dxy_series', [])}
- **解读**: DXY {'高于' if ms.get('dxy_last') and ms.get('dxy_ema20') and ms.get('dxy_last') > ms.get('dxy_ema20') else '低于'} EMA20 → 黄金{'承压' if ms.get('dxy_last') and ms.get('dxy_ema20') and ms.get('dxy_last') > ms.get('dxy_ema20') else '偏多'}

### 美国10年期国债收益率 (^TNX)
- **当前**: {_fmt_val(ms.get('tnx_last'), '%')}  |  **5日趋势**: {ms.get('tnx_trend', 'N/A')}
- **近5日收益率**: {ms.get('tnx_series', [])}
- **解读**: 收益率{'上升→实际利率压力增加→黄金承压' if ms.get('tnx_trend','').startswith('↑') else '下降→实际利率压力减轻→黄金偏多'}

### VIX 恐慌指数 (^VIX)
- **当前**: {_fmt_val(ms.get('vix_last'))}  |  **市场状态**: {ms.get('vix_regime', 'N/A')}  |  **5日趋势**: {ms.get('vix_trend', 'N/A')}
- **近5日VIX**: {ms.get('vix_series', [])}
- **解读**: VIX{'>20 → Risk-Off，黄金避险需求上升' if ms.get('vix_last') and ms.get('vix_last') > 20 else ' ≤20 → 市场平静，黄金避险溢价有限'}

### 黄金/白银比率 (Gold/Silver Ratio)
- **当前比率**: {_fmt_val(ms.get('gs_ratio_last'))}  |  **5日趋势**: {ms.get('gs_ratio_trend', 'N/A')}
- **近5日比率**: {ms.get('gs_ratio_series', [])}
- **解读**: 比率{'上升→黄金相对白银强势→避险属性驱动，非工业需求' if ms.get('gs_ratio_trend','').startswith('↑') else '下降→白银相对黄金强势→风险偏好改善，工业需求主导'}

### PAXG 链上代币化黄金 (Ethereum)
"""
    # PAXG 区块单独拼接，避免 f-string 里嵌套条件过长
    if paxg and paxg.get("price"):
        paxg_price  = paxg["price"]
        paxg_chg    = paxg.get("change_24h")
        spread      = round(paxg_price - current_price, 2)
        spread_pct  = round(spread / current_price * 100, 3)
        chg_str     = f"{paxg_chg:+.2f}%" if paxg_chg is not None else "N/A"
        spread_note = "PAXG 溢价（链上需求旺盛）" if spread > 0 else "PAXG 折价（链上流动性偏弱）"
        macro_section += f"""- **PAXG 现价**: ${paxg_price:.2f}  |  **24h涨跌**: {chg_str}
- **PAXG vs GC=F 价差**: {spread:+.2f} ({spread_pct:+.3f}%)  → {spread_note}
- **解读**: PAXG 与 GC=F 价差反映链上黄金需求，持续溢价表明机构通过以太坊持仓黄金意愿增强
"""
    else:
        macro_section += "- PAXG 数据暂不可用（网络超时或 CoinGecko 限速）\n"

    today_str = datetime.now().strftime("%Y-%m-%d")

    # ── 预计算入场锚点（基于周线 ATR-14，中长期定位）──
    macd_w_val  = round(float(w_ind['macd'].dropna().iloc[-1]), 2)
    rsi14_w_val = round(float(w_ind['rsi14'].dropna().iloc[-1]), 2)
    rsi7_w_val  = round(float(w_ind['rsi7'].dropna().iloc[-1]), 2)

    long_entry_lo  = round(current_price - 0.3 * atr14_weekly, 1)
    long_entry_hi  = round(current_price + 0.3 * atr14_weekly, 1)
    long_stop      = round(current_price - 1.5 * atr14_weekly, 1)
    long_target    = round(current_price + 3.0 * atr14_weekly, 1)
    short_entry_lo = round(current_price - 0.3 * atr14_weekly, 1)
    short_entry_hi = round(current_price + 0.3 * atr14_weekly, 1)
    short_stop     = round(current_price + 1.5 * atr14_weekly, 1)
    short_target   = round(current_price - 3.0 * atr14_weekly, 1)

    # ── 性能反馈区块 ──
    perf_section = ""
    if perf_metrics:
        wrf = perf_metrics.get("win_rate_float")
        cl  = perf_metrics.get("consecutive_losses", 0)

        calibration_warnings = []
        if wrf is not None and wrf < 0.40:
            calibration_warnings.append(
                "⚠️ 胜率低于40%警戒线：本次 bias_score 门槛强制提升至 ≥0.65，低于此值一律输出 no_trade"
            )
        if cl >= 2:
            calibration_warnings.append(
                f"⚠️ 已连续亏损 {cl} 次：本次除非出现极高置信度信号(bias_score≥0.75)，否则输出 no_trade"
            )

        warn_str = "\n".join(calibration_warnings) if calibration_warnings else "当前表现正常，维持标准决策流程。"

        perf_section = f"""
---

## 近期回测表现反馈（必读，据此校准本次决策）

| 指标 | 数值 |
|------|------|
| 胜率 | {perf_metrics.get('win_rate', 'N/A')} |
| 平均盈利 | {perf_metrics.get('avg_win', 'N/A')} |
| 平均亏损 | {perf_metrics.get('avg_loss', 'N/A')} |
| 盈利因子 | {perf_metrics.get('profit_factor', 'N/A')} |
| 总收益 | {perf_metrics.get('total_return', 'N/A')} |
| 最近连续亏损次数 | {cl} |

{warn_str}
"""

    prompt = f"""
# 黄金 (XAU/USD) 中长期趋势分析请求
**分析日期**: {today_str}
**分析周期**: 中长期（持仓周期：2周 ~ 3个月）
**数据来源**: COMEX 黄金期货 (GC=F)
{perf_section}
---

## 价格概要

- **当前价格**: ${current_price}
- **近1周涨跌幅**: {week_chg:+.2f}%
- **近1个月涨跌幅**: {month_chg:+.2f}%
- **近3个月涨跌幅**: {three_month_chg:+.2f}%
- **今日成交量**: {vol_current_d:,}  vs.  **20日均量**: {vol_avg_d:,}

---

## 当前技术指标快照（周线为主，月线为辅）

| 指标 | 当前值 | 框架 |
|------|--------|------|
| 当前价格 | {current_price} | 最新收盘 |
| EMA-20 (周线) | {current_ema20_w} | 主框架 |
| EMA-50 (周线) | {current_ema50_w} | 主框架 |
| EMA-200 (周线) | {current_ema200_w} | 主框架 |
| MACD (周线) | {current_macd_w} | 主框架 |
| RSI-7 (周线) | {current_rsi7_w} | 主框架 |
| RSI-14 (周线) | {current_rsi14_w} | 主框架 |
| ATR-14 (周线) | {atr14_weekly} | 止损定位 |
| EMA-20 (月线) | {current_ema20_m} | 长期背景 |
| EMA-50 (月线) | {current_ema50_m} | 长期背景 |
| MACD (月线) | {current_macd_m} | 长期背景 |
| RSI-14 (月线) | {current_rsi14_m} | 长期背景 |

---

## 周线序列数据（最近 {n_w} 周 ≈ 半年，**从旧到新排列**）

⚠️ 最后一个数值 = 最新数据

周收盘价: [{series_weekly_close}]
EMA-20:   [{series_weekly_ema20}]
EMA-50:   [{series_weekly_ema50}]
MACD:     [{series_weekly_macd}]
RSI-7:    [{series_weekly_rsi7}]
RSI-14:   [{series_weekly_rsi14}]

---

## 月线序列数据（最近 {n_m} 个月 ≈ 2年，**从旧到新排列**）

⚠️ 最后一个数值 = 最新数据

月收盘价: [{series_monthly_close}]
EMA-20:   [{series_monthly_ema20}]
MACD:     [{series_monthly_macd}]
RSI-14:   [{series_monthly_rsi14}]

---

## 长期趋势背景

周线 EMA20: {current_ema20_w}  vs.  EMA50: {current_ema50_w}  vs.  EMA200: {current_ema200_w}
月线 EMA20: {current_ema20_m}  vs.  EMA50: {current_ema50_m}
周线 ATR-3: {atr3_weekly}   vs.  ATR-14: {atr14_weekly}

---

## 52周价格结构

- **52周高点**: {high_52w}  |  **距高点**: {pct_from_high:+.1f}%
- **52周低点**: {low_52w}   |  **距低点**: {pct_from_low:+.1f}%
- **当前价在周线布林带中的位置 (%B)**: {bb_pct_b_w:.3f}  （0=下轨，0.5=中轨，1=上轨，>1=突破上轨，<0=跌破下轨）
- **布林带上轨**: {bb_upper_w}  |  **下轨**: {bb_lower_w}  |  **带宽**: {bb_bw_w:.2f}%

---

## 高级技术指标快照（周线）

| 指标 | 当前值 | 信号解读 |
|------|--------|----------|
| Stochastic %K (周) | {stoch_k_w} | {'超买区 >80' if stoch_k_w > 80 else ('超卖区 <20' if stoch_k_w < 20 else '中性区间')} |
| Stochastic %D (周) | {stoch_d_w} | {'K>D 金叉' if stoch_k_w > stoch_d_w else 'K<D 死叉'} |
| ADX (周) | {adx_w} | {'强趋势 >25' if adx_w > 25 else ('弱趋势 <20，市场振荡' if adx_w < 20 else '趋势形成中')} |
| +DI / -DI (周) | {plus_di_w} / {minus_di_w} | {'+DI>-DI 多头主导' if plus_di_w > minus_di_w else '-DI>+DI 空头主导'} |
| ROC(10周) | {roc10_w:+.2f}% | {'正动量' if roc10_w > 0 else '负动量'} |
| ROC(20周) | {roc20_w:+.2f}% | {'正动量' if roc20_w > 0 else '负动量'} |
| OBV趋势(近5周) | {obv_trend_w} | {'量价配合上涨' if obv_trend_w == '上升' else '量价配合下跌'} |

**近15周序列（从旧到新）**：
Stochastic %K: [{series_stoch_k}]
Stochastic %D: [{series_stoch_d}]
ADX:           [{series_adx}]
ROC(10):       [{series_roc10}]
BB %B:         [{series_bb_pctb}]
{macro_section}
---

## 预计算入场锚点（基于周线 ATR-14={atr14_weekly}，中长期定位）

> 周线 ATR-14 代表约1周的平均真实波幅，中长期止损须置于 1.5×周线ATR 之外以避免被周线噪音止损。
> entry_zone 必须在此范围内，否则将被判定为无效信号 (INVALID_RR)。

| 方向 | entry_zone 参考 | stop_loss 参考 (1.5×ATR) | profit_target 参考 (2.0×R) |
|------|-----------------|--------------------------|----------------------------|
| 做多 | {long_entry_lo} – {long_entry_hi} | {long_stop} | {long_target} |
| 做空 | {short_entry_lo} – {short_entry_hi} | {short_stop} | {short_target} |

你可以在上述参考值基础上小幅调整（±0.5×ATR），但不得大幅偏离。

## 仓位管理（position_size_pct）

请在 JSON 中输出 `position_size_pct`（0.0–1.0），表示建议使用总资金的比例：

| 条件 | 建议仓位 |
|------|---------|
| bias_score 0.50–0.59，Choppy 制度 | 0.1–0.2（轻仓试探） |
| bias_score 0.60–0.69，Trending 制度 | 0.3–0.5（标准仓） |
| bias_score 0.70–0.79，多时间框架共振 | 0.5–0.7（加重仓） |
| bias_score ≥ 0.80，趋势强劲+量价配合 | 0.7–1.0（满仓） |
| 周线 RSI-7 > 75 追高入场 | 上限 0.3（无论 bias 多高）|
| 价格偏离 EMA-20 > 10% | 上限 0.3（避免过度追涨）|

---

## 分析任务

请基于以上数据，按照中长期大宗商品分析框架，完成以下任务：

1. **判断当前市场制度**（Trending / Mean-Reverting / Choppy），以**周线**为准
2. **判断整体市场情绪**（Risk-On / Risk-Off / Neutral）
3. **分析 DXY 对黄金中长期走势的影响**
4. **针对黄金给出中长期交易建议**（持仓周期 2周~3个月），严格按以下 JSON 格式输出：

```json
{{
  "period": "Weekly",
  "overall_market_sentiment": "Risk-On | Risk-Off | Neutral",
  "dxy_assessment": "<DXY 趋势及对黄金中长期走势的影响>",
  "asset_analysis": [
    {{
      "asset": "GOLD",
      "regime": "Trending | Mean-Reverting | Choppy",
      "action": "long | short | no_trade",
      "bias_score": <0.0 到 1.0>,
      "entry_zone": "<价格区间，必须基于上方预计算锚点>",
      "profit_target": <数字 或 null>,
      "stop_loss": <数字 或 null>,
      "risk_reward_ratio": <数字 或 null>,
      "estimated_holding_weeks": <预计持仓周数，整数 2~12，或 null if no_trade>,
      "position_size_pct": <建议仓位占比 0.0–1.0，no_trade 时填 0.0>,
      "invalidation_condition": "<使该观点失效的具体信号（周线收盘为准）>",
      "macro_catalyst": "<驱动中长期行情的宏观逻辑>",
      "technical_setup": "<关键周线/月线指标信号综合描述>",
      "justification": "<不超过300字的综合判断>"
    }}
  ]
}}
```

**硬性约束（违反任意一条必须改为 no_trade）**：
- entry_zone 必须包含当前价格 ±1×周线ATR-14 范围，不得设置脱离市场的理想价格
- profit_target 做多时必须高于 entry_zone 上限，做空时必须低于 entry_zone 下限
- risk_reward_ratio 必须 ≥ 2.0
- stop_loss 距离 entry 不得小于 1.0×周线ATR-14（中长期需容纳周线波动噪音）
- 当 action = no_trade 时，profit_target / stop_loss / risk_reward_ratio / estimated_holding_weeks 填 null，position_size_pct 填 0.0

**信号质量过滤规则（全部适用）**：
- 周线 MACD ({macd_w_val}) < 0 时，**禁止**在 Trending 制度下做多；可评估做空
- 周线 RSI-14 ({rsi14_w_val}) > 75 时，做多 bias_score 自动上限 0.55；RSI-14 < 30 时，做空 bias_score 自动上限 0.55
- 价格偏离周线 EMA-20 ({current_ema20_w}) 超过 10% 时，bias_score 上限 0.55（无论方向）
- **Mean-Reverting 制度下，强制输出 no_trade**（历史回测显示该制度胜率接近 0%，禁止入场）
- Choppy 制度下，bias_score 上限 0.45
- 当 bias_score < 0.50 时，一律输出 no_trade
- 多空均衡：价格低于周线 EMA-20 且 MACD 为负时，**必须认真评估做空机会**，不得默认 no_trade

**宏观与多时间框架因子使用规则**：
- DXY 趋势向上（价格>EMA20）时，做多黄金需额外降低 bias_score 0.05–0.10
- 10Y 收益率持续上升趋势时，做多黄金需额外降低 bias_score 0.05
- VIX > 25 时，Risk-Off 环境，黄金避险需求升级，可适当上调 bias_score 0.05
- ADX ({adx_w}) < 20：市场处于振荡，Trending 信号可靠性下降，降级为 Choppy
- ADX > 30：趋势强劲，可适当上调 bias_score 0.05
- 月线 MACD 与周线 MACD 同向（均为正或均为负）：多时间框架共振，bias_score 上调 0.05
- 月线 MACD 与周线 MACD 背离（方向相反）：信号可靠性下降，bias_score 降低 0.05
- Stochastic %K ({stoch_k_w}) > 80 且 %K < %D：超买死叉，做多信号降级
- Stochastic %K < 20 且 %K > %D：超卖金叉，做空信号降级
- OBV 趋势与价格趋势背离时，降低 bias_score 0.10（量价不配合）
- 金银比快速上升（避险驱动）且 VIX 上升：黄金 safe-haven 信号加强
"""
    return prompt.strip()


# ─────────────────────────────────────────────
# Anthropic API 调用（方案二）
# ─────────────────────────────────────────────

def call_claude_api(prompt: str) -> str:
    """
    直接调用 Anthropic Claude API 获取分析结果。
    使用 ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY / ANTHROPIC_MODEL 配置。
    含 3 次重试逻辑，每次间隔递增。受聚合平台每日配额限制。
    """
    import time
    if not _check_platform_quota():
        return ""
    print(f"\n正在调用 Claude API（模型: {ANTHROPIC_MODEL}）...")
    client = Anthropic(
        base_url=ANTHROPIC_BASE_URL,
        api_key=ANTHROPIC_API_KEY,
        http_client=httpx.Client(verify=False, timeout=120.0),
    )
    for attempt in range(3):
        try:
            message = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=8096,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            # 提取文本内容
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


def call_deepseek_api(prompt: str, model: str) -> str:
    """
    通过 OpenAI 兼容接口调用 DeepSeek API，含3次重试逻辑。
    支持 deepseek-reasoner（R1）和 deepseek-chat。
    """
    print(f"\n正在调用 DeepSeek API（模型: {model}）...")
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content or ""
            # 去除 DeepSeek R1 的 <think>...</think> 推理过程
            import re
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            return raw
        except Exception as e:
            print(f"  第 {attempt + 1} 次调用失败: {e}")
            if attempt < 2:
                import time
                time.sleep(5)
    return ""


def call_openai_api(prompt: str, model: str) -> str:
    """
    通过聚合平台（openai-proxy.org）调用 GPT 系列模型。
    使用与 Claude 相同的 API Key，base_url 为 /v1 兼容接口。
    """
    if not _check_platform_quota():
        return ""
    import time
    print(f"\n正在调用 OpenAI API（模型: {model}，via 聚合平台）...")
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"  第 {attempt + 1} 次调用失败: {e}")
            if attempt < 2:
                time.sleep(5)
    return ""


def _call_any_model(prompt: str, model: str) -> str:
    """统一模型路由：根据 model 名称自动分发到对应 API"""
    if model in DEEPSEEK_MODELS:
        return call_deepseek_api(prompt, model)
    elif model in OPENAI_MODELS:
        return call_openai_api(prompt, model)
    else:
        # Claude 系列（含 claude-sonnet / opus / haiku）
        return call_claude_api(prompt)


# ─────────────────────────────────────────────
# 信号解析
# ─────────────────────────────────────────────

def parse_signal(response: str) -> dict | None:
    """从 LLM 响应中提取交易信号 JSON，按优先级尝试三种解析方式"""
    import json, re

    # 方法1：整体直接解析
    try:
        return json.loads(response)
    except Exception:
        pass

    # 方法2：提取 markdown ```json ... ``` 代码块
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # 方法3：大括号层级匹配
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


def extract_asset_signal(parsed: dict, asset: str = "GOLD") -> dict | None:
    """从解析后的 JSON 中提取指定资产的信号字典"""
    if not parsed:
        return None
    for item in parsed.get("asset_analysis", []):
        if item.get("asset", "").upper() == asset.upper():
            return item
    return None


def call_dual_model_api(
    prompt: str,
    asset_name: str = "GOLD",
    screener_model: str = "deepseek-reasoner",
    confirm_model: str = None,
    bias_threshold: float = 0.55,
) -> str:
    """
    双模型交叉验证：
    1. screener_model 初筛（低成本）
    2. 初筛给出 long/short 且 bias >= bias_threshold 时，触发 confirm_model
    3. 两模型方向一致 → 使用确认模型信号
    4. 方向分歧 → 强制 no_trade
    """
    import json as _json

    if confirm_model is None:
        confirm_model = ANTHROPIC_MODEL

    # ── Step 1: 初筛 ──
    print(f"\n[双模型] Step 1 初筛 ({screener_model})...")
    screener_resp   = _call_any_model(prompt, screener_model)
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

    # ── Step 2: 确认 ──
    print(f"  [双模型] 初筛: {screener_action} (bias={screener_bias:.2f}) → 触发确认模型 ({confirm_model})...")
    confirm_resp   = _call_any_model(prompt, confirm_model)
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


def _force_no_trade(parsed: dict, raw_resp: str, asset_name: str, reason: str) -> str:
    """将 parsed JSON 中指定资产的信号强制改为 no_trade 并序列化返回"""
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


def call_voting_model_api(
    prompt: str,
    asset_name: str = "GOLD",
    models: list = None,
    bias_threshold: float = 0.55,
    prefer_model: str = None,
) -> str:
    """
    多模型投票（推荐三模型）：
    - 调用 models 列表中的所有模型，各自输出 action
    - bias < bias_threshold 的信号计为 no_trade
    - 多数票（>= ceil(N/2)+1 ）决定最终方向
    - 无多数（三票全不同）→ 强制 no_trade
    - 最终信号取多数中 prefer_model 的结果（无则取第一个多数模型）
    """
    from collections import Counter

    if models is None:
        models = ["deepseek-reasoner", ANTHROPIC_MODEL, "gpt-4o"]
    if prefer_model is None:
        prefer_model = ANTHROPIC_MODEL

    majority_threshold = len(models) // 2 + 1  # 3 模型 → 需 2 票

    votes = []  # (model, action, bias, parsed, raw_resp)
    for model in models:
        print(f"\n[投票] 调用 {model}...")
        resp   = _call_any_model(prompt, model)
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
        # 用得票最多（或第一个）的 parsed 结果来格式化 no_trade
        ref_vote = votes[0]
        for v in votes:
            if v[0] == prefer_model:
                ref_vote = v
                break
        note = note_prefix + " | 结果: 无共识，强制 no_trade -->"
        return _force_no_trade(ref_vote[3], ref_vote[4], asset_name,
                               "多模型投票无共识") + note

    if winning_action == "no_trade":
        print("[投票] 多数投 no_trade → 不入场")
        majority_votes = [v for v in votes if v[1] == "no_trade"]
        selected = next((v for v in majority_votes if v[0] == prefer_model), majority_votes[0])
        note = note_prefix + f" | 结果: 多数 no_trade -->"
        return selected[4] + note

    print(f"[投票] ✓ 多数共识: {winning_action}")
    majority_votes = [v for v in votes if v[1] == winning_action]
    selected = next((v for v in majority_votes if v[0] == prefer_model), majority_votes[0])
    note = note_prefix + f" | 结果: 多数={winning_action}，采用 {selected[0]} 信号 -->"
    return selected[4] + note


# ─────────────────────────────────────────────
# Binance 交易执行模块
# ─────────────────────────────────────────────

def is_trading_halted() -> bool:
    if os.path.exists(HALT_FILE):
        print(f"\n[HALT] 交易已暂停：检测到 {HALT_FILE} 文件。确认安全后手动删除该文件以恢复。")
        return True
    return False


def load_trade_log() -> pd.DataFrame:
    if os.path.exists(TRADE_LOG_PATH):
        return pd.read_csv(TRADE_LOG_PATH)
    return pd.DataFrame(columns=TRADE_LOG_COLUMNS)


def append_trade_log(record: dict):
    df = load_trade_log()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(TRADE_LOG_PATH, index=False)


def check_daily_limit(trade_log: pd.DataFrame) -> bool:
    """今日是否已有实盘开仓记录，True = 可继续交易"""
    today = datetime.now().strftime("%Y-%m-%d")
    real = trade_log[(trade_log["dry_run"] == False) & (trade_log["date"] == today)]
    if len(real) > 0:
        print("  [安全锁] 今日已有实盘记录，每日限 1 笔，跳过。")
        return False
    return True


def check_circuit_breaker(trade_log: pd.DataFrame) -> bool:
    """最近 CIRCUIT_BREAKER_N 笔实盘全部止损 → 触发熔断，创建 HALT 文件"""
    real = trade_log[
        (trade_log["dry_run"] == False) &
        (trade_log["status"] == "STOP_LOSS")
    ]
    if len(real) >= CIRCUIT_BREAKER_N:
        last_n = trade_log[trade_log["dry_run"] == False].tail(CIRCUIT_BREAKER_N)
        if (last_n["status"] == "STOP_LOSS").all():
            print(f"\n[熔断] 最近 {CIRCUIT_BREAKER_N} 笔实盘全部止损，自动暂停交易。")
            print(f"  正在创建 {HALT_FILE}，请人工审查后删除该文件以恢复。")
            open(HALT_FILE, "w").close()
            return True
    return False


def _floor_to_step(value: float, step: float) -> float:
    """数量按 Binance 步长向下截断"""
    import math
    if step <= 0:
        return value
    decimals = len(f"{step:.10f}".rstrip("0").split(".")[-1]) if "." in str(step) else 0
    return round(math.floor(value / step) * step, decimals)


def _round_to_tick(value: float, tick: float) -> float:
    """价格按 Binance tick 精度四舍五入"""
    if tick <= 0:
        return value
    decimals = len(f"{tick:.10f}".rstrip("0").split(".")[-1]) if "." in str(tick) else 0
    return round(round(value / tick) * tick, decimals)


def get_symbol_filters(client, symbol: str) -> dict:
    """查询交易对精度及最小下单要求"""
    info = client.get_symbol_info(symbol)
    result = {"step_size": 0.0001, "tick_size": 0.01, "min_qty": 0.0001, "min_notional": 5.0}
    for f in info.get("filters", []):
        ft = f["filterType"]
        if ft == "LOT_SIZE":
            result["step_size"] = float(f["stepSize"])
            result["min_qty"]   = float(f["minQty"])
        elif ft == "PRICE_FILTER":
            result["tick_size"] = float(f["tickSize"])
        elif ft in ("MIN_NOTIONAL", "NOTIONAL"):
            result["min_notional"] = float(f.get("minNotional", 5.0))
    return result


def parse_entry_price(entry_zone: str) -> float | None:
    """解析 entry_zone 字符串（如 '3300 - 3350' 或 '$3325'），返回中点价格"""
    import re
    nums = re.findall(r'[\d]+(?:\.\d+)?', entry_zone.replace(',', ''))
    if len(nums) >= 2:
        return round((float(nums[0]) + float(nums[1])) / 2, 2)
    elif len(nums) == 1:
        return float(nums[0])
    return None


def execute_trade(signal: dict, dry_run: bool = True, max_usdt: float = 50.0):
    """
    根据 LLM 信号执行 PAXG/USDT 交易。

    安全锁（按顺序检查）：
      1. TRADING_HALT 文件
      2. 熔断检测（连续 N 笔止损）
      3. 每日限 1 笔实盘
      4. 实时余额校验
      5. 单笔上限 max_usdt
      6. Binance 最小下单量校验
    """
    action        = signal.get("action", "no_trade").lower()
    bias_score    = signal.get("bias_score", 0.0)
    entry_zone    = signal.get("entry_zone", "")
    stop_loss     = signal.get("stop_loss")
    profit_target = signal.get("profit_target")

    prefix = "[DRY RUN] " if dry_run else "[LIVE] "
    print(f"\n{prefix}交易执行器启动")
    print(f"  信号: {action}  |  bias_score: {bias_score}  |  entry_zone: {entry_zone}")

    if action == "no_trade":
        print("  -> no_trade 信号，不操作。")
        return

    # ── 安全检查（实盘才执行）──
    trade_log = load_trade_log()
    if not dry_run:
        if is_trading_halted():
            return
        if check_circuit_breaker(trade_log):
            return
        if not check_daily_limit(trade_log):
            return

    # ── 解析入场价 ──
    entry_price = parse_entry_price(entry_zone)
    if entry_price is None:
        print(f"  [警告] 无法解析 entry_zone: '{entry_zone}'，跳过。")
        return

    # ── 连接 Binance 或使用模拟数据 ──
    if not dry_run:
        if not BINANCE_API_KEY or not BINANCE_API_SECRET:
            print("  [错误] 未设置 BINANCE_API_KEY / BINANCE_API_SECRET 环境变量，无法交易。")
            return
        try:
            from binance.client import Client as BinanceClient
            client   = BinanceClient(BINANCE_API_KEY, BINANCE_API_SECRET)
            account  = client.get_account()
            balances = {b["asset"]: float(b["free"]) for b in account["balances"]
                        if b["asset"] in ("PAXG", "USDT")}
            filters  = get_symbol_filters(client, TRADING_SYMBOL)
        except Exception as e:
            print(f"  [错误] Binance 连接失败: {e}")
            return
    else:
        # 模拟账户余额
        balances = {"PAXG": 0.06, "USDT": 50.0}
        filters  = {"step_size": 0.0001, "tick_size": 0.01,
                    "min_qty": 0.0001, "min_notional": 5.0}

    paxg_free  = balances.get("PAXG", 0.0)
    usdt_free  = balances.get("USDT", 0.0)
    step_size  = filters["step_size"]
    tick_size  = filters["tick_size"]
    min_qty    = filters["min_qty"]
    min_notional = filters["min_notional"]

    print(f"  账户余额 -> PAXG: {paxg_free:.6f}  USDT: {usdt_free:.2f}")

    # ── 计算下单数量 ──
    if action == "long":
        side           = "BUY"
        usdt_to_use    = min(max_usdt, usdt_free)
        quantity       = _floor_to_step(usdt_to_use / entry_price, step_size)
        balance_ok     = usdt_free >= min_notional
        balance_desc   = f"USDT 可用 {usdt_free:.2f}"
    else:  # short = 现货卖出 PAXG
        side           = "SELL"
        paxg_to_sell   = _floor_to_step(min(max_usdt / entry_price, paxg_free), step_size)
        quantity       = paxg_to_sell
        balance_ok     = paxg_free >= min_qty
        balance_desc   = f"PAXG 可用 {paxg_free:.6f}"

    notional = quantity * entry_price

    print(f"  方向: {side}  数量: {quantity} PAXG  限价: {entry_price}  名义金额: ${notional:.2f}")
    print(f"  止损: {stop_loss}  目标: {profit_target}")

    # ── 最终验证 ──
    if not balance_ok:
        print(f"  [安全锁] 余额不足（{balance_desc}），跳过。")
        return
    if quantity < min_qty:
        print(f"  [安全锁] 数量 {quantity} < 最小值 {min_qty}，跳过。")
        return
    if notional < min_notional:
        print(f"  [安全锁] 名义金额 ${notional:.2f} < 最小值 ${min_notional}，跳过。")
        return

    # ── 下单 ──
    now_str   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_str = datetime.now().strftime("%Y-%m-%d")
    order_id  = None
    status    = "PENDING"

    if dry_run:
        order_id = f"DRY_{today_str}_{side}"
        status   = "DRY_RUN"
        print(f"\n  ===== 模拟订单（未真实下单）=====")
        print(f"  交易对  : {TRADING_SYMBOL}")
        print(f"  方向    : {side}")
        print(f"  类型    : LIMIT GTC")
        print(f"  数量    : {quantity} PAXG")
        print(f"  限价    : {entry_price}")
        print(f"  止损参考: {stop_loss}  （需手动在 Binance App 挂止损单）")
        print(f"  目标参考: {profit_target}")
        print(f"  ===================================")
        print(f"  [DRY RUN] 完成，未实际下单。确认逻辑正确后去掉 --dry-run 进行真实交易。")
    else:
        try:
            limit_price = _round_to_tick(entry_price, tick_size)
            from binance.client import Client as BinanceClient
            order = client.create_order(
                symbol      = TRADING_SYMBOL,
                side        = side,
                type        = "LIMIT",
                timeInForce = "GTC",
                quantity    = str(quantity),
                price       = str(limit_price),
            )
            order_id = str(order.get("orderId", ""))
            status   = order.get("status", "UNKNOWN")
            print(f"  [成功] 订单已提交: orderId={order_id}  status={status}")
            print(f"  [提醒] 止损价 {stop_loss} 请立即在 Binance App 手动设置止损单。")
            print(f"         路径: 现货订单 -> 找到此订单 -> 设置止损")
        except Exception as e:
            print(f"  [失败] 下单报错: {e}")
            status = f"ERROR: {e}"

    # ── 记录日志 ──
    append_trade_log({
        "timestamp":     now_str,
        "date":          today_str,
        "signal_action": action,
        "symbol":        TRADING_SYMBOL,
        "side":          side,
        "quantity":      quantity,
        "price":         entry_price,
        "stop_loss":     stop_loss,
        "profit_target": profit_target,
        "order_id":      order_id,
        "status":        status,
        "dry_run":       dry_run,
        "bias_score":    bias_score,
        "notes":         f"entry_zone={entry_zone}",
    })
    print(f"  [日志] 已记录到 {TRADE_LOG_PATH}")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="黄金分析脚本")
    parser.add_argument(
        "--api",
        action="store_true",
        help="直接调用 API 获取分析结果（默认：只生成提示词文件）",
    )
    parser.add_argument(
        "--model",
        default=ANTHROPIC_MODEL,
        help=(
            f"指定调用的模型 ID（需配合 --api 使用）。"
            f"Claude 模型示例: {', '.join(sorted(CLAUDE_MODELS))}；"
            f"DeepSeek 模型示例: {', '.join(sorted(DEEPSEEK_MODELS))}。"
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
                        help="第三模型，启用后自动切换为三模型投票制（例: gpt-4o）")
    parser.add_argument("--prefer-model",        default=ANTHROPIC_MODEL,
                        help=f"投票胜出时优先采用哪个模型的信号（默认: {ANTHROPIC_MODEL}）")
    parser.add_argument("--dual-bias-threshold", default=0.55, type=float,
                        help="触发确认模型/计票时的 bias 阈值（默认: 0.55）")
    parser.add_argument(
        "--trade",
        action="store_true",
        help="分析完成后自动执行 PAXG/USDT 交易（需配合 --api 使用）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="模拟交易：打印订单但不真实下单（需配合 --trade 使用）",
    )
    parser.add_argument(
        "--max-usdt",
        type=float,
        default=50.0,
        help="单笔最大 USDT 金额（默认 50，建议测试阶段保持 30-50）",
    )
    args = parser.parse_args()

    daily, weekly, monthly = fetch_gold_data()

    print("\n正在获取宏观跨资产数据...")
    macro = fetch_macro_data()

    print("\n正在获取 PAXG 链上黄金价格（CoinGecko）...")
    paxg = fetch_paxg_price()

    perf_metrics = load_perf_metrics()
    if perf_metrics:
        cl = perf_metrics.get("consecutive_losses", 0)
        print(f"\n已载入回测指标 — 胜率: {perf_metrics.get('win_rate')}  "
              f"盈利因子: {perf_metrics.get('profit_factor')}  "
              f"连续亏损: {cl} 次")
    else:
        print("\n未找到回测指标文件，将生成不含性能反馈的提示词")

    prompt = build_prompt(daily, weekly, monthly, perf_metrics=perf_metrics, macro=macro, paxg=paxg)

    # ── 方案一：保存提示词文件（默认）──
    output_path = "gold_prompt_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"\n提示词已保存到文件: {output_path}")

    if args.api:
        # ── 直接调用 API ──
        if args.dual_model and args.third_model:
            # 三模型投票
            analysis = call_voting_model_api(
                prompt,
                asset_name="GOLD",
                models=[args.screener_model, args.confirm_model, args.third_model],
                bias_threshold=args.dual_bias_threshold,
                prefer_model=args.prefer_model,
            )
        elif args.dual_model:
            # 双模型交叉验证
            analysis = call_dual_model_api(
                prompt,
                asset_name="GOLD",
                screener_model=args.screener_model,
                confirm_model=args.confirm_model,
                bias_threshold=args.dual_bias_threshold,
            )
        else:
            model = args.model
            analysis = _call_any_model(prompt, model)

        api_output_path = "gold_api_output.txt"
        with open(api_output_path, "w", encoding="utf-8") as f:
            f.write(analysis)
        print("\n" + "=" * 60)
        print(analysis)
        print("=" * 60)
        print(f"\nAPI 分析结果已保存到文件: {api_output_path}")

        # ── 交易执行 ──
        if args.trade:
            if not args.dry_run and args.max_usdt > 200:
                print(f"\n[安全锁] --max-usdt {args.max_usdt} 超过 200 上限，已强制设为 200。")
                args.max_usdt = 200.0
            parsed_signal = parse_signal(analysis)
            asset_signal  = extract_asset_signal(parsed_signal)
            if asset_signal:
                execute_trade(
                    signal   = asset_signal,
                    dry_run  = args.dry_run,
                    max_usdt = args.max_usdt,
                )
            else:
                print("\n[警告] 无法从 API 响应中解析 GOLD 交易信号，跳过交易。")
                print("  请检查 gold_api_output.txt 确认 LLM 输出格式是否正确。")
    else:
        # ── 只生成提示词文件，供手动使用 ──
        print("\n" + "=" * 60)
        print(prompt)
        print("=" * 60)
        print("\n请将上方内容复制粘贴到 Claude.ai 对话框，即可获得分析结果。")
        print("或使用 --api 参数直接调用 API，例如：")
        print("  python gold_analysis.py --api                                          # 只分析")
        print("  python gold_analysis.py --api --trade --dry-run                        # 模拟交易（推荐先用这个）")
        print("  python gold_analysis.py --api --trade --dry-run --max-usdt 30          # 模拟，限额 $30")
        print("  python gold_analysis.py --api --trade --max-usdt 30                    # 真实下单，限额 $30")
        print("  python gold_analysis.py --api --model deepseek-reasoner --trade --dry-run  # DeepSeek 模拟")


if __name__ == "__main__":
    main()
