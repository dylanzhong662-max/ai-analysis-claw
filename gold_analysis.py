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

# 支持的模型列表（用于 --model 参数提示）
CLAUDE_MODELS   = {"claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"}
DEEPSEEK_MODELS = {"deepseek-reasoner", "deepseek-chat"}

# 禁用 SSL 警告（企业网络/VPN 自签名证书环境）
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
    session = curl_requests.Session(impersonate="chrome", verify=False)
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    return session


def fetch_gold_data():
    ticker = "GC=F"
    print(f"正在获取黄金期货数据 ({ticker})...")

    session = _make_session()

    # 日线：6个月（延长以支持52周高低位和更长指标序列）
    daily = yf.download(
        ticker, period="6mo", interval="1d",
        auto_adjust=True, progress=False, session=session
    )
    # 周线：1年
    weekly = yf.download(
        ticker, period="1y", interval="1wk",
        auto_adjust=True, progress=False, session=session
    )

    if daily.empty or weekly.empty:
        raise ValueError("数据获取失败，请检查网络或 ticker 是否正确")

    print(f"日线数据：{len(daily)} 条  |  周线数据：{len(weekly)} 条")
    return daily, weekly


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

def build_prompt(daily: pd.DataFrame, weekly: pd.DataFrame,
                 perf_metrics: dict | None = None,
                 macro: dict | None = None,
                 paxg: dict | None = None) -> str:
    d_ind = compute_indicators(daily)
    w_ind = compute_indicators(weekly)

    close_d = daily['Close'].squeeze()
    current_price   = round(float(close_d.iloc[-1]), 2)
    current_ema20   = round(float(d_ind['ema20'].iloc[-1]), 2)
    current_macd    = round(float(d_ind['macd'].iloc[-1]), 2)
    current_rsi7    = round(float(d_ind['rsi7'].iloc[-1]), 2)
    current_rsi14   = round(float(d_ind['rsi14'].iloc[-1]), 2)

    last_price, day_chg, week_chg, price_5d_ago = weekly_change(daily)

    # 日线最新交易日 OHLCV
    last_day = daily.iloc[-1]
    today_open  = round(float(last_day['Open'].squeeze()), 2)
    today_high  = round(float(last_day['High'].squeeze()), 2)
    today_low   = round(float(last_day['Low'].squeeze()), 2)
    today_close = current_price
    today_vol   = int(last_day['Volume'].squeeze())

    # 日线序列（最近15条）
    n = 15
    daily_closes  = fmt_series(close_d, 2, n)
    daily_ema20   = fmt_series(d_ind['ema20'], 2, n)
    daily_macd    = fmt_series(d_ind['macd'], 2, n)
    daily_rsi7    = fmt_series(d_ind['rsi7'], 2, n)
    daily_rsi14   = fmt_series(d_ind['rsi14'], 2, n)

    # 周线序列（最近10条）
    close_w = weekly['Close'].squeeze()
    weekly_closes = fmt_series(close_w, 2, 10)
    weekly_macd   = fmt_series(w_ind['macd'], 2, 10)
    weekly_rsi14  = fmt_series(w_ind['rsi14'], 2, 10)

    # 4h 替代用日线长期指标
    ema20_4h  = round(float(d_ind['ema20'].iloc[-1]), 2)
    ema50_4h  = round(float(d_ind['ema50'].iloc[-1]), 2)
    atr3_4h   = round(float(d_ind['atr3'].iloc[-1]), 2)
    atr14_4h  = round(float(d_ind['atr14'].iloc[-1]), 2)

    vol_current = int(daily['Volume'].squeeze().iloc[-1])
    vol_avg     = int(daily['Volume'].squeeze().tail(20).mean())

    # ── 新增指标快照 ──
    stoch_k_val  = round(float(d_ind['stoch_k'].dropna().iloc[-1]), 1)
    stoch_d_val  = round(float(d_ind['stoch_d'].dropna().iloc[-1]), 1)
    adx_val      = round(float(d_ind['adx'].dropna().iloc[-1]), 1)
    plus_di_val  = round(float(d_ind['plus_di'].dropna().iloc[-1]), 1)
    minus_di_val = round(float(d_ind['minus_di'].dropna().iloc[-1]), 1)
    bb_pct_b_val = round(float(d_ind['bb_pct_b'].dropna().iloc[-1]), 3)
    bb_bw_val    = round(float(d_ind['bb_bw'].dropna().iloc[-1]), 2)
    bb_upper_val = round(float(d_ind['bb_upper'].dropna().iloc[-1]), 1)
    bb_lower_val = round(float(d_ind['bb_lower'].dropna().iloc[-1]), 1)
    roc10_val    = round(float(d_ind['roc10'].dropna().iloc[-1]), 2)
    roc20_val    = round(float(d_ind['roc20'].dropna().iloc[-1]), 2)

    # OBV 趋势（最近5日方向）
    obv_series = d_ind['obv'].dropna().tail(5).tolist()
    obv_trend  = "上升" if obv_series[-1] > obv_series[0] else "下降"

    # 52周高低位
    close_d_full  = daily['Close'].squeeze().dropna()
    high_52w = round(float(close_d_full.tail(252).max()), 1)
    low_52w  = round(float(close_d_full.tail(252).min()), 1)
    pct_from_high = round((current_price - high_52w) / high_52w * 100, 1)
    pct_from_low  = round((current_price - low_52w)  / low_52w  * 100, 1)

    # 日线序列扩展（Stochastic, ADX, ROC, BB%B）
    daily_stoch_k = fmt_series(d_ind['stoch_k'], 1, n)
    daily_stoch_d = fmt_series(d_ind['stoch_d'], 1, n)
    daily_adx     = fmt_series(d_ind['adx'],     1, n)
    daily_roc10   = fmt_series(d_ind['roc10'],   2, n)
    daily_bb_pctb = fmt_series(d_ind['bb_pct_b'], 3, n)

    # ── 宏观跨资产摘要 ──
    ms = summarize_macro(macro or {}, close_d_full) if macro is not None else summarize_macro({}, close_d_full)

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

    # ── 预计算入场锚点（强制 entry_zone 贴近当前价，解决 INVALID_RR 问题）──
    atr14_val = round(float(d_ind['atr14'].iloc[-1]), 2)
    macd_val  = round(float(d_ind['macd'].iloc[-1]), 2)
    rsi14_val = round(float(d_ind['rsi14'].iloc[-1]), 2)
    rsi7_val  = round(float(d_ind['rsi7'].iloc[-1]), 2)
    ema20_val = round(float(d_ind['ema20'].iloc[-1]), 2)

    long_entry_lo  = round(current_price - 0.25 * atr14_val, 1)
    long_entry_hi  = round(current_price + 0.25 * atr14_val, 1)
    long_stop      = round(current_price - 1.2 * atr14_val, 1)
    long_target    = round(current_price + 2.4 * atr14_val, 1)
    short_entry_lo = round(current_price - 0.25 * atr14_val, 1)
    short_entry_hi = round(current_price + 0.25 * atr14_val, 1)
    short_stop     = round(current_price + 1.2 * atr14_val, 1)
    short_target   = round(current_price - 2.4 * atr14_val, 1)

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
# 黄金 (XAU/USD) 宏观大宗商品分析请求
**分析日期**: {today_str}
**数据来源**: COMEX 黄金期货 (GC=F)
{perf_section}
---

## 价格概要

- **当前价格**: ${current_price}
- **今日 O/H/L/C**: {today_open} / {today_high} / {today_low} / {today_close}
- **今日涨跌幅**: {day_chg:+.2f}%
- **过去5交易日涨跌**: ${price_5d_ago:.2f} → ${last_price:.2f}  ({week_chg:+.2f}%)
- **今日成交量**: {today_vol:,}

---

## 当前技术指标快照

- current_price = {current_price}
- current_ema20 (日线) = {current_ema20}
- current_macd (日线) = {current_macd}
- current_rsi7 (日线) = {current_rsi7}
- current_rsi14 (日线) = {current_rsi14}

---

## 日线序列数据（最近 {n} 个交易日，**从旧到新排列**）

⚠️ 最后一个数值 = 最新数据

收盘价:   [{daily_closes}]
EMA-20:   [{daily_ema20}]
MACD:     [{daily_macd}]
RSI-7:    [{daily_rsi7}]
RSI-14:   [{daily_rsi14}]

---

## 周线序列数据（最近 10 周，**从旧到新排列**）

⚠️ 最后一个数值 = 最新数据

周收盘价: [{weekly_closes}]
MACD:     [{weekly_macd}]
RSI-14:   [{weekly_rsi14}]

---

## 长期趋势背景

20日EMA: {ema20_4h}  vs.  50日EMA: {ema50_4h}
ATR-3:   {atr3_4h}   vs.  ATR-14: {atr14_4h}
今日成交量: {vol_current:,}  vs.  20日均量: {vol_avg:,}

---

## 52周价格结构

- **52周高点**: {high_52w}  |  **距高点**: {pct_from_high:+.1f}%
- **52周低点**: {low_52w}   |  **距低点**: {pct_from_low:+.1f}%
- **当前价在布林带中的位置 (%B)**: {bb_pct_b_val:.3f}  （0=下轨，0.5=中轨，1=上轨，>1=突破上轨，<0=跌破下轨）
- **布林带上轨**: {bb_upper_val}  |  **下轨**: {bb_lower_val}  |  **带宽**: {bb_bw_val:.2f}%

---

## 高级技术指标快照（日线）

| 指标 | 当前值 | 信号解读 |
|------|--------|----------|
| Stochastic %K | {stoch_k_val} | {'超买区 >80' if stoch_k_val > 80 else ('超卖区 <20' if stoch_k_val < 20 else '中性区间')} |
| Stochastic %D | {stoch_d_val} | {'K>D 金叉' if stoch_k_val > stoch_d_val else 'K<D 死叉'} |
| ADX | {adx_val} | {'强趋势 >25' if adx_val > 25 else ('弱趋势 <20，市场振荡' if adx_val < 20 else '趋势形成中')} |
| +DI / -DI | {plus_di_val} / {minus_di_val} | {'+DI>-DI 多头主导' if plus_di_val > minus_di_val else '-DI>+DI 空头主导'} |
| ROC(10日) | {roc10_val:+.2f}% | {'正动量' if roc10_val > 0 else '负动量'} |
| ROC(20日) | {roc20_val:+.2f}% | {'正动量' if roc20_val > 0 else '负动量'} |
| OBV趋势(5日) | {obv_trend} | {'量价配合上涨' if obv_trend == '上升' else '量价背离下跌'} |

**近15日序列（从旧到新）**：
Stochastic %K: [{daily_stoch_k}]
Stochastic %D: [{daily_stoch_d}]
ADX:           [{daily_adx}]
ROC(10):       [{daily_roc10}]
BB %B:         [{daily_bb_pctb}]
{macro_section}
---

## 预计算入场锚点（基于当前价格与 ATR-14={atr14_val}）

> 这些数值由系统预先计算，**entry_zone 必须在此范围内**，否则将被判定为无效信号 (INVALID_RR)。

| 方向 | entry_zone 参考 | stop_loss 参考 | profit_target 参考 (2.0×R) |
|------|-----------------|----------------|----------------------------|
| 做多 | {long_entry_lo} – {long_entry_hi} | {long_stop} | {long_target} |
| 做空 | {short_entry_lo} – {short_entry_hi} | {short_stop} | {short_target} |

你可以在上述参考值基础上小幅调整（±0.3×ATR），但不得大幅偏离。

---

## 分析任务

请基于以上数据，按照大宗商品分析框架，完成以下任务：

1. **判断当前市场制度**（Trending / Mean-Reverting / Choppy）
2. **判断整体市场情绪**（Risk-On / Risk-Off / Neutral）
3. **分析 DXY 对黄金的潜在压力方向**
4. **针对黄金给出交易建议**，严格按以下 JSON 格式输出：

```json
{{
  "period": "Daily",
  "overall_market_sentiment": "Risk-On | Risk-Off | Neutral",
  "dxy_assessment": "<DXY 趋势及对黄金的影响>",
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
      "invalidation_condition": "<使该观点失效的具体信号>",
      "macro_catalyst": "<驱动此次行情的宏观逻辑>",
      "technical_setup": "<指标信号综合描述>",
      "justification": "<不超过300字的综合判断>"
    }}
  ]
}}
```

**硬性约束（违反任意一条必须改为 no_trade）**：
- entry_zone 必须包含当前价格 ±1×ATR-14 范围，不得设置脱离市场的理想价格
- profit_target 做多时必须高于 entry_zone 上限，做空时必须低于 entry_zone 下限
- risk_reward_ratio 必须 ≥ 2.0
- stop_loss 距离 entry 不得小于 0.8×ATR-14（避免被噪音止损）
- 当 action = no_trade 时，profit_target / stop_loss / risk_reward_ratio 填 null

**信号质量过滤规则（全部适用）**：
- 日线 MACD ({macd_val}) < 0 时，**禁止**在 Trending 制度下做多；可评估做空
- 日线 RSI-7 ({rsi7_val}) > 75 时，做多 bias_score 自动上限 0.55；RSI-7 < 25 时，做空 bias_score 自动上限 0.55
- 价格偏离 EMA-20 ({ema20_val}) 超过 3% 时，bias_score 上限 0.55（无论方向）
- Choppy 或 Mean-Reverting 制度下，bias_score 上限 0.55
- 当 bias_score < 0.50 时，一律输出 no_trade
- 多空均衡：价格低于 EMA-20 且 MACD 为负时，**必须认真评估做空机会**，不得默认 no_trade

**宏观因子使用规则**：
- DXY 趋势向上（价格>EMA20）时，做多黄金需额外降低 bias_score 0.05–0.10
- 10Y 收益率持续上升趋势时，做多黄金需额外降低 bias_score 0.05
- VIX > 25 时，Risk-Off 环境，黄金避险需求升级，可适当上调 bias_score 0.05
- ADX ({adx_val}) < 20：市场处于振荡，Trending 信号可靠性下降，降级为 Choppy
- ADX > 30：趋势强劲，可适当上调 bias_score 0.05
- Stochastic %K ({stoch_k_val}) > 80 且 %K < %D：超买死叉，做多信号降级
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
    """
    print(f"\n正在调用 Claude API（模型: {ANTHROPIC_MODEL}）...")
    client = Anthropic(
        base_url=ANTHROPIC_BASE_URL,
        api_key=ANTHROPIC_API_KEY,
        http_client=httpx.Client(verify=False),
    )
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
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
    args = parser.parse_args()

    daily, weekly = fetch_gold_data()

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

    prompt = build_prompt(daily, weekly, perf_metrics=perf_metrics, macro=macro, paxg=paxg)

    # ── 方案一：保存提示词文件（默认）──
    output_path = "gold_prompt_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"\n提示词已保存到文件: {output_path}")

    if args.api:
        # ── 直接调用 API（根据 --model 自动路由）──
        model = args.model
        if model in DEEPSEEK_MODELS:
            analysis = call_deepseek_api(prompt, model)
        else:
            # 默认走 Claude（含自定义 base_url 的代理场景）
            analysis = call_claude_api(prompt)

        api_output_path = "gold_api_output.txt"
        with open(api_output_path, "w", encoding="utf-8") as f:
            f.write(analysis)
        print("\n" + "=" * 60)
        print(analysis)
        print("=" * 60)
        print(f"\nAPI 分析结果已保存到文件: {api_output_path}")
    else:
        # ── 只生成提示词文件，供手动使用 ──
        print("\n" + "=" * 60)
        print(prompt)
        print("=" * 60)
        print("\n请将上方内容复制粘贴到 Claude.ai 对话框，即可获得分析结果。")
        print("或使用 --api 参数直接调用 API，例如：")
        print("  python gold_analysis.py --api                              # 默认 Claude")
        print("  python gold_analysis.py --api --model deepseek-reasoner    # DeepSeek R1")
        print("  python gold_analysis.py --api --model deepseek-chat        # DeepSeek Chat")


if __name__ == "__main__":
    main()
