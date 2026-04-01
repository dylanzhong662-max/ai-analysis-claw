"""
BTC 交易策略回测引擎
- 防时间泄漏：Prompt 不含具体日期
- 加密货币专用指标：减半周期阶段、资金费率代理、Fear&Greed 代理
- 支持阿里云 DashScope / 原生 DeepSeek 两种 API 端点

三种运行模式：
  【模式一】生成 Prompt 文件（无需 API Key）
    python3 btc_backtest_engine.py --generate --start 2024-01-01 --end 2024-12-31 --step 5

  【模式二】评估已有响应（无需 API Key）
    python3 btc_backtest_engine.py --evaluate

  【模式三】全自动回测（需要 API Key）
    python3 btc_backtest_engine.py --start 2024-01-01 --end 2024-12-31 --step 5 --model deepseek-r1
    python3 btc_backtest_engine.py --start 2025-01-01 --end 2025-12-31 --step 5 --resume

依赖：pip install yfinance pandas numpy openai curl_cffi urllib3
"""

import argparse
import json
import re
import time
import os
import tempfile
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd
import urllib3
import yfinance as yf
from curl_cffi import requests as curl_requests
from openai import OpenAI

yf.set_tz_cache_location(tempfile.mkdtemp())
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 复用 gold_analysis.py 的指标计算函数
from gold_analysis import (
    calc_ema, calc_macd, calc_rsi, calc_atr,
    calc_bollinger_bands, calc_stochastic, calc_adx, calc_obv, calc_roc,
    fmt_series, compute_indicators,
)

# ─────────────────────────────────────────────
# API 配置（优先使用阿里云 DashScope，否则 fallback 到 DeepSeek 官方）
# ─────────────────────────────────────────────

ALIYUN_API_KEY  = os.environ.get("ALIYUN_API_KEY", "")
ALIYUN_BASE_URL = os.environ.get("ALIYUN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "sk-9574b3366dfd41178a5493d0f6af33c0")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

def _get_api_client():
    """返回 (client, api_key来源描述)，优先使用阿里云。"""
    if ALIYUN_API_KEY:
        return OpenAI(api_key=ALIYUN_API_KEY, base_url=ALIYUN_BASE_URL), "阿里云DashScope"
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL), "DeepSeek官方"

# ─────────────────────────────────────────────
# 常量配置
# ─────────────────────────────────────────────

TICKER        = "BTC-USD"
OUTPUT_DIR    = Path("btc_backtest_results")
PROMPTS_DIR   = Path("btc_backtest_prompts")
RESPONSES_DIR = Path("btc_backtest_responses")
SIGNALS_FILE  = OUTPUT_DIR / "signals.csv"
PERF_FILE     = OUTPUT_DIR / "performance.csv"

EVAL_DAYS     = 60    # BTC 中长线持仓，捕捉完整趋势段（原20天太短，随机噪声主导）
LOOKBACK_DAYS = 200   # 向前取数据天数

# BTC 减半历史（用于计算周期阶段）
HALVING_DATES = [
    date(2012, 11, 28),
    date(2016, 7,  9),
    date(2020, 5,  11),
    date(2024, 4,  19),
]
NEXT_HALVING = date(2028, 4, 1)

# ─────────────────────────────────────────────
# 减半周期定位
# ─────────────────────────────────────────────

def _get_halving_phase(ref_date: date) -> dict:
    """计算给定日期处于哪个减半周期及阶段。"""
    past = [d for d in HALVING_DATES if d <= ref_date]
    if not past:
        return {"days_since_halving": None, "cycle_pct": None, "phase": "Pre-Genesis"}
    last_halving = max(past)
    next_halving = NEXT_HALVING if last_halving == HALVING_DATES[-1] else None
    if next_halving is None:
        idx = HALVING_DATES.index(last_halving)
        next_halving = HALVING_DATES[idx + 1] if idx + 1 < len(HALVING_DATES) else NEXT_HALVING

    days_since = (ref_date - last_halving).days
    cycle_len  = (next_halving - last_halving).days
    cycle_pct  = days_since / cycle_len * 100 if cycle_len > 0 else 0

    if cycle_pct < 20:
        phase = "Early-Bull (0-20%)"
    elif cycle_pct < 55:
        phase = "Mid-Bull (20-55%)"
    elif cycle_pct < 75:
        phase = "Late-Bull / Topping (55-75%)"
    elif cycle_pct < 90:
        phase = "Bear-Decline (75-90%)"
    else:
        phase = "Accumulation (90-100%)"

    return {
        "days_since_halving": days_since,
        "cycle_pct": round(cycle_pct, 1),
        "phase": phase,
        "last_halving": last_halving.strftime("%Y-%m-%d"),
        "next_halving_est": next_halving.strftime("%Y-%m-%d"),
    }


# ─────────────────────────────────────────────
# System Prompt（防时间泄漏版）
# ─────────────────────────────────────────────

BTC_SYSTEM_PROMPT = """# ROLE DEFINITION

You are a Senior Crypto Macro Strategist specializing in Bitcoin swing trading.

Your mission: Analyze Bitcoin (BTC/USD) to generate high-probability daily swing trading signals with disciplined risk management. You operate on a daily timeframe with expected holding periods of 3–20 days.

---

# BITCOIN TRADING ENVIRONMENT

- **Asset**: Bitcoin (BTC/USD)
- **Timeframe**: Daily (D1) for entry/exit; Weekly (W1) for trend bias
- **Holding Period**: 3 to 20 days
- **Objective**: Capture regime shifts and trend continuations — NOT scalp noise
- **Risk Profile**: BTC is 3–5× more volatile than gold. Stops must account for this.

---

# DATA INTERPRETATION GUIDELINES

## ⚠️ CRITICAL: DATA ORDERING

**ALL series are ordered: OLDEST → NEWEST. The LAST element = MOST RECENT data point.**

## Technical Indicators

- **EMA (20/50/200-day)**: Golden Cross (50>200) = Bull regime; Death Cross (50<200) = Bear
- **MACD**: Positive = bullish momentum; Negative = bearish; Histogram narrowing = exhaustion
- **RSI — Regime-Dependent**:
  - **Trending**: RSI >70 = momentum CONFIRMATION. BTC has held RSI >80 for weeks during major bull runs. RSI alone is NOT a sell signal in trending markets. Only act on RSI if you see clear **bearish divergence**.
  - **Mean-Reverting / Choppy**: RSI >70 = overbought, reversal risk. RSI <30 = oversold.
- **ATR-14**: BTC's ATR is often 3-8% of price. Use 1.2×ATR for stop placement.
- **Volume**: Rising price + rising volume = confirmed move. But crypto volume can be noisy.
- **Bollinger Bands**: %B > 1.0 = overextended. %B < 0 = oversold.

## BTC Macro Context

- **DXY**: Strong inverse correlation. DXY uptrend = BTC headwind
- **10Y Yield (TNX)**: Rising real yields compress risk assets including BTC
- **VIX**: High VIX = risk-off = BTC under pressure
- **ETH relative strength**: ETH often leads BTC in risk-on phases
- **Halving Cycle Phase**: The single most powerful multi-year framework. Note current cycle position.

---

# ANALYSIS FRAMEWORK

## 1. Halving Cycle Context (Primary Framework)
- Early-Bull (0-20%): Strongest buy bias; momentum typically accelerating
- Mid-Bull (20-55%): Continuation buys on dips; watch for parabolic extension
- Late-Bull / Topping (55-75%): Reduce long bias; watch for RSI divergence and volume exhaustion
- Bear-Decline (75-90%): Avoid longs; only short with high conviction
- Accumulation (90-100%): Begin rebuilding long bias; key support levels matter

## 2. Macro Context
- DXY direction (inverse BTC driver)
- 10Y yield trajectory (risk appetite signal)
- VIX level (fear/greed)

## 3. Technical Analysis
- EMA structure (Golden/Death Cross)
- MACD histogram trend
- RSI (regime-dependent)
- ATR for volatility-adjusted stops

## 4. Regime Classification

| Regime | Signals | Approach |
|--------|---------|----------|
| **Trending** | Price > EMA20 & EMA50, MACD +ve, EMA50 > EMA200 | Buy dips; ride trend; RSI overbought = confirmation |
| **Mean-Reverting** | Price oscillates around EMA, RSI at extremes | Fade extremes; tight stops |
| **Choppy** | Flat EMAs, MACD near zero, no catalyst | no_trade |

---

# ACTION SPACE

1. **long**: Bullish position
   - Trending: Golden Cross, MACD +ve, price above EMA50 — buy pullbacks
   - Mean-Reverting: RSI < 35, near key support, MACD turning up

2. **short**: Bearish position
   - Trending: Death Cross, MACD -ve, price below EMA50 — sell rallies
   - Mean-Reverting: RSI > 70, near key resistance, MACD turning down
   - ⚠️ BTC shorts are asymmetric — violent squeezes can exceed 20% in hours. Use only in clear bear regimes.

3. **no_trade**: No position
   - Choppy / unclear regime
   - Cannot achieve R:R ≥ 2.0 with stop ≥ 1.0×ATR-14 (larger than gold due to BTC volatility)
   - VIX > 35 (crisis mode)

---

# RISK MANAGEMENT (MANDATORY)

For EVERY long/short, you MUST specify:

1. **entry_zone**: Current price ± 0.5×ATR-14
2. **profit_target**: R:R ≥ 2.0 measured from **current_price**
3. **stop_loss**: ≥ 1.0×ATR-14 away from current_price (BTC needs wider stops than equities)
4. **risk_reward_ratio**: calculated from current_price

⚠️ MANDATORY SELF-CHECK:

| Check | Long | Short |
|-------|------|-------|
| Direction | profit_target > current_price > stop_loss | stop_loss > current_price > profit_target |
| R:R ≥ 2.0 | (target − current) / (current − stop) | (current − target) / (stop − current) |
| Stop ≥ 1.0×ATR | current − stop ≥ 1.0×ATR-14 | stop − current ≥ 1.0×ATR-14 |

**If ANY check fails → no_trade, profit_target/stop_loss/risk_reward_ratio = null**

---

# OUTPUT FORMAT (JSON)

Return ONLY valid JSON:

```json
{
  "period": "Daily",
  "overall_market_sentiment": "Risk-On" | "Risk-Off" | "Neutral",
  "dxy_assessment": "<DXY trend and BTC impact>",
  "halving_phase_assessment": "<how cycle phase affects directional bias>",
  "asset_analysis": [
    {
      "asset": "BTC",
      "regime": "Trending" | "Mean-Reverting" | "Choppy",
      "action": "long" | "short" | "no_trade",
      "bias_score": <float 0.0-1.0>,
      "entry_zone": "<price range>",
      "profit_target": <float | null>,
      "stop_loss": <float | null>,
      "risk_reward_ratio": <float | null>,
      "invalidation_condition": "<objective signal that voids thesis>",
      "macro_catalyst": "<key macro driver>",
      "technical_setup": "<indicator alignment>",
      "justification": "<max 300 characters>"
    }
  ]
}
```

**Validation**:
- Long: profit_target > current_price > stop_loss
- Short: stop_loss > current_price > profit_target
- R:R ≥ 2.0; stop ≥ 1.0×ATR-14; bias_score < 0.5 → no_trade
- no_trade → null for profit_target, stop_loss, risk_reward_ratio

---

# COMMON BTC PITFALLS

- ⚠️ **RSI paralysis**: BTC held RSI >80 for weeks in 2024 bull run. "Overbought" in a trending market = buy signal, not sell.
- ⚠️ **Shorting parabolic moves**: BTC can squeeze 30%+ in hours. Short only in clear bear regime.
- ⚠️ **Stop too tight**: BTC daily noise routinely exceeds 3-5%. Stops < 1×ATR-14 will be triggered by noise.
- ⚠️ **Fighting the cycle**: Buying during Bear-Decline phase is historically high-risk. In accumulation, add gradually.
- ⚠️ **Weekend gap risk**: BTC trades 24/7, but large weekend moves are common. Factor this into stop placement.

---

# FINAL INSTRUCTIONS

1. Check halving cycle phase first — it sets the structural bias
2. Assess macro backdrop (DXY, VIX, rates)
3. Classify BTC regime from EMA structure and MACD
4. Apply correct RSI interpretation based on regime
5. Calculate R:R from current_price; apply mandatory self-check
6. Output ONLY valid JSON — no commentary outside the JSON block

**CRITICAL**: Base analysis ONLY on provided data. Do NOT infer the specific date. Output ONLY valid JSON."""


# ─────────────────────────────────────────────
# 数据获取
# ─────────────────────────────────────────────

def _make_session():
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    return curl_requests.Session(impersonate="chrome", verify=False, proxy=proxy)


def _download_with_retry(ticker, start, end, interval, retries=3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            session = _make_session()
            df = yf.download(ticker, start=start, end=end,
                             interval=interval, auto_adjust=True, progress=False, session=session)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                print(f"  [数据获取失败] {ticker} {start}~{end}: {e}")
    return pd.DataFrame()


def fetch_data_up_to(ref_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """严格截断：只返回 ref_date 当日及之前的数据。"""
    end_dt    = pd.Timestamp(ref_date)
    end_str   = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    start_str = (end_dt - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    daily  = _download_with_retry(TICKER, start_str, end_str, "1d")
    weekly = _download_with_retry(TICKER, start_str, end_str, "1wk")
    return daily, weekly


def fetch_macro_for_date(ref_date: str) -> dict:
    """获取 BTC 宏观参考数据：ETH、QQQ、TNX、VIX、DXY。"""
    end_dt    = pd.Timestamp(ref_date)
    end_str   = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    start_str = (end_dt - timedelta(days=60)).strftime("%Y-%m-%d")
    macro = {}
    for key, ticker in [("eth", "ETH-USD"), ("qqq", "QQQ"), ("tnx", "^TNX"), ("vix", "^VIX"), ("dxy", "DX-Y.NYB")]:
        try:
            df = _download_with_retry(ticker, start_str, end_str, "1d")
            macro[key] = df if not df.empty else pd.DataFrame()
        except Exception:
            macro[key] = pd.DataFrame()
    return macro


def fetch_future_data(ref_date: str) -> pd.DataFrame:
    """获取 ref_date 之后的 BTC 数据，用于交易评估。"""
    start_str = (pd.Timestamp(ref_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    end_str   = (pd.Timestamp(ref_date) + timedelta(days=(EVAL_DAYS + 5) * 2)).strftime("%Y-%m-%d")
    df = _download_with_retry(TICKER, start_str, end_str, "1d")
    return df.iloc[:EVAL_DAYS + 5] if not df.empty else df


def get_trading_days(start: str, end: str, step: int) -> list[str]:
    """返回区间内每隔 step 个交易日的日期列表（带重试）。"""
    df = _download_with_retry(TICKER, start, end, "1d", retries=5)
    all_days = [d.strftime("%Y-%m-%d") for d in df.index]
    return all_days[::step]


# ─────────────────────────────────────────────
# 构建防泄漏 Prompt
# ─────────────────────────────────────────────

def _macro_summary(macro: dict, btc_close: pd.Series) -> dict:
    """将宏观 DataFrame 转为摘要字典。"""
    def _last(df, col, n=5):
        if df.empty or col not in df.columns:
            return []
        return df[col].squeeze().dropna().tail(n).round(3).tolist()

    def _trend(vals):
        if len(vals) < 2:
            return "N/A"
        chg = (vals[-1] - vals[0]) / abs(vals[0]) * 100 if vals[0] != 0 else 0
        return f"{'↑' if chg > 0 else '↓'} {abs(chg):.1f}% (5日)"

    result = {}

    # ETH
    eth_df = macro.get("eth", pd.DataFrame())
    eth_closes = _last(eth_df, "Close")
    result["eth_last"]  = round(eth_closes[-1], 2) if eth_closes else None
    result["eth_trend"] = _trend(eth_closes)

    # BTC/ETH relative strength
    if not eth_df.empty and "Close" in eth_df.columns and len(btc_close) > 5:
        eth_aligned = eth_df["Close"].squeeze().reindex(btc_close.index, method="ffill").dropna()
        btc_al = btc_close.reindex(eth_aligned.index).dropna()
        common = btc_al.index.intersection(eth_aligned.index)
        if len(common) >= 5:
            rs = (btc_al.loc[common] / eth_aligned.loc[common]).tail(5)
            rs_list = rs.round(4).tolist()
            result["btc_eth_rs"]       = rs_list
            result["btc_eth_rs_trend"] = _trend(rs_list)
        else:
            result["btc_eth_rs"] = []
            result["btc_eth_rs_trend"] = "N/A"
    else:
        result["btc_eth_rs"] = []
        result["btc_eth_rs_trend"] = "N/A"

    # TNX
    tnx_closes = _last(macro.get("tnx", pd.DataFrame()), "Close")
    result["tnx_last"]  = round(tnx_closes[-1], 3) if tnx_closes else None
    result["tnx_trend"] = _trend(tnx_closes)

    # VIX
    vix_closes = _last(macro.get("vix", pd.DataFrame()), "Close")
    result["vix_last"]  = round(vix_closes[-1], 2) if vix_closes else None
    result["vix_trend"] = _trend(vix_closes)
    if result["vix_last"]:
        v = result["vix_last"]
        result["vix_regime"] = "危机/恐慌" if v > 35 else ("高波动/Risk-Off" if v > 25 else ("中性" if v > 15 else "低波动/Risk-On"))
    else:
        result["vix_regime"] = "N/A"

    # DXY
    dxy_closes = _last(macro.get("dxy", pd.DataFrame()), "Close")
    result["dxy_last"]  = round(dxy_closes[-1], 2) if dxy_closes else None
    result["dxy_trend"] = _trend(dxy_closes)

    return result


def build_blind_prompt(daily: pd.DataFrame, weekly: pd.DataFrame,
                       macro: dict | None = None, ref_date_str: str = "") -> str:
    """构建防时间泄漏的 BTC 分析 Prompt。"""
    if daily.empty or weekly.empty or len(daily) < 30:
        return ""

    d_ind = compute_indicators(daily)
    w_ind = compute_indicators(weekly)

    close_d        = daily["Close"].squeeze()
    current_price  = round(float(close_d.iloc[-1]), 2)
    current_ema20  = round(float(d_ind["ema20"].iloc[-1]), 2)
    current_ema50  = round(float(d_ind["ema50"].iloc[-1]), 2)
    ema200_arr     = d_ind["ema200"].dropna()
    current_ema200 = round(float(ema200_arr.iloc[-1]), 2) if len(ema200_arr) > 0 else None
    current_macd   = round(float(d_ind["macd"].iloc[-1]), 2)
    current_rsi7   = round(float(d_ind["rsi7"].iloc[-1]), 2)
    current_rsi14  = round(float(d_ind["rsi14"].iloc[-1]), 2)
    atr14          = round(float(d_ind["atr14"].iloc[-1]), 2)
    atr3           = round(float(d_ind["atr3"].iloc[-1]), 2)

    last_close = float(close_d.iloc[-1])
    prev_close = float(close_d.iloc[-2]) if len(close_d) >= 2 else last_close
    close_5d   = float(close_d.iloc[-6]) if len(close_d) >= 6 else float(close_d.iloc[0])
    day_chg    = (last_close - prev_close) / prev_close * 100
    week_chg   = (last_close - close_5d) / close_5d * 100

    last_row   = daily.iloc[-1]
    today_open = round(float(last_row["Open"].squeeze()), 2)
    today_high = round(float(last_row["High"].squeeze()), 2)
    today_low  = round(float(last_row["Low"].squeeze()), 2)
    today_vol  = int(last_row["Volume"].squeeze())

    n = 15
    daily_closes = fmt_series(close_d, 2, n)
    daily_ema20  = fmt_series(d_ind["ema20"], 2, n)
    daily_ema50  = fmt_series(d_ind["ema50"], 2, n)
    daily_macd   = fmt_series(d_ind["macd"], 2, n)
    daily_rsi7   = fmt_series(d_ind["rsi7"], 2, n)
    daily_rsi14  = fmt_series(d_ind["rsi14"], 2, n)

    close_w       = weekly["Close"].squeeze()
    weekly_closes = fmt_series(close_w, 2, 10)
    weekly_macd   = fmt_series(w_ind["macd"], 2, 10)
    weekly_rsi14  = fmt_series(w_ind["rsi14"], 2, 10)

    vol_current = int(daily["Volume"].squeeze().iloc[-1])
    vol_avg     = int(daily["Volume"].squeeze().tail(20).mean())

    # 高级指标
    adx_val  = round(float(d_ind["adx"].dropna().iloc[-1]), 1)
    plus_di  = round(float(d_ind["plus_di"].dropna().iloc[-1]), 1)
    minus_di = round(float(d_ind["minus_di"].dropna().iloc[-1]), 1)
    bb_pctb  = round(float(d_ind["bb_pct_b"].dropna().iloc[-1]), 3)
    bb_bw    = round(float(d_ind["bb_bw"].dropna().iloc[-1]), 2)
    bb_upper = round(float(d_ind["bb_upper"].dropna().iloc[-1]), 2)
    bb_lower = round(float(d_ind["bb_lower"].dropna().iloc[-1]), 2)
    roc10    = round(float(d_ind["roc10"].dropna().iloc[-1]), 2)
    roc20    = round(float(d_ind["roc20"].dropna().iloc[-1]), 2)
    obv_arr  = d_ind["obv"].dropna().tail(5).tolist()
    obv_trend = "上升" if obv_arr[-1] > obv_arr[0] else "下降"

    close_full  = close_d.dropna()
    high_52w    = round(float(close_full.tail(252).max()), 2)
    low_52w     = round(float(close_full.tail(252).min()), 2)
    pct_high    = round((current_price - high_52w) / high_52w * 100, 1)
    pct_low     = round((current_price - low_52w)  / low_52w  * 100, 1)

    ema_str = "EMA20 > EMA50" if current_ema20 > current_ema50 else "EMA20 < EMA50"
    if current_ema200:
        ema_str  += f" {'>' if current_ema50 > current_ema200 else '<'} EMA200"
        cross_str = "Golden Cross (EMA50>EMA200)" if current_ema50 > current_ema200 else "Death Cross (EMA50<EMA200)"
    else:
        cross_str = "N/A"

    is_death_cross = (current_ema200 is not None and current_ema50 < current_ema200)

    # 预计算入场锚点（BTC 用 1.2×ATR 作为止损基准，因为日内波动更大）
    long_stop    = round(current_price - 1.2 * atr14, 2)
    long_target  = round(current_price + 3.0 * atr14, 2)
    short_stop   = round(current_price + 1.2 * atr14, 2)
    short_target = round(current_price - 3.0 * atr14, 2)

    # 减半周期
    try:
        halv = _get_halving_phase(pd.Timestamp(ref_date_str).date() if ref_date_str else date.today())
    except Exception:
        halv = {"days_since_halving": None, "cycle_pct": None, "phase": "N/A",
                "last_halving": "N/A", "next_halving_est": "N/A"}

    # 宏观摘要
    ms = _macro_summary(macro or {}, close_full)

    def _fv(v, u=""):
        return f"{v}{u}" if v is not None else "N/A"

    dc_warning = ""
    if is_death_cross and current_price < (current_ema200 or current_price):
        dc_warning = f"""
⚠️ **【Death Cross 过滤】** EMA50({current_ema50}) < EMA200({_fv(current_ema200)})，价格低于EMA200：
- 做多 bias_score 强制 ≤ 0.45（低于0.50门槛 → 自动 no_trade）
- 趋势未反转前，逢低买入属系统性亏损"""

    short_ban = ""
    if pct_high < -25:
        short_ban = f"\n⚠️ **【禁止追空】** 价格距52周高点已跌 {pct_high:.1f}%（超过25%），做空风险回报极差，强制 no_trade"

    prompt = f"""# BTC/USD 加密货币摆动交易分析请求
**数据来源**: Yahoo Finance (BTC-USD)
**重要说明**: 严格基于以下数据分析。不得引用数据窗口之外的任何具体事件。

---

## 价格概要

- **当前价格**: ${current_price:,}
- **今日 O/H/L/C**: {today_open:,} / {today_high:,} / {today_low:,} / {current_price:,}
- **今日涨跌幅**: {day_chg:+.2f}%
- **过去5交易日**: ${close_5d:,.2f} → ${last_close:,.2f}  ({week_chg:+.2f}%)
- **成交量**: {today_vol:,}  vs.  20日均量: {vol_avg:,}  ({'放量' if vol_current > vol_avg * 1.2 else ('缩量' if vol_current < vol_avg * 0.8 else '正常')})

---

## 减半周期定位

- **距上次减半**: {_fv(halv['days_since_halving'], '天')}  |  **周期进度**: {_fv(halv['cycle_pct'], '%')}
- **当前阶段**: {halv['phase']}
- **上次减半日期**: {halv['last_halving']}  |  **预计下次减半**: {halv['next_halving_est']}

---

## EMA 趋势结构

- **EMA20**: {current_ema20:,}  |  **EMA50**: {current_ema50:,}  |  **EMA200**: {_fv(current_ema200)}
- **排列**: {ema_str}  |  **均线状态**: {cross_str}

---

## 当前技术指标快照

- current_price = {current_price:,}
- ema20         = {current_ema20:,}
- ema50         = {current_ema50:,}
- ema200        = {_fv(current_ema200)}
- macd          = {current_macd}
- rsi7          = {current_rsi7}
- rsi14         = {current_rsi14}
- atr14         = {atr14:,}  （止损基准：1.0×ATR = {atr14:,}，1.2×ATR = {round(1.2*atr14,2):,}）

---

## 日线序列数据（最近 {n} 个交易日，从旧到新）

⚠️ 最后一个数值 = 最新数据

收盘价:   [{daily_closes}]
EMA-20:   [{daily_ema20}]
EMA-50:   [{daily_ema50}]
MACD:     [{daily_macd}]
RSI-7:    [{daily_rsi7}]
RSI-14:   [{daily_rsi14}]

---

## 周线序列数据（最近 10 周，从旧到新）

⚠️ 最后一个数值 = 最新数据

周收盘价: [{weekly_closes}]
MACD:     [{weekly_macd}]
RSI-14:   [{weekly_rsi14}]

---

## 52周价格结构

- **52周高点**: {high_52w:,}  |  **距高点**: {pct_high:+.1f}%
- **52周低点**: {low_52w:,}   |  **距低点**: {pct_low:+.1f}%
- **布林带 %B**: {bb_pctb:.3f}  （0=下轨, 0.5=中轨, 1=上轨）
- **布林带上轨**: {bb_upper:,}  |  **下轨**: {bb_lower:,}  |  **带宽**: {bb_bw:.2f}%

---

## 高级技术指标

| 指标 | 当前值 | 信号 |
|------|--------|------|
| ADX | {adx_val} | {'强趋势 >25' if adx_val > 25 else ('弱趋势 <20' if adx_val < 20 else '趋势形成')} |
| +DI / -DI | {plus_di} / {minus_di} | {'+DI>-DI 多头' if plus_di > minus_di else '-DI>+DI 空头'} |
| ROC(10日) | {roc10:+.2f}% | {'正动量' if roc10 > 0 else '负动量'} |
| ROC(20日) | {roc20:+.2f}% | {'正动量' if roc20 > 0 else '负动量'} |
| OBV趋势(5日) | {obv_trend} | {'量价配合' if obv_trend == '上升' else '量价背离'} |

---

## 宏观背景

### ETH / 加密市场
- **ETH 最新价**: {_fv(ms.get('eth_last'), ' USD')}  |  **5日趋势**: {ms.get('eth_trend', 'N/A')}
- **BTC/ETH 相对强度（5日）**: {ms.get('btc_eth_rs', [])}  |  **趋势**: {ms.get('btc_eth_rs_trend', 'N/A')}

### 宏观利率与风险环境
- **10Y 国债收益率**: {_fv(ms.get('tnx_last'), '%')}  |  **趋势**: {ms.get('tnx_trend', 'N/A')}
- **VIX 恐慌指数**: {_fv(ms.get('vix_last'))}  |  **状态**: {ms.get('vix_regime', 'N/A')}  |  **趋势**: {ms.get('vix_trend', 'N/A')}
- **美元指数 (DXY)**: {_fv(ms.get('dxy_last'))}  |  **趋势**: {ms.get('dxy_trend', 'N/A')}
  - {'⚠️ 强美元趋势 → BTC 面临逆风' if ms.get('dxy_trend', '').startswith('↑') else '弱美元/美元走平 → BTC 无汇率逆风'}

---

## 预计算入场锚点（基于 ATR-14={atr14:,}）

| 方向 | stop_loss | profit_target (≥3×R) |
|------|-----------|---------------------|
| 做多 | {long_stop:,} | {long_target:,} |
| 做空 | {short_stop:,} | {short_target:,} |

---

## 特殊过滤规则
{dc_warning}
{short_ban}

**通用规则**：
- bias_score < 0.50 → 强制 no_trade
- ADX < 20 → Trending 信号降级为 Choppy，bias_score 上限 0.45
- OBV 5日下降且价格上涨 → bias_score 降低 0.10（量价背离）
- VIX > 25 → 做多 bias_score 上限 0.60；VIX > 35 → 一律 no_trade
- DXY 持续上升 → 做多 bias_score 降低 0.05-0.10

---

## 分析任务

按照系统指令框架，分析 BTC/USD 当前形态与宏观背景，输出 JSON。

**硬性约束**：
- risk_reward_ratio ≥ 2.0（BTC 更高波动要求更高回报）
- stop_loss 距离 current_price ≥ 1.0×ATR-14 = {atr14:,}
- Long: profit_target > current_price > stop_loss
- Short: stop_loss > current_price > profit_target
- 违反任意一条 → 改为 no_trade""".strip()

    return prompt


# ─────────────────────────────────────────────
# JSON 解析
# ─────────────────────────────────────────────

def _extract_json_by_braces(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escape_next = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_signal(raw: str) -> dict:
    if not raw:
        return {}
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if code_block:
        candidate = code_block.group(1).strip()
        extracted = _extract_json_by_braces(candidate) or candidate
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass
    extracted = _extract_json_by_braces(raw)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass
    print(f"  [警告] JSON 解析失败，原始前200字：{raw[:200]}")
    return {}


# ─────────────────────────────────────────────
# API 调用
# ─────────────────────────────────────────────

def call_api(prompt: str, model: str, rate_limit: int = 20) -> dict:
    """调用 LLM API，含限速重试逻辑。"""
    client, source = _get_api_client()
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": BTC_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ]
            )
            return parse_signal(resp.choices[0].message.content)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower() or "quota" in err_str.lower():
                wait = max(rate_limit * 2, 60 * (attempt + 1))
                print(f"  [限速/{source}] 等待 {wait}s 后重试 (attempt {attempt+1}/3)...")
                time.sleep(wait)
            else:
                print(f"  [API错误/{source}] attempt {attempt+1}/3: {e}")
                if attempt < 2:
                    time.sleep(10)
    return {}


# ─────────────────────────────────────────────
# 交易模拟
# ─────────────────────────────────────────────

def simulate_trade(signal: dict, future_df: pd.DataFrame) -> dict:
    base = {"action": None, "entry_price": None, "exit_price": None,
            "exit_reason": "PENDING", "pnl_pct": None, "win": None, "days_held": None}

    if future_df.empty or not signal:
        base["exit_reason"] = "NO_DATA"
        return base

    asset_list  = signal.get("asset_analysis", [])
    btc_signal  = next((x for x in asset_list if x.get("asset") == "BTC"), None)
    if not btc_signal:
        base["exit_reason"] = "PARSE_ERROR"
        return base

    action        = btc_signal.get("action", "no_trade")
    profit_target = btc_signal.get("profit_target")
    stop_loss     = btc_signal.get("stop_loss")
    base["action"] = action

    if action == "no_trade":
        base["exit_reason"] = "NO_TRADE"
        return base

    if profit_target is None or stop_loss is None:
        base["exit_reason"] = "MISSING_LEVELS"
        return base

    profit_target = float(profit_target)
    stop_loss     = float(stop_loss)

    entry_price = float(future_df.iloc[0]["Open"].squeeze())
    base["entry_price"] = entry_price

    if action == "long":
        risk, reward = entry_price - stop_loss, profit_target - entry_price
    else:
        risk, reward = stop_loss - entry_price, entry_price - profit_target

    if risk <= 0 or reward <= 0 or (reward / risk) < 1.5:
        base["exit_reason"] = "INVALID_RR"
        return base

    for i, (_, row) in enumerate(future_df.iloc[:EVAL_DAYS].iterrows()):
        high  = float(row["High"].squeeze())
        low   = float(row["Low"].squeeze())
        close = float(row["Close"].squeeze())

        if action == "long":
            if low <= stop_loss:
                base.update(exit_price=stop_loss,     exit_reason="STOP_LOSS",   days_held=i + 1)
                break
            if high >= profit_target:
                base.update(exit_price=profit_target, exit_reason="TAKE_PROFIT", days_held=i + 1)
                break
        else:
            if high >= stop_loss:
                base.update(exit_price=stop_loss,     exit_reason="STOP_LOSS",   days_held=i + 1)
                break
            if low <= profit_target:
                base.update(exit_price=profit_target, exit_reason="TAKE_PROFIT", days_held=i + 1)
                break

        if i == EVAL_DAYS - 1:
            base.update(exit_price=close, exit_reason="TIMEOUT", days_held=i + 1)

    if base["exit_reason"] == "PENDING":
        last_close = float(future_df.iloc[-1]["Close"].squeeze())
        base.update(exit_price=last_close, exit_reason="TIMEOUT", days_held=len(future_df))

    if base["exit_price"] is not None:
        if action == "long":
            pnl = (base["exit_price"] - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - base["exit_price"]) / entry_price * 100
        base["pnl_pct"] = round(pnl, 4)
        base["win"]     = pnl > 0

    return base


# ─────────────────────────────────────────────
# 绩效统计
# ─────────────────────────────────────────────

def compute_performance(records: list[dict]) -> dict:
    df        = pd.DataFrame(records)
    traded    = df[df["action"].isin(["long", "short"])].copy()
    no_trade  = (df["action"] == "no_trade").sum()
    invalid_rr = (df["exit_reason"] == "INVALID_RR").sum()

    if traded.empty:
        return {"error": "无有效交易信号"}

    executed = traded[~traded["exit_reason"].isin(["INVALID_RR", "MISSING_LEVELS"])]
    wins     = executed[executed["win"] == True]
    losses   = executed[executed["win"] == False]

    win_rate      = len(wins) / len(executed) * 100 if len(executed) > 0 else 0
    avg_pnl       = executed["pnl_pct"].mean()     if not executed.empty else 0
    avg_win       = wins["pnl_pct"].mean()         if not wins.empty     else 0
    avg_loss      = losses["pnl_pct"].mean()       if not losses.empty   else 0
    total_profit  = wins["pnl_pct"].sum()
    total_loss    = abs(losses["pnl_pct"].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    cum    = executed["pnl_pct"].cumsum()
    max_dd = (cum - cum.cummax()).min() if not cum.empty else 0

    if "date" in df.columns and not executed.empty:
        ec = executed.copy()
        ec["month"] = pd.to_datetime(ec["date"]).dt.to_period("M")
        monthly = ec.groupby("month")["win"].mean() * 100
        monthly_str = "  |  ".join(f"{m}: {v:.0f}%" for m, v in monthly.items())
    else:
        monthly_str = "N/A"

    longs  = executed[executed["action"] == "long"]
    shorts = executed[executed["action"] == "short"]
    long_wr  = f"{len(longs[longs['win']==True]) / len(longs) * 100:.0f}%" if len(longs) > 0 else "N/A"
    short_wr = f"{len(shorts[shorts['win']==True]) / len(shorts) * 100:.0f}%" if len(shorts) > 0 else "N/A"

    return {
        "total_signals":   len(df),
        "traded_signals":  len(traded),
        "executed_trades": len(executed),
        "no_trade_cnt":    no_trade,
        "invalid_rr_cnt":  invalid_rr,
        "no_trade_rate":   f"{no_trade / len(df) * 100:.1f}%",
        "win_count":       len(wins),
        "loss_count":      len(losses),
        "win_rate":        f"{win_rate:.1f}%",
        "long_win_rate":   long_wr,
        "short_win_rate":  short_wr,
        "avg_pnl_pct":     f"{avg_pnl:.2f}%",
        "avg_win_pct":     f"{avg_win:.2f}%",
        "avg_loss_pct":    f"{avg_loss:.2f}%",
        "profit_factor":   f"{profit_factor:.2f}",
        "max_drawdown":    f"{max_dd:.2f}%",
        "total_return":    f"{executed['pnl_pct'].sum():.2f}%" if not executed.empty else "0.00%",
        "monthly_winrate": monthly_str,
    }


# ─────────────────────────────────────────────
# 模式一：生成 Prompt 文件
# ─────────────────────────────────────────────

def run_generate(start: str, end: str, step: int):
    PROMPTS_DIR.mkdir(exist_ok=True)
    RESPONSES_DIR.mkdir(exist_ok=True)
    trading_days = get_trading_days(start, end, step)
    print(f"共 {len(trading_days)} 个节点 → {PROMPTS_DIR}/\n")

    for i, d in enumerate(trading_days):
        out_path = PROMPTS_DIR / f"{d}.txt"
        if out_path.exists():
            print(f"[{i+1:>3}/{len(trading_days)}] {d}  已存在，跳过")
            continue
        daily, weekly = fetch_data_up_to(d)
        if daily.empty or len(daily) < 30:
            print(f"[{i+1:>3}/{len(trading_days)}] {d}  数据不足，跳过")
            continue
        macro  = fetch_macro_for_date(d)
        prompt = build_blind_prompt(daily, weekly, macro, d)
        if not prompt:
            continue
        out_path.write_text(prompt, encoding="utf-8")
        price = round(float(daily["Close"].squeeze().iloc[-1]), 2)
        print(f"[{i+1:>3}/{len(trading_days)}] {d}  BTC=${price:,}  → {out_path.name}")

    print(f"\n完成！将每个 .txt 粘贴到 LLM，把响应 JSON 保存到 {RESPONSES_DIR}/<日期>.json")
    print(f"完成后运行：python3 btc_backtest_engine.py --evaluate")


# ─────────────────────────────────────────────
# 模式二：评估已有响应
# ─────────────────────────────────────────────

def run_evaluate():
    OUTPUT_DIR.mkdir(exist_ok=True)
    files = sorted(RESPONSES_DIR.glob("*.json"))
    if not files:
        print(f"[错误] {RESPONSES_DIR}/ 下没有 .json 文件")
        return

    print(f"找到 {len(files)} 个响应文件，开始评估...\n")
    all_records = []

    for i, fpath in enumerate(files):
        d = fpath.stem
        print(f"[{i+1:>3}/{len(files)}] {d}", end="  ")
        try:
            signal = parse_signal(fpath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"读取失败: {e}")
            continue
        if not signal:
            print("JSON 解析失败，跳过")
            continue
        btc_sig = next((x for x in signal.get("asset_analysis", []) if x.get("asset") == "BTC"), None)
        if not btc_sig:
            print("未找到 BTC 信号，跳过")
            continue

        print(f"action={btc_sig.get('action')}  bias={btc_sig.get('bias_score')}", end="  ")
        future_df = fetch_future_data(d)
        trade = simulate_trade(signal, future_df)
        all_records.append({
            "date":        d,
            "action":      trade["action"],
            "entry_price": trade["entry_price"],
            "exit_price":  trade["exit_price"],
            "exit_reason": trade["exit_reason"],
            "pnl_pct":     trade["pnl_pct"],
            "win":         trade["win"],
            "days_held":   trade["days_held"],
            "bias_score":  btc_sig.get("bias_score"),
            "regime":      btc_sig.get("regime"),
            "sentiment":   signal.get("overall_market_sentiment"),
            "halving_phase": signal.get("halving_phase_assessment", ""),
            "raw_signal":  json.dumps(btc_sig, ensure_ascii=False),
        })
        print(f"→ {trade['exit_reason']}  pnl={trade['pnl_pct']}%")

    if not all_records:
        print("\n无有效记录。")
        return
    _save_and_print(all_records)


# ─────────────────────────────────────────────
# 模式三：全自动回测
# ─────────────────────────────────────────────

def run_backtest(start: str, end: str, step: int, model: str,
                 dry_run: bool, rate_limit: int = 20,
                 resume: bool = False, start_from: str = None):
    OUTPUT_DIR.mkdir(exist_ok=True)
    _, src = _get_api_client()
    print(f"BTC 回测: {start} ~ {end}  |  step={step}  |  model={model}  |  API={src}")
    print(f"评估窗口: {EVAL_DAYS} 天  |  rate_limit={rate_limit}s  |  dry_run={dry_run}")
    print("-" * 60)

    trading_days = get_trading_days(start, end, step)
    if start_from:
        trading_days = [d for d in trading_days if d >= start_from]

    existing_records, done_dates = [], set()
    if resume and SIGNALS_FILE.exists():
        existing_df    = pd.read_csv(SIGNALS_FILE)
        done_dates     = set(existing_df["date"].astype(str).tolist())
        existing_records = existing_df.to_dict("records")
        print(f"已加载 {len(done_dates)} 条现有记录")

    pending = [d for d in trading_days if d not in done_dates]
    print(f"共 {len(trading_days)} 个节点，待处理 {len(pending)} 个\n")
    all_records = list(existing_records)

    for i, d in enumerate(pending):
        print(f"[{i+1:>3}/{len(pending)}] {d}", end="  ")
        daily, weekly = fetch_data_up_to(d)
        if daily.empty or len(daily) < 30:
            print("-> 数据不足，跳过")
            continue

        macro  = fetch_macro_for_date(d)
        prompt = build_blind_prompt(daily, weekly, macro, d)
        if not prompt:
            print("-> prompt 构建失败，跳过")
            continue

        if dry_run:
            price = round(float(daily["Close"].squeeze().iloc[-1]), 2)
            print(f"-> [DRY RUN] BTC=${price:,}  prompt={len(prompt)}字符")
            continue

        signal = call_api(prompt, model, rate_limit)
        if not signal:
            print("-> 信号解析失败，跳过")
            continue

        btc_sig = next((x for x in signal.get("asset_analysis", []) if x.get("asset") == "BTC"), {})
        action  = btc_sig.get("action", "?")
        bias    = btc_sig.get("bias_score", "?")
        target  = btc_sig.get("profit_target")
        stop    = btc_sig.get("stop_loss")
        print(f"-> action={action}  bias={bias}  target={target}  stop={stop}", end="  ")

        future_df = fetch_future_data(d)
        trade = simulate_trade(signal, future_df)

        all_records.append({
            "date":        d,
            "action":      trade["action"],
            "entry_price": trade["entry_price"],
            "exit_price":  trade["exit_price"],
            "exit_reason": trade["exit_reason"],
            "pnl_pct":     trade["pnl_pct"],
            "win":         trade["win"],
            "days_held":   trade["days_held"],
            "bias_score":  btc_sig.get("bias_score"),
            "regime":      btc_sig.get("regime"),
            "sentiment":   signal.get("overall_market_sentiment"),
            "halving_phase": signal.get("halving_phase_assessment", ""),
            "raw_signal":  json.dumps(btc_sig, ensure_ascii=False),
        })
        print(f"-> {trade['exit_reason']}  pnl={trade['pnl_pct']}%")

        time.sleep(rate_limit)

    if not all_records:
        print("\n无有效记录（dry_run 或全部跳过）。")
        return
    _save_and_print(all_records)


def _save_and_print(records: list[dict]):
    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(SIGNALS_FILE, index=False, encoding="utf-8-sig")
    perf = compute_performance(records)
    print("\n" + "=" * 60)
    print("BTC 回测绩效汇总")
    print("=" * 60)
    for k, v in perf.items():
        if k != "monthly_winrate":
            print(f"  {k:<25}: {v}")
    print(f"\n  逐月胜率: {perf.get('monthly_winrate', 'N/A')}")
    pd.DataFrame([perf]).to_csv(PERF_FILE, index=False, encoding="utf-8-sig")
    print(f"\n信号明细 → {SIGNALS_FILE}")
    print(f"绩效汇总 → {PERF_FILE}")


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BTC 加密货币 LLM 交易策略回测引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 生成 Prompt 文件（无需 API Key）
  python3 btc_backtest_engine.py --generate --start 2024-01-01 --end 2024-12-31 --step 5

  # 评估已有响应文件
  python3 btc_backtest_engine.py --evaluate

  # 全自动回测（阿里云 DashScope DeepSeek R1）
  ALIYUN_API_KEY=sk-xxx python3 btc_backtest_engine.py --start 2024-01-01 --end 2024-12-31 --model deepseek-r1 --rate-limit 30

  # 2025 年全年回测，断点续跑
  ALIYUN_API_KEY=sk-xxx python3 btc_backtest_engine.py --start 2025-01-01 --end 2025-12-31 --model deepseek-r1 --resume
        """
    )
    parser.add_argument("--generate",   action="store_true",        help="生成 Prompt 文件（无需 API Key）")
    parser.add_argument("--evaluate",   action="store_true",        help="评估已有响应文件（无需 API Key）")
    parser.add_argument("--start",      default="2024-01-01",       help="回测开始日期 YYYY-MM-DD")
    parser.add_argument("--end",        default="2024-12-31",       help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--step",       default=5,    type=int,     help="每隔N个交易日触发一次（默认5）")
    parser.add_argument("--model",      default="deepseek-r1",      help="模型 ID（默认 deepseek-r1）")
    parser.add_argument("--rate-limit", default=20,   type=int,     help="API 调用间隔秒数（默认20）")
    parser.add_argument("--dry-run",    action="store_true",        help="只验证数据，不调用 API")
    parser.add_argument("--resume",     action="store_true",        help="跳过已完成节点，追加合并")
    parser.add_argument("--start-from", default=None,               help="从指定日期开始（YYYY-MM-DD）")
    args = parser.parse_args()

    if args.generate:
        run_generate(start=args.start, end=args.end, step=args.step)
    elif args.evaluate:
        run_evaluate()
    else:
        run_backtest(
            start      = args.start,
            end        = args.end,
            step       = args.step,
            model      = args.model,
            dry_run    = args.dry_run,
            rate_limit = args.rate_limit,
            resume     = args.resume,
            start_from = args.start_from,
        )
