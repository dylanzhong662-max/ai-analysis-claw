"""
科技股通用 LLM 回测引擎（--ticker 参数化）
- 支持：NVDA, MSFT, GOOGL, AAPL, META, AMZN 及任意 yfinance ticker
- 防时间泄漏：Prompt 不含具体日期
- 支持阿里云 DashScope / 原生 DeepSeek 两种 API 端点

三种运行模式：
  【模式一】生成 Prompt 文件（无需 API Key）
    python3 tech_backtest_engine.py --ticker NVDA --generate --start 2024-01-01 --end 2024-12-31

  【模式二】评估已有响应
    python3 tech_backtest_engine.py --ticker NVDA --evaluate

  【模式三】全自动回测
    ALIYUN_API_KEY=sk-xxx python3 tech_backtest_engine.py --ticker NVDA --start 2024-01-01 --end 2024-12-31 --model deepseek-r1

依赖：pip install yfinance pandas numpy openai curl_cffi urllib3
"""

import argparse
import json
import re
import time
import os
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import urllib3
import yfinance as yf
from curl_cffi import requests as curl_requests
from openai import OpenAI

yf.set_tz_cache_location(tempfile.mkdtemp())
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from gold_analysis import (
    calc_ema, calc_macd, calc_rsi, calc_atr,
    calc_bollinger_bands, calc_stochastic, calc_adx, calc_obv, calc_roc,
    fmt_series, compute_indicators,
)

# ─────────────────────────────────────────────
# API 配置（优先阿里云 DashScope）
# ─────────────────────────────────────────────

ALIYUN_API_KEY  = os.environ.get("ALIYUN_API_KEY", "")
ALIYUN_BASE_URL = os.environ.get("ALIYUN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "sk-9574b3366dfd41178a5493d0f6af33c0")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

def _get_api_client():
    if ALIYUN_API_KEY:
        return OpenAI(api_key=ALIYUN_API_KEY, base_url=ALIYUN_BASE_URL), "阿里云DashScope"
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL), "DeepSeek官方"

# ─────────────────────────────────────────────
# 每个 ticker 的专属行业背景（注入 System Prompt）
# ─────────────────────────────────────────────

TICKER_CONTEXT = {
    "NVDA": {
        "name": "NVIDIA Corporation",
        "sector": "Semiconductors / AI Infrastructure",
        "description": """
## Key Drivers for NVDA
1. **AI Accelerator Demand**: Datacenter GPU (H100/H200/Blackwell) shipment cadence; hyperscaler CapEx cycle
2. **Export Controls**: US restrictions on advanced chip exports to China create revenue uncertainty
3. **Competitive Moat**: CUDA ecosystem lock-in vs AMD MI300 series alternatives
4. **Inventory Cycles**: Crypto mining demand (cyclical), gaming GPU segment (consumer)
5. **Earnings Cadence**: Reports quarterly (late Jan, late May, late Aug, late Nov); guidance moves stock ±10%
6. **Valuation Premium**: High P/E requires sustained growth narrative — any deceleration signal = sharp correction""",
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "sector": "Cloud Computing / Enterprise Software / AI",
        "description": """
## Key Drivers for MSFT
1. **Azure Growth**: Cloud revenue growth rate vs. AWS/GCP — deceleration > 2pp = negative catalyst
2. **Copilot Monetization**: Microsoft 365 Copilot seat adds ($30/user/mo premium) — monetization velocity
3. **OpenAI Partnership**: $13B investment creates AI narrative upside; deprecation risk if OpenAI pivots
4. **Enterprise Stickiness**: O365, Teams, GitHub, Azure DevOps create high switching costs = defensive moat
5. **Earnings Cadence**: Reports quarterly (late Jan, late April, late July, late Oct)
6. **Dividend + Buyback**: Yield ~0.7%; consistent buybacks provide floor""",
    },
    "GOOGL": {
        "name": "Alphabet Inc. (Google)",
        "sector": "Digital Advertising / Cloud Computing / AI",
        "description": """
## Key Drivers for GOOGL
1. **Search Ad Revenue**: Core business; cyclically sensitive to macro slowdowns
2. **Google Cloud**: Margin expansion trajectory; GCP gaining share vs. AWS/Azure
3. **AI/Gemini**: Competitive response to ChatGPT/Copilot — search disruption risk is the bear case
4. **YouTube**: Ad monetization + YouTube Premium subscriptions; Reels competitor dynamics
5. **Regulatory Risk**: DOJ antitrust rulings on Search/Ad tech; potential structural remedies
6. **Earnings Cadence**: Reports quarterly (late Jan, late April, late July, late Oct)""",
    },
    "AAPL": {
        "name": "Apple Inc.",
        "sector": "Consumer Electronics / Software / Services",
        "description": """
## Key Drivers for AAPL
1. **iPhone Upgrade Cycle**: Annual September launch; China demand critical (30%+ of revenue)
2. **Services Revenue**: App Store, iCloud, Apple TV+, Apple Pay — highest-margin segment, ~25% revenue
3. **India Manufacturing**: China supply chain diversification via Foxconn India — geopolitical risk hedge
4. **AI Integration (Apple Intelligence)**: On-device AI features drive upgrade cycle for iPhone 16+
5. **Share Buybacks**: Largest buyback program globally (~$90B/year) provides consistent demand
6. **Earnings Cadence**: Reports quarterly (late Jan, early May, early Aug, early Nov)""",
    },
    "META": {
        "name": "Meta Platforms, Inc.",
        "sector": "Social Media / Digital Advertising / AR/VR",
        "description": """
## Key Drivers for META
1. **Advertising CPM**: Core revenue; extremely cyclically sensitive — ad budgets cut in economic downturns
2. **Reels Monetization**: Instagram/Facebook Reels driving engagement + ad load expansion
3. **AI Ad Targeting**: Advantage+ automated ads outperforming; recovering from iOS 14 ATT headwind
4. **Reality Labs**: Burning $15B+/year on AR/VR; Quest headsets gaining traction but not yet profitable
5. **Threads vs. X**: Social engagement competition for advertising time
6. **Earnings Cadence**: Reports quarterly (late Jan, late April, late July, late Oct)""",
    },
    "AMZN": {
        "name": "Amazon.com, Inc.",
        "sector": "E-Commerce / Cloud Computing / Advertising / Logistics",
        "description": """
## Key Drivers for AMZN
1. **AWS Growth**: Cloud segment (80%+ of operating profit); growth rate vs. Azure/GCP is key signal
2. **Retail Profitability**: Operating margin improvement in North America vs. International segment
3. **Advertising Revenue**: High-margin $50B+ business growing 20%+ YoY — hidden gem
4. **AI / Bedrock**: AWS AI services (Bedrock, Trainium chips) competing with Azure OpenAI Service
5. **Prime Ecosystem**: 200M+ Prime members; subscription + loyalty creates defensive retail moat
6. **Earnings Cadence**: Reports quarterly (late Jan, late April, late July, late Oct)""",
    },
}

DEFAULT_CONTEXT = {
    "name": "{ticker}",
    "sector": "Equity",
    "description": """
## Key Drivers
1. Sector momentum and relative strength vs. broad market (QQQ, SPY)
2. Earnings calendar and analyst estimate revisions
3. Rate environment sensitivity (growth vs. value positioning)
4. Technical trend structure (EMA Golden/Death Cross)
5. Volume-confirmed momentum signals""",
}


def _build_system_prompt(ticker: str) -> str:
    ctx = TICKER_CONTEXT.get(ticker.upper(), {**DEFAULT_CONTEXT, "name": ticker, "sector": "Equity"})
    name   = ctx["name"].replace("{ticker}", ticker)
    sector = ctx["sector"]
    extra  = ctx["description"].replace("{ticker}", ticker)

    return f"""# ROLE DEFINITION

You are a Senior Technology Equity Strategist specializing in medium-to-long-term position trading of large-cap US growth stocks.

Your mission: Analyze {name} ({ticker}) to generate high-probability medium-term signals with strict risk management and position sizing. Hold time: 2–5 weeks. Focus on QUALITY over QUANTITY — it is far better to output no_trade than to enter a marginal setup.

---

# TRADING ENVIRONMENT

- **Asset**: {ticker} ({name})
- **Sector**: {sector}
- **Timeframe**: Daily (D1) for entry/exit; Weekly (W1) for trend bias
- **Position Duration**: 2 to 5 weeks (medium-term only; no short-term scalps)
- **Capital Model**: Starting from 100% cash; each trade deploys a fraction of available capital based on conviction
{extra}

---

# DATA INTERPRETATION GUIDELINES

## ⚠️ CRITICAL: DATA ORDERING

**ALL series ordered: OLDEST → NEWEST. LAST element = MOST RECENT data point.**

## Technical Indicators

- **EMA (20/50/200-day)**: Golden Cross (50>200) = Bull; Death Cross (50<200) = Bear
- **MACD**: Positive = bullish; histogram narrowing = exhaustion
- **RSI — Regime-Dependent**:
  - **Trending**: RSI >70 = momentum CONFIRMATION. Tech stocks can sustain RSI >75 for months in bull runs. Only bearish divergence warrants caution.
  - **Mean-Reverting/Choppy**: RSI >70 = overbought; RSI <30 = oversold
- **ATR-14**: Sets stop distance; typical tech ATR = 2-5% of price
- **Volume**: Rising price + rising OBV = confirmed accumulation
- **QQQ RS**: {ticker}/QQQ ratio trend = alpha vs. sector

---

# ANALYSIS FRAMEWORK

## 1. Macro & Sector Context First
- **10Y Yield**: Rising yields > 4.5% compress growth multiples → reduce long bias
- **VIX**: >25 = risk-off, reduce longs; >35 = no_trade (wait for stabilization)
- **DXY**: Strong dollar = international revenue headwind for US multinationals
- **QQQ Trend**: Sector tide; fighting a QQQ downtrend requires high conviction

## 2. Regime Classification

| Regime | Signals | RSI Rule | Approach |
|--------|---------|----------|----------|
| **Trending** | Price > EMA20&50, MACD +ve, EMA50>EMA200 | RSI >70 = momentum; look for divergence only | Buy pullbacks; ride trend |
| **Mean-Reverting** | Price oscillates around EMA, RSI extreme | RSI >70 = overbought (fade); RSI <30 = oversold | Fade extremes; tight stops |
| **Choppy** | Flat EMAs, MACD near zero | RSI unhelpful | no_trade |

---

# ACTION SPACE

1. **long**: Bullish
   - Trending: Golden Cross, MACD +ve, price > EMA50 — buy dips
   - Mean-Reverting: RSI <35, near support, MACD turning up

2. **short**: Bearish — STRICT CONDITIONS REQUIRED (all three must be met):
   - ① QQQ Death Cross confirmed: QQQ EMA50 < QQQ EMA200
   - ② VIX > 20 (risk-off environment supporting downside)
   - ③ {ticker} price below EMA200 with bearish MACD
   - ⚠️ If ANY condition is missing → output no_trade, NOT short

3. **no_trade**: Choppy/unclear regime; cannot achieve R:R ≥ 2.5; VIX > 35; setup doesn't meet all entry criteria

---

# RISK MANAGEMENT (MANDATORY)

For EVERY long/short, specify:
1. **entry_zone**: current_price ± 0.5×ATR-14
2. **profit_target**: R:R ≥ 2.5 from current_price (gap + hold-period buffer)
3. **stop_loss**: ≥ 1.5×ATR-14 from current_price (wider stop for medium-term holds)
4. **risk_reward_ratio**: from current_price
5. **position_size_pct**: fraction of available cash to deploy (0.0–1.0), based on conviction and market conditions

## Position Sizing Rules (position_size_pct)

| Condition | Size |
|-----------|------|
| bias_score ≥ 0.80, Trending, RSI 45–65 | 0.80–1.00 (full) |
| bias_score 0.65–0.79, Trending | 0.50–0.70 |
| bias_score 0.50–0.64 | 0.30–0.50 |
| RSI-14 > 72 (chasing overbought) | **Reduce by 40–60%** regardless of bias |
| RSI-14 > 80 (extreme overbought) | **Max 0.25** — anti-chase rule |
| Price > EMA20 by > 5% | Reduce by 30% |
| VIX > 25 | Reduce by 30% |
| Mean-Reverting regime | Max 0.40 |

⚠️ MANDATORY SELF-CHECK:
- Long: profit_target > current_price > stop_loss; R:R ≥ 2.5; stop ≥ 1.5×ATR
- Short: stop_loss > current_price > profit_target; same R:R and stop rules
- If ANY fails → no_trade, set levels to null
- position_size_pct must reflect current RSI/regime/VIX conditions

---

# OUTPUT FORMAT (JSON)

Return ONLY valid JSON:

```json
{{
  "period": "Daily",
  "overall_market_sentiment": "Risk-On" | "Risk-Off" | "Neutral",
  "sector_assessment": "<QQQ trend and sector backdrop>",
  "rate_assessment": "<10Y yield direction and impact on {ticker} multiple>",
  "asset_analysis": [
    {{
      "asset": "{ticker}",
      "regime": "Trending" | "Mean-Reverting" | "Choppy",
      "action": "long" | "short" | "no_trade",
      "bias_score": <float 0.0-1.0>,
      "position_size_pct": <float 0.0-1.0>,
      "entry_zone": "<price range>",
      "profit_target": <float | null>,
      "stop_loss": <float | null>,
      "risk_reward_ratio": <float | null>,
      "estimated_holding_weeks": <int 2-5 | null>,
      "invalidation_condition": "<objective signal that voids thesis>",
      "macro_catalyst": "<key macro/sector driver>",
      "technical_setup": "<indicator alignment>",
      "justification": "<max 300 characters>"
    }}
  ]
}}
```

**Validation**:
- R:R ≥ 2.5; stop ≥ 1.5×ATR-14; bias_score < 0.5 → no_trade
- Long: target > price > stop; Short: stop > price > target
- short requires: QQQ EMA50 < EMA200 AND VIX > 20 AND price below EMA200

---

# COMMON PITFALLS

- ⚠️ **RSI paralysis**: {ticker} can sustain RSI >75 for weeks/months in trending markets. Never refuse a long solely because RSI is "high" in a Trending regime.
- ⚠️ **Anti-chase rule**: RSI-14 > 72 = reduce position_size_pct by 40–60%. RSI > 80 = max position_size_pct 0.25. Entering at cycle highs is the #1 cause of large losses.
- ⚠️ **Fighting rate headwinds**: Rapidly rising yields above 4.5% compress growth multiples.
- ⚠️ **Sector rotation**: If QQQ is in a clear downtrend, standalone {ticker} longs will struggle.
- ⚠️ **Death Cross trap**: When EMA50 < EMA200 and price below EMA200 — buying "support" leads to serial stop-outs. Force bias_score ≤ 0.45 (auto no_trade).
- ⚠️ **Short discipline**: Shorts only when QQQ EMA50 < EMA200 + VIX > 20 + price below EMA200. Do NOT short into oversold bounces.
- ⚠️ **Gap risk**: Tech stocks gap ±2-4% on macro news/earnings. Stop at 1.5×ATR provides buffer; target at 2.5×R minimum.

---

**CRITICAL**: Base analysis ONLY on provided data. Do NOT infer specific dates. Output ONLY valid JSON.

Now analyze the {ticker} market data provided below."""


# ─────────────────────────────────────────────
# 数据获取（含 Parquet 持久化缓存 + 内存缓存）
# ─────────────────────────────────────────────

# 内存缓存：(ticker, interval) -> 完整 DataFrame
_FULL_CACHE: dict[tuple, pd.DataFrame] = {}
# 持久化缓存目录（Parquet 文件）
_CACHE_DIR = Path("data_cache")

MACRO_TICKERS = [("qqq", "QQQ"), ("tnx", "^TNX"), ("vix", "^VIX"), ("dxy", "DX-Y.NYB")]


def _cache_path(ticker: str, interval: str) -> Path:
    safe = ticker.replace("^", "").replace("/", "_").replace("=", "")
    return _CACHE_DIR / f"{safe}_{interval}.parquet"


def _load_disk_cache(ticker: str, interval: str) -> pd.DataFrame:
    p = _cache_path(ticker, interval)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            p.unlink(missing_ok=True)  # 损坏的缓存文件直接删除
    return pd.DataFrame()


def _save_disk_cache(ticker: str, interval: str, df: pd.DataFrame):
    _CACHE_DIR.mkdir(exist_ok=True)
    df.to_parquet(_cache_path(ticker, interval))


def _make_session():
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    return curl_requests.Session(impersonate="chrome", verify=False, proxy=proxy)


def _download_with_retry(ticker, start, end, interval, retries=5) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            session = _make_session()
            df = yf.download(ticker, start=start, end=end,
                             interval=interval, auto_adjust=True, progress=False, session=session)
            if df is not None and not df.empty:
                return df
            # 返回空但未抛异常，短暂等待后重试
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print(f"  [数据获取失败] {ticker}: {e}")
    return pd.DataFrame()


def _cached_slice(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """优先从内存缓存切片；内存未命中则走磁盘缓存；均未命中才实时下载。"""
    key = (ticker, interval)
    if key in _FULL_CACHE:
        df = _FULL_CACHE[key]
        mask = (df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))
        return df.loc[mask].copy()
    return _download_with_retry(ticker, start, end, interval)


def prefetch_all_data(asset_ticker: str, start: str, end: str,
                      lookback: int = 200, eval_days: int = 15):
    """
    回测前一次性准备所有历史数据：
    1. 优先从 data_cache/*.parquet 读取本地缓存
    2. 如本地缓存覆盖范围不足，则从 Yahoo Finance 补充下载并更新本地文件
    覆盖范围：(start - lookback天) ~ (end + eval_days缓冲)
    """
    pre_start = pd.Timestamp(start) - timedelta(days=lookback + 30)
    post_end  = pd.Timestamp(end)   + timedelta(days=(eval_days + 5) * 2 + 30)
    pre_start_str = pre_start.strftime("%Y-%m-%d")
    post_end_str  = post_end.strftime("%Y-%m-%d")

    print(f"\n[预下载] 数据范围: {pre_start_str} ~ {post_end_str}")
    all_tickers = [(asset_ticker, "1d"), (asset_ticker, "1wk")] + \
                  [(t, "1d") for _, t in MACRO_TICKERS]

    for ticker, interval in all_tickers:
        key = (ticker, interval)
        if key in _FULL_CACHE:
            print(f"  {ticker:12s} [{interval}]  已在内存，跳过")
            continue

        # 尝试从磁盘加载
        disk_df = _load_disk_cache(ticker, interval)
        if not disk_df.empty:
            cached_start = disk_df.index.min()
            cached_end   = disk_df.index.max()
            need_start   = pre_start - timedelta(days=5)
            need_end     = post_end  + timedelta(days=5)
            if cached_start <= need_start and cached_end >= need_end:
                _FULL_CACHE[key] = disk_df
                print(f"  {ticker:12s} [{interval}]  本地缓存命中 ({len(disk_df)} 条，{cached_start.date()}~{cached_end.date()})")
                continue
            else:
                print(f"  {ticker:12s} [{interval}]  本地缓存范围不足，重新下载...", end=" ", flush=True)
        else:
            print(f"  {ticker:12s} [{interval}]  无本地缓存，下载中...", end=" ", flush=True)

        df = _download_with_retry(ticker, pre_start_str, post_end_str, interval, retries=5)
        if not df.empty:
            # 与磁盘已有数据合并（保留更宽范围）
            if not disk_df.empty:
                df = pd.concat([disk_df, df]).sort_index()
                df = df[~df.index.duplicated(keep="last")]
            _FULL_CACHE[key] = df
            _save_disk_cache(ticker, interval, df)
            print(f"✓  {len(df)} 条 → 已保存 {_cache_path(ticker, interval)}")
        else:
            if not disk_df.empty:
                _FULL_CACHE[key] = disk_df
                print(f"✗  下载失败，使用旧缓存 ({len(disk_df)} 条)")
            else:
                print("✗  下载失败，回测时将实时重试")
    print()


def fetch_data_up_to(ticker: str, ref_date: str, lookback: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    end_dt    = pd.Timestamp(ref_date)
    end_str   = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    start_str = (end_dt - timedelta(days=lookback)).strftime("%Y-%m-%d")
    daily  = _cached_slice(ticker, start_str, end_str, "1d")
    weekly = _cached_slice(ticker, start_str, end_str, "1wk")
    return daily, weekly


def fetch_macro_for_date(ref_date: str, asset_ticker: str) -> dict:
    end_dt    = pd.Timestamp(ref_date)
    end_str   = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    start_str = (end_dt - timedelta(days=60)).strftime("%Y-%m-%d")
    macro = {}
    for key, ticker in MACRO_TICKERS:
        try:
            df = _cached_slice(ticker, start_str, end_str, "1d")
            macro[key] = df if not df.empty else pd.DataFrame()
        except Exception:
            macro[key] = pd.DataFrame()
    return macro


def fetch_future_data(ticker: str, ref_date: str, eval_days: int) -> pd.DataFrame:
    start_str = (pd.Timestamp(ref_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    end_str   = (pd.Timestamp(ref_date) + timedelta(days=(eval_days + 5) * 2)).strftime("%Y-%m-%d")
    df = _cached_slice(ticker, start_str, end_str, "1d")
    return df.iloc[:eval_days + 5] if not df.empty else df


def get_trading_days(ticker: str, start: str, end: str, step: int) -> list[str]:
    """返回区间内每隔 step 个交易日的日期列表（带重试）。"""
    df = _cached_slice(ticker, start, end, "1d")
    if df.empty:
        df = _download_with_retry(ticker, start, end, "1d", retries=5)
    all_days = [d.strftime("%Y-%m-%d") for d in df.index]
    return all_days[::step]


# ─────────────────────────────────────────────
# 构建防泄漏 Prompt
# ─────────────────────────────────────────────

def _summarize_macro(macro: dict, stock_close: pd.Series, asset_ticker: str) -> dict:
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

    qqq_df = macro.get("qqq", pd.DataFrame())
    qqq_closes = _last(qqq_df, "Close")
    result["qqq_last"]  = round(qqq_closes[-1], 2) if qqq_closes else None
    result["qqq_trend"] = _trend(qqq_closes)
    if not qqq_df.empty and "Close" in qqq_df.columns:
        qqq_ema20 = calc_ema(qqq_df["Close"].squeeze(), 20).dropna()
        result["qqq_ema20"] = round(float(qqq_ema20.iloc[-1]), 2) if not qqq_ema20.empty else None
    else:
        result["qqq_ema20"] = None

    # Stock vs QQQ RS
    if not qqq_df.empty and "Close" in qqq_df.columns and len(stock_close) > 5:
        qqq_al  = qqq_df["Close"].squeeze().reindex(stock_close.index, method="ffill").dropna()
        stk_al  = stock_close.reindex(qqq_al.index).dropna()
        common  = stk_al.index.intersection(qqq_al.index)
        if len(common) >= 5:
            rs = (stk_al.loc[common] / qqq_al.loc[common]).tail(5)
            result["stock_qqq_rs"]       = rs.round(4).tolist()
            result["stock_qqq_rs_trend"] = _trend(rs.tolist())
        else:
            result["stock_qqq_rs"] = []
            result["stock_qqq_rs_trend"] = "N/A"
    else:
        result["stock_qqq_rs"] = []
        result["stock_qqq_rs_trend"] = "N/A"

    tnx_closes = _last(macro.get("tnx", pd.DataFrame()), "Close")
    result["tnx_last"]   = round(tnx_closes[-1], 3) if tnx_closes else None
    result["tnx_trend"]  = _trend(tnx_closes)
    result["tnx_series"] = tnx_closes

    vix_closes = _last(macro.get("vix", pd.DataFrame()), "Close")
    result["vix_last"]  = round(vix_closes[-1], 2) if vix_closes else None
    result["vix_trend"] = _trend(vix_closes)
    if result["vix_last"]:
        v = result["vix_last"]
        result["vix_regime"] = "危机/恐慌" if v > 35 else ("高波动/Risk-Off" if v > 25 else ("中性" if v > 15 else "低波动/Risk-On"))
    else:
        result["vix_regime"] = "N/A"

    dxy_closes = _last(macro.get("dxy", pd.DataFrame()), "Close")
    result["dxy_last"]  = round(dxy_closes[-1], 2) if dxy_closes else None
    result["dxy_trend"] = _trend(dxy_closes)

    return result


def build_blind_prompt(asset_ticker: str, daily: pd.DataFrame, weekly: pd.DataFrame,
                       macro: dict | None = None, perf_metrics: dict | None = None) -> str:
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
    current_macd   = round(float(d_ind["macd"].iloc[-1]), 4)
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
    daily_macd   = fmt_series(d_ind["macd"], 4, n)
    daily_rsi7   = fmt_series(d_ind["rsi7"], 2, n)
    daily_rsi14  = fmt_series(d_ind["rsi14"], 2, n)

    close_w       = weekly["Close"].squeeze()
    weekly_closes = fmt_series(close_w, 2, 10)
    weekly_macd   = fmt_series(w_ind["macd"], 4, 10)
    weekly_rsi14  = fmt_series(w_ind["rsi14"], 2, 10)

    vol_current = int(daily["Volume"].squeeze().iloc[-1])
    vol_avg     = int(daily["Volume"].squeeze().tail(20).mean())

    adx_val  = round(float(d_ind["adx"].dropna().iloc[-1]), 1)
    plus_di  = round(float(d_ind["plus_di"].dropna().iloc[-1]), 1)
    minus_di = round(float(d_ind["minus_di"].dropna().iloc[-1]), 1)
    bb_pctb  = round(float(d_ind["bb_pct_b"].dropna().iloc[-1]), 3)
    bb_bw    = round(float(d_ind["bb_bw"].dropna().iloc[-1]), 2)
    bb_upper = round(float(d_ind["bb_upper"].dropna().iloc[-1]), 2)
    bb_lower = round(float(d_ind["bb_lower"].dropna().iloc[-1]), 2)
    roc10    = round(float(d_ind["roc10"].dropna().iloc[-1]), 2)
    roc20    = round(float(d_ind["roc20"].dropna().iloc[-1]), 2)
    stoch_k  = round(float(d_ind["stoch_k"].dropna().iloc[-1]), 1)
    stoch_d  = round(float(d_ind["stoch_d"].dropna().iloc[-1]), 1)
    obv_arr  = d_ind["obv"].dropna().tail(5).tolist()
    obv_trend = "上升" if obv_arr[-1] > obv_arr[0] else "下降"

    close_full  = close_d.dropna()
    high_52w    = round(float(close_full.tail(252).max()), 2)
    low_52w     = round(float(close_full.tail(252).min()), 2)
    pct_high    = round((current_price - high_52w) / high_52w * 100, 1)
    pct_low     = round((current_price - low_52w)  / low_52w  * 100, 1)

    ema_str   = "EMA20 > EMA50" if current_ema20 > current_ema50 else "EMA20 < EMA50"
    if current_ema200:
        ema_str  += f" {'>' if current_ema50 > current_ema200 else '<'} EMA200"
        cross_str = "Golden Cross (EMA50>EMA200)" if current_ema50 > current_ema200 else "Death Cross (EMA50<EMA200)"
    else:
        cross_str = "N/A"

    is_death_cross = current_ema200 is not None and current_ema50 < current_ema200
    price_below_200 = current_ema200 is not None and current_price < current_ema200

    # 预计算入场锚点
    long_stop    = round(current_price - 1.2 * atr14, 2)
    long_target  = round(current_price + 3.5 * atr14, 2)
    short_stop   = round(current_price + 1.2 * atr14, 2)
    short_target = round(current_price - 3.5 * atr14, 2)

    # 宏观摘要
    ms = _summarize_macro(macro or {}, close_full, asset_ticker)

    def _fv(v, u=""):
        return f"{v}{u}" if v is not None else "N/A"

    # 动态过滤规则
    filter_rules = []
    if is_death_cross and price_below_200:
        pct_200 = round((current_price - current_ema200) / current_ema200 * 100, 1) if current_ema200 else None
        filter_rules.append(
            f"⚠️ **【Death Cross 过滤】** EMA50({current_ema50}) < EMA200({_fv(current_ema200)})，"
            f"价格 vs EMA200: {_fv(pct_200, '%')}：做多 bias_score 强制 ≤ 0.45 → 自动 no_trade。"
            f"趋势未反转前，逢低买入属系统性亏损。"
        )
    if pct_high < -20:
        filter_rules.append(
            f"⚠️ **【禁止追空】** 价格距52周高点已跌 {pct_high:.1f}%（>20%），"
            f"做空风险回报极差（轧空风险），强制 no_trade。"
        )

    perf_feedback = ""
    if perf_metrics:
        lines = []
        wr = perf_metrics.get("win_rate", 50)
        cl = perf_metrics.get("consecutive_losses", 0)
        if wr < 40:
            lines.append(f"⚠️ 历史绩效反馈：近期胜率 {wr:.1f}% < 40%，bias_score 门槛提升至 ≥ 0.65")
        if cl >= 2:
            lines.append(f"⚠️ 历史绩效反馈：连续亏损 {cl} 笔，需 bias_score ≥ 0.75 才入场")
        perf_feedback = "\n".join(lines)

    prompt = f"""# {asset_ticker} ({TICKER_CONTEXT.get(asset_ticker.upper(), DEFAULT_CONTEXT)['name'].replace('{ticker}', asset_ticker)}) 科技股摆动交易分析请求
**数据来源**: Yahoo Finance ({asset_ticker})
**重要说明**: 严格基于以下数据分析，不得引用数据窗口之外的具体事件。

---

## 价格概要

- **当前价格**: ${current_price}
- **今日 O/H/L/C**: {today_open} / {today_high} / {today_low} / {current_price}
- **今日涨跌幅**: {day_chg:+.2f}%
- **过去5交易日**: ${close_5d:.2f} → ${last_close:.2f}  ({week_chg:+.2f}%)
- **成交量**: {today_vol:,}  vs.  20日均量: {vol_avg:,}  ({'放量' if vol_current > vol_avg * 1.2 else ('缩量' if vol_current < vol_avg * 0.8 else '正常')})

---

## EMA 趋势结构

- **EMA20**: {current_ema20}  |  **EMA50**: {current_ema50}  |  **EMA200**: {_fv(current_ema200)}
- **排列**: {ema_str}  |  **均线状态**: {cross_str}

---

## 当前技术指标快照

- current_price = {current_price}
- ema20         = {current_ema20}
- ema50         = {current_ema50}
- ema200        = {_fv(current_ema200)}
- macd          = {current_macd}
- rsi7          = {current_rsi7}
- rsi14         = {current_rsi14}
- atr14         = {atr14}

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

- **52周高点**: {high_52w}  |  **距高点**: {pct_high:+.1f}%
- **52周低点**: {low_52w}   |  **距低点**: {pct_low:+.1f}%
- **布林带 %B**: {bb_pctb:.3f}  |  **上轨**: {bb_upper}  |  **下轨**: {bb_lower}  |  **带宽**: {bb_bw:.2f}%

---

## 高级技术指标

| 指标 | 当前值 | 信号 |
|------|--------|------|
| Stochastic %K/%D | {stoch_k} / {stoch_d} | {'K>D 金叉' if stoch_k > stoch_d else 'K<D 死叉'} |
| ADX | {adx_val} | {'强趋势 >25' if adx_val > 25 else ('弱趋势 <20' if adx_val < 20 else '趋势形成中')} |
| +DI / -DI | {plus_di} / {minus_di} | {'+DI>-DI 多头' if plus_di > minus_di else '-DI>+DI 空头'} |
| ROC(10日) | {roc10:+.2f}% | {'正动量' if roc10 > 0 else '负动量'} |
| ROC(20日) | {roc20:+.2f}% | {'正动量' if roc20 > 0 else '负动量'} |
| OBV趋势(5日) | {obv_trend} | {'量价配合' if obv_trend == '上升' else '量价背离'} |
| ATR-14 | {atr14} | 止损基准 |

---

## 宏观与板块背景

### Nasdaq-100 ETF (QQQ)
- **QQQ 最新价**: {_fv(ms.get('qqq_last'))}  |  **5日趋势**: {ms.get('qqq_trend', 'N/A')}  |  **QQQ EMA20**: {_fv(ms.get('qqq_ema20'))}
- **QQQ 状态**: {'QQQ高于EMA20 — 科技板块偏多' if ms.get('qqq_last') and ms.get('qqq_ema20') and ms.get('qqq_last') > ms.get('qqq_ema20') else 'QQQ低于EMA20 — 科技板块承压'}

### {asset_ticker} vs QQQ 相对强度
- **近5日 {asset_ticker}/QQQ 比率**: {ms.get('stock_qqq_rs', [])}
- **RS 趋势**: {ms.get('stock_qqq_rs_trend', 'N/A')}

### 宏观利率与风险环境
- **10Y 收益率**: {_fv(ms.get('tnx_last'), '%')}  |  **趋势**: {ms.get('tnx_trend', 'N/A')}
- {'⚠️ 收益率 > 4.5%，成长股估值承压' if ms.get('tnx_last') and ms.get('tnx_last') > 4.5 else '收益率处于合理范围'}
- **VIX**: {_fv(ms.get('vix_last'))}  |  **状态**: {ms.get('vix_regime', 'N/A')}  |  {'⚠️ VIX>25，谨慎做多' if ms.get('vix_last') and ms.get('vix_last') > 25 else '市场相对平静'}
- **DXY**: {_fv(ms.get('dxy_last'))}  |  **趋势**: {ms.get('dxy_trend', 'N/A')}

---

## 预计算入场锚点（ATR-14={atr14}）

| 方向 | stop_loss | profit_target (≥3.5×ATR) |
|------|-----------|--------------------------|
| 做多 | {long_stop} | {long_target} |
| 做空 | {short_stop} | {short_target} |

---

## 特殊过滤规则

{chr(10).join(filter_rules) if filter_rules else '当前无特殊过滤规则触发'}

**通用规则**：
- bias_score < 0.50 → 强制 no_trade
- ADX < 20 → Trending 信号降级为 Choppy，bias_score 上限 0.45
- OBV 5日下降且价格上涨 → bias_score 降低 0.10
- MACD ({current_macd}) < 0 且 EMA20 < EMA50 → 禁止在 Trending 制度做多
- VIX > 25 → 做多 bias_score 上限 0.60；VIX > 35 → 一律 no_trade
- RSI-7 ({current_rsi7}) > 82 且价格 > EMA20+2% → 做多 bias_score 上限 0.65
{perf_feedback}

---

## 分析任务

按照系统指令框架，分析 {asset_ticker} 当前技术形态与宏观背景，输出 JSON。资产名称必须为 "{asset_ticker}"。

**硬性约束**：
- risk_reward_ratio ≥ 2.0（推荐 2.5 以缓解次日跳空）
- stop_loss 距 current_price ≥ 0.8×ATR-14 = {round(0.8*atr14, 2)}
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

def call_api(prompt: str, model: str, system_prompt: str, rate_limit: int = 20) -> dict:
    client, source = _get_api_client()
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": system_prompt},
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

def _calc_position_size(bias_score: float, regime: str, position_size_pct_llm=None) -> float:
    """
    计算实际入场仓位占比（0.0–1.0）。
    优先使用 LLM 输出的 position_size_pct，若缺失则根据 bias + regime 计算兜底值。
    """
    if position_size_pct_llm is not None:
        try:
            return round(min(1.0, max(0.0, float(position_size_pct_llm))), 2)
        except (ValueError, TypeError):
            pass

    bias = float(bias_score) if bias_score is not None else 0.5
    # Base: bias 0.5→0.0, 0.75→0.5, 1.0→1.0
    base = min(1.0, max(0.0, (bias - 0.5) * 2))

    regime_mult = {
        "Trending": 1.0, "Trending-Up": 1.0, "Trending-Down": 0.8,
        "Mean-Reverting": 0.5, "Consolidation": 0.5, "Choppy": 0.3,
    }.get(regime, 0.7)

    return round(min(1.0, max(0.1, base * regime_mult)), 2)


def simulate_trade(signal: dict, future_df: pd.DataFrame, asset_ticker: str, eval_days: int,
                   macro: dict = None) -> dict:
    base = {"action": None, "entry_price": None, "exit_price": None,
            "exit_reason": "PENDING", "pnl_pct": None, "win": None,
            "days_held": None, "position_size": None, "portfolio_pnl": None}

    if future_df.empty or not signal:
        base["exit_reason"] = "NO_DATA"
        return base

    asset_list = signal.get("asset_analysis", [])
    sig = next((x for x in asset_list if x.get("asset") == asset_ticker), None)
    if not sig:
        sig = next((x for x in asset_list if x.get("asset", "").upper() == asset_ticker.upper()), None)
    if not sig:
        base["exit_reason"] = "PARSE_ERROR"
        return base

    action        = sig.get("action", "no_trade")
    profit_target = sig.get("profit_target")
    stop_loss     = sig.get("stop_loss")
    bias_score    = sig.get("bias_score", 0.5)
    regime        = sig.get("regime", "Trending")
    base["action"] = action

    if action == "no_trade":
        base["exit_reason"] = "NO_TRADE"
        return base

    # ── 逆势做空过滤：QQQ 未死叉时禁止做空 ──────────────────────
    if action == "short" and macro is not None:
        qqq_df = macro.get("qqq", pd.DataFrame())
        vix_df = macro.get("vix", pd.DataFrame())
        qqq_in_downtrend = False
        if not qqq_df.empty and "Close" in qqq_df.columns:
            qqq_close = qqq_df["Close"].squeeze().dropna()
            ema50  = calc_ema(qqq_close, 50).dropna()
            ema200 = calc_ema(qqq_close, 200).dropna()
            if len(ema50) > 0 and len(ema200) > 0:
                qqq_in_downtrend = float(ema50.iloc[-1]) < float(ema200.iloc[-1])
        vix_elevated = False
        if not vix_df.empty and "Close" in vix_df.columns:
            vix_val = vix_df["Close"].squeeze().dropna()
            if len(vix_val) > 0:
                vix_elevated = float(vix_val.iloc[-1]) > 20.0
        if not (qqq_in_downtrend and vix_elevated):
            base["exit_reason"] = "SHORT_FILTERED"
            base["action"] = "no_trade"
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

    # ── 仓位计算 ──────────────────────────────────────────────────
    pos_size = _calc_position_size(bias_score, regime, sig.get("position_size_pct"))

    for i, (_, row) in enumerate(future_df.iloc[:eval_days].iterrows()):
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

        if i == eval_days - 1:
            base.update(exit_price=close, exit_reason="TIMEOUT", days_held=i + 1)

    if base["exit_reason"] == "PENDING":
        last_close = float(future_df.iloc[-1]["Close"].squeeze())
        base.update(exit_price=last_close, exit_reason="TIMEOUT", days_held=len(future_df))

    if base["exit_price"] is not None:
        if action == "long":
            pnl = (base["exit_price"] - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - base["exit_price"]) / entry_price * 100
        base["pnl_pct"]       = round(pnl, 4)
        base["win"]           = pnl > 0
        base["position_size"] = pos_size
        base["portfolio_pnl"] = round(pnl * pos_size, 4)  # 仓位加权收益

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
# 历史绩效反馈（从 signals.csv 读取）
# ─────────────────────────────────────────────

def load_perf_metrics(signals_file: Path, perf_file: Path) -> dict:
    perf = {}
    try:
        if perf_file.exists():
            df = pd.read_csv(perf_file)
            if not df.empty:
                row = df.iloc[0]
                perf["win_rate"]    = float(str(row.get("win_rate", "50%")).replace("%", ""))
                perf["max_drawdown"] = float(str(row.get("max_drawdown", "0%")).replace("%", ""))
    except Exception:
        pass
    try:
        if signals_file.exists():
            df = pd.read_csv(signals_file)
            executed = df[df["exit_reason"].isin(["STOP_LOSS", "TAKE_PROFIT", "TIMEOUT"])].tail(10)
            if not executed.empty:
                wins = executed["win"].tolist()
                consec = 0
                for w in reversed(wins):
                    if w is False or str(w).lower() == "false":
                        consec += 1
                    else:
                        break
                perf["consecutive_losses"] = consec
    except Exception:
        perf["consecutive_losses"] = 0
    return perf


# ─────────────────────────────────────────────
# 三种运行模式
# ─────────────────────────────────────────────

def run_generate(asset_ticker: str, start: str, end: str, step: int, eval_days: int):
    output_dir    = Path(f"{asset_ticker.lower()}_backtest_results")
    prompts_dir   = Path(f"{asset_ticker.lower()}_backtest_prompts")
    responses_dir = Path(f"{asset_ticker.lower()}_backtest_responses")
    prompts_dir.mkdir(exist_ok=True)
    responses_dir.mkdir(exist_ok=True)

    system_prompt = _build_system_prompt(asset_ticker)
    trading_days  = get_trading_days(asset_ticker, start, end, step)
    perf_metrics  = load_perf_metrics(output_dir / "signals.csv", output_dir / "performance.csv")
    print(f"[{asset_ticker}] 共 {len(trading_days)} 个节点 → {prompts_dir}/\n")

    for i, d in enumerate(trading_days):
        out_path = prompts_dir / f"{d}.txt"
        if out_path.exists():
            print(f"[{i+1:>3}/{len(trading_days)}] {d}  已存在，跳过")
            continue
        daily, weekly = fetch_data_up_to(asset_ticker, d)
        if daily.empty or len(daily) < 30:
            print(f"[{i+1:>3}/{len(trading_days)}] {d}  数据不足，跳过")
            continue
        macro  = fetch_macro_for_date(d, asset_ticker)
        prompt = build_blind_prompt(asset_ticker, daily, weekly, macro, perf_metrics)
        if not prompt:
            continue
        out_path.write_text(prompt, encoding="utf-8")
        price = round(float(daily["Close"].squeeze().iloc[-1]), 2)
        print(f"[{i+1:>3}/{len(trading_days)}] {d}  {asset_ticker}=${price}  → {out_path.name}")

    print(f"\n完成！将 .txt 粘贴到 LLM，把响应 JSON 保存到 {responses_dir}/<日期>.json")


def run_evaluate(asset_ticker: str, eval_days: int):
    output_dir    = Path(f"{asset_ticker.lower()}_backtest_results")
    responses_dir = Path(f"{asset_ticker.lower()}_backtest_responses")
    signals_file  = output_dir / "signals.csv"
    perf_file     = output_dir / "performance.csv"
    output_dir.mkdir(exist_ok=True)

    files = sorted(responses_dir.glob("*.json"))
    if not files:
        print(f"[错误] {responses_dir}/ 下没有 .json 文件")
        return

    print(f"找到 {len(files)} 个响应文件...\n")
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
        asset_list = signal.get("asset_analysis", [])
        sig = next((x for x in asset_list if x.get("asset", "").upper() == asset_ticker.upper()), None)
        if not sig:
            print(f"未找到 {asset_ticker} 信号，跳过")
            continue
        print(f"action={sig.get('action')}  bias={sig.get('bias_score')}", end="  ")
        future_df = fetch_future_data(asset_ticker, d, eval_days)
        trade = simulate_trade(signal, future_df, asset_ticker, eval_days)
        all_records.append({
            "date": d, "action": trade["action"], "entry_price": trade["entry_price"],
            "exit_price": trade["exit_price"], "exit_reason": trade["exit_reason"],
            "pnl_pct": trade["pnl_pct"], "win": trade["win"], "days_held": trade["days_held"],
            "bias_score": sig.get("bias_score"), "regime": sig.get("regime"),
            "sentiment": signal.get("overall_market_sentiment"),
            "raw_signal": json.dumps(sig, ensure_ascii=False),
        })
        print(f"→ {trade['exit_reason']}  pnl={trade['pnl_pct']}%")

    if not all_records:
        print("\n无有效记录。")
        return
    _save_and_print(asset_ticker, all_records, signals_file, perf_file)


def run_backtest(asset_ticker: str, start: str, end: str, step: int, model: str,
                 eval_days: int, dry_run: bool, rate_limit: int = 20,
                 resume: bool = False, start_from: str = None):
    output_dir   = Path(f"{asset_ticker.lower()}_backtest_results")
    signals_file = output_dir / "signals.csv"
    perf_file    = output_dir / "performance.csv"
    output_dir.mkdir(exist_ok=True)

    _, src = _get_api_client()
    system_prompt = _build_system_prompt(asset_ticker)
    print(f"[{asset_ticker}] 回测: {start} ~ {end}  |  step={step}  |  model={model}  |  API={src}")
    print(f"评估窗口: {eval_days} 天  |  rate_limit={rate_limit}s  |  dry_run={dry_run}")
    print("-" * 60)

    # 预下载所有历史数据到内存缓存，避免逐节点重复拉取
    prefetch_all_data(asset_ticker, start, end, lookback=200, eval_days=eval_days)

    perf_metrics  = load_perf_metrics(signals_file, perf_file)
    trading_days  = get_trading_days(asset_ticker, start, end, step)
    if start_from:
        trading_days = [d for d in trading_days if d >= start_from]

    existing_records, done_dates = [], set()
    if resume and signals_file.exists():
        existing_df    = pd.read_csv(signals_file)
        done_dates     = set(existing_df["date"].astype(str).tolist())
        existing_records = existing_df.to_dict("records")
        print(f"已加载 {len(done_dates)} 条现有记录")

    pending = [d for d in trading_days if d not in done_dates]
    print(f"共 {len(trading_days)} 个节点，待处理 {len(pending)} 个\n")
    all_records = list(existing_records)

    for i, d in enumerate(pending):
        print(f"[{i+1:>3}/{len(pending)}] {d}", end="  ")
        daily, weekly = fetch_data_up_to(asset_ticker, d)
        if daily.empty or len(daily) < 30:
            print("-> 数据不足，跳过")
            continue

        macro  = fetch_macro_for_date(d, asset_ticker)
        prompt = build_blind_prompt(asset_ticker, daily, weekly, macro, perf_metrics)
        if not prompt:
            print("-> prompt 构建失败，跳过")
            continue

        if dry_run:
            price = round(float(daily["Close"].squeeze().iloc[-1]), 2)
            print(f"-> [DRY RUN] {asset_ticker}=${price}  prompt={len(prompt)}字符")
            continue

        signal = call_api(prompt, model, system_prompt, rate_limit)
        if not signal:
            print("-> 信号解析失败，跳过")
            continue

        asset_list = signal.get("asset_analysis", [])
        sig = next((x for x in asset_list if x.get("asset", "").upper() == asset_ticker.upper()), {})
        action = sig.get("action", "?")
        bias   = sig.get("bias_score", "?")
        target = sig.get("profit_target")
        stop   = sig.get("stop_loss")
        print(f"-> action={action}  bias={bias}  target={target}  stop={stop}", end="  ")

        future_df = fetch_future_data(asset_ticker, d, eval_days)
        trade = simulate_trade(signal, future_df, asset_ticker, eval_days, macro=macro)

        all_records.append({
            "date": d, "action": trade["action"], "entry_price": trade["entry_price"],
            "exit_price": trade["exit_price"], "exit_reason": trade["exit_reason"],
            "pnl_pct": trade["pnl_pct"], "win": trade["win"], "days_held": trade["days_held"],
            "position_size": trade.get("position_size"),
            "portfolio_pnl": trade.get("portfolio_pnl"),
            "bias_score": sig.get("bias_score"), "regime": sig.get("regime"),
            "sentiment": signal.get("overall_market_sentiment"),
            "sector_assessment": signal.get("sector_assessment", ""),
            "rate_assessment":   signal.get("rate_assessment", ""),
            "raw_signal": json.dumps(sig, ensure_ascii=False),
        })
        print(f"-> {trade['exit_reason']}  pnl={trade['pnl_pct']}%")

        time.sleep(rate_limit)

    if not all_records:
        print("\n无有效记录（dry_run 或全部跳过）。")
        return
    _save_and_print(asset_ticker, all_records, signals_file, perf_file)


def _save_and_print(asset_ticker: str, records: list[dict], signals_file: Path, perf_file: Path):
    pd.DataFrame(records).to_csv(signals_file, index=False, encoding="utf-8-sig")
    perf = compute_performance(records)
    print("\n" + "=" * 60)
    print(f"{asset_ticker} 回测绩效汇总")
    print("=" * 60)
    for k, v in perf.items():
        if k != "monthly_winrate":
            print(f"  {k:<25}: {v}")
    print(f"\n  逐月胜率: {perf.get('monthly_winrate', 'N/A')}")
    pd.DataFrame([perf]).to_csv(perf_file, index=False, encoding="utf-8-sig")
    print(f"\n信号明细 → {signals_file}")
    print(f"绩效汇总 → {perf_file}")


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="科技股通用 LLM 回测引擎（支持 NVDA/MSFT/GOOGL/AAPL/META/AMZN 等）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 生成 Prompt 文件（无需 API Key）
  python3 tech_backtest_engine.py --ticker NVDA --generate --start 2024-01-01 --end 2024-12-31

  # 全自动回测（阿里云 DashScope DeepSeek R1）
  ALIYUN_API_KEY=sk-xxx python3 tech_backtest_engine.py --ticker NVDA --start 2024-01-01 --end 2024-12-31 --model deepseek-r1 --rate-limit 30

  # 断点续跑
  ALIYUN_API_KEY=sk-xxx python3 tech_backtest_engine.py --ticker MSFT --start 2025-01-01 --end 2025-12-31 --model deepseek-r1 --resume

  # 评估已有响应文件
  python3 tech_backtest_engine.py --ticker GOOGL --evaluate
        """
    )
    parser.add_argument("--ticker",     required=True,             help="股票代码，如 NVDA, MSFT, GOOGL")
    parser.add_argument("--generate",   action="store_true",       help="生成 Prompt 文件（无需 API Key）")
    parser.add_argument("--evaluate",   action="store_true",       help="评估已有响应文件（无需 API Key）")
    parser.add_argument("--start",      default="2024-01-01",      help="回测开始日期 YYYY-MM-DD")
    parser.add_argument("--end",        default="2024-12-31",      help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--step",       default=5,   type=int,     help="每隔N个交易日触发一次（默认5）")
    parser.add_argument("--eval-days",  default=22,  type=int,     help="最长持仓天数（默认22，中线持仓）")
    parser.add_argument("--model",      default="deepseek-r1",     help="模型 ID（默认 deepseek-r1）")
    parser.add_argument("--rate-limit", default=20,  type=int,     help="API 调用间隔秒数（默认20）")
    parser.add_argument("--dry-run",    action="store_true",       help="只验证数据，不调用 API")
    parser.add_argument("--resume",     action="store_true",       help="跳过已完成节点，追加合并")
    parser.add_argument("--start-from", default=None,              help="从指定日期开始（YYYY-MM-DD）")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    if args.generate:
        run_generate(ticker, args.start, args.end, args.step, args.eval_days)
    elif args.evaluate:
        run_evaluate(ticker, args.eval_days)
    else:
        run_backtest(
            asset_ticker = ticker,
            start        = args.start,
            end          = args.end,
            step         = args.step,
            model        = args.model,
            eval_days    = args.eval_days,
            dry_run      = args.dry_run,
            rate_limit   = args.rate_limit,
            resume       = args.resume,
            start_from   = args.start_from,
        )
