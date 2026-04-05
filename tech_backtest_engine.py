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
from typing import Optional
from curl_cffi import requests as curl_requests
from openai import OpenAI

try:
    import httpx
    from anthropic import Anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

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
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "sk-6BV9Xfa9AJ09pkt0AHFPQtZUtlM28pCOnon6ArdIJW1fVyDP")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.openai-proxy.org/anthropic")
ANTHROPIC_MODEL    = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

DEEPSEEK_MODELS = {"deepseek-reasoner", "deepseek-chat"}
CLAUDE_MODELS   = {"claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"}

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

Your mission: Analyze {name} ({ticker}) to generate high-probability medium-term signals with disciplined risk management. Hold time: 2–5 weeks. Balance CONVICTION with OPPORTUNITY CAPTURE — both missing a genuine trending/recovery setup AND taking a false setup are costly errors. Enter when the weekly structure is clear; stand aside only when regime is genuinely ambiguous or risk/reward fails. Do NOT bias toward no_trade in clear uptrends or early recovery phases.

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
- **VIX**: >25 = risk-off, reduce longs; >35 AND still rising = no_trade; >35 BUT clearly declining from spike peak (fallen ≥20% from high) = recovery window, cautious long with 0.2 position allowed
- **DXY**: Strong dollar = international revenue headwind for US multinationals
- **QQQ Trend**: Sector tide; fighting a QQQ downtrend requires high conviction

## 2. Regime Classification

| Regime | Signals | RSI Rule | Approach |
|--------|---------|----------|----------|
| **Trending** | Price > EMA20 > EMA50 > EMA200, MACD +ve, ADX ≥ 18 | RSI >70 = momentum confirmation | Buy pullbacks; ride trend |
| **Trending-Recovery** | Price bounced ≥15% from recent bottom AND weekly MACD histogram improving (turning less negative or positive) AND weekly RSI-14 crossing above 40 — price may still be below EMA200 | RSI 40–60 = healthy early recovery | **Highest-alpha regime**: buy dips with 2×ATR stops; bias_score 0.55–0.65 appropriate |
| **Mean-Reverting** | Price oscillates around flat EMAs, RSI extreme | RSI >70 = overbought; RSI <30 = oversold | Fade extremes; tight stops |
| **Choppy** | Flat EMAs, MACD near zero, ADX < 18, no clear direction | RSI unhelpful | no_trade |

⚠️ **CRITICAL — Early Recovery Detection (HIGHEST ALPHA)**: When price has fallen ≥20% from 52-week high AND has since bounced ≥15% from the bottom low AND weekly MACD histogram is improving → classify as **Trending-Recovery**, even if price is still below EMA200. Do NOT classify as Choppy or Consolidation — that is the most expensive mistake. Missing this setup costs more alpha than any false entry.

⚠️ **Monthly Data = Background Only**: Monthly MACD negative or RSI < 50 does NOT veto a bullish weekly signal. Monthly data shifts bias_score by ±0.05 at most. A clear weekly Trending or Trending-Recovery setup takes priority. Exception: monthly RSI > 75 = add anti-chase note, reduce bias 0.05.

---

# ACTION SPACE

1. **long**: Bullish
   - Trending: Golden Cross, MACD +ve, price > EMA50 — buy dips
   - Mean-Reverting: RSI <35, near support, MACD turning up

2. **short**: Bearish — STRICT CONDITIONS REQUIRED (all three must be met):
   - ① QQQ Death Cross confirmed: QQQ EMA50 < QQQ EMA200
   - ② VIX > 25 (risk-off environment supporting downside)
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
- Long: profit_target > current_price > stop_loss; R:R ≥ 2.0; stop ≥ 1.5×ATR (2×ATR preferred in Trending/Trending-Recovery)
- Short: stop_loss > current_price > profit_target; same R:R and stop rules
- If ANY fails → no_trade, set levels to null
- position_size_pct must reflect current RSI/regime/VIX conditions

---

# OUTPUT FORMAT (JSON)

Return ONLY valid JSON:

```json
{{
  "period": "Weekly",
  "stock_ticker": "{ticker}",
  "overall_market_sentiment": "Risk-On" | "Risk-Off" | "Neutral",
  "qqq_assessment": "<QQQ trend and directional impact on {ticker}>",
  "sector_assessment": "<XLK sector rotation signal — bullish or bearish for tech>",
  "macro_rate_environment": "<10Y yield trend and impact on growth stock multiples>",
  "asset_analysis": [
    {{
      "asset": "{ticker}",
      "regime": "Trending-Up" | "Trending-Down" | "Trending-Recovery" | "Mean-Reverting" | "Consolidation" | "Choppy",
      "action": "long" | "short" | "no_trade",
      "bias_score": <float 0.0-1.0>,
      "entry_zone": "<price range>",
      "profit_target": <float | null>,
      "stop_loss": <float | null>,
      "risk_reward_ratio": <float | null>,
      "estimated_holding_weeks": <int 2-5 | null>,
      "position_size_pct": <float 0.0-1.0>,
      "invalidation_condition": "<objective signal that voids thesis>",
      "price_action_analysis": {{
        "trend_structure": "<EMA alignment, weekly golden/death cross, monthly MACD>",
        "momentum_signals": "<MACD, RSI, Stochastic>",
        "volatility_context": "<ATR, BB%B, bandwidth>",
        "volume_obv": "<OBV trend, volume/price confirmation or divergence>",
        "relative_strength_vs_qqq": "<outperforming or underperforming QQQ>"
      }},
      "structured_analysis": {{
        "market_hierarchy_alignment": "<L1-L5 alignment: all aligned / X layers conflicting>",
        "sector_rotation_signal": "<XLK vs QQQ impact on this stock>",
        "regime_justification": "<core basis for regime classification>"
      }},
      "justification": "<max 300 words>"
    }}
  ]
}}
```

**Validation**:
- R:R ≥ 1.5; stop ≥ 1.5×weekly ATR-14; bias_score < 0.45 → no_trade
- Long: target > price > stop; Short: stop > price > target
- Long: QQQ EMA50 > EMA200 required AND {ticker} weekly EMA50 > EMA200 required (no longs if either in death cross)
- Short requires ALL THREE: QQQ EMA50 < EMA200 AND VIX > 25 AND price below weekly EMA-200
- regime field must be one of: "Trending-Up", "Trending-Down", "Trending-Recovery", "Mean-Reverting", "Consolidation", "Choppy"

---

# COMMON PITFALLS

- ⚠️ **RSI paralysis**: {ticker} can sustain RSI >75 for weeks/months in trending markets. Never refuse a long solely because RSI is "high" in a Trending regime.
- ⚠️ **Anti-chase rule**: RSI-14 > 72 = reduce position_size_pct by 40–60%. RSI > 80 = max position_size_pct 0.25. Entering at cycle highs is the #1 cause of large losses.
- ⚠️ **Fighting rate headwinds**: Rapidly rising yields above 4.5% compress growth multiples.
- ⚠️ **Sector rotation**: If QQQ is in a clear downtrend, standalone {ticker} longs will struggle.
- ⚠️ **Death Cross trap (HARD RULE)**: If {ticker} weekly EMA50 < EMA200 → force no_trade for longs regardless of other signals. Serial stop-outs are inevitable when going long against a confirmed weekly downtrend. The system will also block this in code. You MUST output no_trade when {ticker}'s own EMA50 < EMA200.
- ⚠️ **Trending-Up momentum continuation**: When {ticker} is in Trending-Up regime AND price > EMA50 > EMA200 AND QQQ EMA50 > EMA200, bias_score 0.55–0.75 is appropriate for momentum continuation entries. Do NOT be overly conservative in confirmed uptrends — under-signaling in bull markets causes large alpha leakage vs buy-and-hold.
- ⚠️ **Recovery Phase recognition (CRITICAL)**: After a ≥20% correction, when price has bounced ≥15% from the bottom AND weekly MACD histogram is improving → classify as Trending-Recovery even BEFORE price recrosses EMA200. Wider stops (2×ATR), bias_score 0.55–0.65. Do NOT call this Choppy. This is the single highest-alpha entry in tech stocks.
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

MACRO_TICKERS = [("qqq", "QQQ"), ("xlk", "XLK"), ("spy", "SPY"), ("tnx", "^TNX"), ("vix", "^VIX"), ("dxy", "DX-Y.NYB")]


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
                      lookback: int = 200, eval_days: int = 15,
                      reproducible: bool = False):
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
    all_tickers = [(asset_ticker, "1d"), (asset_ticker, "1wk"), (asset_ticker, "1mo")] + \
                  [(t, "1wk") for _, t in MACRO_TICKERS]

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
            if reproducible:
                # 复现模式：只要有磁盘缓存就直接使用，确保回测可复现（不受Yahoo数据修订影响）
                _FULL_CACHE[key] = disk_df
                print(f"  {ticker:12s} [{interval}]  [复现模式] 磁盘缓存 ({len(disk_df)} 条，{cached_start.date()}~{cached_end.date()})")
                continue
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


def _drop_incomplete_weekly_bar(weekly_df: pd.DataFrame, ref_date: str) -> pd.DataFrame:
    """
    去除最后一根不完整的周线 Bar。
    yfinance 周线 Bar 从周一开始；若 ref_date 距该周起始不足 4 个日历日（周一~周四），
    则认为该周尚未收盘，删除最后一根 Bar，避免不完整数据污染 ATR/RSI/MACD 指标。
    保留至少 2 根 Bar，防止数据过少。
    """
    if weekly_df.empty or len(weekly_df) < 2:
        return weekly_df
    ref_dt = pd.Timestamp(ref_date)
    last_bar_start = weekly_df.index[-1]
    if hasattr(last_bar_start, "normalize"):
        last_bar_start = last_bar_start.normalize()
    if (ref_dt - last_bar_start).days < 4:
        return weekly_df.iloc[:-1]
    return weekly_df


def fetch_data_up_to(ticker: str, ref_date: str, lookback: int = 200) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    end_dt    = pd.Timestamp(ref_date)
    end_str   = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    start_str = (end_dt - timedelta(days=lookback)).strftime("%Y-%m-%d")
    # 月线需要更长的回望期（5年背景）
    monthly_start = (end_dt - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    daily   = _cached_slice(ticker, start_str, end_str, "1d")
    weekly  = _drop_incomplete_weekly_bar(
        _cached_slice(ticker, start_str, end_str, "1wk"), ref_date)
    monthly = _cached_slice(ticker, monthly_start, end_str, "1mo")
    return daily, weekly, monthly


def fetch_macro_for_date(ref_date: str, asset_ticker: str) -> dict:
    end_dt    = pd.Timestamp(ref_date)
    end_str   = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    # 120天=约17周，确保拿到12周以上的周线数据（与实盘保持一致）
    start_str = (end_dt - timedelta(days=120)).strftime("%Y-%m-%d")
    macro = {}
    for key, ticker in MACRO_TICKERS:
        try:
            df = _cached_slice(ticker, start_str, end_str, "1wk")
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
    def _last(df, col, n=8):
        if df.empty or col not in df.columns:
            return []
        return df[col].squeeze().dropna().tail(n).round(3).tolist()

    def _trend(vals):
        if len(vals) < 2:
            return "N/A"
        chg = (vals[-1] - vals[0]) / abs(vals[0]) * 100 if vals[0] != 0 else 0
        return f"{'↑' if chg > 0 else '↓'} {abs(chg):.1f}% (近{len(vals)}周)"

    def _ema_cross(df, fast=20, slow=50):
        if df.empty or "Close" not in df.columns:
            return "N/A"
        close = df["Close"].squeeze().dropna()
        if len(close) < slow:
            return "N/A"
        ema_f = calc_ema(close, fast).dropna()
        ema_s = calc_ema(close, slow).dropna()
        if ema_f.empty or ema_s.empty:
            return "N/A"
        return "金叉(EMA20>EMA50，多头)" if float(ema_f.iloc[-1]) > float(ema_s.iloc[-1]) else "死叉(EMA20<EMA50，空头)"

    result = {}

    # ── QQQ ──
    qqq_df = macro.get("qqq", pd.DataFrame())
    qqq_closes = _last(qqq_df, "Close")
    result["qqq_last"]   = round(qqq_closes[-1], 2) if qqq_closes else None
    result["qqq_trend"]  = _trend(qqq_closes)
    result["qqq_series"] = qqq_closes
    result["qqq_cross"]  = _ema_cross(qqq_df)
    if not qqq_df.empty and "Close" in qqq_df.columns:
        qqq_ema20 = calc_ema(qqq_df["Close"].squeeze(), 20).dropna()
        result["qqq_ema20"] = round(float(qqq_ema20.iloc[-1]), 2) if not qqq_ema20.empty else None
    else:
        result["qqq_ema20"] = None

    # ── XLK 板块轮动 ──
    xlk_df = macro.get("xlk", pd.DataFrame())
    xlk_closes = _last(xlk_df, "Close")
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

    # ── SPY ──
    spy_df = macro.get("spy", pd.DataFrame())
    spy_closes = _last(spy_df, "Close")
    result["spy_last"]  = round(spy_closes[-1], 2) if spy_closes else None
    result["spy_trend"] = _trend(spy_closes)

    # ── 个股 vs QQQ 相对强弱（周线 RS，近20周） ──
    if not qqq_df.empty and "Close" in qqq_df.columns and len(stock_close) >= 5:
        try:
            qqq_wk = qqq_df["Close"].squeeze().dropna()
            stk_wk = stock_close.dropna()
            # 将两个序列对齐到共同日期
            common = stk_wk.index.intersection(qqq_wk.index)
            if len(common) >= 5:
                period = min(20, len(common) - 1)
                sc = stk_wk.loc[common]
                bc = qqq_wk.loc[common]
                rs = (sc / sc.shift(period)) / (bc / bc.shift(period)) - 1
                rs_clean = rs.dropna()
                rs_last = round(float(rs_clean.iloc[-1]) * 100, 2)
                result["stock_qqq_rs"]       = [round(x * 100, 2) for x in rs_clean.tail(6).tolist()]
                result["stock_qqq_rs_trend"] = _trend(result["stock_qqq_rs"])
                result["stock_qqq_rs_last"]  = rs_last
                result["stock_qqq_signal"]   = "跑赢QQQ" if rs_last > 0 else "跑输QQQ"
            else:
                result["stock_qqq_rs"] = []
                result["stock_qqq_rs_trend"] = "N/A"
                result["stock_qqq_rs_last"]  = None
                result["stock_qqq_signal"]   = "N/A"
        except Exception:
            result["stock_qqq_rs"] = []
            result["stock_qqq_rs_trend"] = "N/A"
            result["stock_qqq_rs_last"]  = None
            result["stock_qqq_signal"]   = "N/A"
    else:
        result["stock_qqq_rs"] = []
        result["stock_qqq_rs_trend"] = "N/A"
        result["stock_qqq_rs_last"]  = None
        result["stock_qqq_signal"]   = "N/A"

    # ── 10Y 收益率 ──
    tnx_closes = _last(macro.get("tnx", pd.DataFrame()), "Close")
    result["tnx_last"]   = round(tnx_closes[-1], 3) if tnx_closes else None
    result["tnx_trend"]  = _trend(tnx_closes)
    result["tnx_series"] = tnx_closes

    # ── VIX ──
    vix_closes = _last(macro.get("vix", pd.DataFrame()), "Close")
    result["vix_last"]  = round(vix_closes[-1], 2) if vix_closes else None
    result["vix_trend"] = _trend(vix_closes)
    if result["vix_last"]:
        v = result["vix_last"]
        result["vix_regime"] = (
            "极度恐慌(>35)" if v > 35 else
            ("恐慌/Risk-Off(>25)" if v > 25 else
            ("高波动(>20)" if v > 20 else
            ("中性(>15)" if v > 15 else "低波动/乐观(<15)")))
        )
    else:
        result["vix_regime"] = "N/A"

    # ── DXY ──
    dxy_closes = _last(macro.get("dxy", pd.DataFrame()), "Close")
    result["dxy_last"]  = round(dxy_closes[-1], 2) if dxy_closes else None
    result["dxy_trend"] = _trend(dxy_closes)

    return result


def build_blind_prompt(asset_ticker: str, daily: pd.DataFrame, weekly: pd.DataFrame,
                       macro: dict | None = None, perf_metrics: dict | None = None,
                       monthly: pd.DataFrame | None = None) -> str:
    if daily.empty or weekly.empty or len(daily) < 30:
        return ""

    d_ind = compute_indicators(daily)
    w_ind = compute_indicators(weekly)

    close_d       = daily["Close"].squeeze()
    close_w       = weekly["Close"].squeeze()
    current_price = round(float(close_d.iloc[-1]), 2)

    # ── 日线快照（用于入场精确定时） ──
    current_ema20  = round(float(d_ind["ema20"].dropna().iloc[-1]), 2)
    current_ema50  = round(float(d_ind["ema50"].dropna().iloc[-1]), 2)
    ema200_arr     = d_ind["ema200"].dropna()
    current_ema200 = round(float(ema200_arr.iloc[-1]), 2) if len(ema200_arr) > 0 else None
    current_macd   = round(float(d_ind["macd"].dropna().iloc[-1]), 4)
    current_rsi14  = round(float(d_ind["rsi14"].dropna().iloc[-1]), 2)
    current_rsi7   = round(float(d_ind["rsi7"].dropna().iloc[-1]), 2)
    atr14          = round(float(d_ind["atr14"].dropna().iloc[-1]), 2)

    last_row   = daily.iloc[-1]
    today_open = round(float(last_row["Open"].squeeze()), 2)
    today_high = round(float(last_row["High"].squeeze()), 2)
    today_low  = round(float(last_row["Low"].squeeze()), 2)
    today_vol  = int(last_row["Volume"].squeeze())
    vol_avg    = int(daily["Volume"].squeeze().tail(20).mean())
    prev_close = float(close_d.iloc[-2]) if len(close_d) >= 2 else current_price
    close_5d   = float(close_d.iloc[-6]) if len(close_d) >= 6 else float(close_d.iloc[0])
    day_chg    = (current_price - prev_close) / prev_close * 100
    week_chg   = (current_price - close_5d) / close_5d * 100

    # ── 周线关键指标快照（主分析框架，与实盘一致） ──
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
    w_adx     = round(float(w_ind['adx'].dropna().iloc[-1]), 1)    if w_ind['adx'].dropna().shape[0]     > 0 else None
    w_pdi     = round(float(w_ind['plus_di'].dropna().iloc[-1]), 1) if w_ind['plus_di'].dropna().shape[0] > 0 else None
    w_mdi     = round(float(w_ind['minus_di'].dropna().iloc[-1]),1) if w_ind['minus_di'].dropna().shape[0]> 0 else None
    w_bb_pctb = round(float(w_ind['bb_pct_b'].dropna().iloc[-1]),3) if w_ind['bb_pct_b'].dropna().shape[0]> 0 else None
    w_bb_bw   = round(float(w_ind['bb_bw'].dropna().iloc[-1]),2)    if w_ind['bb_bw'].dropna().shape[0]   > 0 else None
    w_stoch_k = round(float(w_ind['stoch_k'].dropna().iloc[-1]),1)  if w_ind['stoch_k'].dropna().shape[0] > 0 else None
    w_stoch_d = round(float(w_ind['stoch_d'].dropna().iloc[-1]),1)  if w_ind['stoch_d'].dropna().shape[0] > 0 else None
    w_roc20   = round(float(w_ind['roc20'].dropna().iloc[-1]),2)    if w_ind['roc20'].dropna().shape[0]   > 0 else None
    w_obv_arr = w_ind['obv'].dropna().tail(6).tolist()
    w_obv_trend = "上升" if len(w_obv_arr) >= 2 and w_obv_arr[-1] > w_obv_arr[0] else "下降"

    # ── 月线长期背景 ──
    m_ema20 = m_macd = m_rsi14 = None
    if monthly is not None and not monthly.empty and len(monthly) >= 5:
        m_ind   = compute_indicators(monthly)
        m_ema20  = _safe_last(m_ind['ema20'])
        m_macd   = _safe_last(m_ind['macd'])
        m_rsi14  = _safe_last(m_ind['rsi14'])

    # ── 52周价格结构 ──
    close_full  = close_d.dropna()
    high_52w    = round(float(close_full.tail(252).max()), 2)
    low_52w     = round(float(close_full.tail(252).min()), 2)
    pct_high    = round((current_price - high_52w) / high_52w * 100, 1)
    pct_low     = round((current_price - low_52w)  / low_52w  * 100, 1)

    # ── EMA 状态 ──
    ema_cross_str = (
        "金叉 (EMA50 > EMA200，多头结构)" if w_ema50 and w_ema200 and w_ema50 > w_ema200
        else "死叉 (EMA50 < EMA200，空头结构)" if w_ema50 and w_ema200 else "N/A"
    )
    is_death_cross  = w_ema200 is not None and w_ema50 is not None and w_ema50 < w_ema200
    price_below_200 = w_ema200 is not None and current_price < w_ema200

    # ── 宏观摘要（使用周线数据，与实盘一致） ──
    # _summarize_macro 现在接收 weekly close 用于 RS 计算
    ms = _summarize_macro(macro or {}, close_w, asset_ticker)

    # ── 预计算入场锚点（基于周线 ATR-14，中线持仓） ──
    atr = w_atr14 or atr14 or 1.0
    long_stop    = round(current_price - 2.0 * atr, 2)
    long_target  = round(current_price + 4.0 * atr, 2)
    short_stop   = round(current_price + 2.0 * atr, 2)
    short_target = round(current_price - 4.0 * atr, 2)

    # ── 日线序列 ──
    n = 15
    daily_closes = fmt_series(close_d, 2, n)
    daily_ema20  = fmt_series(d_ind["ema20"], 2, n)
    daily_ema50  = fmt_series(d_ind["ema50"], 2, n)
    daily_macd   = fmt_series(d_ind["macd"], 4, n)
    daily_rsi14  = fmt_series(d_ind["rsi14"], 2, n)

    # ── 周线序列 ──
    n_w = 12
    weekly_closes  = fmt_series(close_w, 2, n_w)
    weekly_macd    = fmt_series(w_ind['macd'], 2, n_w)
    weekly_rsi14   = fmt_series(w_ind['rsi14'], 2, n_w)
    weekly_ema20   = fmt_series(w_ind['ema20'], 2, n_w)
    weekly_ema50   = fmt_series(w_ind['ema50'], 2, n_w)
    weekly_adx     = fmt_series(w_ind['adx'], 1, n_w)
    weekly_stoch_k = fmt_series(w_ind['stoch_k'], 1, n_w)
    weekly_bb_pctb = fmt_series(w_ind['bb_pct_b'], 3, n_w)

    # ── 月线序列 ──
    monthly_closes = monthly_macd = monthly_rsi14 = "N/A"
    if monthly is not None and not monthly.empty:
        close_m = monthly["Close"].squeeze()
        m_ind_for_series = compute_indicators(monthly)
        monthly_closes = fmt_series(close_m, 2, 12)
        monthly_macd   = fmt_series(m_ind_for_series['macd'], 2, 12)
        monthly_rsi14  = fmt_series(m_ind_for_series['rsi14'], 2, 12)

    def _fv(v, u=""):
        return f"{v}{u}" if v is not None else "N/A"

    # ── 动态过滤规则 ──
    filter_rules = []
    if is_death_cross and price_below_200:
        pct_200 = round((current_price - w_ema200) / w_ema200 * 100, 1) if w_ema200 else None
        filter_rules.append(
            f"⚠️ **【Death Cross 过滤】** 周线 EMA50({w_ema50}) < EMA200({_fv(w_ema200)})，"
            f"价格 vs 周线EMA200: {_fv(pct_200, '%')}：做多 bias_score 强制 ≤ 0.45 → 自动 no_trade。"
        )
    if pct_high < -20:
        filter_rules.append(
            f"⚠️ **【禁止追空】** 价格距52周高点已跌 {pct_high:.1f}%（>20%），做空风险回报极差，强制 no_trade。"
        )

    # ── 绩效反馈 ──
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

    # ── 行业专属上下文（与实盘一致，直接注入用户 prompt） ──
    ctx = TICKER_CONTEXT.get(asset_ticker.upper(), DEFAULT_CONTEXT)
    industry_context = ctx.get("description", "").replace("{ticker}", asset_ticker)

    # ── L3 板块状态 ──
    xlk_vs_qqq_pct = ms.get("xlk_vs_qqq_pct")
    sector_rotation = ms.get("sector_rotation", "N/A")
    l3_direction = "有利" if xlk_vs_qqq_pct is not None and xlk_vs_qqq_pct > 0 else "不利"

    # ── L4 个股 RS 状态 ──
    rs_last  = ms.get("stock_qqq_rs_last")
    rs_signal = ms.get("stock_qqq_signal", "N/A")
    l4_direction = "有利" if rs_last is not None and rs_last > 0 else "不利"

    prompt = f"""# {asset_ticker} ({ctx['name'].replace('{ticker}', asset_ticker)}) 纳斯达克科技股中长线分析请求
**数据来源**: Yahoo Finance ({asset_ticker})
**分析框架**: 周线(W1)为主 + 月线(M1)长期背景 + 日线辅助入场
**持仓目标周期**: 2–5 周（中线摆动交易）
**重要说明**: 严格基于以下数据推断，不得引用数据窗口之外的具体事件。
{perf_feedback}
---

## 一、价格概要

- **当前价格**: ${current_price}
- **今日 O/H/L/C**: {today_open} / {today_high} / {today_low} / {current_price}
- **今日涨跌幅**: {day_chg:+.2f}%  |  **近5日涨跌**: {week_chg:+.2f}%
- **成交量**: {today_vol:,}  vs 20日均量 {vol_avg:,}  ({'放量' if today_vol > vol_avg * 1.2 else ('缩量' if today_vol < vol_avg * 0.8 else '正常')})
- **52周高点**: ${high_52w}  距高点: {pct_high:+.1f}%
- **52周低点**: ${low_52w}   距低点: {pct_low:+.1f}%

---

## 二、周线关键指标快照（主分析框架）

| 指标 | 当前值 | 信号解读 |
|------|--------|----------|
| EMA-20 (周) | {_fv(w_ema20)} | {'价格高于EMA20，短期偏多' if w_ema20 and current_price > w_ema20 else '价格低于EMA20，短期偏空'} |
| EMA-50 (周) | {_fv(w_ema50)} | {'价格高于EMA50，中期偏多' if w_ema50 and current_price > w_ema50 else '价格低于EMA50，中期偏空'} |
| EMA-200 (周) | {_fv(w_ema200)} | {'价格高于EMA200，长期牛市' if w_ema200 and current_price > w_ema200 else '价格低于EMA200，长期熊市'} |
| EMA 金/死叉 (50/200周) | {ema_cross_str} | 长期趋势方向 |
| MACD (周) | {_fv(w_macd)} | {'正值，多头动能' if w_macd and w_macd > 0 else '负值，空头动能'} |
| RSI-14 (周) | {_fv(w_rsi14)} | {'超买 >70，谨慎追多' if w_rsi14 and w_rsi14 > 70 else ('超卖 <30，关注反弹' if w_rsi14 and w_rsi14 < 30 else '中性区间 30–70')} |
| RSI-7 (周) | {_fv(w_rsi7)} | {'极度超买 >80' if w_rsi7 and w_rsi7 > 80 else ('极度超卖 <20' if w_rsi7 and w_rsi7 < 20 else '正常范围')} |
| ADX (周) | {_fv(w_adx)} | {'强趋势 >25' if w_adx and w_adx > 25 else ('弱趋势/振荡 <20' if w_adx and w_adx < 20 else '趋势形成中 20–25')} |
| +DI / -DI | {_fv(w_pdi)} / {_fv(w_mdi)} | {'+DI>-DI 多头主导' if w_pdi and w_mdi and w_pdi > w_mdi else '-DI>+DI 空头主导'} |
| Stochastic %K/%D | {_fv(w_stoch_k)} / {_fv(w_stoch_d)} | {'超买死叉，谨慎做多' if w_stoch_k and w_stoch_d and w_stoch_k > 80 and w_stoch_k < w_stoch_d else ('超卖金叉，关注做多' if w_stoch_k and w_stoch_d and w_stoch_k < 20 and w_stoch_k > w_stoch_d else '中性区间')} |
| BB %B (周) | {_fv(w_bb_pctb)} | {'突破上轨 >1' if w_bb_pctb and w_bb_pctb > 1 else ('跌破下轨 <0' if w_bb_pctb and w_bb_pctb < 0 else '布林带内运行')} |
| BB 带宽 % | {_fv(w_bb_bw)} | {'带宽扩张，趋势加速' if w_bb_bw and w_bb_bw > 10 else '带宽收缩，蓄势'} |
| ROC-20 (周) | {_fv(w_roc20, '%')} | {'正动量' if w_roc20 and w_roc20 > 0 else '负动量'} |
| OBV 趋势 (近6周) | {w_obv_trend} | {'量价配合，机构积累' if w_obv_trend == '上升' else '量价背离，机构派发'} |
| ATR-14 (周) | {_fv(w_atr14)} | 止损基准（中线持仓） |

---

## 三、月线长期趋势背景（⚠️ 背景参考，权重 ≤ ±0.05，不构成否决权）

> **月线权重规则**：月线数据仅调整 bias_score ±0.05，不得单独阻止周线级别的入场信号。
> 月线 MACD < 0 或价格低于月线 EMA-20 → bias_score 最多 -0.05。
> 月线 RSI > 75 → bias_score 最多 -0.05（反追高）。
> 月线 RSI < 35 且周线 RSI 向上 → 这是**恢复期确认信号**，bias_score 可 +0.05。
> **周线 Trending / Trending-Recovery 信号优先于月线空头背景。**

| 指标 | 当前值 | 解读（仅背景参考） |
|------|--------|------|
| EMA-20 (月) | {_fv(m_ema20)} | {'价格高于月线EMA20，长期趋势向上' if m_ema20 and current_price > m_ema20 else ('价格低于月线EMA20，长期趋势向下——不影响周线多头信号' if m_ema20 else 'N/A')} |
| MACD (月) | {_fv(m_macd)} | {'月线多头动能，背景利好' if m_macd and m_macd > 0 else ('月线空头动能——bias_score 最多 -0.05，不否决周线信号' if m_macd is not None else 'N/A')} |
| RSI-14 (月) | {_fv(m_rsi14)} | {'月线超买 >70，注意追高风险，bias_score -0.05' if m_rsi14 and m_rsi14 > 70 else ('月线超卖 <35，结合周线向上 = 恢复期确认' if m_rsi14 and m_rsi14 < 35 else '月线中性')} |

---

## 四、序列数据（从旧到新排列，⚠️最后一个值 = 最新）

**周线数据（近{n_w}周）**：
收盘价:     [{weekly_closes}]
EMA-20:     [{weekly_ema20}]
EMA-50:     [{weekly_ema50}]
MACD:       [{weekly_macd}]
RSI-14:     [{weekly_rsi14}]
ADX:        [{weekly_adx}]
Stoch %K:   [{weekly_stoch_k}]
BB %B:      [{weekly_bb_pctb}]

**日线数据（近{n}个交易日，辅助入场定时）**：
收盘价: [{daily_closes}]
EMA-20: [{daily_ema20}]
EMA-50: [{daily_ema50}]
MACD:   [{daily_macd}]
RSI-14: [{daily_rsi14}]

**月线数据（近12个月，仅背景参考，权重 ≤ ±0.05）**：
收盘价: [{monthly_closes}]
MACD:   [{monthly_macd}]
RSI-14: [{monthly_rsi14}]

---

## 五、结构化分析 — 五层市场层级

| 层级 | 维度 | 当前状态 | 方向 |
|------|------|----------|------|
| L1 宏观 | Fed政策 + 10Y收益率 | {_fv(ms.get('tnx_last'), '%')}  趋势: {ms.get('tnx_trend', 'N/A')} | {'收益率上升→成长股承压' if ms.get('tnx_trend','').startswith('↑') else '收益率下降→成长股受益'} |
| L2 指数 | QQQ 趋势 | {_fv(ms.get('qqq_last'))}  趋势: {ms.get('qqq_trend', 'N/A')}  均线: {ms.get('qqq_cross', 'N/A')} | {'多头' if ms.get('qqq_cross','').startswith('金') else '空头'} |
| L3 板块 | XLK vs QQQ | XLK超额: {_fv(ms.get('xlk_vs_qqq_pct'), '%')}  {sector_rotation} | {l3_direction} |
| L4 个股 | {asset_ticker} vs QQQ (RS) | 超额收益: {_fv(rs_last, '%')}  {rs_signal} | {l4_direction} |
| L5 技术 | 入场时机 | ADX={_fv(w_adx)}  RSI7={_fv(w_rsi7)}  BB%B={_fv(w_bb_pctb)} | 待模型判断 |

---

## 六、宏观与板块背景

### Nasdaq-100 (QQQ)
- **最新价**: {_fv(ms.get('qqq_last'))}  |  **趋势**: {ms.get('qqq_trend', 'N/A')}  |  **均线状态**: {ms.get('qqq_cross', 'N/A')}
- **近8周序列**: {ms.get('qqq_series', [])}
- **QQQ EMA20**: {_fv(ms.get('qqq_ema20'))}  |  **状态**: {'QQQ高于EMA20 — 科技板块偏多' if ms.get('qqq_last') and ms.get('qqq_ema20') and ms.get('qqq_last') > ms.get('qqq_ema20') else 'QQQ低于EMA20 — 科技板块承压'}

### 科技板块轮动 (XLK vs QQQ)
- **XLK最新价**: {_fv(ms.get('xlk_last'))}  |  **趋势**: {ms.get('xlk_trend', 'N/A')}
- **XLK vs QQQ 超额**: {_fv(ms.get('xlk_vs_qqq_pct'), '%')}  |  **板块信号**: {sector_rotation}
- {'✅ 科技板块领涨，顺势做多有利' if ms.get('xlk_vs_qqq_pct') and ms.get('xlk_vs_qqq_pct') > 0 else '⚠️ 科技板块落后大盘，做多需额外谨慎'}

### 大盘风险环境 (SPY + VIX + TNX)
- **SPY**: {_fv(ms.get('spy_last'))}  趋势: {ms.get('spy_trend', 'N/A')}
- **10Y 收益率**: {_fv(ms.get('tnx_last'), '%')}  趋势: {ms.get('tnx_trend', 'N/A')}
- {'⚠️ 收益率 > 4.5%，成长股估值承压' if ms.get('tnx_last') and ms.get('tnx_last') > 4.5 else '收益率处于合理范围'}
- **VIX**: {_fv(ms.get('vix_last'))}  状态: {ms.get('vix_regime', 'N/A')}  趋势: {ms.get('vix_trend', 'N/A')}
- {'⚠️ VIX>25，做多 bias_score 上限 0.60' if ms.get('vix_last') and ms.get('vix_last') > 25 else '市场情绪相对平稳'}
- **DXY**: {_fv(ms.get('dxy_last'))}  趋势: {ms.get('dxy_trend', 'N/A')}

### {asset_ticker} vs QQQ 相对强弱
- **RS 超额收益**: {_fv(rs_last, '%')}  |  **信号**: {rs_signal}
- **近6期RS序列**: {ms.get('stock_qqq_rs', [])}
- **RS 趋势**: {ms.get('stock_qqq_rs_trend', 'N/A')}

---

## 七、预计算入场锚点（基于周线 ATR-14 = {_fv(w_atr14)}）

> 中线持仓止损和目标均基于**周线 ATR**，给价格充分的波动空间。

| 方向 | stop_loss (2×wATR) | profit_target (4×wATR, R:R≥2.0) |
|------|-------------------|----------------------------------|
| 做多 | {long_stop} | {long_target} |
| 做空 | {short_stop} | {short_target} |

---

## 八、行业专属分析维度

{industry_context}

---

## 九、特殊过滤规则

{chr(10).join(filter_rules) if filter_rules else '当前无特殊过滤规则触发'}

**通用规则（硬性约束）**：
- `bias_score < 0.50` → 强制 no_trade
- QQQ 处于死叉（EMA50 < EMA200）→ 禁止做多
- 做空需同时满足：QQQ 死叉 **且** VIX > 25 **且** 股价低于周线 EMA-200；三者缺一禁止做空
- `risk_reward_ratio < 2.0` → 盈亏比不足，no_trade
- 止损距 entry < 2.0×周ATR（{_fv(w_atr14)}）→ 止损太紧，中长线需给价格足够呼吸空间

**技术过滤规则**：
- L1–L5 五层全部对齐 → bias_score 允许 > 0.65
- 有 1 层冲突 → bias_score 上限 0.60
- 有 2 层及以上冲突 → bias_score 上限 0.50（建议 no_trade）
- 周线 MACD ({_fv(w_macd)}) < 0 且周线 Trending 制度 → 禁止做多
- 周线 RSI-7 ({_fv(w_rsi7)}) > 75 → **仅 Mean-Reverting/Consolidation 制度**时做多 bias_score 上限 0.55；Trending-Up 制度下 RSI>75 为动能确认，不强制限制 bias_score，但须降低仓位（position_size_pct ≤ 0.3）
- 价格偏离周线 EMA-20 ({_fv(w_ema20)}) 超过 5% → bias_score 上限 0.55
- 周线 ADX ({_fv(w_adx)}) < 20 → 制度降级为 Consolidation，bias_score 上限 0.45
- OBV 近6周下降且价格上涨 → bias_score 降低 0.10
- XLK 落后 QQQ（科技板块不领涨）→ 做多 bias_score -0.05
- VIX > 25 → 做多 bias_score 上限 0.60；VIX > 35 → 一律 no_trade

---

## 十、分析任务

按照系统指令框架，分析 {asset_ticker} 当前技术形态、宏观背景与板块轮动，输出 JSON。资产名称必须为 "{asset_ticker}"。

**JSON 输出要求**：
```json
{{
  "period": "Weekly",
  "stock_ticker": "{asset_ticker}",
  "overall_market_sentiment": "Risk-On | Risk-Off | Neutral",
  "qqq_assessment": "<QQQ趋势及其对{asset_ticker}的方向性影响>",
  "sector_assessment": "<XLK板块轮动信号，利好还是利空科技股>",
  "macro_rate_environment": "<10Y收益率趋势和Fed政策对成长股估值的影响>",
  "asset_analysis": [
    {{
      "asset": "{asset_ticker}",
      "regime": "Trending-Up | Trending-Down | Trending-Recovery | Mean-Reverting | Consolidation | Choppy",
      "action": "long | short | no_trade",
      "bias_score": <0.0–1.0>,
      "entry_zone": "<价格区间>",
      "profit_target": <数字 或 null>,
      "stop_loss": <数字 或 null>,
      "risk_reward_ratio": <数字 或 null>,
      "estimated_holding_weeks": <2–5>,
      "position_size_pct": <0.0–1.0>,
      "invalidation_condition": "<使该观点失效的具体市场信号>",
      "price_action_analysis": {{
        "trend_structure": "<EMA对齐情况，周线金叉/死叉状态，月线MACD方向>",
        "momentum_signals": "<MACD、RSI、Stochastic综合描述>",
        "volatility_context": "<ATR、BB%B、带宽>",
        "volume_obv": "<OBV趋势，量价配合还是背离>",
        "relative_strength_vs_qqq": "<跑赢或跑输QQQ，幅度，近6期RS序列趋势>"
      }},
      "structured_analysis": {{
        "market_hierarchy_alignment": "<L1–L5五层对齐情况>",
        "sector_rotation_signal": "<XLK vs QQQ，对本股利好还是利空>",
        "regime_justification": "<判断当前制度的核心依据>"
      }},
      "justification": "<不超过300字的综合判断>"
    }}
  ]
}}
```""".strip()

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


def _call_claude_raw(prompt: str, system_prompt: str, rate_limit: int = 20) -> dict:
    """通过 Anthropic SDK 调用 Claude，返回解析后的信号 dict。"""
    if not _ANTHROPIC_AVAILABLE:
        print("  [双模型] Anthropic SDK 未安装，跳过确认")
        return {}
    try:
        client = Anthropic(
            base_url=ANTHROPIC_BASE_URL,
            api_key=ANTHROPIC_API_KEY,
            http_client=httpx.Client(verify=False, timeout=120.0),
        )
    except Exception as e:
        print(f"  [双模型] Claude 客户端初始化失败: {e}")
        return {}
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = "".join(b.text for b in msg.content if hasattr(b, "text"))
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            return parse_signal(raw)
        except Exception as e:
            print(f"  [双模型/Claude] attempt {attempt+1}/3: {e}")
            if attempt < 2:
                time.sleep(15 * (attempt + 1))
    return {}


def call_dual_api(prompt: str, primary_model: str, system_prompt: str,
                  rate_limit: int, second_model: str = None,
                  asset_ticker: str = "", dual_threshold: float = 0.55) -> dict:
    """
    双模型确认：主模型初筛，若 action=long/short 且 bias >= threshold 则触发确认模型。
    两者一致 → 使用确认模型结果；分歧 → 强制 no_trade。
    """
    primary_signal = call_api(prompt, primary_model, system_prompt, rate_limit)
    if not second_model or not primary_signal:
        return primary_signal

    asset_list = primary_signal.get("asset_analysis", [])
    sig = next((x for x in asset_list if x.get("asset", "").upper() == asset_ticker.upper()), None)
    if not sig:
        return primary_signal

    action = sig.get("action", "no_trade")
    bias   = float(sig.get("bias_score") or 0)

    # Trending-Up 制度下降低确认门槛，避免牛市错失过多信号
    regime = sig.get("regime", "")
    effective_threshold = dual_threshold
    if regime in ("Trending-Up", "Trending-Recovery") and action == "long":
        effective_threshold = 0.50  # 趋势向上时0.50即触发双模型确认

    if action == "no_trade" or bias < effective_threshold:
        print(f"  [双模型] 初筛: {action} bias={bias:.2f} → 低于阈值，跳过确认")
        return primary_signal

    print(f"  [双模型] 初筛: {action} bias={bias:.2f} → 触发确认模型 ({second_model})...", flush=True)
    time.sleep(rate_limit)  # 两次 API 调用之间让 rate limit 冷却

    if second_model in CLAUDE_MODELS or second_model.startswith("claude"):
        confirm_signal = _call_claude_raw(prompt, system_prompt, rate_limit)
    else:
        confirm_signal = call_api(prompt, second_model, system_prompt, rate_limit)

    if not confirm_signal:
        print("  [双模型] 确认模型失败，沿用初筛结果")
        return primary_signal

    c_list = confirm_signal.get("asset_analysis", [])
    c_sig  = next((x for x in c_list if x.get("asset", "").upper() == asset_ticker.upper()), None)
    if not c_sig:
        return primary_signal

    c_action = c_sig.get("action", "no_trade")
    c_bias   = float(c_sig.get("bias_score") or 0)

    if action == c_action:
        print(f"  [双模型] ✓ 一致: {action} | 初筛 bias={bias:.2f}, 确认 bias={c_bias:.2f}")
        return confirm_signal
    else:
        print(f"  [双模型] ✗ 分歧: 初筛={action}({bias:.2f}), 确认={c_action}({c_bias:.2f}) → 强制 no_trade")
        for item in primary_signal.get("asset_analysis", []):
            if item.get("asset", "").upper() == asset_ticker.upper():
                item["action"]           = "no_trade"
                item["bias_score"]       = 0.0
                item["profit_target"]    = None
                item["stop_loss"]        = None
                item["risk_reward_ratio"] = None
                item["position_size_pct"] = 0.0
        return primary_signal


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
                   macro: dict = None, slippage_pct: float = 0.001) -> dict:
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
                vix_elevated = float(vix_val.iloc[-1]) > 25.0
        if not (qqq_in_downtrend and vix_elevated):
            base["exit_reason"] = "SHORT_FILTERED"
            base["action"] = "no_trade"
            return base

    if profit_target is None or stop_loss is None:
        base["exit_reason"] = "MISSING_LEVELS"
        return base

    profit_target = float(profit_target)
    stop_loss     = float(stop_loss)

    raw_open = float(future_df.iloc[0]["Open"].squeeze())
    # 滑点模型：做多付ask（价格偏高），做空收bid（价格偏低）
    if action == "long":
        entry_price = raw_open * (1 + slippage_pct)
    else:
        entry_price = raw_open * (1 - slippage_pct)
    base["entry_price"] = round(entry_price, 4)

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
        # 滑点模型：出场时做多收bid（偏低），做空付ask（偏高）
        raw_exit = base["exit_price"]
        if action == "long":
            effective_exit = raw_exit * (1 - slippage_pct)
            pnl = (effective_exit - entry_price) / entry_price * 100
        else:
            effective_exit = raw_exit * (1 + slippage_pct)
            pnl = (entry_price - effective_exit) / entry_price * 100
        base["exit_price"]    = round(effective_exit, 4)
        base["pnl_pct"]       = round(pnl, 4)
        # TIMEOUT 出场不计入胜负（未完成持仓），仅 STOP_LOSS / TAKE_PROFIT 才算"已确认交易"
        base["win"]           = (pnl > 0) if base.get("exit_reason") in ("STOP_LOSS", "TAKE_PROFIT") else None
        base["position_size"] = pos_size
        base["portfolio_pnl"] = round(pnl * pos_size, 4)  # 仓位加权收益

    return base


# ─────────────────────────────────────────────
# 绩效统计
# ─────────────────────────────────────────────

def compute_performance(records: list[dict],
                        start_date: str = None, end_date: str = None) -> dict:
    df         = pd.DataFrame(records)
    traded     = df[df["action"].isin(["long", "short"])].copy()
    no_trade   = (df["action"] == "no_trade").sum()
    invalid_rr = (df["exit_reason"] == "INVALID_RR").sum()

    if traded.empty:
        return {"error": "无有效交易信号"}

    executed     = traded[~traded["exit_reason"].isin(["INVALID_RR", "MISSING_LEVELS"])]
    # 仅 STOP_LOSS / TAKE_PROFIT 为"已确认交易"，计入胜率分母
    completed    = executed[executed["exit_reason"].isin(["STOP_LOSS", "TAKE_PROFIT"])]
    timeout_cnt  = int((executed["exit_reason"] == "TIMEOUT").sum())
    wins         = completed[completed["win"] == True]
    losses       = completed[completed["win"] == False]

    win_rate      = len(wins) / len(completed) * 100 if len(completed) > 0 else 0
    avg_pnl       = executed["pnl_pct"].mean()     if not executed.empty else 0
    avg_win       = wins["pnl_pct"].mean()         if not wins.empty     else 0
    avg_loss      = losses["pnl_pct"].mean()       if not losses.empty   else 0
    total_profit  = wins["pnl_pct"].sum()
    total_loss    = abs(losses["pnl_pct"].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    # 最大回撤：基于持仓加权复利资金曲线，而非 P&L 简单累加
    _equity = 1.0
    _equity_vals = [1.0]
    for _, _row in executed.iterrows():
        _sz = float(_row.get("position_size") or 1.0) if "position_size" in executed.columns else 1.0
        _equity *= (1 + _row["pnl_pct"] / 100 * _sz)
        _equity_vals.append(_equity)
    _eq_s  = pd.Series(_equity_vals)
    max_dd = float((((_eq_s - _eq_s.cummax()) / _eq_s.cummax()).min()) * 100)

    # ── CAGR / Sharpe / Sortino / Calmar ─────────────────────────
    cagr = sharpe = sortino = calmar = float("nan")

    # 时间跨度（优先用显式传入的日期区间）
    if start_date and end_date:
        years = max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25, 0.01)
    elif "date" in executed.columns and len(executed) >= 2:
        dates = pd.to_datetime(executed["date"])
        years = max((dates.max() - dates.min()).days / 365.25, 0.01)
    else:
        years = 1.0

    if not executed.empty:
        # CAGR：用持仓加权复利曲线（_equity_vals 已在 MDD 块构建）
        if _equity > 0:
            cagr = (_equity ** (1 / years) - 1) * 100

    # Sharpe / Sortino：用 sqrt(年化交易笔数) 作为年化因子
    # 这是 per-trade Sharpe 的标准做法，比 sqrt(252/avg_hold) 更稳健
    pnl_dec = executed["pnl_pct"] / 100
    if len(pnl_dec) >= 2 and pnl_dec.std() > 0:
        n_trades_per_year = max(len(executed) / years, 1.0)
        ann_factor        = n_trades_per_year ** 0.5
        sharpe            = float(pnl_dec.mean() / pnl_dec.std() * ann_factor)
        down = pnl_dec[pnl_dec < 0]
        if len(down) > 1 and down.std() > 0:
            sortino = float(pnl_dec.mean() / down.std() * ann_factor)

    if not np.isnan(cagr) and max_dd < 0:
        calmar = cagr / abs(max_dd)

    def _f(v, sfx=""):
        return f"{v:.2f}{sfx}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"

    if "date" in df.columns and not executed.empty:
        ec = executed.copy()
        ec["month"] = pd.to_datetime(ec["date"]).dt.to_period("M")
        monthly = ec.groupby("month")["win"].mean() * 100
        monthly_str = "  |  ".join(f"{m}: {v:.0f}%" for m, v in monthly.items())
    else:
        monthly_str = "N/A"

    comp_longs  = completed[completed["action"] == "long"]
    comp_shorts = completed[completed["action"] == "short"]
    long_wr  = f"{len(comp_longs[comp_longs['win']==True]) / len(comp_longs) * 100:.0f}%" if len(comp_longs) > 0 else "N/A"
    short_wr = f"{len(comp_shorts[comp_shorts['win']==True]) / len(comp_shorts) * 100:.0f}%" if len(comp_shorts) > 0 else "N/A"

    return {
        "total_signals":    len(df),
        "traded_signals":   len(traded),
        "executed_trades":  len(executed),
        "completed_trades": len(completed),
        "timeout_count":    timeout_cnt,
        "no_trade_cnt":     no_trade,
        "invalid_rr_cnt":   invalid_rr,
        "no_trade_rate":    f"{no_trade / len(df) * 100:.1f}%",
        "win_count":        len(wins),
        "loss_count":       len(losses),
        "win_rate":         f"{win_rate:.1f}%  (基于 {len(completed)} 笔确认交易，不含 {timeout_cnt} 笔超时)",
        "long_win_rate":    long_wr,
        "short_win_rate":   short_wr,
        "avg_pnl_pct":     f"{avg_pnl:.2f}%",
        "avg_win_pct":     f"{avg_win:.2f}%",
        "avg_loss_pct":    f"{avg_loss:.2f}%",
        "profit_factor":   f"{profit_factor:.2f}",
        "max_drawdown":    f"{max_dd:.2f}%",
        "total_return":    f"{executed['pnl_pct'].sum():.2f}%" if not executed.empty else "0.00%",
        "cagr":            _f(cagr, "%"),
        "sharpe_ratio":    _f(sharpe),
        "sortino_ratio":   _f(sortino),
        "calmar_ratio":    _f(calmar),
        "monthly_winrate": monthly_str,
    }


# ─────────────────────────────────────────────
# Buy-and-Hold 基准对比
# ─────────────────────────────────────────────

def compute_buyhold_return(asset_ticker: str, start: str, end: str) -> dict:
    """计算同期 Buy-and-Hold 收益，作为策略超额收益基准。"""
    df = _cached_slice(asset_ticker, start, end, "1d")
    if df.empty or len(df) < 2:
        return {}
    buy_price  = float(df.iloc[0]["Close"].squeeze())
    sell_price = float(df.iloc[-1]["Close"].squeeze())
    bh_return  = (sell_price - buy_price) / buy_price * 100
    days  = (df.index[-1] - df.index[0]).days
    years = max(days / 365.25, 0.01)
    bh_cagr = ((1 + bh_return / 100) ** (1 / years) - 1) * 100
    return {
        "bh_total_return": f"{bh_return:.2f}%",
        "bh_cagr":         f"{bh_cagr:.2f}%",
        "bh_start_price":  round(buy_price, 2),
        "bh_end_price":    round(sell_price, 2),
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
                 resume: bool = False, start_from: str = None,
                 oos_split: float = 0.0, slippage_pct: float = 0.001,
                 reproducible: bool = False, second_model: str = None):
    output_dir   = Path(f"{asset_ticker.lower()}_backtest_results")
    signals_file = output_dir / "signals.csv"
    perf_file    = output_dir / "performance.csv"
    output_dir.mkdir(exist_ok=True)

    _, src = _get_api_client()
    system_prompt = _build_system_prompt(asset_ticker)
    print(f"[{asset_ticker}] 回测: {start} ~ {end}  |  step={step}  |  model={model}  |  API={src}")
    print(f"评估窗口: {eval_days} 天  |  rate_limit={rate_limit}s  |  dry_run={dry_run}")
    print(f"滑点: {slippage_pct*100:.2f}%/边  |  OOS比例: {oos_split*100:.0f}%  |  复现模式: {reproducible}")
    print("-" * 60)

    # 预下载所有历史数据到内存缓存，避免逐节点重复拉取
    prefetch_all_data(asset_ticker, start, end, lookback=200, eval_days=eval_days,
                      reproducible=reproducible)

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
        daily, weekly, monthly = fetch_data_up_to(asset_ticker, d)
        if daily.empty or len(daily) < 30:
            print("-> 数据不足，跳过")
            continue

        macro  = fetch_macro_for_date(d, asset_ticker)
        prompt = build_blind_prompt(asset_ticker, daily, weekly, macro, perf_metrics, monthly)
        if not prompt:
            print("-> prompt 构建失败，跳过")
            continue

        if dry_run:
            price = round(float(daily["Close"].squeeze().iloc[-1]), 2)
            print(f"-> [DRY RUN] {asset_ticker}=${price}  prompt={len(prompt)}字符")
            continue

        signal = call_dual_api(prompt, model, system_prompt, rate_limit,
                               second_model=second_model, asset_ticker=asset_ticker)
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
        trade = simulate_trade(signal, future_df, asset_ticker, eval_days, macro=macro,
                               slippage_pct=slippage_pct)

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

    # ── OOS 样本外验证 ────────────────────────────────────────────
    if oos_split > 0 and len(trading_days) >= 4:
        split_idx  = max(1, int(len(trading_days) * (1 - oos_split)))
        split_date = trading_days[split_idx]
        is_records  = [r for r in all_records if str(r.get("date", "")) < split_date]
        oos_records = [r for r in all_records if str(r.get("date", "")) >= split_date]
        print(f"\n{'='*60}")
        print(f"【样本内 IS】{start} ~ {split_date}  ({len(is_records)} 条信号)")
        print("="*60)
        is_perf = compute_performance(is_records, start, split_date)
        for k, v in is_perf.items():
            if k != "monthly_winrate":
                print(f"  {k:<25}: {v}")
        if oos_records:
            print(f"\n{'='*60}")
            print(f"【样本外 OOS】{split_date} ~ {end}  ({len(oos_records)} 条信号)  ← 关键验证")
            print("="*60)
            oos_perf = compute_performance(oos_records, split_date, end)
            for k, v in oos_perf.items():
                if k != "monthly_winrate":
                    print(f"  {k:<25}: {v}")
            bh = compute_buyhold_return(asset_ticker, split_date, end)
            if bh:
                print(f"\n  ── Buy-and-Hold 基准对比 (OOS) ──")
                oos_ret = float(oos_perf.get("total_return", "0%").replace("%", "") or 0)
                bh_ret  = float(bh["bh_total_return"].replace("%", ""))
                alpha   = oos_ret - bh_ret
                print(f"  {'策略总收益':<25}: {oos_perf.get('total_return', 'N/A')}")
                print(f"  {'B&H 总收益':<25}: {bh['bh_total_return']}")
                print(f"  {'超额收益 (Alpha)':<25}: {alpha:+.2f}%  {'✓ 跑赢' if alpha > 0 else '✗ 跑输'} Buy-and-Hold")

    _save_and_print(asset_ticker, all_records, signals_file, perf_file, start, end)


def _save_and_print(asset_ticker: str, records: list[dict], signals_file: Path, perf_file: Path,
                    start_date: str = None, end_date: str = None):
    pd.DataFrame(records).to_csv(signals_file, index=False, encoding="utf-8-sig")
    perf = compute_performance(records, start_date, end_date)
    print("\n" + "=" * 60)
    print(f"{asset_ticker} 回测绩效汇总")
    print("=" * 60)
    for k, v in perf.items():
        if k != "monthly_winrate":
            print(f"  {k:<25}: {v}")
    print(f"\n  逐月胜率: {perf.get('monthly_winrate', 'N/A')}")

    # Buy-and-Hold 基准对比
    if start_date and end_date:
        bh = compute_buyhold_return(asset_ticker, start_date, end_date)
        if bh:
            print(f"\n  ── Buy-and-Hold 基准对比 ({start_date} ~ {end_date}) ──")
            print(f"  {'策略总收益':<25}: {perf.get('total_return', 'N/A')}")
            print(f"  {'B&H 总收益':<25}: {bh['bh_total_return']}")
            print(f"  {'策略 CAGR':<25}: {perf.get('cagr', 'N/A')}")
            print(f"  {'B&H CAGR':<25}: {bh['bh_cagr']}")
            strat_ret = float(perf.get("total_return", "0%").replace("%", "") or 0)
            bh_ret    = float(bh["bh_total_return"].replace("%", ""))
            alpha     = strat_ret - bh_ret
            print(f"  {'超额收益 (Alpha)':<25}: {alpha:+.2f}%  {'✓ 跑赢' if alpha > 0 else '✗ 跑输'} Buy-and-Hold")

    pd.DataFrame([perf]).to_csv(perf_file, index=False, encoding="utf-8-sig")
    print(f"\n信号明细 → {signals_file}")
    print(f"绩效汇总 → {perf_file}")


# ─────────────────────────────────────────────
# 组合回测（串行持仓 + 真实佣金 + 资金曲线）
# ─────────────────────────────────────────────

def _safe_float_bt(val) -> Optional[float]:
    """鲁棒价格解析，兼容 LLM 幻觉格式（'145.5 (approx)', '~145', None）。"""
    if val is None:
        return None
    s = str(val).strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def _get_all_trading_days(asset_ticker: str, start: str, end: str) -> list[str]:
    """返回区间内所有交易日（日粒度）。"""
    df = _cached_slice(asset_ticker, start, end, "1d")
    if df.empty:
        df = _download_with_retry(asset_ticker, start, end, "1d")
    return [d.strftime("%Y-%m-%d") for d in df.index]


def _ohlcv_from_cache(asset_ticker: str, date_str: str) -> Optional[dict]:
    """从内存缓存快速取单日 OHLCV，避免重复 slice。"""
    key = (asset_ticker, "1d")
    full_df = _FULL_CACHE.get(key)
    if full_df is None or full_df.empty:
        return None

    ts = pd.Timestamp(date_str)
    try:
        idx_norm = full_df.index.normalize()
    except Exception:
        idx_norm = full_df.index
    mask = idx_norm == ts.normalize()
    if not mask.any():
        return None

    row = full_df[mask].iloc[0]

    def _sc(col):
        v = row[col]
        if hasattr(v, "squeeze"):
            v = v.squeeze()
        if hasattr(v, "item"):
            return float(v.item())
        return float(v)

    try:
        return {"open": _sc("Open"), "high": _sc("High"),
                "low": _sc("Low"),  "close": _sc("Close")}
    except (KeyError, ValueError):
        return None


def run_portfolio_backtest(
    asset_ticker: str,
    start: str,
    end: str,
    model: str,
    eval_days: int          = 65,
    step: int               = 2,
    initial_capital: float  = 100_000,
    commission_pct: float   = 0.001,
    slippage_pct: float     = 0.001,
    risk_per_trade: float   = 0.03,
    max_position_pct: float = 0.50,
    min_rr: float           = 1.5,
    min_atr_mult: float     = 0.8,
    stop_cooldown: int      = 5,
    rate_limit: int         = 20,
    resume: bool            = False,
    oos_split: float        = 0.0,
    reproducible: bool      = False,
    second_model: str       = None,
    consec_stop_limit: int  = 2,
    circuit_breaker_days: int = 15,
):
    """
    组合回测：串行持仓 + 真实资金曲线 + 双边佣金。

    相比旧版 run_backtest() 的改进：
      1. 同一时刻只允许一笔持仓（串行，不重叠）
      2. 每日检查止损/止盈是否触发（而非只在评估日检查）
      3. 买入/卖出各扣 commission_pct 佣金
      4. 资金复利：每笔交易基于当前组合净值定仓
      5. 止损后 stop_cooldown 个交易日冷却期
      6. 信号验证使用当日收盘价；次日开盘价实际入场
      7. 双模型确认：second_model 不为 None 时触发（DeepSeek 初筛 → Claude 确认）
      8. 连续止损熔断：连续 consec_stop_limit 次 STOP_LOSS → 暂停入场 circuit_breaker_days 个交易日
      9. 移动止损：盈利 ≥5% 上移至入场价（保本），≥10% 跟踪至 50% 盈利位，延长持仓至触发
    """
    output_dir = Path(f"{asset_ticker.lower()}_portfolio_backtest")
    output_dir.mkdir(exist_ok=True)

    system_prompt = _build_system_prompt(asset_ticker)
    prefetch_all_data(asset_ticker, start, end, lookback=200, eval_days=eval_days,
                      reproducible=reproducible)

    all_days = _get_all_trading_days(asset_ticker, start, end)
    print(f"\n{'='*60}")
    print(f"[组合回测] {asset_ticker}  {start} ~ {end}")
    print(f"  模型={model}  初始资金=${initial_capital:,.0f}")
    print(f"  佣金={commission_pct*100:.2f}%/边  滑点={slippage_pct*100:.2f}%/边  总摩擦={((commission_pct+slippage_pct)*2)*100:.2f}%(往返)")
    print(f"  风险/笔={risk_per_trade*100:.0f}%  仓位上限={max_position_pct*100:.0f}%  R:R≥{min_rr}")
    print(f"  评估间隔={step}天  最长持仓={eval_days}天  止损冷却={stop_cooldown}天")
    print(f"  OOS比例={oos_split*100:.0f}%  复现模式={reproducible}  共 {len(all_days)} 个交易日")
    print("=" * 60)

    # ── 状态变量 ──────────────────────────────────────────────────
    cash: float            = float(initial_capital)
    position: Optional[dict] = None  # 当前持仓
    cooldown: int          = 0
    eval_counter: int      = 0
    pending_entry: Optional[dict] = None  # 昨日信号，今日开盘入场

    # 连续止损熔断
    consecutive_stops: int    = 0
    circuit_breaker_end_idx: int = -1  # all_days 里的索引上界（含），到此日前不入场

    equity_curve: list[dict]  = []
    trade_records: list[dict] = []
    signal_records: list[dict] = []

    # ── 主循环：逐日迭代 ───────────────────────────────────────────
    for idx, today_str in enumerate(all_days):
        ohlcv = _ohlcv_from_cache(asset_ticker, today_str)
        if ohlcv is None:
            continue
        today_open  = ohlcv["open"]
        today_high  = ohlcv["high"]
        today_low   = ohlcv["low"]
        today_close = ohlcv["close"]

        # ── A. 次日开盘入场（前一日信号排队的订单）─────────────────
        if pending_entry is not None and position is None:
            pe            = pending_entry
            pending_entry = None
            pe_action     = pe.get("action", "long")

            # 滑点：做多付 ask（+slippage），做空收 bid（-slippage）
            if pe_action == "long":
                entry_price = today_open * (1 + slippage_pct)
                risk        = entry_price - pe["stop"]
                reward      = pe["target"] - entry_price
            else:  # short
                entry_price = today_open * (1 - slippage_pct)
                risk        = pe["stop"] - entry_price
                reward      = entry_price - pe["target"]

            if risk > 0 and reward > 0 and (reward / risk) >= min_rr:
                portfolio_val = cash  # 此时尚未建仓，cash = 全部净值
                risk_budget   = portfolio_val * risk_per_trade
                qty_by_risk   = int(risk_budget / risk) if risk > 0 else 0
                qty_by_cap    = int(cash * max_position_pct / entry_price)
                quantity      = min(qty_by_risk, qty_by_cap)

                # 统一将名义持仓价值从 cash 中扣除（多空均锁定等额保证金）
                cost = quantity * entry_price * (1 + commission_pct)
                if quantity > 0 and cost <= cash:
                    cash -= cost
                    actual_pct = quantity * entry_price / portfolio_val
                    position = {
                        "action":      pe_action,
                        "entry_price": entry_price,  # 含滑点的实际成交价
                        "stop":        pe["stop"],
                        "target":      pe["target"],
                        "quantity":    quantity,
                        "entry_date":  today_str,
                        "hold_days":   0,
                        "signal_date": pe["signal_date"],
                        "bias":        pe["bias"],
                        "regime":      pe["regime"],
                    }
                    print(f"  [ENTER {pe_action.upper()}] {today_str} @ {entry_price:.2f}(含滑点) ×{quantity}"
                          f"  stop={pe['stop']:.2f}  target={pe['target']:.2f}"
                          f"  rr={reward/risk:.2f}  size={actual_pct:.0%}")
            else:
                rr = reward / risk if risk > 0 else 0
                print(f"  [ENTRY_SKIP] {today_str} 入场时 rr={rr:.2f} 不足，放弃")

        # ── B. 检查现有持仓（每日止损/止盈/移动止损/超时）──────────
        if position is not None:
            position["hold_days"] += 1
            pos_action = position["action"]

            # ── 移动止损更新（仅做多）─────────────────────────────────
            if pos_action == "long":
                unrealized_pct = (today_close - position["entry_price"]) / position["entry_price"] * 100
                if unrealized_pct >= 10.0:
                    # 跟踪止损：锁定 50% 盈利
                    new_trail = position["entry_price"] + (today_close - position["entry_price"]) * 0.5
                    if new_trail > position["stop"]:
                        position["stop"] = round(new_trail, 2)
                        position["trailing"] = True
                        print(f"  [TRAIL+] {today_str} 止损上移至 {position['stop']:.2f} (锁定50%盈利, 浮盈{unrealized_pct:.1f}%)")
                elif unrealized_pct >= 8.0 and not position.get("trailing", False):
                    if position["entry_price"] > position["stop"]:
                        position["stop"] = round(position["entry_price"], 2)
                        position["trailing"] = True
                        print(f"  [TRAIL] {today_str} 止损移至入场价 {position['stop']:.2f} (保本, 浮盈{unrealized_pct:.1f}%)")

            exit_price  = None
            exit_reason = None

            if pos_action == "long":
                if today_low <= position["stop"]:
                    exit_price, exit_reason = position["stop"], "STOP_LOSS"
                elif today_high >= position["target"]:
                    exit_price, exit_reason = position["target"], "TAKE_PROFIT"
                elif position["hold_days"] >= eval_days and not position.get("trailing", False):
                    exit_price, exit_reason = today_close, "TIMEOUT"
                elif position["hold_days"] >= eval_days * 2:
                    exit_price, exit_reason = today_close, "TIMEOUT_EXTENDED"
            else:  # short：止损触发方向相反
                if today_high >= position["stop"]:
                    exit_price, exit_reason = position["stop"], "STOP_LOSS"
                elif today_low <= position["target"]:
                    exit_price, exit_reason = position["target"], "TAKE_PROFIT"
                elif position["hold_days"] >= eval_days:
                    # 空头无移动止损，直接超时平仓
                    exit_price, exit_reason = today_close, "TIMEOUT"

            if exit_price is not None:
                if pos_action == "long":
                    # 做多卖出：收 bid（价格 - 滑点）
                    effective_exit = exit_price * (1 - slippage_pct)
                    fee            = position["quantity"] * effective_exit * commission_pct
                    # 归还资本 + 盈亏
                    proceeds       = position["quantity"] * effective_exit - fee
                    pnl_pct        = (effective_exit - position["entry_price"]) / position["entry_price"] * 100
                else:
                    # 做空回补：付 ask（价格 + 滑点）
                    effective_exit = exit_price * (1 + slippage_pct)
                    fee            = position["quantity"] * effective_exit * commission_pct
                    # 归还保证金 + 空头盈亏（entry - exit）
                    proceeds       = position["quantity"] * (2 * position["entry_price"] - effective_exit) - fee
                    pnl_pct        = (position["entry_price"] - effective_exit) / position["entry_price"] * 100
                cash           += proceeds
                portfolio_value = cash  # 平仓后全为现金

                trade_records.append({
                    "entry_date":      position["entry_date"],
                    "exit_date":       today_str,
                    "signal_date":     position.get("signal_date", ""),
                    "action":          position["action"],
                    "entry_price":     round(position["entry_price"], 4),
                    "exit_price":      round(effective_exit, 4),
                    "stop":            position["stop"],
                    "target":          position["target"],
                    "exit_reason":     exit_reason,
                    "quantity":        position["quantity"],
                    "pnl_pct":         round(pnl_pct, 4),
                    "win":             pnl_pct > 0,
                    "hold_days":       position["hold_days"],
                    "bias_score":      position["bias"],
                    "regime":          position["regime"],
                    "portfolio_value": round(portfolio_value, 2),
                    "trailing":        position.get("trailing", False),
                })
                print(f"  [EXIT]  {today_str} @ {effective_exit:.2f}(含滑点)  {exit_reason:<16}"
                      f"  pnl={pnl_pct:+.2f}%  portfolio=${portfolio_value:,.0f}"
                      + ("  [移动止损]" if position.get("trailing") else ""))

                if exit_reason == "STOP_LOSS":
                    cooldown     = stop_cooldown + 1
                    eval_counter = 0
                    # ── 连续止损熔断 ────────────────────────────────
                    if not position.get("trailing", False):
                        # 只有"原始止损"（非移动止损保本出场）才累加
                        consecutive_stops += 1
                        if consecutive_stops >= consec_stop_limit:
                            circuit_breaker_end_idx = idx + circuit_breaker_days
                            print(f"  [熔断] 连续止损 {consecutive_stops} 次 → 暂停入场 {circuit_breaker_days} 个交易日"
                                  f"（至 {all_days[min(circuit_breaker_end_idx, len(all_days)-1)]}）")
                            consecutive_stops = 0  # 重置计数
                    else:
                        # 移动止损出场（保本或盈利）= 不算亏损，重置计数
                        consecutive_stops = 0
                else:
                    cooldown          = 0
                    eval_counter      = 0
                    consecutive_stops = 0  # 止盈/超时出场重置
                position = None

        # ── C. 更新资金曲线 ──────────────────────────────────────
        if position is not None:
            if position["action"] == "long":
                pos_value = position["quantity"] * today_close
            else:  # short：保证金 + 未实现盈亏
                pos_value = position["quantity"] * (2 * position["entry_price"] - today_close)
        else:
            pos_value = 0
        mark_to_market = cash + pos_value
        equity_curve.append({"date": today_str, "portfolio_value": round(mark_to_market, 2)})

        # ── D. 冷却期 ────────────────────────────────────────────
        if cooldown > 0:
            cooldown -= 1
            continue

        # ── D2. 连续止损熔断期 ────────────────────────────────────
        if idx <= circuit_breaker_end_idx:
            remaining = circuit_breaker_end_idx - idx
            if remaining % 5 == 0:  # 每5天打印一次提示
                print(f"  [熔断中] {today_str} 熔断期剩余 {remaining} 天，跳过信号")
            continue

        # ── E. 持仓中跳过信号生成 ────────────────────────────────
        if position is not None:
            continue

        # ── F. 评估日门控 ────────────────────────────────────────
        eval_counter += 1
        if (eval_counter - 1) % step != 0:  # step=1→每日, step=2→隔日, 以此类推
            continue

        # ── G. 调用 LLM 生成信号 ─────────────────────────────────
        daily, weekly, monthly = fetch_data_up_to(asset_ticker, today_str)
        if daily.empty or len(daily) < 30:
            continue

        macro  = fetch_macro_for_date(today_str, asset_ticker)
        prompt = build_blind_prompt(asset_ticker, daily, weekly, macro, None, monthly)
        if not prompt:
            continue

        print(f"[{idx+1:>3}/{len(all_days)}] {today_str}  LLM...", end="  ", flush=True)
        signal = call_dual_api(prompt, model, system_prompt, rate_limit,
                               second_model=second_model, asset_ticker=asset_ticker)
        if not signal:
            print("API_FAIL")
            continue

        asset_list = signal.get("asset_analysis", [])
        sig = next((x for x in asset_list if x.get("asset", "").upper() == asset_ticker.upper()), None)
        if not sig:
            print("PARSE_FAIL")
            continue

        action = sig.get("action", "no_trade")
        bias   = float(sig.get("bias_score") or 0)
        regime = sig.get("regime", "")
        pt     = _safe_float_bt(sig.get("profit_target"))
        sl     = _safe_float_bt(sig.get("stop_loss"))

        print(f"action={action}  bias={bias:.2f}  regime={regime}", end="  ")
        signal_records.append({"date": today_str, "action": action,
                                "bias_score": bias, "regime": regime,
                                "profit_target": pt, "stop_loss": sl})

        if action not in ("long", "short") or bias < 0.45:
            print("→ SKIP")
            continue

        # ── 个股周线死叉：硬过滤，禁止做多 ────────────────────────────
        # 回测结果显示：2022年熊市中做多4次连续止损(-18%~-24%)，根本原因是个股
        # 自身周线EMA50<EMA200（死叉）时QQQ尚未确认死叉，仅靠提示词约束不足。
        # 此处强制拦截，避免在确认下跌趋势的个股上做多。
        if action == "long" and not weekly.empty and "Close" in weekly.columns:
            wc = weekly["Close"].squeeze().dropna()
            w_e50  = calc_ema(wc, 50).dropna()
            w_e200 = calc_ema(wc, 200).dropna()
            if len(w_e50) > 0 and len(w_e200) > 0:
                if float(w_e50.iloc[-1]) < float(w_e200.iloc[-1]):
                    print(f"→ LONG_BLOCKED(周线死叉 EMA50={w_e50.iloc[-1]:.1f}<EMA200={w_e200.iloc[-1]:.1f})")
                    signal_records[-1]["action"] = "blocked_death_cross"
                    continue

        if pt is None or sl is None:
            print("→ MISSING_LEVELS")
            continue

        # 用当日收盘价验证 R:R（入场在次日开盘，这里做预筛）
        if action == "long":
            risk_chk   = today_close - sl
            reward_chk = pt - today_close
        else:  # short
            risk_chk   = sl - today_close
            reward_chk = today_close - pt
        if risk_chk <= 0 or reward_chk <= 0 or (reward_chk / risk_chk) < min_rr:
            rr = reward_chk / risk_chk if risk_chk > 0 else 0
            print(f"→ BAD_RR({rr:.2f})")
            continue

        # 排队，次日开盘入场
        pending_entry = {"action": action, "stop": round(sl, 2), "target": round(pt, 2),
                         "signal_date": today_str, "bias": bias, "regime": regime}
        print(f"→ QUEUED({action})  stop={sl:.2f}  target={pt:.2f}")

    # ── 回测结束：强平未平仓位 ────────────────────────────────────
    if position is not None and all_days:
        last_ohlcv = _ohlcv_from_cache(asset_ticker, all_days[-1])
        if last_ohlcv:
            if position["action"] == "long":
                ep       = last_ohlcv["close"] * (1 - slippage_pct)
                fee      = position["quantity"] * ep * commission_pct
                cash    += position["quantity"] * ep - fee
                pnl      = (ep - position["entry_price"]) / position["entry_price"] * 100
            else:  # short
                ep       = last_ohlcv["close"] * (1 + slippage_pct)
                fee      = position["quantity"] * ep * commission_pct
                cash    += position["quantity"] * (2 * position["entry_price"] - ep) - fee
                pnl      = (position["entry_price"] - ep) / position["entry_price"] * 100
            trade_records.append({
                "entry_date": position["entry_date"], "exit_date": all_days[-1],
                "signal_date": position.get("signal_date", ""),
                "action": position["action"],
                "entry_price": round(position["entry_price"], 4),
                "exit_price": round(ep, 4),
                "stop": position["stop"], "target": position["target"],
                "exit_reason": "END_OF_BACKTEST",
                "quantity": position["quantity"], "pnl_pct": round(pnl, 4),
                "win": pnl > 0, "hold_days": position["hold_days"],
                "bias_score": position["bias"], "regime": position["regime"],
                "portfolio_value": round(cash, 2),
            })
            position = None

    # ── 保存结果 ──────────────────────────────────────────────────
    trades_df  = pd.DataFrame(trade_records)
    equity_df  = pd.DataFrame(equity_curve)
    signals_df = pd.DataFrame(signal_records)

    trades_df.to_csv(output_dir / "trades.csv",   index=False, encoding="utf-8-sig")
    equity_df.to_csv(output_dir / "equity.csv",   index=False, encoding="utf-8-sig")
    signals_df.to_csv(output_dir / "signals.csv", index=False, encoding="utf-8-sig")

    # ── 绩效统计 ──────────────────────────────────────────────────
    final_value  = cash
    total_return = (final_value - initial_capital) / initial_capital * 100

    print("\n" + "=" * 60)
    print(f"[组合回测结果] {asset_ticker}  初始资金 ${initial_capital:,.0f}")
    print("=" * 60)
    print(f"  最终资金          : ${final_value:>12,.2f}")
    print(f"  总收益            : {total_return:>+10.2f}%")

    if trades_df.empty:
        print("  无有效交易")
        return

    executed = trades_df[trades_df["exit_reason"].isin(
        ["STOP_LOSS", "TAKE_PROFIT", "TIMEOUT", "END_OF_BACKTEST"])]
    # 仅 STOP_LOSS / TAKE_PROFIT 为"已确认交易"，计入胜率分母；TIMEOUT/END_OF_BACKTEST 为未完成持仓
    pf_completed   = executed[executed["exit_reason"].isin(["STOP_LOSS", "TAKE_PROFIT"])]
    pf_timeout_cnt = int((executed["exit_reason"].isin(["TIMEOUT", "END_OF_BACKTEST"])).sum())
    wins   = pf_completed[pf_completed["win"] == True]
    losses = pf_completed[pf_completed["win"] == False]

    win_rate      = len(wins) / len(pf_completed) * 100 if len(pf_completed) > 0 else 0
    avg_win       = float(wins["pnl_pct"].mean())   if not wins.empty   else 0.0
    avg_loss      = float(losses["pnl_pct"].mean()) if not losses.empty else 0.0
    profit_factor = (wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum())
                     if not losses.empty and losses["pnl_pct"].sum() != 0 else float("inf"))

    # 最大回撤（从资金曲线计算，比逐笔累计更准确）
    if not equity_df.empty:
        pv     = equity_df["portfolio_value"]
        max_dd = float(((pv - pv.cummax()) / pv.cummax()).min() * 100)
    else:
        max_dd = 0.0

    # ── Sharpe / Sortino（基于日度资金曲线）──────────────────────
    sharpe = sortino = float("nan")
    if not equity_df.empty and len(equity_df) > 2:
        pv_vals      = equity_df["portfolio_value"].values
        daily_ret    = np.diff(pv_vals) / pv_vals[:-1]
        if daily_ret.std() > 0:
            sharpe   = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        dn = daily_ret[daily_ret < 0]
        if len(dn) > 1 and dn.std() > 0:
            sortino  = float(daily_ret.mean() / dn.std() * np.sqrt(252))

    # ── CAGR / Calmar ────────────────────────────────────────────
    cagr = calmar = float("nan")
    backtest_days = len(all_days)
    years         = max(backtest_days / 252, 0.01)
    if total_return > -100:
        cagr   = ((1 + total_return / 100) ** (1 / years) - 1) * 100
    if not np.isnan(cagr) and max_dd < 0:
        calmar = cagr / abs(max_dd)

    def _ff(v, sfx=""):
        return f"{v:.2f}{sfx}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"

    total_days  = len(all_days)
    held_days   = int(executed["hold_days"].sum()) if not executed.empty else 0
    time_in_mkt = held_days / total_days * 100 if total_days > 0 else 0

    llm_calls  = len(signals_df)
    entry_rate = len(executed) / llm_calls * 100 if llm_calls > 0 else 0

    print(f"  总交易笔数        : {len(executed)}  (其中超时/期末平仓: {pf_timeout_cnt} 笔不计入胜率)")
    print(f"  胜率              : {win_rate:.1f}%  ({len(wins)}胜 / {len(losses)}负，基于 {len(pf_completed)} 笔确认交易)")
    print(f"  平均盈利          : {avg_win:+.2f}%")
    print(f"  平均亏损          : {avg_loss:+.2f}%")
    print(f"  盈利因子          : {profit_factor:.2f}")
    print(f"  最大回撤          : {max_dd:.2f}%")
    print(f"  CAGR              : {_ff(cagr, '%')}")
    print(f"  Sharpe Ratio      : {_ff(sharpe)}  (日度收益，年化)")
    print(f"  Sortino Ratio     : {_ff(sortino)}  (仅下行波动)")
    print(f"  Calmar Ratio      : {_ff(calmar)}  (CAGR/最大回撤)")
    print(f"  持仓时间占比      : {time_in_mkt:.1f}%  ({held_days}/{total_days} 天)")
    print(f"  LLM 调用次数      : {llm_calls}")
    print(f"  信号触发入场率    : {entry_rate:.1f}%")

    # ── Buy-and-Hold 基准对比 ─────────────────────────────────────
    bh = compute_buyhold_return(asset_ticker, start, end)
    if bh:
        print(f"\n  ── Buy-and-Hold 基准对比 ({start} ~ {end}) ──")
        print(f"  {'策略总收益':<25}: {total_return:+.2f}%")
        print(f"  {'B&H 总收益':<25}: {bh['bh_total_return']}")
        print(f"  {'策略 CAGR':<25}: {_ff(cagr, '%')}")
        print(f"  {'B&H CAGR':<25}: {bh['bh_cagr']}")
        bh_ret = float(bh["bh_total_return"].replace("%", ""))
        alpha  = total_return - bh_ret
        print(f"  {'超额收益 (Alpha)':<25}: {alpha:+.2f}%  {'✓ 跑赢' if alpha > 0 else '✗ 跑输'} Buy-and-Hold")

    # ── OOS 样本外对比报告 ────────────────────────────────────────
    if oos_split > 0 and not executed.empty:
        oos_split_date = None
        if not executed.empty and "entry_date" in executed.columns:
            dates_sorted = sorted(executed["entry_date"].unique())
            split_idx    = max(1, int(len(dates_sorted) * (1 - oos_split)))
            if split_idx < len(dates_sorted):
                oos_split_date = dates_sorted[split_idx]
        if oos_split_date:
            is_trades  = executed[executed["entry_date"] < oos_split_date]
            oos_trades = executed[executed["entry_date"] >= oos_split_date]

            def _subset_stats(df_sub, label):
                if df_sub.empty:
                    print(f"\n  [{label}] 无交易记录")
                    return
                w  = df_sub[df_sub["win"] == True]
                l  = df_sub[df_sub["win"] == False]
                wr = len(w) / len(df_sub) * 100
                pf = w["pnl_pct"].sum() / abs(l["pnl_pct"].sum()) if not l.empty and l["pnl_pct"].sum() != 0 else float("inf")
                print(f"\n  ── {label} ({df_sub['entry_date'].min()} ~ {df_sub['entry_date'].max()}) ──")
                print(f"    交易笔数: {len(df_sub)}  胜率: {wr:.1f}%  ({len(w)}胜/{len(l)}负)  盈利因子: {pf:.2f}")
                print(f"    平均盈利: {float(w['pnl_pct'].mean()):+.2f}%  平均亏损: {float(l['pnl_pct'].mean()):+.2f}%" if not w.empty and not l.empty else "")

            print(f"\n  {'='*50}")
            print(f"  OOS 分割点: {oos_split_date}  (后{oos_split*100:.0f}%为OOS)")
            _subset_stats(is_trades,  "样本内 IS")
            _subset_stats(oos_trades, "样本外 OOS ← 关键验证")

    print()
    if not executed.empty:
        print("  交易明细:")
        for _, row in executed.iterrows():
            print(f"    {row['entry_date']} → {str(row['exit_date']):>12}"
                  f"  {str(row['exit_reason']):<16}  {row['pnl_pct']:>+6.2f}%"
                  f"  ×{int(row['quantity'])}  portfolio=${row['portfolio_value']:>10,.0f}")
    print(f"\n  交易明细  → {output_dir}/trades.csv")
    print(f"  资金曲线  → {output_dir}/equity.csv")
    print(f"  信号记录  → {output_dir}/signals.csv")


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

  # 旧版回测 + OOS验证（后20%为样本外）+ 真实滑点
  ALIYUN_API_KEY=sk-xxx python3 tech_backtest_engine.py --ticker NVDA --start 2025-01-01 --end 2025-12-31 --oos-split 0.2 --slippage 0.001

  # 【推荐】组合回测（串行持仓 + 真实佣金 + 资金曲线 + Sharpe/CAGR + Buy-and-Hold对比）
  ALIYUN_API_KEY=sk-xxx python3 tech_backtest_engine.py --ticker NVDA --portfolio --start 2025-01-01 --end 2025-12-31 --oos-split 0.2 --slippage 0.001

  # 复现模式：强制使用磁盘缓存，确保回测可复现（不受Yahoo数据修订影响）
  ALIYUN_API_KEY=sk-xxx python3 tech_backtest_engine.py --ticker NVDA --portfolio --start 2025-01-01 --end 2025-12-31 --reproducible

  # 断点续跑（旧版）
  ALIYUN_API_KEY=sk-xxx python3 tech_backtest_engine.py --ticker MSFT --start 2025-01-01 --end 2025-12-31 --model deepseek-r1 --resume
        """
    )
    parser.add_argument("--ticker",         required=True,              help="股票代码，如 NVDA, MSFT, GOOGL")
    parser.add_argument("--generate",       action="store_true",        help="生成 Prompt 文件（无需 API Key）")
    parser.add_argument("--evaluate",       action="store_true",        help="评估已有响应文件（无需 API Key）")
    parser.add_argument("--portfolio",      action="store_true",        help="【推荐】组合回测：串行持仓+真实佣金+资金曲线")
    parser.add_argument("--start",          default="2024-01-01",       help="回测开始日期 YYYY-MM-DD")
    parser.add_argument("--end",            default="2024-12-31",       help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--step",           default=1,    type=int,     help="每隔N个交易日触发LLM一次（默认1=每日评估）")
    parser.add_argument("--eval-days",      default=65,   type=int,     help="最长持仓天数（默认65，约3个月，与中长线策略对齐）")
    parser.add_argument("--model",          default="deepseek-reasoner", help="模型 ID（默认 deepseek-reasoner）")
    parser.add_argument("--rate-limit",     default=20,   type=int,     help="API 调用间隔秒数（默认20）")
    parser.add_argument("--dry-run",        action="store_true",        help="只验证数据，不调用 API")
    parser.add_argument("--resume",         action="store_true",        help="跳过已完成节点，追加合并（旧版用）")
    parser.add_argument("--start-from",     default=None,               help="从指定日期开始（YYYY-MM-DD，旧版用）")
    parser.add_argument("--capital",        default=100000, type=float, help="组合回测初始资金（默认 100000）")
    parser.add_argument("--commission",     default=0.001,  type=float, help="单边佣金比例（默认 0.001 = 0.1%%）")
    parser.add_argument("--slippage",       default=0.001,  type=float, help="单边滑点（买卖价差+冲击，默认 0.001 = 0.1%%）")
    parser.add_argument("--risk-per-trade", default=0.03,   type=float, help="每笔最大亏损占净值比（默认 0.03 = 3%%）")
    parser.add_argument("--stop-cooldown",  default=5,      type=int,   help="止损后冷却天数（默认5）")
    parser.add_argument("--oos-split",      default=0.0,    type=float, help="样本外比例，0.2=后20%%为OOS（默认0=不分割）")
    parser.add_argument("--reproducible",   action="store_true",        help="复现模式：强制使用磁盘缓存，不重新下载数据")
    parser.add_argument("--second-model",   default=None,               help="双模型确认模型（如 claude-sonnet-4-6），None=单模型")
    parser.add_argument("--consec-stop-limit", default=2, type=int,     help="连续止损次数触发熔断（默认2）")
    parser.add_argument("--circuit-breaker-days", default=15, type=int, help="熔断后暂停入场天数（默认15）")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    if args.generate:
        run_generate(ticker, args.start, args.end, args.step, args.eval_days)
    elif args.evaluate:
        run_evaluate(ticker, args.eval_days)
    elif args.portfolio:
        run_portfolio_backtest(
            asset_ticker         = ticker,
            start                = args.start,
            end                  = args.end,
            model                = args.model,
            eval_days            = args.eval_days,
            step                 = args.step,
            initial_capital      = args.capital,
            commission_pct       = args.commission,
            slippage_pct         = args.slippage,
            risk_per_trade       = args.risk_per_trade,
            stop_cooldown        = args.stop_cooldown,
            rate_limit           = args.rate_limit,
            oos_split            = args.oos_split,
            reproducible         = args.reproducible,
            second_model         = args.second_model,
            consec_stop_limit    = args.consec_stop_limit,
            circuit_breaker_days = args.circuit_breaker_days,
        )
    else:
        run_backtest(
            asset_ticker  = ticker,
            start         = args.start,
            end           = args.end,
            step          = args.step,
            model         = args.model,
            eval_days     = args.eval_days,
            dry_run       = args.dry_run,
            rate_limit    = args.rate_limit,
            resume        = args.resume,
            start_from    = args.start_from,
            oos_split     = args.oos_split,
            slippage_pct  = args.slippage,
            reproducible  = args.reproducible,
            second_model  = args.second_model,
        )
