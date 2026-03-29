"""
Google (GOOGL) 科技股 LLM 交易策略回测引擎
- 防时间泄漏：Prompt 不含具体日期
- 科技股专用指标：QQQ相对强度、利率敏感度、VIX 风险环境
- 复用 gold_analysis.py 的技术指标计算函数

三种运行模式：

  【模式一】生成 Prompt 文件（无需 API Key）
    python3 google_backtest.py --generate --start 2024-01-01 --end 2025-12-31 --step 5

  【模式二】评估已有响应（无需 API Key）
    python3 google_backtest.py --evaluate

  【模式三】全自动回测（需要 DeepSeek API Key）
    python3 google_backtest.py --start 2024-01-01 --end 2025-12-31 --step 5

依赖：
    pip install anthropic yfinance pandas numpy curl_cffi urllib3 openai
"""

import argparse
import json
import re
import time
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import urllib3
import yfinance as yf
from curl_cffi import requests as curl_requests
from openai import OpenAI

# 修复 yfinance SQLite 时区缓存冲突
yf.set_tz_cache_location(tempfile.mkdtemp())

# 复用 gold_analysis.py 的指标计算函数
from gold_analysis import (
    calc_ema, calc_macd, calc_rsi, calc_atr,
    calc_bollinger_bands, calc_stochastic, calc_adx, calc_obv, calc_roc,
    fmt_series, compute_indicators,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ─────────────────────────────────────────────
# 常量配置
# ─────────────────────────────────────────────

TICKER         = "GOOGL"
OUTPUT_DIR     = Path("googl_backtest_results")
PROMPTS_DIR    = Path("googl_backtest_prompts")
RESPONSES_DIR  = Path("googl_backtest_responses")
SIGNALS_FILE   = OUTPUT_DIR / "signals.csv"
PERF_FILE      = OUTPUT_DIR / "performance.csv"

EVAL_DAYS      = 15     # 最长持仓天数
LOOKBACK_DAYS  = 200    # 回测节点向前取数据天数

DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "sk-9574b3366dfd41178a5493d0f6af33c0")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


# ─────────────────────────────────────────────
# 历史绩效反馈（从 signals.csv / performance.csv 读取）
# ─────────────────────────────────────────────

def load_googl_perf_metrics() -> dict:
    """
    读取 GOOGL 回测绩效数据，注入到 Prompt 中形成性能反馈闭环。
    返回 dict，包含 win_rate、consecutive_losses 等关键指标。
    """
    perf: dict = {}

    # 从 performance.csv 读取汇总指标
    try:
        if PERF_FILE.exists():
            df = pd.read_csv(PERF_FILE)
            if not df.empty:
                row = df.iloc[0]
                perf["win_rate"]     = float(str(row.get("win_rate", "50%")).replace("%", ""))
                perf["total_return"] = float(str(row.get("total_return", "0%")).replace("%", ""))
                perf["max_drawdown"] = float(str(row.get("max_drawdown", "0%")).replace("%", ""))
                perf["avg_win_pct"]  = float(str(row.get("avg_win_pct", "0%")).replace("%", ""))
                perf["avg_loss_pct"] = float(str(row.get("avg_loss_pct", "0%")).replace("%", ""))
    except Exception:
        pass

    # 从 signals.csv 计算最近连续亏损笔数
    try:
        if SIGNALS_FILE.exists():
            df = pd.read_csv(SIGNALS_FILE)
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
# 科技股专用 System Prompt（防时间泄漏）
# ─────────────────────────────────────────────

TECH_STOCK_SYSTEM = """# ROLE DEFINITION

You are a Senior Technology Equity Strategist with 15+ years of experience trading FAANG/Magnificent-7 stocks.

Your mission: Analyze Google (GOOGL/Alphabet) to generate high-probability daily/weekly swing trading signals with disciplined risk management.

---

# TRADING ENVIRONMENT

- **Asset**: GOOGL (Alphabet Inc. Class A)
- **Timeframe Focus**: Daily (D1) for entry/exit; Weekly (W1) for trend bias
- **Decision Frequency**: Once per signal period
- **Objective**: Identify trend continuations, momentum breakouts, and key reversal setups
- **Position Duration**: Expected hold time is 3 to 15 days
- **Sector**: Technology / Digital Advertising / Cloud Computing

---

# TECH STOCK CONTEXT

## Key Drivers for GOOGL

1. **AI/Cloud Growth**: Google Cloud margin expansion and Gemini AI adoption trajectory
2. **Digital Ad Market**: Cyclical sensitivity to macro slowdowns (ad budgets cut first in recessions)
3. **Rate Sensitivity**: Growth stocks are highly sensitive to 10Y Treasury yields — rising yields compress P/E multiples
4. **Sector Rotation**: Monitor QQQ (Nasdaq ETF) relative strength — when QQQ is weak, individual tech names face headwinds
5. **Earnings Cadence**: GOOGL reports quarterly (late Jan, late April, late July, late October)
6. **Regulatory Risk**: Antitrust proceedings in Search/Ad Tech can create headline risk

---

# DATA INTERPRETATION GUIDELINES

## ⚠️ CRITICAL: DATA ORDERING

**ALL price and indicator series are ordered: OLDEST → NEWEST**

**The LAST element in each array is the MOST RECENT data point.**

Do NOT confuse the order. This is the most common error causing wrong decisions.

## Technical Indicators

- **EMA (20/50/200-day)**:
  - Price > EMA200 = Long-term bull trend; Price < EMA200 = Bear trend
  - EMA20 > EMA50 = Short-term bullish momentum
  - EMA50 > EMA200 = Golden Cross (strong bull); EMA50 < EMA200 = Death Cross (bear)

- **MACD**:
  - Positive = Bullish momentum; Negative = Bearish momentum
  - Histogram widening = trend accelerating; Histogram narrowing = exhaustion warning
  - MACD crossover (signal line) = near-term momentum shift

- **RSI — Regime-Dependent Interpretation**:
  - **In a Trending regime**: RSI >70 = Momentum CONFIRMATION for tech growth stocks. In strong bull trends (like GOOGL's 2023-2024 AI rally), RSI can stay above 70 for weeks. RSI >85 = note extension, but RSI alone is NOT a sell signal in a Trending market. Only concern yourself with RSI in trending markets if you see clear **bearish divergence** (price makes new high but RSI makes lower high).
  - **In a Mean-Reverting or Choppy regime**: RSI >70 = Overbought, reversal risk. RSI <30 = Oversold, bounce risk.
  - ⚠️ **Biggest mistake**: Refusing to go long a trending tech stock because RSI is >70. Technology stocks with AI/cloud tailwinds can sustain RSI >70 for months.

- **ATR (14-day)**: Sets stop distance for swing trades. For GOOGL, typical ATR is 2-5% of price.

- **Volume**:
  - Rising price + Rising volume = Confirmed institutional accumulation
  - Rising price + Declining volume = Distribution warning, suspect breakout
  - Note: Single-day low volume anomaly (e.g., holiday) is NOT a veto signal.

- **QQQ Relative Strength (RS)**:
  - GOOGL/QQQ ratio trending UP = Alpha generation vs. sector (outperforming)
  - GOOGL/QQQ ratio trending DOWN = GOOGL lagging sector (rotation away from name)
  - If GOOGL is underperforming QQQ in a sector uptrend, reduce conviction

---

# ANALYSIS FRAMEWORK

## 1. Macro & Sector Context First

1. **Rate Environment**: 10Y Treasury yield direction is critical for growth stock multiples
   - Yields rising + high levels (>4.5%) = multiple compression headwind, reduce long bias
   - Yields falling = multiple expansion tailwind, support for growth stocks
2. **VIX / Risk Appetite**:
   - VIX < 15 = Low volatility, risk-on, tech stocks typically bid
   - VIX 15-25 = Normal volatility, evaluate on technicals
   - VIX > 25 = Elevated fear, growth stocks under pressure, reduce long bias
   - VIX > 35 = Panic/crisis, wait for stabilization before entering longs
3. **DXY (USD)**: Strong dollar creates headwinds for multinational tech revenue (GOOGL earns ~50% internationally)
4. **QQQ Trend**: The Nasdaq composite trend is the sector tide — swimming against it requires high conviction

## 2. Technical Context

- **Primary Trend**: EMA50 vs EMA200 for Golden/Death Cross
- **Momentum**: MACD histogram and RSI direction
- **Volatility**: Bollinger Band position (%B) and ATR sizing
- **Volume Confirmation**: OBV trend alignment with price

## 3. Regime Classification

| Regime | Signals | RSI Rule | Approach |
|--------|---------|----------|----------|
| **Trending** | Price > EMA20 & EMA50, MACD positive, EMA50 > EMA200 | RSI >70 = momentum, NOT reversal signal. Look for divergence only | Buy pullbacks to EMA20/50; ride with trailing stop |
| **Mean-Reverting** | Price oscillating around EMA, RSI at extremes then reverting | RSI >70 = Overbought (fade); RSI <30 = Oversold (bounce) | Fade extremes with tight stops; target EMA mean |
| **Choppy** | Flat EMAs, MACD near zero, range-bound, no sector catalyst | RSI unhelpful | no_trade — wait for regime break |

---

# ACTION SPACE

1. **long**: Bullish position
   - Trending: Golden Cross, MACD positive, price > EMA50 — buy dips to EMA20/50
   - Mean-Reverting: RSI < 35, price at key support, MACD turning up from negative
   - QQQ must NOT be in severe downtrend (don't fight the sector)

2. **short**: Bearish position (use only when high conviction)
   - Trending: Death Cross, MACD negative, price < EMA50
   - Mean-Reverting: RSI > 65, price at key resistance, MACD turning down
   - Note: Shorting mega-cap tech has asymmetric risk — only short in clear downtrend

3. **no_trade**: No position
   - Choppy/Unclear regime
   - Cannot achieve R:R ≥ 2.0 with stop ≥ 0.8 × ATR-14
   - VIX > 35 (crisis mode, wait for stabilization)
   - Conflicting signals across timeframes

**Default to no_trade when in doubt — but do NOT default to no_trade because RSI is "too high" in a Trending regime.**

---

# RISK MANAGEMENT (MANDATORY)

For EVERY long/short signal, specify:

1. **entry_zone** (string): Current price ± 0.3 × ATR-14 (stay near market price)
2. **profit_target** (float): R:R ≥ 2.0 from **current_price** (not entry_zone)
3. **stop_loss** (float): ≥ 0.8 × ATR-14 from current_price; for longs place below key support
4. **risk_reward_ratio** (float): Calculated from current_price:
   - Long:  (profit_target − current_price) / (current_price − stop_loss)
   - Short: (current_price − profit_target) / (stop_loss − current_price)

⚠️ MANDATORY SELF-CHECK before writing JSON:

| Check | Long | Short |
|-------|------|-------|
| Direction | profit_target > current_price > stop_loss | stop_loss > current_price > profit_target |
| R:R ≥ 2.5 (gap buffer) | (profit_target − current_price) / (current_price − stop_loss) | (current_price − profit_target) / (stop_loss − current_price) |
| Stop distance | current_price − stop_loss ≥ 0.8 × ATR-14 | stop_loss − current_price ≥ 0.8 × ATR-14 |

**R:R target is 2.5 (not 2.0)** to provide buffer against overnight gap entries. A trade valid at 2.0 R:R today may become invalid at 1.1 R:R after a 1% gap open — causing INVALID_RR at execution. Use the pre-calculated anchor targets (≥3.5×ATR) as reference.

If ANY check fails → action = "no_trade", set profit_target/stop_loss/risk_reward_ratio = null.

---

# OUTPUT FORMAT (JSON)

Return ONLY valid JSON:

```json
{
  "period": "Daily",
  "overall_market_sentiment": "Risk-On" | "Risk-Off" | "Neutral",
  "sector_assessment": "<QQQ trend and tech sector backdrop>",
  "rate_assessment": "<10Y yield direction and impact on GOOGL multiple>",
  "asset_analysis": [
    {
      "asset": "GOOGL",
      "regime": "Trending" | "Mean-Reverting" | "Choppy",
      "action": "long" | "short" | "no_trade",
      "bias_score": <float 0.0–1.0>,
      "entry_zone": "<price range string>",
      "profit_target": <float | null>,
      "stop_loss": <float | null>,
      "risk_reward_ratio": <float | null>,
      "invalidation_condition": "<objective signal that voids thesis>",
      "macro_catalyst": "<key macro/sector driver>",
      "technical_setup": "<key indicator alignment>",
      "justification": "<max 300 characters>"
    }
  ]
}
```

**Validation Rules**:
- Long: profit_target > current_price > stop_loss
- Short: stop_loss > current_price > profit_target
- R:R ≥ 2.0 always; if not, force no_trade
- Stop ≥ 0.8 × ATR-14 from current_price
- no_trade → set profit_target/stop_loss/risk_reward_ratio = null
- bias_score for no_trade typically < 0.4

---

# COMMON PITFALLS

- ⚠️ **RSI paralysis in tech trends**: The #1 mistake — GOOGL, META, MSFT all had RSI >75 for months in 2023-2024 AI rally. Refusing to buy because RSI is "overbought" in a Trending regime means missing 40%+ moves.
- ⚠️ **Fighting rate headwinds**: When 10Y yields are rapidly rising above 4.5%, growth stock multiples compress. Don't fight this with aggressive longs.
- ⚠️ **Ignoring sector rotation**: If QQQ is in a clear downtrend, standalone GOOGL longs will struggle. Wait for sector alignment.
- ⚠️ **Shorting strength**: GOOGL has structural advantages (AI, Search monopoly, Cloud). Short only in clear technical downtrends.
- ⚠️ **Volume artifacts**: Single day low volume (holiday, contract rollover) is NOT a veto.
- ⚠️ **Gap risk on entry**: GOOGL frequently gaps ±1-3% overnight due to macro news, AI narrative shifts, and pre-earnings positioning. To remain valid after typical gaps, set profit_target to achieve R:R ≥ 2.5 from current_price (not just ≥ 2.0). If a valid target cannot be identified that achieves R:R ≥ 2.5, use no_trade.
- ⚠️ **Over-extension stops**: When RSI-7 > 82 AND price is >2% above EMA20, the stock is over-extended intraday. Long entries at these levels are frequently stopped out on the next open. Raise long bias_score threshold to ≥ 0.80 in this condition, or use no_trade if the extension is extreme (RSI-7 > 88).
- ⚠️ **OBV divergence in uptrends**: Rising price with falling OBV in a Trending regime signals institutional distribution. This is NOT a minor -0.10 deduction — if OBV is falling for 3+ days while price rises, treat it as a medium-confidence warning and require bias_score ≥ 0.70 to enter long.
- ⚠️ **Death Cross 抄底陷阱（高频亏损来源）**: 当 EMA50 < EMA200（死叉）且价格低于 EMA200 时，严禁做多。历史回测显示，死叉期间"买入技术性支撑位"会演变为连续止损——价格会持续寻底，每次反弹都是新的卖出机会。只有当 EMA50 重新上穿 EMA200（黄金交叉确认）后，才可恢复做多偏好。提示词中若标注"当前 Death Cross"，则 bias_score 强制 ≤ 0.45。
- ⚠️ **大幅下跌后追空（低效空头）**: 当价格从近期高点已下跌 >20% 时，做空的风险回报极差。大跌后任何利好消息（政策、数据、技术修复）都可能引发 10%+ 的快速轧空，止损空头。空头信号的最佳时机是死叉刚形成、价格跌幅 <12% 时介入。提示词中若标注"禁止做空"，则强制 no_trade。

---

# FINAL INSTRUCTIONS

1. Assess macro backdrop first (VIX, rates, DXY, QQQ trend)
2. Classify GOOGL regime based on EMA structure and MACD
3. Apply CORRECT RSI interpretation based on regime
4. Calculate R:R from current_price; verify all risk constraints
5. Output ONLY valid JSON — no commentary outside the JSON block

**CRITICAL**: Base analysis ONLY on provided data. Do NOT use knowledge of specific dates, earnings reports, or events outside the data window. Output ONLY valid JSON.

Now analyze the GOOGL market data provided below."""


# ─────────────────────────────────────────────
# 数据获取（严格时间截断）
# ─────────────────────────────────────────────

def _make_session():
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    session = curl_requests.Session(impersonate="chrome", verify=False, proxy=proxy)
    return session


def _download_with_retry(ticker, start, end, interval, retries=3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            session = _make_session()
            df = yf.download(
                ticker, start=start, end=end,
                interval=interval, auto_adjust=True, progress=False, session=session
            )
            if df is not None and not df.empty:
                return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                print(f"  [数据获取失败] {ticker} {start}~{end}: {e}")
    return pd.DataFrame()


def fetch_data_up_to(date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """严格截断到 date 当日，不包含未来数据。"""
    end_dt    = pd.Timestamp(date)
    end_str   = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    start_str = (end_dt - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    daily  = _download_with_retry(TICKER, start_str, end_str, "1d")
    weekly = _download_with_retry(TICKER, start_str, end_str, "1wk")
    return daily, weekly


def fetch_macro_for_date(date: str) -> dict:
    """
    获取 date 当日及之前的科技股宏观数据：
    - QQQ: Nasdaq-100 ETF（科技板块标杆）
    - ^TNX: 美国10年期国债收益率
    - ^VIX: CBOE 恐慌指数
    - DX-Y.NYB: 美元指数
    """
    end_dt    = pd.Timestamp(date)
    end_str   = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    start_str = (end_dt - timedelta(days=60)).strftime("%Y-%m-%d")

    tickers = {"qqq": "QQQ", "tnx": "^TNX", "vix": "^VIX", "dxy": "DX-Y.NYB"}
    macro = {}
    for key, ticker in tickers.items():
        try:
            df = _download_with_retry(ticker, start_str, end_str, "1d")
            macro[key] = df if not df.empty else pd.DataFrame()
        except Exception:
            macro[key] = pd.DataFrame()
    return macro


def fetch_future_data(date: str) -> pd.DataFrame:
    """获取 date 之后的 GOOGL 数据，用于交易结果评估。"""
    start_str = (pd.Timestamp(date) + timedelta(days=1)).strftime("%Y-%m-%d")
    end_str   = (pd.Timestamp(date) + timedelta(days=(EVAL_DAYS + 5) * 2)).strftime("%Y-%m-%d")
    df = _download_with_retry(TICKER, start_str, end_str, "1d")
    if df.empty:
        return df
    return df.iloc[: EVAL_DAYS + 5]


def get_trading_days(start: str, end: str, step: int) -> list[str]:
    """返回区间内每隔 step 个交易日的日期列表。"""
    session = _make_session()
    df = yf.download(
        TICKER, start=start, end=end,
        interval="1d", auto_adjust=True, progress=False, session=session
    )
    all_days = [d.strftime("%Y-%m-%d") for d in df.index]
    return all_days[::step]


# ─────────────────────────────────────────────
# 构建科技股盲化 Prompt
# ─────────────────────────────────────────────

def _summarize_macro(macro: dict, googl_close: pd.Series) -> dict:
    """将宏观原始 DataFrame 转为摘要字典。"""
    result = {}

    def _last_n(df, col, n=5):
        if df.empty or col not in df.columns:
            return []
        return df[col].squeeze().dropna().tail(n).round(3).tolist()

    def _trend(vals):
        if len(vals) < 2:
            return "N/A"
        chg = (vals[-1] - vals[0]) / abs(vals[0]) * 100 if vals[0] != 0 else 0
        return f"{'↑' if chg > 0 else '↓'} {abs(chg):.1f}% (5日)"

    # QQQ
    qqq_df = macro.get("qqq", pd.DataFrame())
    qqq_closes = _last_n(qqq_df, "Close")
    result["qqq_last"]  = round(qqq_closes[-1], 2) if qqq_closes else None
    result["qqq_trend"] = _trend(qqq_closes)
    if not qqq_df.empty and "Close" in qqq_df.columns:
        qqq_ema20 = calc_ema(qqq_df["Close"].squeeze(), 20).dropna()
        result["qqq_ema20"] = round(float(qqq_ema20.iloc[-1]), 2) if not qqq_ema20.empty else None
    else:
        result["qqq_ema20"] = None

    # GOOGL vs QQQ 相对强度（最近10日比值趋势）
    if not qqq_df.empty and "Close" in qqq_df.columns and len(googl_close) > 10:
        qqq_aligned = qqq_df["Close"].squeeze().reindex(googl_close.index, method="ffill").dropna()
        googl_aligned = googl_close.reindex(qqq_aligned.index).dropna()
        common = googl_aligned.index.intersection(qqq_aligned.index)
        if len(common) >= 5:
            rs = (googl_aligned.loc[common] / qqq_aligned.loc[common]).tail(5)
            rs_list = rs.round(4).tolist()
            result["googl_qqq_rs"]       = rs_list
            result["googl_qqq_rs_trend"] = _trend(rs_list)
        else:
            result["googl_qqq_rs"]       = []
            result["googl_qqq_rs_trend"] = "N/A"
    else:
        result["googl_qqq_rs"]       = []
        result["googl_qqq_rs_trend"] = "N/A"

    # 10Y Yield
    tnx_df = macro.get("tnx", pd.DataFrame())
    tnx_closes = _last_n(tnx_df, "Close")
    result["tnx_last"]   = round(tnx_closes[-1], 3) if tnx_closes else None
    result["tnx_trend"]  = _trend(tnx_closes)
    result["tnx_series"] = tnx_closes

    # VIX
    vix_df = macro.get("vix", pd.DataFrame())
    vix_closes = _last_n(vix_df, "Close")
    result["vix_last"]  = round(vix_closes[-1], 2) if vix_closes else None
    result["vix_trend"] = _trend(vix_closes)
    if result["vix_last"]:
        v = result["vix_last"]
        result["vix_regime"] = "危机/恐慌" if v > 35 else ("高波动/Risk-Off" if v > 25 else ("中性" if v > 15 else "低波动/Risk-On"))
    else:
        result["vix_regime"] = "N/A"

    # DXY
    dxy_df = macro.get("dxy", pd.DataFrame())
    dxy_closes = _last_n(dxy_df, "Close")
    result["dxy_last"]  = round(dxy_closes[-1], 2) if dxy_closes else None
    result["dxy_trend"] = _trend(dxy_closes)

    return result


def build_blind_prompt(daily: pd.DataFrame, weekly: pd.DataFrame,
                       macro: dict | None = None,
                       perf_metrics: dict | None = None) -> str:
    """
    构建防时间泄漏的 GOOGL 分析 Prompt。
    不包含具体日期，防止模型利用训练知识作弊。
    """
    if daily.empty or weekly.empty or len(daily) < 30:
        return ""

    d_ind = compute_indicators(daily)
    w_ind = compute_indicators(weekly)

    close_d       = daily["Close"].squeeze()
    current_price = round(float(close_d.iloc[-1]), 2)
    current_ema20 = round(float(d_ind["ema20"].iloc[-1]), 2)
    current_ema50 = round(float(d_ind["ema50"].iloc[-1]), 2)
    current_ema200= round(float(d_ind["ema200"].dropna().iloc[-1]), 2) if len(d_ind["ema200"].dropna()) > 0 else None
    current_macd  = round(float(d_ind["macd"].iloc[-1]), 2)
    current_rsi7  = round(float(d_ind["rsi7"].iloc[-1]), 2)
    current_rsi14 = round(float(d_ind["rsi14"].iloc[-1]), 2)
    atr14_val     = round(float(d_ind["atr14"].iloc[-1]), 2)
    atr3_val      = round(float(d_ind["atr3"].iloc[-1]), 2)

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

    # 高级指标
    stoch_k_val  = round(float(d_ind["stoch_k"].dropna().iloc[-1]), 1)
    stoch_d_val  = round(float(d_ind["stoch_d"].dropna().iloc[-1]), 1)
    adx_val      = round(float(d_ind["adx"].dropna().iloc[-1]), 1)
    plus_di_val  = round(float(d_ind["plus_di"].dropna().iloc[-1]), 1)
    minus_di_val = round(float(d_ind["minus_di"].dropna().iloc[-1]), 1)
    bb_pct_b_val = round(float(d_ind["bb_pct_b"].dropna().iloc[-1]), 3)
    bb_bw_val    = round(float(d_ind["bb_bw"].dropna().iloc[-1]), 2)
    bb_upper_val = round(float(d_ind["bb_upper"].dropna().iloc[-1]), 2)
    bb_lower_val = round(float(d_ind["bb_lower"].dropna().iloc[-1]), 2)
    roc10_val    = round(float(d_ind["roc10"].dropna().iloc[-1]), 2)
    roc20_val    = round(float(d_ind["roc20"].dropna().iloc[-1]), 2)

    obv_series = d_ind["obv"].dropna().tail(5).tolist()
    obv_trend  = "上升" if obv_series[-1] > obv_series[0] else "下降"

    close_d_full = close_d.dropna()
    high_52w = round(float(close_d_full.tail(252).max()), 2)
    low_52w  = round(float(close_d_full.tail(252).min()), 2)
    pct_from_high = round((current_price - high_52w) / high_52w * 100, 1)
    pct_from_low  = round((current_price - low_52w) / low_52w * 100, 1)

    # 中期趋势状态
    is_death_cross  = (current_ema50 < current_ema200) if current_ema200 else False
    pct_from_ema200 = round((current_price - current_ema200) / current_ema200 * 100, 1) if current_ema200 else None

    daily_stoch_k = fmt_series(d_ind["stoch_k"], 1, n)
    daily_adx     = fmt_series(d_ind["adx"], 1, n)
    daily_roc10   = fmt_series(d_ind["roc10"], 2, n)
    daily_bb_pctb = fmt_series(d_ind["bb_pct_b"], 3, n)

    # EMA 结构判断
    ema_structure = "EMA20 > EMA50" if current_ema20 > current_ema50 else "EMA20 < EMA50"
    if current_ema200:
        ema_structure += f" > EMA200" if current_ema50 > current_ema200 else f" < EMA200"
        golden_cross = "Golden Cross (EMA50>EMA200, 多头排列)" if current_ema50 > current_ema200 else "Death Cross (EMA50<EMA200, 空头排列)"
    else:
        golden_cross = "N/A (数据不足)"

    # 预计算入场锚点
    long_entry_lo  = round(current_price - 0.3 * atr14_val, 2)
    long_entry_hi  = round(current_price + 0.3 * atr14_val, 2)
    long_stop      = round(current_price - 1.2 * atr14_val, 2)
    long_target    = round(current_price + 3.5 * atr14_val, 2)  # 扩宽至3.5x，缓解次日跳空导致的INVALID_RR
    short_entry_lo = round(current_price - 0.3 * atr14_val, 2)
    short_entry_hi = round(current_price + 0.3 * atr14_val, 2)
    short_stop     = round(current_price + 1.2 * atr14_val, 2)
    short_target   = round(current_price - 3.5 * atr14_val, 2)  # 扩宽至3.5x，缓解次日跳空导致的INVALID_RR

    # 宏观摘要
    ms = _summarize_macro(macro or {}, close_d_full) if macro is not None else _summarize_macro({}, close_d_full)

    # 中期趋势过滤与空头限制规则（动态生成，基于当前数据）
    # 触发条件：DC + 价格仍低于 EMA200（真正的下跌趋势，而非已恢复的假死叉）
    trend_filter_rules = []
    price_below_ema200 = (pct_from_ema200 is not None and pct_from_ema200 < 0)
    if is_death_cross and price_below_ema200:
        ema200_str = str(current_ema200) if current_ema200 else 'N/A'
        pf_ema200_str = f'{pct_from_ema200:+.1f}%' if pct_from_ema200 is not None else 'N/A'
        trend_filter_rules.append(
            f'- ⚠️ 【中期下跌趋势过滤 — Death Cross + 价格低于EMA200】EMA50({current_ema50}) < EMA200({ema200_str})，'
            f'价格 vs EMA200: {pf_ema200_str}：做多 bias_score 强制 <= 0.45（低于0.50门槛，自动 no_trade）。'
            f'死叉且价格未能站回EMA200，趋势尚未反转，逢低买入属系统性亏损'
        )
        if adx_val > 20:
            trend_filter_rules.append(
                f'- ⚠️ 【中期下跌趋势过滤 — Death Cross + ADX {adx_val} > 20 + 价格低于EMA200】'
                f'三重确认下跌趋势。做多 bias_score 需 >= 0.80 方可考虑（实际等同于禁止）'
            )
    if pct_from_high is not None and pct_from_high < -20:
        trend_filter_rules.append(
            f'- ⚠️ 【空头入场限制 — 大幅下跌后禁止追空】价格距52周高点已跌 {pct_from_high:.1f}%（超过20%阈值），'
            f'空头风险回报极差（轧空风险高）：禁止做空，强制 no_trade'
        )
    elif is_death_cross and price_below_ema200 and pct_from_high is not None and pct_from_high < -12:
        trend_filter_rules.append(
            f'- ⚠️ 【空头入场限制】价格距52周高点已跌 {pct_from_high:.1f}%（>12%）且低于EMA200，追空效率低，'
            f'空头 bias_score 上限 0.60（需高确信度才可做空，否则 no_trade）'
        )
    trend_filter_section = "\n".join(trend_filter_rules)

    # 性能反馈段落
    perf_feedback_section = ""
    if perf_metrics:
        lines = []
        wr  = perf_metrics.get("win_rate", 50)
        cl  = perf_metrics.get("consecutive_losses", 0)
        mdd = perf_metrics.get("max_drawdown", 0)
        if wr < 40:
            lines.append(f"- ⚠️ 历史绩效反馈：近期系统胜率 {wr:.1f}% < 40%，bias_score 门槛自动提升至 ≥ 0.65")
        if cl >= 2:
            lines.append(f"- ⚠️ 历史绩效反馈：近期连续亏损 {cl} 笔，需 bias_score ≥ 0.75 才允许入场，否则 no_trade")
        if mdd < -15:
            lines.append(f"- ⚠️ 历史绩效反馈：最大回撤已达 {mdd:.1f}%，当前处于深度回撤期，需 bias_score ≥ 0.70 才入场")
        perf_feedback_section = "\n".join(lines)

    def _fv(v, unit=""):
        return f"{v}{unit}" if v is not None else "N/A"

    macro_section = f"""
---

## 宏观与板块背景

### Nasdaq-100 ETF (QQQ)
- **QQQ 最新价**: {_fv(ms.get('qqq_last'))}  |  **5日趋势**: {ms.get('qqq_trend', 'N/A')}  |  **QQQ EMA20**: {_fv(ms.get('qqq_ema20'))}
- **QQQ 状态**: {'QQQ 高于 EMA20 — 科技板块偏多' if ms.get('qqq_last') and ms.get('qqq_ema20') and ms.get('qqq_last') > ms.get('qqq_ema20') else 'QQQ 低于 EMA20 — 科技板块承压'}

### GOOGL vs QQQ 相对强度
- **近5日 GOOGL/QQQ 比率**: {ms.get('googl_qqq_rs', [])}
- **RS 趋势**: {ms.get('googl_qqq_rs_trend', 'N/A')}
- **解读**: {'GOOGL 跑赢 QQQ — 个股 Alpha 积累' if ms.get('googl_qqq_rs_trend', '').startswith('↑') else 'GOOGL 跑输 QQQ — 注意板块内轮动'}

### 美国10年期国债收益率 (^TNX)
- **当前**: {_fv(ms.get('tnx_last'), '%')}  |  **5日趋势**: {ms.get('tnx_trend', 'N/A')}
- **近5日收益率**: {ms.get('tnx_series', [])}
- **解读**: {'收益率上升 → 成长股估值压缩，GOOGL 多头偏谨慎' if ms.get('tnx_trend','').startswith('↑') else '收益率下降 → 成长股估值扩张利好，支持多头'}
- **警示**: {'⚠️ 收益率 > 4.5%，成长股多头需额外谨慎' if ms.get('tnx_last') and ms.get('tnx_last') > 4.5 else '收益率处于合理范围'}

### VIX 恐慌指数 (^VIX)
- **当前**: {_fv(ms.get('vix_last'))}  |  **市场状态**: {ms.get('vix_regime', 'N/A')}  |  **5日趋势**: {ms.get('vix_trend', 'N/A')}
- **解读**: {'⚠️ VIX > 25，Risk-Off 环境，科技股承压，谨慎做多' if ms.get('vix_last') and ms.get('vix_last') > 25 else '市场相对平静，技术面主导'}

### 美元指数 (DXY)
- **当前**: {_fv(ms.get('dxy_last'))}  |  **5日趋势**: {ms.get('dxy_trend', 'N/A')}
- **解读**: {'强美元 → GOOGL 海外收入折算压力，适当降低多头 bias_score' if ms.get('dxy_trend','').startswith('↑') else '弱美元/美元走平 → 海外收入无汇率逆风'}
"""

    prompt = f"""# GOOGL (Alphabet Inc.) 科技股摆动交易分析请求
**数据来源**: Yahoo Finance (GOOGL)
**重要说明**: 请严格基于以下提供的数据进行分析。不得引用任何数据窗口之外的具体事件（如具体财报、监管新闻等）。

---

## 价格概要

- **当前价格**: ${current_price}
- **今日 O/H/L/C**: {today_open} / {today_high} / {today_low} / {current_price}
- **今日涨跌幅**: {day_chg:+.2f}%
- **过去5交易日**: ${close_5d:.2f} → ${last_close:.2f}  ({week_chg:+.2f}%)
- **今日成交量**: {today_vol:,}  vs.  20日均量: {vol_avg:,}  ({'放量' if vol_current > vol_avg * 1.2 else ('缩量' if vol_current < vol_avg * 0.8 else '正常')})

---

## EMA 趋势结构

- **EMA20**: {current_ema20}  |  **EMA50**: {current_ema50}  |  **EMA200**: {_fv(current_ema200)}
- **EMA排列**: {ema_structure}
- **均线交叉状态**: {golden_cross}
- **价格 vs EMA20**: {'价格高于 EMA20 ({:+.1f}%)'.format((current_price - current_ema20) / current_ema20 * 100)}

---

## 当前技术指标快照

- current_price   = {current_price}
- current_ema20   = {current_ema20}
- current_ema50   = {current_ema50}
- current_macd    = {current_macd}
- current_rsi7    = {current_rsi7}
- current_rsi14   = {current_rsi14}
- atr14           = {atr14_val}

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

- **52周高点**: {high_52w}  |  **距高点**: {pct_from_high:+.1f}%
- **52周低点**: {low_52w}   |  **距低点**: {pct_from_low:+.1f}%
- **布林带 %B**: {bb_pct_b_val:.3f}  （0=下轨, 0.5=中轨, 1=上轨）
- **布林带上轨**: {bb_upper_val}  |  **下轨**: {bb_lower_val}  |  **带宽**: {bb_bw_val:.2f}%

---

## 高级技术指标（日线）

| 指标 | 当前值 | 信号 |
|------|--------|------|
| Stochastic %K | {stoch_k_val} | {'超买 >80' if stoch_k_val > 80 else ('超卖 <20' if stoch_k_val < 20 else '中性')} |
| Stochastic %D | {stoch_d_val} | {'K>D 金叉' if stoch_k_val > stoch_d_val else 'K<D 死叉'} |
| ADX | {adx_val} | {'强趋势 >25' if adx_val > 25 else ('弱趋势 <20' if adx_val < 20 else '趋势形成')} |
| +DI / -DI | {plus_di_val} / {minus_di_val} | {'+DI>-DI 多头' if plus_di_val > minus_di_val else '-DI>+DI 空头'} |
| ROC(10日) | {roc10_val:+.2f}% | {'正动量' if roc10_val > 0 else '负动量'} |
| ROC(20日) | {roc20_val:+.2f}% | {'正动量' if roc20_val > 0 else '负动量'} |
| OBV趋势(5日) | {obv_trend} | {'量价配合' if obv_trend == '上升' else '量价背离'} |
| ATR-3 | {atr3_val} | 短期波动 |
| ATR-14 | {atr14_val} | 止损基准 |

**近15日序列（从旧到新）**：
Stochastic %K: [{daily_stoch_k}]
ADX:           [{daily_adx}]
ROC(10):       [{daily_roc10}]
BB %B:         [{daily_bb_pctb}]
{macro_section}
---

## 预计算入场锚点（基于 ATR-14={atr14_val}）

> entry_zone 必须在此范围内，否则将被判定无效 (INVALID_RR)

| 方向 | entry_zone | stop_loss | profit_target (≥2.5×R，含跳空缓冲) |
|------|------------|-----------|----------------------|
| 做多 | {long_entry_lo} – {long_entry_hi} | {long_stop} | {long_target} |
| 做空 | {short_entry_lo} – {short_entry_hi} | {short_stop} | {short_target} |

---

## 分析任务

按照系统指令的框架，分析 GOOGL 当前技术形态与宏观背景，输出如下 JSON：

```json
{{
  "period": "Daily",
  "overall_market_sentiment": "Risk-On | Risk-Off | Neutral",
  "sector_assessment": "<QQQ趋势及科技板块背景>",
  "rate_assessment": "<10Y利率方向对GOOGL估值的影响>",
  "asset_analysis": [
    {{
      "asset": "GOOGL",
      "regime": "Trending | Mean-Reverting | Choppy",
      "action": "long | short | no_trade",
      "bias_score": <0.0-1.0>,
      "entry_zone": "<必须基于上方预计算锚点>",
      "profit_target": <数字 或 null>,
      "stop_loss": <数字 或 null>,
      "risk_reward_ratio": <数字 或 null>,
      "invalidation_condition": "<使该观点失效的具体信号>",
      "macro_catalyst": "<宏观/板块驱动逻辑>",
      "technical_setup": "<指标信号综合>",
      "justification": "<不超过300字>"
    }}
  ]
}}
```

**硬性约束**：
- entry_zone 必须包含当前价格 ±1×ATR-14，不得设置离市场太远的理想价格
- risk_reward_ratio 必须 ≥ 2.0
- stop_loss 距离 current_price 不得小于 0.8×ATR-14
- Long: profit_target > current_price > stop_loss
- Short: stop_loss > current_price > profit_target
- 违反任意一条 → 改为 no_trade

**科技股专用规则**：
- MACD ({current_macd}) < 0 且 EMA20 < EMA50 时，禁止在 Trending 制度下做多
- VIX > 25 时，做多 bias_score 上限 0.60；VIX > 35 时，一律 no_trade
- 10Y 收益率 > 4.5% 且上升时，做多 bias_score 额外降低 0.05–0.10
- GOOGL 跑输 QQQ（RS 趋势下降）时，做多 bias_score 降低 0.05
- ADX ({adx_val}) < 20 时，Trending 信号降级为 Choppy
- OBV 与价格背离时，bias_score 降低 0.10；若同时 RSI-7 > 75，额外降低 0.05
- RSI-7 ({current_rsi7}) > 82 且价格偏离 EMA20 超过 2% 时，做多 bias_score 上限 0.65（延伸过度，次日开盘被扫止损风险极高）
- bias_score < 0.50 → 强制 no_trade
{trend_filter_section}
{perf_feedback_section}"""

    return prompt.strip()


# ─────────────────────────────────────────────
# JSON 解析（兼容 DeepSeek R1 think 标签）
# ─────────────────────────────────────────────

def _extract_json_by_braces(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
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
                return text[start: i + 1]
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
# DeepSeek API 调用
# ─────────────────────────────────────────────

def call_deepseek(prompt: str, model: str = "deepseek-reasoner") -> dict:
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=4000,
                messages=[
                    {"role": "system", "content": TECH_STOCK_SYSTEM},
                    {"role": "user",   "content": prompt},
                ]
            )
            return parse_signal(resp.choices[0].message.content)
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                wait = 30 * (attempt + 1)
                print(f"  [限速] 等待 {wait}s 后重试...")
                time.sleep(wait)
            else:
                print(f"  [API错误] attempt {attempt+1}/3: {e}")
                if attempt < 2:
                    time.sleep(5)
    return {}


# ─────────────────────────────────────────────
# 交易模拟（单笔）
# ─────────────────────────────────────────────

def simulate_trade(signal: dict, future_df: pd.DataFrame) -> dict:
    base = {
        "action":      None,
        "entry_price": None,
        "exit_price":  None,
        "exit_reason": "PENDING",
        "pnl_pct":     None,
        "win":         None,
        "days_held":   None,
    }

    if future_df.empty or not signal:
        base["exit_reason"] = "NO_DATA"
        return base

    asset_list    = signal.get("asset_analysis", [])
    googl_signal  = next((x for x in asset_list if x.get("asset") == "GOOGL"), None)
    if not googl_signal:
        base["exit_reason"] = "PARSE_ERROR"
        return base

    action        = googl_signal.get("action", "no_trade")
    profit_target = googl_signal.get("profit_target")
    stop_loss     = googl_signal.get("stop_loss")
    base["action"] = action

    if action == "no_trade":
        base["exit_reason"] = "NO_TRADE"
        return base

    if profit_target is None or stop_loss is None:
        base["exit_reason"] = "MISSING_LEVELS"
        return base

    profit_target = float(profit_target)
    stop_loss     = float(stop_loss)

    # 次日开盘入场
    entry_price = float(future_df.iloc[0]["Open"].squeeze())
    base["entry_price"] = entry_price

    # R:R 验证（防缺口使目标失效）
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
        elif action == "short":
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
    df       = pd.DataFrame(records)
    traded   = df[df["action"].isin(["long", "short"])].copy()
    no_trade = (df["action"] == "no_trade").sum()
    parse_err = df["exit_reason"].isin(["PARSE_ERROR", "NO_DATA", "MISSING_LEVELS"]).sum()
    invalid_rr = (df["exit_reason"] == "INVALID_RR").sum()

    if traded.empty:
        return {"error": "无有效交易信号"}

    executed = traded[~traded["exit_reason"].isin(["INVALID_RR", "MISSING_LEVELS"])]
    wins     = executed[executed["win"] == True]
    losses   = executed[executed["win"] == False]

    win_rate      = len(wins) / len(executed) * 100 if len(executed) > 0 else 0
    avg_pnl       = executed["pnl_pct"].mean() if not executed.empty else 0
    avg_win       = wins["pnl_pct"].mean()   if not wins.empty   else 0
    avg_loss      = losses["pnl_pct"].mean() if not losses.empty else 0
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

    # 做多/做空统计
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
        "parse_err_cnt":   parse_err,
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

    perf_metrics = load_googl_perf_metrics()
    trading_days = get_trading_days(start, end, step)
    print(f"共生成 {len(trading_days)} 个 Prompt 文件 → {PROMPTS_DIR}/\n")

    for i, date in enumerate(trading_days):
        out_path = PROMPTS_DIR / f"{date}.txt"
        if out_path.exists():
            print(f"[{i+1:>3}/{len(trading_days)}] {date}  已存在，跳过")
            continue

        daily, weekly = fetch_data_up_to(date)
        if daily.empty or len(daily) < 30:
            print(f"[{i+1:>3}/{len(trading_days)}] {date}  数据不足，跳过")
            continue

        macro = fetch_macro_for_date(date)
        prompt = build_blind_prompt(daily, weekly, macro, perf_metrics)
        if not prompt:
            continue

        out_path.write_text(prompt, encoding="utf-8")
        price = round(float(daily["Close"].squeeze().iloc[-1]), 2)
        print(f"[{i+1:>3}/{len(trading_days)}] {date}  price=${price}  → {out_path.name}")

    print(f"\n完成！文件保存到 {PROMPTS_DIR}/")
    print(f"将每个 .txt 粘贴到 Claude.ai，把 JSON 响应保存为 {RESPONSES_DIR}/<日期>.json")
    print(f"完成后运行：python3 google_backtest.py --evaluate")


# ─────────────────────────────────────────────
# 模式二：评估已有响应
# ─────────────────────────────────────────────

def run_evaluate():
    OUTPUT_DIR.mkdir(exist_ok=True)
    response_files = sorted(RESPONSES_DIR.glob("*.json"))
    if not response_files:
        print(f"[错误] {RESPONSES_DIR}/ 下没有 .json 文件")
        return

    print(f"找到 {len(response_files)} 个响应文件，开始评估...\n")
    all_records = []

    for i, fpath in enumerate(response_files):
        date = fpath.stem
        print(f"[{i+1:>3}/{len(response_files)}] {date}", end="  ")

        try:
            raw = fpath.read_text(encoding="utf-8")
            signal = parse_signal(raw)
        except Exception as e:
            print(f"读取失败: {e}")
            continue

        if not signal:
            print("JSON 解析失败，跳过")
            continue

        asset_list   = signal.get("asset_analysis", [])
        googl_signal = next((x for x in asset_list if x.get("asset") == "GOOGL"), None)
        if not googl_signal:
            print("未找到 GOOGL 信号，跳过")
            continue

        action = googl_signal.get("action", "no_trade")
        print(f"action={action}", end="  ")

        future_df = fetch_future_data(date)
        trade = simulate_trade(signal, future_df)

        record = {
            "date":        date,
            "action":      trade["action"],
            "entry_price": trade["entry_price"],
            "exit_price":  trade["exit_price"],
            "exit_reason": trade["exit_reason"],
            "pnl_pct":     trade["pnl_pct"],
            "win":         trade["win"],
            "days_held":   trade["days_held"],
            "bias_score":  googl_signal.get("bias_score"),
            "regime":      googl_signal.get("regime"),
            "sentiment":   signal.get("overall_market_sentiment"),
            "raw_signal":  json.dumps(googl_signal, ensure_ascii=False),
        }
        all_records.append(record)
        print(f"→ {trade['exit_reason']}  pnl={trade['pnl_pct']}%")

    if not all_records:
        print("\n无有效记录。")
        return

    _save_and_print(all_records)


# ─────────────────────────────────────────────
# 模式三：全自动回测
# ─────────────────────────────────────────────

def run_backtest(start: str, end: str, step: int, model: str,
                 dry_run: bool, resume: bool = False, start_from: str = None):
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"GOOGL 回测参数: {start} ~ {end}  |  step={step}  |  model={model}")
    print(f"评估周期: {EVAL_DAYS} 天  |  dry_run={dry_run}")
    print("-" * 60)

    perf_metrics = load_googl_perf_metrics()
    trading_days = get_trading_days(start, end, step)
    if start_from:
        trading_days = [d for d in trading_days if d >= start_from]

    existing_records = []
    done_dates: set[str] = set()
    if resume and SIGNALS_FILE.exists():
        existing_df    = pd.read_csv(SIGNALS_FILE)
        done_dates     = set(existing_df["date"].astype(str).tolist())
        existing_records = existing_df.to_dict("records")
        print(f"已加载 {len(done_dates)} 条现有记录")

    pending = [d for d in trading_days if d not in done_dates]
    print(f"共 {len(trading_days)} 个节点，待处理 {len(pending)} 个\n")

    all_records = list(existing_records)

    for i, date in enumerate(pending):
        print(f"[{i+1:>3}/{len(pending)}] {date}", end="  ")

        daily, weekly = fetch_data_up_to(date)
        if daily.empty or len(daily) < 30:
            print("-> 数据不足，跳过")
            continue

        macro  = fetch_macro_for_date(date)
        prompt = build_blind_prompt(daily, weekly, macro, perf_metrics)
        if not prompt:
            print("-> prompt 构建失败，跳过")
            continue

        if dry_run:
            price = round(float(daily["Close"].squeeze().iloc[-1]), 2)
            print(f"-> [DRY RUN] price=${price}  prompt={len(prompt)}字符")
            continue

        signal = call_deepseek(prompt, model)
        if not signal:
            print("-> 信号解析失败，跳过")
            continue

        asset_list   = signal.get("asset_analysis", [])
        googl_signal = next((x for x in asset_list if x.get("asset") == "GOOGL"), {})
        action  = googl_signal.get("action", "?")
        bias    = googl_signal.get("bias_score", "?")
        target  = googl_signal.get("profit_target")
        stop    = googl_signal.get("stop_loss")
        print(f"-> action={action}  bias={bias}  target={target}  stop={stop}", end="  ")

        future_df = fetch_future_data(date)
        trade     = simulate_trade(signal, future_df)

        record = {
            "date":        date,
            "action":      trade["action"],
            "entry_price": trade["entry_price"],
            "exit_price":  trade["exit_price"],
            "exit_reason": trade["exit_reason"],
            "pnl_pct":     trade["pnl_pct"],
            "win":         trade["win"],
            "days_held":   trade["days_held"],
            "bias_score":  googl_signal.get("bias_score"),
            "regime":      googl_signal.get("regime"),
            "sentiment":   signal.get("overall_market_sentiment"),
            "sector_assessment": signal.get("sector_assessment", ""),
            "rate_assessment":   signal.get("rate_assessment", ""),
            "raw_signal":  json.dumps(googl_signal, ensure_ascii=False),
        }
        all_records.append(record)
        print(f"-> {trade['exit_reason']}  pnl={trade['pnl_pct']}%")

        time.sleep(2)  # 避免 API 限速

    if not all_records:
        print("\n无有效记录（dry_run 或全部跳过）。")
        return

    _save_and_print(all_records)


def _save_and_print(all_records: list[dict]):
    OUTPUT_DIR.mkdir(exist_ok=True)
    df_records = pd.DataFrame(all_records)
    df_records.to_csv(SIGNALS_FILE, index=False, encoding="utf-8-sig")

    perf = compute_performance(all_records)

    print("\n" + "=" * 60)
    print("GOOGL 回测绩效汇总")
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
        description="Google (GOOGL) 科技股 LLM 回测引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 生成 Prompt 文件（无需 API Key）
  python3 google_backtest.py --generate --start 2024-01-01 --end 2025-12-31 --step 5

  # 评估已有响应文件
  python3 google_backtest.py --evaluate

  # 全自动回测（DeepSeek）
  python3 google_backtest.py --start 2024-01-01 --end 2025-12-31 --step 5

  # 仅验证数据和 Prompt（不调用 API）
  python3 google_backtest.py --start 2024-01-01 --end 2025-12-31 --dry-run

  # 断点续跑
  python3 google_backtest.py --start 2024-01-01 --end 2025-12-31 --resume
        """
    )
    parser.add_argument("--generate",   action="store_true",       help="生成 Prompt 文件")
    parser.add_argument("--evaluate",   action="store_true",       help="评估已有响应文件")
    parser.add_argument("--start",      default="2024-01-01",      help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end",        default="2025-12-31",      help="结束日期 YYYY-MM-DD")
    parser.add_argument("--step",       default=5,   type=int,     help="每隔N个交易日一次（默认5）")
    parser.add_argument("--model",      default="deepseek-chat",   help="模型 ID（deepseek-chat / deepseek-reasoner）")
    parser.add_argument("--dry-run",    action="store_true",       help="只验证数据，不调用 API")
    parser.add_argument("--resume",     action="store_true",       help="跳过已完成节点，追加合并")
    parser.add_argument("--start-from", default=None,              help="从指定日期开始处理")
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
            resume     = args.resume,
            start_from = args.start_from,
        )
