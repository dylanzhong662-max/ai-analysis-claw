"""
因子缓存脚本 — 每周采样生成 LLM 信号，存储为 CSV

核心思路：将 LLM 推理与历史回测解耦。
  Step 1（本脚本）：批量生成过去 N 年的每周 bias_score 信号 → signals_cache/{ticker}.csv
  Step 2（离线）：  读取 CSV，用不同参数组合（bias 阈值 / ATR 倍数）快速做向量化回测

特点：
  - 每周五采样（可配置 --day-of-week），避免过密 API 调用
  - 自动续跑（已缓存日期跳过，不重复调用）
  - 默认使用 deepseek-reasoner（R1），推理能力强；批量省成本可改 --model deepseek-chat
  - 速率限制（--rate-limit，默认 20s/次）

用法：
    python factor_cache.py --ticker NVDA --start 2024-01-01 --end 2025-12-31
    python factor_cache.py --ticker MSFT --start 2023-01-01 --end 2025-12-31
    python factor_cache.py --ticker GOOGL --start 2024-06-01 --end 2025-12-31 --day-of-week 4 --rate-limit 15

CSV 字段：
    date, ticker, action, bias_score, regime, ema_structure,
    qqq_status, w_atr14, w_rsi14, w_adx, notes
"""

import argparse
import csv
import json
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── 复用 backtest engine 里的数据获取 + prompt 构建 ──────────────────────────
from tech_backtest_engine import (
    fetch_data_up_to,
    fetch_macro_for_date,
    build_blind_prompt,
    compute_indicators,
    _python_pre_filter_bt,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_BASE_URL,
    ANTHROPIC_MODEL,
)
from gold_analysis import calc_ema

CACHE_DIR = Path(__file__).parent / "signals_cache"
CACHE_DIR.mkdir(exist_ok=True)

DEEPSEEK_MODELS = {"deepseek-reasoner", "deepseek-chat"}
CLAUDE_MODELS   = {"claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"}

CSV_FIELDS = [
    "date", "ticker", "action", "bias_score", "regime",
    "ema_structure", "qqq_status", "w_atr14", "w_rsi14", "w_adx",
    "pre_filtered", "notes",
]


# ─────────────────────────────────────────────
# 采样日期生成
# ─────────────────────────────────────────────

def generate_sample_dates(start: str, end: str, day_of_week: int = 4) -> list[str]:
    """
    生成采样日期列表。
    day_of_week: 0=周一 … 4=周五（默认）
    """
    d   = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    dates = []
    # 对齐到第一个目标星期
    while d.weekday() != day_of_week:
        d += timedelta(days=1)
    while d <= end_d:
        dates.append(d.isoformat())
        d += timedelta(weeks=1)
    return dates


# ─────────────────────────────────────────────
# 已缓存日期读取
# ─────────────────────────────────────────────

def load_cached_dates(ticker: str) -> set[str]:
    path = CACHE_DIR / f"{ticker.upper()}.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["date"], dtype=str)
    return set(df["date"].dropna().tolist())


# ─────────────────────────────────────────────
# LLM 调用（支持 DeepSeek / Claude）
# ─────────────────────────────────────────────

def _call_llm(prompt: str, model: str, system_prompt: str = "") -> str:
    if model in DEEPSEEK_MODELS:
        from openai import OpenAI
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    max_tokens=4000,
                    messages=[
                        *(([{"role": "system", "content": system_prompt}]) if system_prompt else []),
                        {"role": "user", "content": prompt},
                    ],
                )
                raw = resp.choices[0].message.content or ""
                # 去掉 DeepSeek R1 的 <think> 标签
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                return raw
            except Exception as e:
                print(f"    [重试 {attempt+1}/3] {e}")
                if attempt < 2:
                    time.sleep(10)
        return ""

    elif model in CLAUDE_MODELS:
        import httpx
        from anthropic import Anthropic
        client = Anthropic(
            base_url=ANTHROPIC_BASE_URL,
            api_key=ANTHROPIC_API_KEY,
            http_client=httpx.Client(verify=False, timeout=120.0),
        )
        for attempt in range(3):
            try:
                msg = client.messages.create(
                    model=model,
                    max_tokens=4000,
                    system=system_prompt or "You are a senior equity strategist.",
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(b.text for b in msg.content if hasattr(b, "text"))
            except Exception as e:
                print(f"    [重试 {attempt+1}/3] {e}")
                if attempt < 2:
                    time.sleep(15 * (attempt + 1))
        return ""

    else:
        # 通用 OpenAI 兼容接口
        from openai import OpenAI
        openai_key = os.getenv("OPENAI_API_KEY", ANTHROPIC_API_KEY)
        openai_url = os.getenv("OPENAI_BASE_URL", "https://api.openai-proxy.org/v1")
        client = OpenAI(api_key=openai_key, base_url=openai_url)
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model, max_tokens=4000,
                    messages=[
                        *(([{"role": "system", "content": system_prompt}]) if system_prompt else []),
                        {"role": "user", "content": prompt},
                    ],
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                print(f"    [重试 {attempt+1}/3] {e}")
                if attempt < 2:
                    time.sleep(10)
        return ""


def _parse_signal(raw: str, ticker: str) -> dict:
    """从 LLM 响应中提取信号字段"""
    result = {"action": "parse_error", "bias_score": None, "regime": "", "notes": ""}
    if not raw:
        result["notes"] = "empty_response"
        return result

    # 尝试解析 JSON
    parsed = None
    for pattern in [
        r'```(?:json)?\s*(\{.*?\})\s*```',
        r'(\{[^{}]*"asset_analysis"[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
    ]:
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
                break
            except Exception:
                pass
    if parsed is None:
        start = raw.find('{')
        if start != -1:
            depth, end = 0, -1
            for i, c in enumerate(raw[start:], start):
                depth += (c == '{') - (c == '}')
                if depth == 0:
                    end = i
                    break
            if end != -1:
                try:
                    parsed = json.loads(raw[start:end + 1])
                except Exception:
                    pass

    if parsed is None:
        result["notes"] = "json_parse_failed"
        return result

    # 提取目标资产信号
    sig = None
    for item in parsed.get("asset_analysis", []):
        if item.get("asset", "").upper() == ticker.upper():
            sig = item
            break
    if sig is None:
        result["notes"] = "asset_not_found"
        return result

    result["action"]     = sig.get("action", "no_trade")
    result["bias_score"] = sig.get("bias_score")
    result["regime"]     = sig.get("regime", "")

    # QQQ 状态
    qqq_a = parsed.get("qqq_assessment", "")
    result["qqq_status"] = qqq_a[:80] if qqq_a else ""
    return result


# ─────────────────────────────────────────────
# 单日信号生成
# ─────────────────────────────────────────────

def generate_signal_for_date(ticker: str, date_str: str, model: str,
                              system_prompt: str = "") -> dict:
    """
    为指定日期生成信号。返回一行 CSV 数据的 dict。
    """
    row = {f: "" for f in CSV_FIELDS}
    row["date"]   = date_str
    row["ticker"] = ticker.upper()

    try:
        daily, weekly, monthly = fetch_data_up_to(ticker, date_str)
    except Exception as e:
        row["action"] = "data_error"
        row["notes"]  = str(e)[:120]
        return row

    if daily.empty or len(daily) < 30:
        row["action"] = "insufficient_data"
        return row

    # 计算关键指标（仅用于记录上下文，不传给 LLM）
    try:
        w_ind  = compute_indicators(weekly)
        today_close = float(daily["Close"].squeeze().dropna().iloc[-1])

        def _sl(s):
            s = s.dropna()
            return round(float(s.iloc[-1]), 2) if len(s) > 0 else None

        w_atr14 = _sl(w_ind['atr14'])
        w_rsi14 = _sl(w_ind['rsi14'])
        w_adx   = _sl(w_ind['adx'])
        w_e50   = _sl(w_ind['ema50'])
        w_e200  = _sl(w_ind['ema200'])

        if w_e50 and w_e200 and today_close:
            if today_close > w_e50 > w_e200:
                ema_struct = "Trending-Up"
            elif w_e50 < w_e200 and today_close >= w_e200:
                ema_struct = "Trending-Recovery"
            elif w_e50 < w_e200 and today_close < w_e200:
                ema_struct = "Death-Cross"
            else:
                ema_struct = "Mixed"
        else:
            ema_struct = "N/A"

        row["w_atr14"]      = w_atr14 or ""
        row["w_rsi14"]      = w_rsi14 or ""
        row["w_adx"]        = w_adx or ""
        row["ema_structure"] = ema_struct
    except Exception:
        today_close = 0.0
        w_atr14 = 1.0

    # 前置过滤（复用回测引擎逻辑）
    macro = fetch_macro_for_date(date_str, ticker)
    pre_block = _python_pre_filter_bt(weekly, today_close, macro)
    if pre_block:
        row["action"]       = "no_trade"
        row["bias_score"]   = 0.0
        row["regime"]       = "pre_filtered"
        row["pre_filtered"] = "1"
        row["notes"]        = pre_block[:120]
        return row

    # 构建提示词 + 调用 LLM
    prompt = build_blind_prompt(ticker, daily, weekly, macro, None, monthly)
    if not prompt:
        row["action"] = "prompt_error"
        return row

    raw = _call_llm(prompt, model, system_prompt)
    sig = _parse_signal(raw, ticker)

    row["action"]     = sig.get("action", "parse_error")
    row["bias_score"] = sig.get("bias_score", "")
    row["regime"]     = sig.get("regime", "")
    row["qqq_status"] = sig.get("qqq_status", "")
    row["notes"]      = sig.get("notes", "")
    return row


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="因子缓存脚本：批量生成 LLM 信号 CSV，解耦 API 调用与回测"
    )
    parser.add_argument("--ticker",       required=True,  help="股票代码，例如 NVDA")
    parser.add_argument("--start",        required=True,  help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end",          required=True,  help="结束日期 YYYY-MM-DD")
    parser.add_argument("--model",        default="deepseek-reasoner",
                        help="LLM 模型（默认 deepseek-reasoner，推理能力强；批量省成本可改 deepseek-chat）")
    parser.add_argument("--day-of-week",  type=int, default=4,
                        help="采样星期（0=周一…4=周五，默认4=周五）")
    parser.add_argument("--rate-limit",   type=float, default=20.0,
                        help="两次 API 调用之间的最小间隔秒数（默认 20）")
    parser.add_argument("--force",        action="store_true",
                        help="强制重新生成（忽略已缓存）")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    dates  = generate_sample_dates(args.start, args.end, args.day_of_week)
    cached = set() if args.force else load_cached_dates(ticker)

    pending = [d for d in dates if d not in cached]
    print(f"[{ticker}] 采样日期 {len(dates)} 个，已缓存 {len(cached)} 个，待生成 {len(pending)} 个")
    print(f"  模型: {args.model}  采样星期: {args.day_of_week}  速率限制: {args.rate_limit}s")

    if not pending:
        print("全部已缓存，无需调用 API。")
        return

    out_path = CACHE_DIR / f"{ticker}.csv"
    file_exists = out_path.exists() and not args.force

    with open(out_path, "a" if file_exists else "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()

        for i, date_str in enumerate(pending):
            print(f"\n[{i+1:>3}/{len(pending)}] {date_str}  ", end="", flush=True)
            row = generate_signal_for_date(ticker, date_str, args.model)
            writer.writerow(row)
            f.flush()

            action = row.get("action", "?")
            bias   = row.get("bias_score", "")
            regime = row.get("regime", "")
            notes  = row.get("notes", "")
            print(f"action={action}  bias={bias}  regime={regime}"
                  + (f"  [{notes}]" if notes else ""))

            # 速率限制（最后一条不等待）
            if i < len(pending) - 1:
                time.sleep(args.rate_limit)

    print(f"\n完成！信号已保存到: {out_path}")
    df = pd.read_csv(out_path)
    total    = len(df)
    traded   = (df["action"].isin(["long", "short"])).sum()
    no_trade = (df["action"] == "no_trade").sum()
    filtered = (df["action"] == "no_trade") & (df["pre_filtered"] == "1")
    print(f"  总记录: {total}  交易信号: {traded}  no_trade: {no_trade}"
          f"  (预过滤: {filtered.sum()})")

    if traded > 0:
        avg_bias = df[df["action"].isin(["long", "short"])]["bias_score"].astype(float).mean()
        print(f"  平均 bias_score（入场信号）: {avg_bias:.3f}")

    # 制度分布
    if "regime" in df.columns:
        regime_counts = df[df["action"].isin(["long", "short"])]["regime"].value_counts()
        print("\n  制度分布（入场信号）:")
        for regime, cnt in regime_counts.items():
            print(f"    {regime:<30} {cnt}")


if __name__ == "__main__":
    main()
