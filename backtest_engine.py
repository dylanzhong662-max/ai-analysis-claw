"""
黄金交易策略回测引擎
- 防时间泄漏：数据严格截断到回测日期，Prompt 不含具体日期

三种运行模式：

  【模式一】生成 Prompt 文件（无需 API Key）
    python backtest_engine.py --generate --start 2024-01-01 --end 2024-12-31 --step 5
    → 在 backtest_prompts/ 目录生成每个回测日的提示词文件
    → 将每个 .txt 文件内容粘贴到 Claude.ai，把 JSON 回复保存到 backtest_responses/<日期>.json

  【模式二】评估已有响应（无需 API Key）
    python backtest_engine.py --evaluate
    → 读取 backtest_responses/ 下的所有 JSON 文件，计算 P&L 和绩效

  【模式三】全自动回测（需要 ANTHROPIC_API_KEY）
    python backtest_engine.py --start 2024-01-01 --end 2024-12-31 --step 5

依赖：
    pip install anthropic yfinance pandas numpy curl_cffi urllib3
"""

import argparse
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
import urllib3
import yfinance as yf
from curl_cffi import requests as curl_requests

# 复用现有脚本的指标计算函数
from gold_analysis import calc_ema, calc_macd, calc_rsi, calc_atr, fmt_series, compute_indicators

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ─────────────────────────────────────────────
# 常量配置
# ─────────────────────────────────────────────

TICKER        = "GC=F"
OUTPUT_DIR    = Path("backtest_results")
PROMPTS_DIR   = Path("backtest_prompts")
RESPONSES_DIR = Path("backtest_responses")
SIGNALS_FILE  = OUTPUT_DIR / "signals.csv"
PERF_FILE     = OUTPUT_DIR / "performance.csv"

EVAL_DAYS    = 15    # 最长持仓天数（对应策略3-15天）
LOOKBACK_DAYS = 180  # 每个回测节点向前取多少天的数据

# 防时间泄漏的 System Prompt
ANTI_LEAK_SYSTEM = """你是一个严格的量化交易信号生成器。

核心规则（必须严格遵守）：
1. 你的分析【必须且只能】基于用户提供的价格序列和技术指标数据
2. 你【绝对不能】使用任何关于该数据时间窗口之外发生的事件的知识
3. 你【不得】推断或猜测当前的具体日期，也不得基于猜测日期调整分析结论
4. 你【不得】说"我知道此后发生了..."或"根据当时的宏观背景..."等
5. dxy_assessment / macro_catalyst 字段：只填写可以从价格走势本身推断的内容，不引用外部新闻事件
6. 如果你无法仅凭所提供的数据做出高置信度判断，action 必须返回 "no_trade"

输出格式：必须返回有效的 JSON，且仅返回 JSON，不包含任何其他说明文字。"""


# ─────────────────────────────────────────────
# 数据获取（严格时间截断）
# ─────────────────────────────────────────────

def _make_session():
    return curl_requests.Session(impersonate="chrome", verify=False)


def fetch_data_up_to(date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    严格截断：只获取 date 当日及之前的历史数据，绝不包含未来数据。
    yfinance 的 end 参数是 exclusive，所以传 date+1。
    """
    end_dt   = pd.Timestamp(date)
    end_str  = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    start_str = (end_dt - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    session = _make_session()

    daily = yf.download(
        TICKER, start=start_str, end=end_str,
        interval="1d", auto_adjust=True, progress=False, session=session
    )
    weekly = yf.download(
        TICKER, start=start_str, end=end_str,
        interval="1wk", auto_adjust=True, progress=False, session=session
    )
    return daily, weekly


def fetch_future_data(date: str) -> pd.DataFrame:
    """获取 date 之后的数据，用于评估交易结果（仅在回测评估阶段使用）。"""
    start_str = (pd.Timestamp(date) + timedelta(days=1)).strftime("%Y-%m-%d")
    end_str   = (pd.Timestamp(date) + timedelta(days=(EVAL_DAYS + 5) * 2)).strftime("%Y-%m-%d")

    session = _make_session()
    df = yf.download(
        TICKER, start=start_str, end=end_str,
        interval="1d", auto_adjust=True, progress=False, session=session
    )
    return df.iloc[: EVAL_DAYS + 5]


# ─────────────────────────────────────────────
# 构建防泄漏 Prompt（不含具体日期）
# ─────────────────────────────────────────────

def build_blind_prompt(daily: pd.DataFrame, weekly: pd.DataFrame) -> str:
    """
    与 gold_analysis.py 的 build_prompt 逻辑相同，
    但移除了 today_str（具体日期），防止模型推断时间位置。
    """
    if daily.empty or weekly.empty or len(daily) < 30:
        return ""

    d_ind = compute_indicators(daily)
    w_ind = compute_indicators(weekly)

    close_d = daily["Close"].squeeze()
    current_price = round(float(close_d.iloc[-1]), 2)
    current_ema20 = round(float(d_ind["ema20"].iloc[-1]), 2)
    current_macd  = round(float(d_ind["macd"].iloc[-1]), 2)
    current_rsi7  = round(float(d_ind["rsi7"].iloc[-1]), 2)
    current_rsi14 = round(float(d_ind["rsi14"].iloc[-1]), 2)

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
    daily_macd   = fmt_series(d_ind["macd"], 2, n)
    daily_rsi7   = fmt_series(d_ind["rsi7"], 2, n)
    daily_rsi14  = fmt_series(d_ind["rsi14"], 2, n)

    close_w       = weekly["Close"].squeeze()
    weekly_closes = fmt_series(close_w, 2, 10)
    weekly_macd   = fmt_series(w_ind["macd"], 2, 10)
    weekly_rsi14  = fmt_series(w_ind["rsi14"], 2, 10)

    ema20_d  = round(float(d_ind["ema20"].iloc[-1]), 2)
    ema50_d  = round(float(d_ind["ema50"].iloc[-1]), 2)
    atr3_d   = round(float(d_ind["atr3"].iloc[-1]), 2)
    atr14_d  = round(float(d_ind["atr14"].iloc[-1]), 2)

    vol_current = int(daily["Volume"].squeeze().iloc[-1])
    vol_avg     = int(daily["Volume"].squeeze().tail(20).mean())

    # 注意：此处不包含具体日期
    prompt = f"""
# 黄金 (XAU/USD) 大宗商品分析请求
**数据来源**: COMEX 黄金期货 (GC=F)
**重要说明**: 请严格基于以下提供的数据进行分析，不得引用任何外部事件或数据范围之外的知识。

---

## 价格概要

- **当前价格**: ${current_price}
- **今日 O/H/L/C**: {today_open} / {today_high} / {today_low} / {current_price}
- **今日涨跌幅**: {day_chg:+.2f}%
- **过去5交易日涨跌**: ${close_5d:.2f} → ${last_close:.2f}  ({week_chg:+.2f}%)
- **今日成交量**: {today_vol:,}

---

## 当前技术指标快照

- current_price  = {current_price}
- current_ema20 (日线) = {current_ema20}
- current_macd  (日线) = {current_macd}
- current_rsi7  (日线) = {current_rsi7}
- current_rsi14 (日线) = {current_rsi14}

---

## 日线序列数据（最近 {n} 个交易日，从旧到新排列）

⚠️ 最后一个数值 = 最新数据

收盘价:   [{daily_closes}]
EMA-20:   [{daily_ema20}]
MACD:     [{daily_macd}]
RSI-7:    [{daily_rsi7}]
RSI-14:   [{daily_rsi14}]

---

## 周线序列数据（最近 10 周，从旧到新排列）

⚠️ 最后一个数值 = 最新数据

周收盘价: [{weekly_closes}]
MACD:     [{weekly_macd}]
RSI-14:   [{weekly_rsi14}]

---

## 长期趋势背景

20日EMA: {ema20_d}  vs.  50日EMA: {ema50_d}
ATR-3:   {atr3_d}   vs.  ATR-14: {atr14_d}
今日成交量: {vol_current:,}  vs.  20日均量: {vol_avg:,}

---

## 分析任务

请基于以上数据，按照大宗商品分析框架，完成以下任务：

1. 判断当前市场制度（Trending / Mean-Reverting / Choppy）
2. 判断整体市场情绪（Risk-On / Risk-Off / Neutral）
3. 分析 DXY 对黄金的潜在压力方向（只基于价格走势推断，不引用外部新闻）
4. 针对黄金给出交易建议，严格按以下 JSON 格式输出：

```json
{{
  "period": "Daily",
  "overall_market_sentiment": "Risk-On | Risk-Off | Neutral",
  "dxy_assessment": "<仅基于价格数据推断的DXY方向>",
  "asset_analysis": [
    {{
      "asset": "GOLD",
      "regime": "Trending | Mean-Reverting | Choppy",
      "action": "long | short | no_trade",
      "bias_score": <0.0 到 1.0>,
      "entry_zone": "<价格区间>",
      "profit_target": <数字 或 null>,
      "stop_loss": <数字 或 null>,
      "risk_reward_ratio": <数字 或 null>,
      "invalidation_condition": "<使该观点失效的具体信号>",
      "macro_catalyst": "<只基于技术数据推断的驱动因素，不引用外部新闻>",
      "technical_setup": "<指标信号综合描述>",
      "justification": "<不超过200字的综合判断>"
    }}
  ]
}}
```

注意：
- profit_target 做多时必须高于入场价，做空时必须低于入场价
- risk_reward_ratio 必须 ≥ 2.0，否则改为 no_trade
- 当 action = no_trade 时，profit_target / stop_loss / risk_reward_ratio 填 null
""".strip()

    return prompt


# ─────────────────────────────────────────────
# Claude API 调用
# ─────────────────────────────────────────────

def call_claude(prompt: str, model: str) -> dict:
    """调用 Claude API，含3次重试逻辑。"""
    client = anthropic.Anthropic()

    for attempt in range(3):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=1500,
                system=ANTI_LEAK_SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            )
            return parse_signal(message.content[0].text)
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"  [限速] 等待 {wait}s 后重试...")
            time.sleep(wait)
        except Exception as e:
            print(f"  [API错误] attempt {attempt+1}/3: {e}")
            if attempt < 2:
                time.sleep(5)

    return {}


def parse_signal(raw: str) -> dict:
    """从 LLM 输出中提取 JSON，兼容 markdown 代码块。"""
    # 直接解析
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # 提取 ```json ... ``` 代码块
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 宽松匹配最外层 {}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    print(f"  [警告] JSON 解析失败，原始输出前200字：{raw[:200]}")
    return {}


# ─────────────────────────────────────────────
# 交易模拟（单笔）
# ─────────────────────────────────────────────

def simulate_trade(signal: dict, future_df: pd.DataFrame, entry_date: str) -> dict:
    """
    模拟规则：
    - 在 entry_date 次日开盘价入场
    - 逐日检查：触达止盈 → WIN，触达止损 → LOSS
    - 超过 EVAL_DAYS 天 → TIMEOUT，按最后收盘价结算
    """
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

    asset_list  = signal.get("asset_analysis", [])
    gold_signal = next((x for x in asset_list if x.get("asset") == "GOLD"), None)
    if not gold_signal:
        base["exit_reason"] = "PARSE_ERROR"
        return base

    action        = gold_signal.get("action", "no_trade")
    profit_target = gold_signal.get("profit_target")
    stop_loss     = gold_signal.get("stop_loss")

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

        # 超时平仓
        if i == EVAL_DAYS - 1:
            base.update(exit_price=close, exit_reason="TIMEOUT", days_held=i + 1)

    # 未触发任何条件（数据不足 EVAL_DAYS 天）
    if base["exit_reason"] == "PENDING":
        last_close = float(future_df.iloc[-1]["Close"].squeeze())
        base.update(exit_price=last_close, exit_reason="TIMEOUT", days_held=len(future_df))

    # 计算 P&L
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
    df      = pd.DataFrame(records)
    traded  = df[df["action"].isin(["long", "short"])].copy()
    no_trade_cnt = (df["action"] == "no_trade").sum()

    if traded.empty:
        return {"error": "无有效交易信号"}

    wins   = traded[traded["win"] == True]
    losses = traded[traded["win"] == False]

    win_rate     = len(wins) / len(traded) * 100
    avg_pnl      = traded["pnl_pct"].mean()
    avg_win      = wins["pnl_pct"].mean()   if not wins.empty   else 0.0
    avg_loss     = losses["pnl_pct"].mean() if not losses.empty else 0.0
    total_profit = wins["pnl_pct"].sum()
    total_loss   = abs(losses["pnl_pct"].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    # 最大回撤（基于累计 P&L 曲线）
    cum = traded["pnl_pct"].cumsum()
    max_dd = (cum - cum.cummax()).min()

    # 按月统计胜率
    if "date" in df.columns:
        traded_copy = traded.copy()
        traded_copy["month"] = pd.to_datetime(traded_copy["date"]).dt.to_period("M")
        monthly = traded_copy.groupby("month")["win"].mean() * 100
        monthly_str = "  |  ".join(f"{m}: {v:.0f}%" for m, v in monthly.items())
    else:
        monthly_str = "N/A"

    return {
        "total_signals":   len(df),
        "traded":          len(traded),
        "no_trade":        no_trade_cnt,
        "no_trade_rate":   f"{no_trade_cnt / len(df) * 100:.1f}%",
        "win_count":       len(wins),
        "loss_count":      len(losses),
        "win_rate":        f"{win_rate:.1f}%",
        "avg_pnl_pct":     f"{avg_pnl:.2f}%",
        "avg_win_pct":     f"{avg_win:.2f}%",
        "avg_loss_pct":    f"{avg_loss:.2f}%",
        "profit_factor":   f"{profit_factor:.2f}",
        "max_drawdown":    f"{max_dd:.2f}%",
        "total_return":    f"{traded['pnl_pct'].sum():.2f}%",
        "monthly_winrate": monthly_str,
    }


# ─────────────────────────────────────────────
# 获取回测交易日列表
# ─────────────────────────────────────────────

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
# 模式一：生成 Prompt 文件（无需 API Key）
# ─────────────────────────────────────────────

def run_generate(start: str, end: str, step: int):
    """
    为每个回测日期生成盲化 Prompt 文件，保存到 backtest_prompts/ 目录。
    用户将每个文件内容粘贴到 Claude.ai，把 JSON 回复保存到 backtest_responses/<日期>.json。
    """
    PROMPTS_DIR.mkdir(exist_ok=True)
    RESPONSES_DIR.mkdir(exist_ok=True)

    trading_days = get_trading_days(start, end, step)
    print(f"共生成 {len(trading_days)} 个 Prompt 文件 → {PROMPTS_DIR}/\n")

    skipped = []
    for i, date in enumerate(trading_days):
        out_path = PROMPTS_DIR / f"{date}.txt"
        if out_path.exists():
            print(f"[{i+1:>3}/{len(trading_days)}] {date}  已存在，跳过")
            continue

        daily, weekly = fetch_data_up_to(date)
        if daily.empty or len(daily) < 30:
            print(f"[{i+1:>3}/{len(trading_days)}] {date}  数据不足，跳过")
            skipped.append(date)
            continue

        prompt = build_blind_prompt(daily, weekly)
        if not prompt:
            skipped.append(date)
            continue

        out_path.write_text(prompt, encoding="utf-8")
        price = round(float(daily["Close"].squeeze().iloc[-1]), 2)
        print(f"[{i+1:>3}/{len(trading_days)}] {date}  price=${price}  → {out_path.name}")

    print(f"\n完成！{len(trading_days) - len(skipped)} 个文件已保存到 {PROMPTS_DIR}/")
    print("\n" + "=" * 60)
    print("下一步操作：")
    print("=" * 60)
    print(f"1. 打开 Claude.ai（或任意 LLM 对话框）")
    print(f"2. 依次将 {PROMPTS_DIR}/<日期>.txt 的内容粘贴进去")
    print(f"3. 将 LLM 返回的 JSON 内容保存为 {RESPONSES_DIR}/<日期>.json")
    print(f"   例如：{RESPONSES_DIR}/2024-01-02.json")
    print(f"4. 全部完成后运行：python3 backtest_engine.py --evaluate")


# ─────────────────────────────────────────────
# 模式二：评估已有响应（无需 API Key）
# ─────────────────────────────────────────────

def run_evaluate():
    """
    读取 backtest_responses/ 目录下所有 <日期>.json 文件，
    获取对应日期之后的真实价格数据，模拟交易，输出绩效。
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    response_files = sorted(RESPONSES_DIR.glob("*.json"))
    if not response_files:
        print(f"[错误] {RESPONSES_DIR}/ 目录下没有找到任何 .json 文件。")
        print("请先运行 --generate 生成 Prompt，粘贴到 Claude.ai 后保存 JSON 响应。")
        return

    print(f"找到 {len(response_files)} 个响应文件，开始评估...\n")
    all_records = []

    for i, fpath in enumerate(response_files):
        date = fpath.stem  # 文件名即日期，如 2024-01-02
        print(f"[{i+1:>3}/{len(response_files)}] {date}", end="  ")

        # 读取并解析 JSON
        try:
            raw = fpath.read_text(encoding="utf-8")
            signal = parse_signal(raw)
        except Exception as e:
            print(f"读取失败: {e}")
            continue

        if not signal:
            print("JSON 解析失败，跳过")
            continue

        asset_list  = signal.get("asset_analysis", [])
        gold_signal = next((x for x in asset_list if x.get("asset") == "GOLD"), None)
        if not gold_signal:
            print("未找到 GOLD 信号，跳过")
            continue

        action = gold_signal.get("action", "no_trade")
        target = gold_signal.get("profit_target")
        stop   = gold_signal.get("stop_loss")
        print(f"action={action}  target={target}  stop={stop}", end="  ")

        # 获取未来真实数据
        future_df = fetch_future_data(date)

        # 模拟交易
        trade = simulate_trade(signal, future_df, date)

        record = {
            "date":        date,
            "action":      trade["action"],
            "entry_price": trade["entry_price"],
            "exit_price":  trade["exit_price"],
            "exit_reason": trade["exit_reason"],
            "pnl_pct":     trade["pnl_pct"],
            "win":         trade["win"],
            "days_held":   trade["days_held"],
            "bias_score":  gold_signal.get("bias_score"),
            "regime":      gold_signal.get("regime"),
            "sentiment":   signal.get("overall_market_sentiment"),
            "raw_signal":  json.dumps(gold_signal, ensure_ascii=False),
        }
        all_records.append(record)
        print(f"→ {trade['exit_reason']}  pnl={trade['pnl_pct']}%")

    if not all_records:
        print("\n无有效交易记录。")
        return

    # 保存信号明细
    df_records = pd.DataFrame(all_records)
    df_records.to_csv(SIGNALS_FILE, index=False, encoding="utf-8-sig")

    # 绩效汇总
    perf = compute_performance(all_records)

    print("\n" + "=" * 60)
    print("回测绩效汇总")
    print("=" * 60)
    for k, v in perf.items():
        print(f"  {k:<22}: {v}")

    pd.DataFrame([perf]).to_csv(PERF_FILE, index=False, encoding="utf-8-sig")
    print(f"\n信号明细 → {SIGNALS_FILE}")
    print(f"绩效汇总 → {PERF_FILE}")


# ─────────────────────────────────────────────
# 模式三：全自动回测（需要 ANTHROPIC_API_KEY）
# ─────────────────────────────────────────────

def run_backtest(start: str, end: str, step: int, model: str, dry_run: bool):
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"回测参数: {start} ~ {end}  |  step={step} 交易日  |  model={model}")
    print(f"评估持仓周期: {EVAL_DAYS} 天  |  dry_run={dry_run}")
    print("-" * 60)

    trading_days = get_trading_days(start, end, step)
    print(f"共 {len(trading_days)} 个回测节点\n")

    all_records = []

    for i, date in enumerate(trading_days):
        print(f"[{i + 1:>3}/{len(trading_days)}] {date}", end="  ")

        # 1. 严格截断获取历史数据
        daily, weekly = fetch_data_up_to(date)
        if daily.empty or len(daily) < 30:
            print("-> 数据不足，跳过")
            continue

        # 2. 构建盲化 prompt
        prompt = build_blind_prompt(daily, weekly)
        if not prompt:
            print("-> prompt 构建失败，跳过")
            continue

        if dry_run:
            current_price = round(float(daily["Close"].squeeze().iloc[-1]), 2)
            print(f"-> [DRY RUN] price=${current_price}  prompt={len(prompt)}字符")
            continue

        # 3. 调用 Claude API
        signal = call_claude(prompt, model)
        if not signal:
            print("-> 信号解析失败，跳过")
            continue

        asset_list  = signal.get("asset_analysis", [])
        gold_signal = next((x for x in asset_list if x.get("asset") == "GOLD"), {})
        action      = gold_signal.get("action", "?")
        bias        = gold_signal.get("bias_score", "?")
        target      = gold_signal.get("profit_target")
        stop        = gold_signal.get("stop_loss")

        print(f"-> action={action}  bias={bias}  target={target}  stop={stop}", end="  ")

        # 4. 获取未来数据（仅用于评估）
        future_df = fetch_future_data(date)

        # 5. 模拟交易
        trade = simulate_trade(signal, future_df, date)

        # 6. 记录结果
        record = {
            "date":        date,
            "action":      trade["action"],
            "entry_price": trade["entry_price"],
            "exit_price":  trade["exit_price"],
            "exit_reason": trade["exit_reason"],
            "pnl_pct":     trade["pnl_pct"],
            "win":         trade["win"],
            "days_held":   trade["days_held"],
            "bias_score":  gold_signal.get("bias_score"),
            "regime":      gold_signal.get("regime"),
            "sentiment":   signal.get("overall_market_sentiment"),
            "raw_signal":  json.dumps(gold_signal, ensure_ascii=False),
        }
        all_records.append(record)

        print(f"-> {trade['exit_reason']}  pnl={trade['pnl_pct']}%")

        time.sleep(2)  # 避免 API 限速

    if not all_records:
        print("\n无有效记录（可能是 dry_run 模式或全部跳过）。")
        return

    # 7. 保存信号记录
    df_records = pd.DataFrame(all_records)
    df_records.to_csv(SIGNALS_FILE, index=False, encoding="utf-8-sig")

    # 8. 绩效汇总
    perf = compute_performance(all_records)

    print("\n" + "=" * 60)
    print("回测绩效汇总")
    print("=" * 60)
    for k, v in perf.items():
        print(f"  {k:<22}: {v}")

    pd.DataFrame([perf]).to_csv(PERF_FILE, index=False, encoding="utf-8-sig")

    print(f"\n信号明细 -> {SIGNALS_FILE}")
    print(f"绩效汇总 -> {PERF_FILE}")


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="大模型黄金交易策略回测引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 【无需 API Key】生成所有 Prompt 文件，手动粘贴到 Claude.ai
  python3 backtest_engine.py --generate --start 2024-01-01 --end 2024-12-31 --step 5

  # 【无需 API Key】评估已保存的 Claude.ai 响应，计算绩效
  python3 backtest_engine.py --evaluate

  # 【需要 API Key】全自动回测
  python3 backtest_engine.py --start 2024-01-01 --end 2024-12-31 --step 5
        """
    )
    parser.add_argument("--generate",  action="store_true",       help="生成 Prompt 文件到 backtest_prompts/（无需 API Key）")
    parser.add_argument("--evaluate",  action="store_true",       help="评估 backtest_responses/ 下的响应文件（无需 API Key）")
    parser.add_argument("--start",     default="2024-01-01",      help="回测开始日期 YYYY-MM-DD")
    parser.add_argument("--end",       default="2024-12-31",      help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--step",      default=5,   type=int,     help="每隔N个交易日触发一次 (默认5)")
    parser.add_argument("--model",     default="claude-opus-4-6", help="Claude 模型 ID（仅全自动模式使用）")
    parser.add_argument("--dry-run",   action="store_true",       help="只验证数据和 Prompt，不调用 API")
    args = parser.parse_args()

    if args.generate:
        run_generate(start=args.start, end=args.end, step=args.step)
    elif args.evaluate:
        run_evaluate()
    else:
        run_backtest(
            start   = args.start,
            end     = args.end,
            step    = args.step,
            model   = args.model,
            dry_run = args.dry_run,
        )
