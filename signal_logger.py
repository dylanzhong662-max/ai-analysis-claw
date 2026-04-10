"""
signal_logger.py — 实盘信号持久化工具

用途：
  每次实盘分析完成后，把 *_api_output.txt 中的信号结构化追加到
  live_signal_log.csv，积累历史记录供 IC 分析使用。

运行方式：
  python3 signal_logger.py                    # 记录当前所有资产信号
  python3 signal_logger.py --tickers NVDA GOOGL  # 只记录指定资产
  python3 signal_logger.py --dry-run          # 预览，不写入文件

由 run_daily.sh 在所有分析完成后自动调用。
"""

import argparse
import csv
import json
import os
import re
from datetime import date, datetime
from pathlib import Path

SIGNAL_LOG = Path("live_signal_log.csv")

# 资产 → output 文件映射（与 assets_config.py 一致）
ASSET_OUTPUT_MAP = {
    "NVDA":  "outputs/nvda_api_output.txt",
    "MSFT":  "outputs/msft_api_output.txt",
    "GOOGL": "outputs/googl_api_output.txt",
    "AAPL":  "outputs/aapl_api_output.txt",
    "META":  "outputs/meta_api_output.txt",
    "AMZN":  "outputs/amzn_api_output.txt",
    "GOLD":  "outputs/gold_api_output.txt",
    "BTC":   "outputs/btc_api_output.txt",
}

FIELDNAMES = [
    "signal_date", "ticker", "action", "bias_score",
    "regime", "entry_zone", "stop_loss", "profit_target",
    "risk_reward_ratio", "estimated_holding_weeks",
    "overall_sentiment", "qqq_assessment",
]


def _parse_output_file(path: Path) -> list[dict]:
    """解析单个 *_api_output.txt，返回信号列表（可能包含多个资产）。"""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []

    # 提取 JSON：直接解析 → markdown 代码块 → 大括号匹配
    data = None
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    if data is None:
        # markdown 代码块
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    if data is None:
        # 大括号计数匹配
        start = text.find("{")
        if start >= 0:
            depth, end = 0, start
            for i, ch in enumerate(text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

    if not data:
        return []

    sentiment = data.get("overall_market_sentiment", "")
    qqq_assess = data.get("qqq_assessment", "")[:120] if data.get("qqq_assessment") else ""

    results = []
    for item in data.get("asset_analysis", []):
        results.append({
            "ticker":                item.get("asset", ""),
            "action":                item.get("action", ""),
            "bias_score":            item.get("bias_score"),
            "regime":                item.get("regime", ""),
            "entry_zone":            item.get("entry_zone", ""),
            "stop_loss":             item.get("stop_loss"),
            "profit_target":         item.get("profit_target"),
            "risk_reward_ratio":     item.get("risk_reward_ratio"),
            "estimated_holding_weeks": item.get("estimated_holding_weeks"),
            "overall_sentiment":     sentiment,
            "qqq_assessment":        qqq_assess,
        })
    return results


def _load_existing_keys() -> set[tuple]:
    """加载已记录的 (signal_date, ticker) 组合，防止重复写入。"""
    if not SIGNAL_LOG.exists():
        return set()
    keys = set()
    with open(SIGNAL_LOG, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keys.add((row.get("signal_date", ""), row.get("ticker", "")))
    return keys


def log_signals(tickers: list[str] | None = None, dry_run: bool = False,
                signal_date: str | None = None) -> int:
    """
    读取当前 *_api_output.txt，追加到 live_signal_log.csv。
    返回新写入的行数。
    """
    today = signal_date or str(date.today())
    existing = _load_existing_keys()

    target = tickers or list(ASSET_OUTPUT_MAP.keys())
    new_rows = []

    for ticker in target:
        fname = ASSET_OUTPUT_MAP.get(ticker.upper())
        if not fname:
            print(f"  [跳过] {ticker}：未在 ASSET_OUTPUT_MAP 中注册")
            continue

        fpath = Path(fname)
        signals = _parse_output_file(fpath)
        if not signals:
            print(f"  [跳过] {ticker}：{fname} 解析失败或为空")
            continue

        for sig in signals:
            sig_ticker = sig.get("ticker") or ticker
            key = (today, sig_ticker)
            if key in existing:
                print(f"  [已存在] {today} {sig_ticker}，跳过重复写入")
                continue
            row = {"signal_date": today, **sig}
            new_rows.append(row)
            existing.add(key)
            print(f"  [新增] {today} {sig_ticker}  action={sig.get('action')}  "
                  f"bias={sig.get('bias_score')}  regime={sig.get('regime')}")

    if dry_run:
        print(f"\n[dry-run] 共 {len(new_rows)} 条新信号，未写入文件。")
        return len(new_rows)

    if not new_rows:
        print("无新信号需要写入。")
        return 0

    file_exists = SIGNAL_LOG.exists()
    with open(SIGNAL_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)

    print(f"\n✅ 已追加 {len(new_rows)} 条信号 → {SIGNAL_LOG}")
    return len(new_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实盘信号持久化工具")
    parser.add_argument("--tickers", nargs="+", help="只记录指定资产（默认全部）")
    parser.add_argument("--dry-run", action="store_true", help="预览，不写入文件")
    parser.add_argument("--date", help="覆盖信号日期（默认今天，格式 YYYY-MM-DD）")
    args = parser.parse_args()

    log_signals(
        tickers=args.tickers,
        dry_run=args.dry_run,
        signal_date=args.date,
    )
