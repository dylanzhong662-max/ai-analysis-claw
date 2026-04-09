"""
parse_logs_ic.py — 从历史 daily logs 提取信号 + 计算 IC

流程：
  1. 扫描 logs/daily_*.log，提取每个 JSON 信号块
  2. 用 yfinance 获取信号日 + 后续 5/10/15/21 天的收盘价
  3. 计算前向收益率
  4. 计算 Spearman IC（bias_score vs forward_return）
  5. 输出 IC 汇总 + 信号明细 CSV

运行方式：
  python3 parse_logs_ic.py                        # 使用默认 logs/ 目录
  python3 parse_logs_ic.py --log-dir /opt/finance-analysis/logs
  python3 parse_logs_ic.py --horizons 5 10 21     # 自定义评估周期
  python3 parse_logs_ic.py --tickers NVDA GOOGL   # 只分析指定标的

注意：样本量极小（当前约18天 × 5资产），IC 结论仅供参考，无统计显著性。
"""

import argparse
import json
import os
import re
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

# yfinance 缓存目录
import tempfile
yf.set_tz_cache_location(tempfile.mkdtemp())

# ticker 映射：信号里的 asset 名 → yfinance ticker
TICKER_MAP = {
    "NVDA": "NVDA", "MSFT": "MSFT", "GOOGL": "GOOGL",
    "AAPL": "AAPL", "META": "META", "AMZN": "AMZN",
    "BTC":  "BTC-USD", "GOLD": "GC=F",
}

HORIZONS_DEFAULT = [5, 10, 15, 21]


# ── 日志解析 ────────────────────────────────────────────────────────────────

def _extract_json_blocks(text: str) -> list[dict]:
    """从日志文本中提取所有顶级 JSON 对象（支持嵌套）。"""
    blocks = []
    i = 0
    while i < len(text):
        start = text.find("{", i)
        if start < 0:
            break
        depth, end = 0, start
        for j, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        fragment = text[start : end + 1]
        try:
            obj = json.loads(fragment)
            if isinstance(obj, dict) and "asset_analysis" in obj:
                blocks.append(obj)
        except json.JSONDecodeError:
            pass
        i = end + 1
    return blocks


def parse_log_file(log_path: Path, log_date: str) -> list[dict]:
    """解析单个 daily log 文件，返回信号列表。"""
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"  读取失败 {log_path}: {e}")
        return []

    records = []
    for block in _extract_json_blocks(text):
        sentiment = block.get("overall_market_sentiment", "")
        for item in block.get("asset_analysis", []):
            asset = item.get("asset", "")
            if not asset:
                continue
            records.append({
                "signal_date":    log_date,
                "ticker":         asset,
                "action":         item.get("action", ""),
                "bias_score":     item.get("bias_score"),
                "regime":         item.get("regime", ""),
                "stop_loss":      item.get("stop_loss"),
                "profit_target":  item.get("profit_target"),
                "rr":             item.get("risk_reward_ratio"),
                "sentiment":      sentiment,
            })
    return records


def scan_log_dir(log_dir: Path, tickers: list[str] | None = None) -> pd.DataFrame:
    """扫描 log 目录中所有 daily_*.log 文件，返回信号 DataFrame。"""
    all_records = []
    log_files = sorted(log_dir.glob("daily_*.log"))
    print(f"找到 {len(log_files)} 个 daily log 文件")

    for lf in log_files:
        # 从文件名提取日期：daily_20260406.log → 2026-04-06
        m = re.search(r"daily_(\d{8})", lf.name)
        if not m:
            continue
        raw_date = m.group(1)
        log_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
        records = parse_log_file(lf, log_date)
        print(f"  {lf.name}: {len(records)} 条信号")
        all_records.extend(records)

    if not all_records:
        print("未提取到任何信号！")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["bias_score"]  = pd.to_numeric(df["bias_score"],  errors="coerce")
    df["stop_loss"]   = pd.to_numeric(df["stop_loss"],   errors="coerce")
    df["profit_target"] = pd.to_numeric(df["profit_target"], errors="coerce")
    df["rr"]          = pd.to_numeric(df["rr"],          errors="coerce")

    # 过滤指定标的
    if tickers:
        df = df[df["ticker"].isin([t.upper() for t in tickers])]

    # 去重（同一天同一标的取最后一条，因为每天只跑一次）
    df = df.drop_duplicates(subset=["signal_date", "ticker"], keep="last")
    df = df.sort_values(["ticker", "signal_date"]).reset_index(drop=True)
    print(f"\n去重后共 {len(df)} 条唯一信号（{df['ticker'].nunique()} 个标的）")
    return df


# ── 价格获取 ────────────────────────────────────────────────────────────────

def fetch_forward_prices(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    为每条信号获取信号日收盘价 + 各 horizon 后的收盘价。
    新增列：price_0, price_5, price_10, ...
    """
    tickers_needed = df["ticker"].unique().tolist()
    # 确定下载日期范围
    min_date = df["signal_date"].min() - timedelta(days=3)
    max_date = df["signal_date"].max() + timedelta(days=max(horizons) * 2 + 10)

    price_cache: dict[str, pd.Series] = {}
    for asset in tickers_needed:
        yf_ticker = TICKER_MAP.get(asset, asset)
        print(f"  下载 {asset} ({yf_ticker}) 价格 {min_date.date()} ~ {max_date.date()}...")
        try:
            raw = yf.download(
                yf_ticker,
                start=min_date.strftime("%Y-%m-%d"),
                end=max_date.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                print(f"    ⚠ {asset}: 无数据")
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            price_cache[asset] = raw["Close"].squeeze()
        except Exception as e:
            print(f"    ✗ {asset}: {e}")

    # 添加价格列
    df = df.copy()
    df["price_0"] = float("nan")
    for h in horizons:
        df[f"fwd_{h}d"] = float("nan")
        df[f"ret_{h}d"] = float("nan")

    for idx, row in df.iterrows():
        asset = row["ticker"]
        sig_date = row["signal_date"]
        prices = price_cache.get(asset)
        if prices is None:
            continue

        # 找信号日当天或下一个交易日的收盘价
        avail = prices[prices.index >= sig_date]
        if avail.empty:
            continue
        p0_date = avail.index[0]
        p0 = float(avail.iloc[0])
        df.at[idx, "price_0"] = p0

        for h in horizons:
            # 找 h 个交易日之后的价格
            future = prices[prices.index > p0_date]
            if len(future) >= h:
                ph = float(future.iloc[h - 1])
                df.at[idx, f"fwd_{h}d"] = ph
                df.at[idx, f"ret_{h}d"] = (ph - p0) / p0

    return df


# ── IC 计算 ─────────────────────────────────────────────────────────────────

def compute_ic(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    计算 Spearman IC：bias_score vs forward return。
    分三组：全部信号、只有 long 信号、排除 no_trade。
    """
    from scipy.stats import spearmanr

    rows = []
    for h in horizons:
        ret_col = f"ret_{h}d"
        valid   = df.dropna(subset=["bias_score", ret_col])
        if len(valid) < 3:
            print(f"  horizon={h}d: 样本不足（{len(valid)} 条），跳过")
            continue

        for group_name, subset in [
            ("全部信号",    valid),
            ("long信号",   valid[valid["action"] == "long"]),
            ("非no_trade", valid[valid["action"] != "no_trade"]),
        ]:
            if len(subset) < 3:
                continue
            ic, pval = spearmanr(subset["bias_score"], subset[ret_col])
            rows.append({
                "horizon":  f"{h}d",
                "group":    group_name,
                "n":        len(subset),
                "IC":       round(ic, 4),
                "p_value":  round(pval, 4),
                "sig":      "✅" if pval < 0.05 else ("⚠️" if pval < 0.10 else "❌"),
                "mean_ret": round(subset[ret_col].mean() * 100, 2),
                "std_ret":  round(subset[ret_col].std()  * 100, 2),
            })

    return pd.DataFrame(rows)


def compute_long_only_stats(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """对 long 信号按 bias 分档统计平均收益。"""
    longs = df[df["action"] == "long"].copy()
    if longs.empty:
        return pd.DataFrame()

    longs["bias_bucket"] = pd.cut(
        longs["bias_score"],
        bins=[0, 0.50, 0.55, 0.60, 0.65, 0.70, 1.01],
        labels=["<0.50", "0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "≥0.70"],
    )

    rows = []
    for h in horizons:
        ret_col = f"ret_{h}d"
        valid = longs.dropna(subset=[ret_col])
        if valid.empty:
            continue
        for bucket, grp in valid.groupby("bias_bucket", observed=True):
            rows.append({
                "horizon": f"{h}d",
                "bias_bucket": bucket,
                "n": len(grp),
                "mean_ret_%": round(grp[ret_col].mean() * 100, 2),
                "win_rate_%": round((grp[ret_col] > 0).mean() * 100, 1),
            })
    return pd.DataFrame(rows)


# ── 主流程 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="从历史 logs 提取信号并计算 IC")
    parser.add_argument("--log-dir",  default="logs",        help="日志目录（默认 ./logs）")
    parser.add_argument("--horizons", nargs="+", type=int,   default=HORIZONS_DEFAULT,
                        help="前向评估天数（默认 5 10 15 21）")
    parser.add_argument("--tickers",  nargs="+",             help="只分析指定标的")
    parser.add_argument("--out-csv",  default="ic_analysis_results.csv", help="输出 CSV 文件名")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"日志目录不存在: {log_dir}")
        sys.exit(1)

    print("=" * 60)
    print("步骤 1：解析日志文件")
    print("=" * 60)
    df = scan_log_dir(log_dir, tickers=args.tickers)
    if df.empty:
        sys.exit(1)

    print("\n" + "=" * 60)
    print("步骤 2：获取前向价格数据")
    print("=" * 60)
    df = fetch_forward_prices(df, args.horizons)

    # 保存明细
    detail_csv = Path(args.out_csv).stem + "_detail.csv"
    df.to_csv(detail_csv, index=False)
    print(f"\n信号明细已保存 → {detail_csv}")

    print("\n" + "=" * 60)
    print("步骤 3：IC 计算结果")
    print("=" * 60)

    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("缺少 scipy，请运行：pip install scipy")
        sys.exit(1)

    ic_df = compute_ic(df, args.horizons)
    if ic_df.empty:
        print("样本不足，无法计算 IC。")
    else:
        print(ic_df.to_string(index=False))
        ic_df.to_csv(args.out_csv, index=False)
        print(f"\nIC 汇总已保存 → {args.out_csv}")

    print("\n" + "=" * 60)
    print("步骤 4：Long 信号 Bias 分档收益")
    print("=" * 60)
    bucket_df = compute_long_only_stats(df, args.horizons)
    if bucket_df.empty:
        print("无 long 信号或数据不足。")
    else:
        print(bucket_df.to_string(index=False))
        bucket_csv = Path(args.out_csv).stem + "_buckets.csv"
        bucket_df.to_csv(bucket_csv, index=False)
        print(f"\nBias 分档统计已保存 → {bucket_csv}")

    print("\n" + "=" * 60)
    print("步骤 5：信号分布摘要")
    print("=" * 60)
    print(f"信号总数: {len(df)}")
    print(f"日期范围: {df['signal_date'].min().date()} ~ {df['signal_date'].max().date()}")
    print(f"标的: {sorted(df['ticker'].unique().tolist())}")
    print("\n按 action 分布:")
    print(df["action"].value_counts().to_string())
    print("\n按标的 × action 分布:")
    print(df.groupby(["ticker", "action"]).size().to_string())

    if len(df) < 30:
        print(f"\n⚠️  当前样本量 {len(df)} < 30，IC 结果不具统计显著性。")
        print("   继续积累数据，3 个月后（约 60-90 条）结论才可信。")


if __name__ == "__main__":
    main()
