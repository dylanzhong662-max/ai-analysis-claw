"""
freeze_holdout.py — 封存 2024 全年数据作为真正的 Holdout 集

用途：
  把 2024-01-01 ~ 2024-12-31 的所有股票+宏观数据下载到 holdout_2024/ 目录，
  并写入 .locked 文件，防止意外覆盖。

  之后 tech_backtest_engine.py --start 2024-01-01 --end 2024-12-31 --reproducible
  会自动读取 data_cache/ 中的 parquet 缓存，与此 holdout 数据完全对应。

重要规则：
  - 一旦冻结，此数据不得再用于参数调整或规则修改
  - 只有在所有规则"冻结"后，才能在此数据上运行最终 OOS 验证
  - 违反此规则会导致 holdout 失效（隐式训练 = 过拟合）

使用方法：
  python3 freeze_holdout.py                    # 执行封存（只需运行一次）
  python3 freeze_holdout.py --status           # 查看封存状态
  python3 freeze_holdout.py --force            # 强制重新封存（会清除现有数据）
"""

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
yf.set_tz_cache_location(tempfile.mkdtemp())

# ── 封存配置 ──────────────────────────────────────────────────────────────────
HOLDOUT_DIR  = Path("holdout_2024")
LOCK_FILE    = HOLDOUT_DIR / ".locked"
MANIFEST     = HOLDOUT_DIR / "manifest.json"

HOLDOUT_START = "2023-07-01"   # 多取半年 lookback，保证指标计算够用
HOLDOUT_END   = "2024-12-31"
HOLDOUT_LABEL = "2024-01-01 ~ 2024-12-31"

# 股票标的
TICKERS = ["NVDA", "MSFT", "GOOGL", "AAPL", "META", "AMZN"]
# 宏观数据
MACRO_TICKERS = {
    "QQQ":      "QQQ",
    "SPY":      "SPY",
    "XLK":      "XLK",
    "VIX":      "^VIX",
    "TNX":      "^TNX",
    "DXY":      "DX-Y.NYB",
}

INTERVALS = ["1d", "1wk"]


def _download_df(ticker: str, interval: str) -> pd.DataFrame:
    """下载单个 ticker 的 OHLCV 数据，带重试（兼容 yfinance 0.2.x 和 1.x）。"""
    import inspect
    _yf_dl_params = set(inspect.signature(yf.download).parameters.keys())

    for attempt in range(3):
        try:
            kwargs = dict(
                start=HOLDOUT_START, end=HOLDOUT_END,
                interval=interval, auto_adjust=True,
                progress=False,
            )
            if "show_errors" in _yf_dl_params:
                kwargs["show_errors"] = False
            df = yf.download(ticker, **kwargs)
            if df is not None and not df.empty:
                # yfinance 1.x 返回 MultiIndex columns，打平
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
        except Exception as e:
            if attempt == 2:
                print(f"    ✗ {ticker} ({interval}) 下载失败: {e}")
            else:
                time.sleep(2 * (attempt + 1))
    return pd.DataFrame()


def freeze(force: bool = False):
    """执行数据封存。"""
    if LOCK_FILE.exists() and not force:
        print(f"⚠️  holdout_2024 已经封存（{LOCK_FILE}）。")
        print("   如需重新封存，使用 --force 参数（会清除现有数据）。")
        return

    if force and HOLDOUT_DIR.exists():
        import shutil
        shutil.rmtree(HOLDOUT_DIR)
        print("已清除旧封存数据。")

    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"开始封存 {HOLDOUT_LABEL} 数据 → {HOLDOUT_DIR}/")
    print("=" * 60)
    print("⚠️  封存后此数据不得再用于参数调整！只用于最终 OOS 验证。")
    print("=" * 60)

    manifest = {
        "frozen_at":    datetime.now().isoformat(),
        "holdout_range": HOLDOUT_LABEL,
        "lookback_start": HOLDOUT_START,
        "tickers":       TICKERS,
        "macro":         list(MACRO_TICKERS.keys()),
        "files":         [],
        "warnings":      [],
    }

    # ── 下载股票数据 ──────────────────────────────────────────────
    for ticker in TICKERS:
        for interval in INTERVALS:
            print(f"  下载 {ticker} ({interval})...", end="  ", flush=True)
            df = _download_df(ticker, interval)
            if df.empty:
                msg = f"{ticker} ({interval}) 下载失败，holdout 数据不完整"
                print(f"✗ 失败")
                manifest["warnings"].append(msg)
                continue
            fname = f"{ticker}_{interval}.parquet"
            fpath = HOLDOUT_DIR / fname
            df.to_parquet(fpath)
            rows = len(df)
            print(f"✓ {rows} 行  → {fname}")
            manifest["files"].append({
                "file":     fname,
                "ticker":   ticker,
                "interval": interval,
                "rows":     rows,
                "start":    str(df.index[0].date()),
                "end":      str(df.index[-1].date()),
            })

    # ── 下载宏观数据 ───────────────────────────────────────────────
    for name, yticker in MACRO_TICKERS.items():
        for interval in INTERVALS:
            print(f"  下载 {name}/{yticker} ({interval})...", end="  ", flush=True)
            df = _download_df(yticker, interval)
            if df.empty:
                msg = f"{name} ({interval}) 下载失败"
                print(f"✗ 失败（跳过）")
                manifest["warnings"].append(msg)
                continue
            fname = f"{name}_{interval}.parquet"
            fpath = HOLDOUT_DIR / fname
            df.to_parquet(fpath)
            rows = len(df)
            print(f"✓ {rows} 行  → {fname}")
            manifest["files"].append({
                "file":     fname,
                "ticker":   yticker,
                "interval": interval,
                "rows":     rows,
                "start":    str(df.index[0].date()),
                "end":      str(df.index[-1].date()),
            })

    # ── 写 manifest ────────────────────────────────────────────────
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ── 写锁文件 ───────────────────────────────────────────────────
    with open(LOCK_FILE, "w") as f:
        f.write(f"Frozen at: {manifest['frozen_at']}\n")
        f.write(f"Range: {HOLDOUT_LABEL}\n")
        f.write(f"Tickers: {', '.join(TICKERS)}\n")
        f.write(f"Files: {len(manifest['files'])}\n")
        if manifest["warnings"]:
            f.write(f"Warnings: {len(manifest['warnings'])}\n")
            for w in manifest["warnings"]:
                f.write(f"  - {w}\n")
        f.write("\n")
        f.write("DO NOT MODIFY THIS DATA AFTER FREEZING.\n")
        f.write("This is the holdout set for final OOS validation only.\n")

    print(f"\n{'='*60}")
    print(f"✅ 封存完成！{len(manifest['files'])} 个文件写入 {HOLDOUT_DIR}/")
    if manifest["warnings"]:
        print(f"⚠️  {len(manifest['warnings'])} 个警告（部分数据下载失败）:")
        for w in manifest["warnings"]:
            print(f"   - {w}")
    print(f"\n下一步：")
    print(f"  1. 继续在 2022-2023 和 2025 数据上调参")
    print(f"  2. 策略参数冻结后，运行以下命令进行最终 OOS 验证：")
    print(f"     python3 tech_backtest_engine.py --ticker NVDA --start 2024-01-01 --end 2024-12-31 --reproducible")
    print(f"     python3 tech_backtest_engine.py --ticker NVDA --start 2024-01-01 --end 2024-12-31 --simplified --reproducible")
    print(f"     python3 tech_backtest_engine.py --ticker NVDA --start 2024-01-01 --end 2024-12-31 --beta-floor --reproducible")


def status():
    """查看封存状态。"""
    if not HOLDOUT_DIR.exists():
        print("holdout_2024/ 目录不存在，尚未封存。")
        print("运行 python3 freeze_holdout.py 执行封存。")
        return

    if not LOCK_FILE.exists():
        print("holdout_2024/ 目录存在但未完成封存（.locked 文件缺失）。")
        return

    print(f"✅ holdout_2024 已封存")
    print(f"   锁文件: {LOCK_FILE}")
    print()
    print(LOCK_FILE.read_text())

    if MANIFEST.exists():
        with open(MANIFEST) as f:
            m = json.load(f)
        print(f"封存清单：{len(m['files'])} 个文件")
        for item in m["files"]:
            print(f"  {item['file']:<35} {item['rows']:>4} 行  {item['start']} ~ {item['end']}")
        if m.get("warnings"):
            print(f"\n警告（{len(m['warnings'])} 条）:")
            for w in m["warnings"]:
                print(f"  ⚠️  {w}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="封存 2024 全年数据作为 Holdout 集")
    parser.add_argument("--status", action="store_true", help="查看封存状态")
    parser.add_argument("--force",  action="store_true", help="强制重新封存（清除现有数据）")
    args = parser.parse_args()

    if args.status:
        status()
    else:
        freeze(force=args.force)
