"""
数据预取脚本 — 提前下载并缓存指定年份的所有回测所需数据
用法:
    python3 prefetch_data.py --year 2022
    python3 prefetch_data.py --start 2021-01-01 --end 2023-06-30
    python3 prefetch_data.py --year 2022 --tickers NVDA MSFT GOOGL AAPL META AMZN

数据存储到 data_cache/*.parquet，与现有缓存自动合并（不覆盖已有数据）。
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
import urllib3
from curl_cffi import requests as curl_requests

yf.set_tz_cache_location(tempfile.mkdtemp())
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
_CACHE_DIR = Path("data_cache")

# 科技股 + 大宗商品 ETF（tech_backtest_engine.py 覆盖范围）
TECH_TICKERS = ["NVDA", "MSFT", "GOOGL", "AAPL", "META", "AMZN",
                "SLV", "COPX", "REMX", "USO"]

# 宏观指标（所有回测都需要）
MACRO_TICKERS = ["QQQ", "XLK", "SPY", "^TNX", "^VIX", "DX-Y.NYB"]

# 各 ticker 需要下载的时间粒度
INTERVALS_TECH  = ["1d", "1wk", "1mo"]
INTERVALS_MACRO = ["1d", "1wk"]

# 回测参数常量
LOOKBACK_DAYS = 200   # 指标计算需要的历史长度
EVAL_DAYS     = 65    # 最长持仓（需要在 end 后预留缓冲）
BUFFER_EXTRA  = 30    # 额外余量


def _cache_path(ticker: str, interval: str) -> Path:
    safe = ticker.replace("^", "").replace("/", "_").replace("=", "")
    return _CACHE_DIR / f"{safe}_{interval}.parquet"


def _make_session():
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    return curl_requests.Session(impersonate="chrome", verify=False, proxy=proxy)


def _download_with_retry(ticker: str, start: str, end: str,
                         interval: str, retries: int = 5) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            session = _make_session()
            df = yf.download(ticker, start=start, end=end,
                             interval=interval, auto_adjust=True,
                             progress=False, session=session)
            if df is not None and not df.empty:
                return df
            if attempt < retries - 1:
                time.sleep(6 * (attempt + 1))
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(6 * (attempt + 1))
            else:
                print(f"    [失败] {ticker} {interval}: {e}")
    return pd.DataFrame()


def _merge_and_save(ticker: str, interval: str, new_df: pd.DataFrame):
    """将新下载的数据与磁盘已有数据合并后保存。"""
    _CACHE_DIR.mkdir(exist_ok=True)
    p = _cache_path(ticker, interval)
    if p.exists():
        try:
            existing = pd.read_parquet(p)
            combined = pd.concat([existing, new_df]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
        except Exception:
            combined = new_df
    else:
        combined = new_df
    combined.to_parquet(p)
    return combined


def prefetch_year(year: int = None,
                  start: str = None,
                  end: str = None,
                  tickers: list = None):
    """
    下载并缓存指定范围的所有回测数据。

    日期范围逻辑：
      - 下载起点 = max(start - LOOKBACK_DAYS, 不早于5年前)
      - 下载终点 = end + EVAL_DAYS + BUFFER_EXTRA（缓冲持仓评估期）
    """
    if year and not (start and end):
        start = f"{year}-01-01"
        end   = f"{year}-12-31"

    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=LOOKBACK_DAYS + BUFFER_EXTRA)
                  ).strftime("%Y-%m-%d")
    fetch_end   = (pd.Timestamp(end)   + pd.Timedelta(days=EVAL_DAYS + BUFFER_EXTRA)
                  ).strftime("%Y-%m-%d")

    target_tickers = tickers or TECH_TICKERS

    print(f"\n{'='*60}")
    print(f"  预取目标年份: {start} ~ {end}")
    print(f"  实际下载范围: {fetch_start} ~ {fetch_end}")
    print(f"  股票: {target_tickers}")
    print(f"  宏观: {MACRO_TICKERS}")
    print(f"{'='*60}\n")

    total = (len(target_tickers) * len(INTERVALS_TECH) +
             len(MACRO_TICKERS)  * len(INTERVALS_MACRO))
    done = 0

    # ── 科技股数据 ──
    for ticker in target_tickers:
        for interval in INTERVALS_TECH:
            done += 1
            p = _cache_path(ticker, interval)

            # 检查现有缓存覆盖范围
            if p.exists():
                try:
                    existing = pd.read_parquet(p)
                    if not existing.empty:
                        cmin = existing.index.min()
                        cmax = existing.index.max()
                        need_start = pd.Timestamp(fetch_start) - pd.Timedelta(days=5)
                        need_end   = pd.Timestamp(fetch_end)   + pd.Timedelta(days=5)
                        if cmin <= need_start and cmax >= need_end:
                            print(f"  [{done:3d}/{total}] {ticker:6s} [{interval}]  ✓ 缓存已覆盖 "
                                  f"({cmin.date()} ~ {cmax.date()}, {len(existing)} 条)")
                            continue
                        else:
                            print(f"  [{done:3d}/{total}] {ticker:6s} [{interval}]  ↓ 缓存范围不足，补充下载...", end=" ", flush=True)
                except Exception:
                    print(f"  [{done:3d}/{total}] {ticker:6s} [{interval}]  ↓ 缓存损坏，重新下载...", end=" ", flush=True)
            else:
                print(f"  [{done:3d}/{total}] {ticker:6s} [{interval}]  ↓ 新下载...", end=" ", flush=True)

            df = _download_with_retry(ticker, fetch_start, fetch_end, interval)
            if not df.empty:
                merged = _merge_and_save(ticker, interval, df)
                print(f"✓  {len(merged)} 条  ({merged.index.min().date()} ~ {merged.index.max().date()})")
            else:
                print("✗  下载失败，回测时将实时重试")

    # ── 宏观指标数据 ──
    for macro_ticker in MACRO_TICKERS:
        for interval in INTERVALS_MACRO:
            done += 1
            p = _cache_path(macro_ticker, interval)

            if p.exists():
                try:
                    existing = pd.read_parquet(p)
                    if not existing.empty:
                        cmin = existing.index.min()
                        cmax = existing.index.max()
                        need_start = pd.Timestamp(fetch_start) - pd.Timedelta(days=5)
                        need_end   = pd.Timestamp(fetch_end)   + pd.Timedelta(days=5)
                        if cmin <= need_start and cmax >= need_end:
                            print(f"  [{done:3d}/{total}] {macro_ticker:12s} [{interval}]  ✓ 缓存已覆盖 "
                                  f"({cmin.date()} ~ {cmax.date()}, {len(existing)} 条)")
                            continue
                        else:
                            print(f"  [{done:3d}/{total}] {macro_ticker:12s} [{interval}]  ↓ 补充下载...", end=" ", flush=True)
                except Exception:
                    print(f"  [{done:3d}/{total}] {macro_ticker:12s} [{interval}]  ↓ 重新下载...", end=" ", flush=True)
            else:
                print(f"  [{done:3d}/{total}] {macro_ticker:12s} [{interval}]  ↓ 新下载...", end=" ", flush=True)

            df = _download_with_retry(macro_ticker, fetch_start, fetch_end, interval)
            if not df.empty:
                merged = _merge_and_save(macro_ticker, interval, df)
                print(f"✓  {len(merged)} 条  ({merged.index.min().date()} ~ {merged.index.max().date()})")
            else:
                print("✗  下载失败")

    print(f"\n{'='*60}")
    print(f"  预取完成！数据已存入 {_CACHE_DIR.resolve()}/")
    print(f"  共 {len(list(_CACHE_DIR.glob('*.parquet')))} 个 parquet 文件")
    print(f"{'='*60}\n")

    # 打印缓存汇总
    print("缓存文件汇总:")
    for f in sorted(_CACHE_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(f)
            print(f"  {f.name:35s}  {len(df):5d} 条  "
                  f"{df.index.min().date()} ~ {df.index.max().date()}")
        except Exception as e:
            print(f"  {f.name:35s}  [读取失败: {e}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预下载回测所需历史数据并缓存为 Parquet")
    parser.add_argument("--year",  type=int, help="目标年份，例如 2022")
    parser.add_argument("--start", type=str, help="自定义起始日期，例如 2022-01-01")
    parser.add_argument("--end",   type=str, help="自定义结束日期，例如 2022-12-31")
    parser.add_argument("--tickers", nargs="+",
                        default=None,
                        help="只下载指定 ticker，不填则下载全部科技股")
    args = parser.parse_args()

    if not args.year and not (args.start and args.end):
        parser.error("请提供 --year 或同时提供 --start 和 --end")

    prefetch_year(
        year=args.year,
        start=args.start,
        end=args.end,
        tickers=args.tickers,
    )
