"""
validate_regime.py — 验证2：制度过滤规则（EMA200/EMA50）的历史准确率

分析问题：
  EMA200制度过滤 和 EMA50快速出场 这两条规则到底有多强？
  系统最有价值的部分——"熊市清仓"——是否真的有效？

输出：
  1. 制度准确率：处于"牛市制度"时，未来N周涨的概率
  2. 各制度下平均前向收益（4/8/12/20周）
  3. EMA200出场的回撤规避效果
  4. EMA50快速出场 vs EMA200慢速出场的时序对比
  5. 每个资产的分年统计（识别哪些年份制度过滤有效/无效）

运行：
  python3 validate_regime.py
  python3 validate_regime.py --tickers NVDA MSFT --start 2020-01-01
"""

import argparse
import tempfile
import urllib3
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests as curl_requests

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
yf.set_tz_cache_location(tempfile.mkdtemp())


def _make_session():
    import os
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    return curl_requests.Session(impersonate="chrome", verify=False, proxy=proxy)

# ── 默认配置 ────────────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["NVDA", "MSFT", "GOOGL"]
DEFAULT_START   = "2018-01-01"
DEFAULT_END     = "2026-01-01"
HORIZONS        = [4, 8, 12, 20]   # 前向评估周数
EMA200_PERIOD   = 200
EMA50_PERIOD    = 50
DRAWDOWN_EXIT   = 0.15             # 15% 回撤触发快速出场
REBAL_COST      = 0.002            # 单次再平衡成本（佣金+滑点，往返）


# ── 工具函数 ────────────────────────────────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def download_weekly(ticker: str, start: str, end: str) -> pd.DataFrame:
    session = _make_session()
    df = yf.download(ticker, start=start, end=end,
                     interval="1wk", auto_adjust=True, progress=False,
                     session=session)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


# ── 主分析函数 ───────────────────────────────────────────────────────────────

def analyze_ticker(ticker: str, start: str, end: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  {ticker}  ({start} ~ {end})")
    print('='*60)

    df = download_weekly(ticker, start, end)
    if len(df) < EMA200_PERIOD + max(HORIZONS) + 10:
        print(f"  数据不足（{len(df)}周），跳过")
        return {}

    close = df["Close"].squeeze()
    high  = df["High"].squeeze()

    # ── 技术指标 ────────────────────────────────────────────────────────────
    ema200 = calc_ema(close, EMA200_PERIOD)
    ema50  = calc_ema(close, EMA50_PERIOD)

    # 52周滚动高点（用于回撤计算）
    high_52w = close.rolling(52, min_periods=20).max()

    # ── 制度标记 ─────────────────────────────────────────────────────────────
    df["ema200"]       = ema200
    df["ema50"]        = ema50
    df["high_52w"]     = high_52w
    df["above_ema200"] = (close > ema200).astype(int)
    df["above_ema50"]  = (close > ema50).astype(int)
    df["drawdown"]     = (high_52w - close) / high_52w

    # 熊市制度：价格跌破EMA200
    # 快速出场：跌破EMA50 或 回撤>15%
    df["regime"] = "bull"
    df.loc[close <= ema200, "regime"] = "bear"
    df.loc[(close > ema200) & (
        (close <= ema50) | (df["drawdown"] > DRAWDOWN_EXIT)
    ), "regime"] = "fast_exit"

    # ── 前向收益计算 ──────────────────────────────────────────────────────────
    for h in HORIZONS:
        df[f"fwd_{h}w"] = close.pct_change(h).shift(-h)

    # 只保留有足够EMA历史的行
    df = df.iloc[EMA200_PERIOD:].dropna(subset=["ema200"])

    # ── 统计分析 ──────────────────────────────────────────────────────────────
    results = {}

    print(f"\n  总样本: {len(df)} 周")
    regime_counts = df["regime"].value_counts()
    for r, n in regime_counts.items():
        print(f"    {r:12s}: {n:3d}周 ({n/len(df)*100:.0f}%)")

    # 1. 各制度下前向收益
    print(f"\n  [制度×前向收益] 平均N周后收益（%）")
    print(f"  {'制度':<12}", end="")
    for h in HORIZONS:
        print(f"  {h}周后".rjust(8), end="")
    print(f"  {'样本数':>6}")

    regime_stats = {}
    for regime in ["bull", "fast_exit", "bear"]:
        mask = df["regime"] == regime
        sub  = df[mask]
        if sub.empty:
            continue
        row = {"n": len(sub)}
        print(f"  {regime:<12}", end="")
        for h in HORIZONS:
            col = f"fwd_{h}w"
            v = sub[col].dropna()
            mean_ret = v.mean() * 100 if not v.empty else float("nan")
            row[f"mean_{h}w"] = mean_ret
            row[f"pos_rate_{h}w"] = (v > 0).mean() * 100 if not v.empty else float("nan")
            print(f"  {mean_ret:+6.1f}%", end="")
        print(f"  {len(sub):>6}")
        regime_stats[regime] = row
    results["regime_stats"] = regime_stats

    # 2. 制度识别"准确率"（牛市制度 → 未来确实上涨的概率）
    print(f"\n  [牛市制度准确率] 处于bull制度时，未来N周上涨概率")
    print(f"  {'制度':<12}", end="")
    for h in HORIZONS:
        print(f"  {h}周胜率".rjust(8), end="")
    print()
    for regime in ["bull", "fast_exit", "bear"]:
        mask = df["regime"] == regime
        sub  = df[mask]
        if sub.empty:
            continue
        print(f"  {regime:<12}", end="")
        for h in HORIZONS:
            col = f"fwd_{h}w"
            v = sub[col].dropna()
            rate = (v > 0).mean() * 100 if not v.empty else float("nan")
            print(f"  {rate:5.1f}%  ", end="")
        print()

    # 3. EMA200出场的回撤规避效果
    print(f"\n  [EMA200出场效果] 跌破EMA200后N周内的平均额外下跌")
    bear_start_rows = df[(df["regime"] == "bear") &
                         (df["regime"].shift(1).isin(["bull","fast_exit"]))].copy()
    if not bear_start_rows.empty:
        for h in HORIZONS:
            col = f"fwd_{h}w"
            exits = bear_start_rows[col].dropna()
            if not exits.empty:
                print(f"    出场后{h:2d}周平均收益: {exits.mean()*100:+.1f}%"
                      f"  (下跌概率 {(exits<0).mean()*100:.0f}%,"
                      f" 样本{len(exits)}次)")
    else:
        print("    无EMA200跌破事件（全程牛市？）")

    # 4. EMA50快速出场 vs EMA200慢出场的时序优势
    print(f"\n  [EMA50快速出场时序优势] 跌破EMA50时，距离跌破EMA200还有多少周？")
    fast_exit_rows = df[(df["regime"] == "fast_exit") &
                        (df["regime"].shift(1) == "bull")].copy()
    if not fast_exit_rows.empty:
        weeks_to_ema200 = []
        for idx in fast_exit_rows.index:
            # 往后找第一次跌破EMA200
            future = df.loc[idx:, "regime"]
            bear_after = future[future == "bear"]
            if not bear_after.empty:
                w = (bear_after.index[0] - idx).days // 7
                weeks_to_ema200.append(w)
        if weeks_to_ema200:
            arr = np.array(weeks_to_ema200)
            print(f"    平均提前 {arr.mean():.1f} 周（中位数 {np.median(arr):.0f}周）出场")
            print(f"    最少提前 {arr.min()} 周，最多提前 {arr.max()} 周")
            print(f"    样本数: {len(arr)} 次")
        else:
            print("    快速出场后都未跌破EMA200（快速出场后反弹了）")
            fwd4 = fast_exit_rows["fwd_4w"].dropna()
            print(f"    快速出场后4周均值: {fwd4.mean()*100:+.1f}%"
                  f"  (假阳性率: {(fwd4>0).mean()*100:.0f}%)")
    else:
        print("    无独立的fast_exit事件")

    # 5. 分年统计
    print(f"\n  [分年统计] 每年bull制度时，12周后平均收益 & 胜率")
    print(f"  {'年份':<6}  {'bull周数':>6}  {'bear周数':>6}  "
          f"{'12w均收益':>9}  {'12w胜率':>7}  {'年度趋势':<10}")
    df["year"] = df.index.year
    annual_rows = []
    for yr, grp in df.groupby("year"):
        bull_n  = (grp["regime"] == "bull").sum()
        bear_n  = (grp["regime"] == "bear").sum()
        bull_g  = grp[grp["regime"] == "bull"]
        fwd12   = bull_g["fwd_12w"].dropna()
        mean12  = fwd12.mean() * 100 if not fwd12.empty else float("nan")
        rate12  = (fwd12 > 0).mean() * 100 if not fwd12.empty else float("nan")
        # 年度涨跌
        yr_ret  = (grp["Close"].iloc[-1] / grp["Close"].iloc[0] - 1) * 100 if len(grp) > 1 else float("nan")
        trend   = f"{'↑' if yr_ret > 0 else '↓'}{abs(yr_ret):.0f}%"
        print(f"  {yr:<6}  {bull_n:>6}  {bear_n:>6}  "
              f"  {mean12:>+7.1f}%  {rate12:>6.0f}%  {trend:<10}")
        annual_rows.append({"year": yr, "bull_weeks": bull_n, "bear_weeks": bear_n,
                            "mean_12w": mean12, "rate_12w": rate12, "yr_return": yr_ret})
    results["annual"] = annual_rows

    # 6. 综合评分
    print(f"\n  [综合评分]")
    bull_12w = regime_stats.get("bull", {}).get("mean_12w", float("nan"))
    bear_12w = regime_stats.get("bear", {}).get("mean_12w", float("nan"))
    bull_pos  = regime_stats.get("bull", {}).get("pos_rate_12w", float("nan"))

    if not np.isnan(bull_12w) and not np.isnan(bear_12w):
        spread = bull_12w - bear_12w
        print(f"    牛市12周均收益: {bull_12w:+.1f}%  熊市12周均收益: {bear_12w:+.1f}%")
        print(f"    制度区分度（spread）: {spread:+.1f}pp")
        print(f"    牛市制度12周胜率: {bull_pos:.0f}%")
        if spread > 10:
            print(f"    ✅ 制度过滤有效：牛熊收益差异显著（>{10}pp）")
        elif spread > 5:
            print(f"    ⚠️  制度过滤一般：差异有限（{spread:.1f}pp）")
        else:
            print(f"    ❌ 制度过滤效果弱：牛熊收益差异不明显")

    return results


# ── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="制度过滤规则历史验证")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start",   default=DEFAULT_START)
    parser.add_argument("--end",     default=DEFAULT_END)
    args = parser.parse_args()

    print("=" * 60)
    print("  验证2：EMA200/EMA50 制度过滤规则历史准确率分析")
    print(f"  标的: {args.tickers}  区间: {args.start} ~ {args.end}")
    print("=" * 60)

    all_results = {}
    for ticker in args.tickers:
        try:
            all_results[ticker] = analyze_ticker(ticker, args.start, args.end)
        except Exception as e:
            print(f"  {ticker} 分析失败: {e}")

    # 跨资产汇总
    print(f"\n\n{'='*60}")
    print("  跨资产汇总：各资产制度过滤效果对比")
    print('='*60)
    print(f"  {'资产':<8}  {'牛市4w均':>8}  {'牛市12w均':>9}  "
          f"{'熊市4w均':>8}  {'熊市12w均':>9}  {'制度有效?':>8}")
    for ticker, res in all_results.items():
        rs = res.get("regime_stats", {})
        b4  = rs.get("bull",  {}).get("mean_4w",  float("nan"))
        b12 = rs.get("bull",  {}).get("mean_12w", float("nan"))
        e4  = rs.get("bear",  {}).get("mean_4w",  float("nan"))
        e12 = rs.get("bear",  {}).get("mean_12w", float("nan"))
        spread = b12 - e12 if not (np.isnan(b12) or np.isnan(e12)) else float("nan")
        flag = ("✅" if spread > 10 else ("⚠️" if spread > 5 else "❌")) if not np.isnan(spread) else "N/A"
        print(f"  {ticker:<8}  {b4:>+7.1f}%  {b12:>+8.1f}%  "
              f"{e4:>+7.1f}%  {e12:>+8.1f}%  {flag:>8}")

    print(f"\n注：数据来源 yfinance 周线，EMA200基于{EMA200_PERIOD}周，EMA50基于{EMA50_PERIOD}周")
    print("结论参考：spread>10pp 表示制度过滤显著有效，>5pp 一般有效，<5pp 效果弱")


if __name__ == "__main__":
    main()
