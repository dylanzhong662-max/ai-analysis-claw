"""
validate_ic.py — 验证1：LLM信号 IC（信息系数）分析

检验核心问题：bias_score 对未来收益是否有预测力？

数据来源（按优先级）：
  1. 服务器日志（推荐）：--log-dir /path/to/logs
  2. 本地回测信号 CSV：  --signals-csv nvda_portfolio_backtest/signals.csv
  3. 本地 daily log：   logs/daily_*.log（本地演示用，样本量极少）

输出：
  1. IC（信息系数）= Spearman(bias_score, forward_return)
  2. t-statistic（判断统计显著性，>2 才可信）
  3. 各 bias 分档的平均收益和胜率
  4. no_trade / long / short 各动作的前向收益
  5. 制度（regime）× 前向收益的交叉分析

运行方式：
  # 本地演示（样本量不足，仅展示框架）
  python3 validate_ic.py

  # 服务器完整版（先 scp 日志回本地或直接在服务器运行）
  python3 validate_ic.py --log-dir /opt/finance-analysis/logs

  # 指定回测信号 CSV
  python3 validate_ic.py --signals-csv nvda_portfolio_backtest/signals.csv

  # 同时分析多个 CSV（多资产）
  python3 validate_ic.py --signals-csv nvda_portfolio_backtest/signals.csv \\
                                       msft_portfolio_backtest/signals.csv

统计显著性解读：
  t-stat > 2.0  (p < 0.05): 信号有效，值得使用
  t-stat > 1.5  (p < 0.10): 边缘显著，需更多数据
  t-stat ≤ 1.5            : 信号无效，不具统计意义
"""

import argparse
import json
import os
import re
import sys
import tempfile
import urllib3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests as curl_requests

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
yf.set_tz_cache_location(tempfile.mkdtemp())


def _make_session():
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    return curl_requests.Session(impersonate="chrome", verify=False, proxy=proxy)

TICKER_MAP = {
    "NVDA": "NVDA", "MSFT": "MSFT", "GOOGL": "GOOGL",
    "AAPL": "AAPL", "META": "META",  "AMZN": "AMZN",
    "BTC":  "BTC-USD", "GOLD": "GC=F",
    "SLV":  "SLV",  "COPX": "COPX",
}
HORIZONS = [5, 10, 15, 21]  # 交易日


# ── 数据加载 ─────────────────────────────────────────────────────────────────

def load_from_logs(log_dir: Path, tickers=None) -> pd.DataFrame:
    """解析 daily_*.log 文件中的 JSON 信号块"""
    records = []
    log_files = sorted(log_dir.glob("daily_*.log"))
    print(f"  找到 {len(log_files)} 个日志文件")

    for lf in log_files:
        m = re.search(r"daily_(\d{8})", lf.name)
        if not m:
            continue
        raw = m.group(1)
        log_date = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"

        try:
            text = lf.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # 提取所有 JSON 块
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
            fragment = text[start:end + 1]
            try:
                obj = json.loads(fragment)
                if isinstance(obj, dict) and "asset_analysis" in obj:
                    sentiment = obj.get("overall_market_sentiment", "")
                    for item in obj.get("asset_analysis", []):
                        asset = item.get("asset", "")
                        if tickers and asset not in [t.upper() for t in tickers]:
                            i = end + 1
                            continue
                        records.append({
                            "signal_date":  log_date,
                            "ticker":       asset,
                            "action":       item.get("action", ""),
                            "bias_score":   item.get("bias_score"),
                            "regime":       item.get("regime", ""),
                            "stop_loss":    item.get("stop_loss"),
                            "profit_target":item.get("profit_target"),
                            "rr":           item.get("risk_reward_ratio"),
                            "sentiment":    sentiment,
                            "source":       "log",
                        })
            except json.JSONDecodeError:
                pass
            i = end + 1

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["bias_score"]  = pd.to_numeric(df["bias_score"], errors="coerce")
    df = df.drop_duplicates(subset=["signal_date", "ticker"], keep="last")
    print(f"  提取 {len(df)} 条信号（{df['ticker'].nunique()} 个标的）")
    return df


def load_from_backtest_csv(csv_paths: list[str]) -> pd.DataFrame:
    """从回测产生的 signals.csv 加载信号"""
    frames = []
    for path in csv_paths:
        p = Path(path)
        if not p.exists():
            print(f"  文件不存在: {path}")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        # 兼容不同列名
        col_map = {}
        for col in df.columns:
            lc = col.lower()
            if "date" in lc and "signal" not in lc and "entry" not in lc:
                col_map[col] = "signal_date"
            elif "bias" in lc:
                col_map[col] = "bias_score"
            elif "action" in lc:
                col_map[col] = "action"
            elif "regime" in lc:
                col_map[col] = "regime"
        df = df.rename(columns=col_map)
        # 推断 ticker（从文件路径）
        ticker = p.parent.name.replace("_portfolio_backtest", "").upper()
        if "ticker" not in df.columns:
            df["ticker"] = ticker
        df["source"] = "backtest_csv"
        frames.append(df)
        print(f"  加载 {len(df)} 条信号 from {path}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce")
    df["bias_score"]  = pd.to_numeric(df["bias_score"], errors="coerce")
    return df


def fetch_forward_returns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """获取每条信号后 N 个交易日的前向收益"""
    tickers_needed = df["ticker"].unique().tolist()
    min_date = df["signal_date"].min() - pd.Timedelta(days=5)
    max_date = df["signal_date"].max() + pd.Timedelta(days=max(horizons) * 2 + 10)

    price_cache = {}
    for asset in tickers_needed:
        yf_ticker = TICKER_MAP.get(asset, asset)
        print(f"  下载 {asset} ({yf_ticker}) {min_date.date()} ~ {max_date.date()}")
        try:
            session = _make_session()
            raw = yf.download(
                yf_ticker,
                start=min_date.strftime("%Y-%m-%d"),
                end=max_date.strftime("%Y-%m-%d"),
                interval="1d", auto_adjust=True, progress=False,
                session=session,
            )
            if raw.empty:
                print(f"    ⚠  {asset}: 无数据")
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            price_cache[asset] = raw["Close"].squeeze()
        except Exception as e:
            print(f"    ✗ {asset}: {e}")

    df = df.copy()
    df["price_0"] = np.nan
    for h in horizons:
        df[f"fwd_{h}d"] = np.nan
        df[f"ret_{h}d"] = np.nan

    for idx, row in df.iterrows():
        asset    = row["ticker"]
        sig_date = row["signal_date"]
        prices   = price_cache.get(asset)
        if prices is None:
            continue

        # 信号日当天或下一个交易日
        avail = prices[prices.index >= sig_date]
        if avail.empty:
            continue
        p0_date = avail.index[0]
        p0      = float(avail.iloc[0])
        df.at[idx, "price_0"] = p0

        for h in horizons:
            future = prices[prices.index > p0_date]
            if len(future) >= h:
                ph = float(future.iloc[h - 1])
                df.at[idx, f"fwd_{h}d"] = ph
                df.at[idx, f"ret_{h}d"] = (ph - p0) / p0

    return df


# ── IC 计算 ──────────────────────────────────────────────────────────────────

def compute_spearman_ic(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    """返回 (IC, t-stat, p-value)"""
    from scipy.stats import spearmanr
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 5:
        return np.nan, np.nan, np.nan
    ic, pval = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
    n      = len(valid)
    t_stat = ic * np.sqrt(n - 2) / np.sqrt(max(1 - ic**2, 1e-10))
    return ic, t_stat, pval


def run_ic_analysis(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    计算 bias_score 对 forward_return 的 IC
    分三组：全部信号 / 只有long / 排除no_trade
    """
    rows = []
    for h in horizons:
        ret_col = f"ret_{h}d"
        valid   = df.dropna(subset=["bias_score", ret_col])
        if len(valid) < 5:
            print(f"  horizon={h}d: 样本不足({len(valid)}条)，跳过")
            continue

        for group_name, subset in [
            ("全部信号",    valid),
            ("long信号",   valid[valid["action"] == "long"]),
            ("非no_trade", valid[valid["action"] != "no_trade"]),
        ]:
            if len(subset) < 5:
                continue
            ic, t_stat, pval = compute_spearman_ic(subset["bias_score"], subset[ret_col])
            if np.isnan(ic):
                continue

            if pval < 0.05:
                sig = "✅ 显著"
            elif pval < 0.10:
                sig = "⚠️ 边缘"
            else:
                sig = "❌ 不显著"

            rows.append({
                "horizon":   f"{h}d",
                "group":     group_name,
                "n":         len(subset),
                "IC":        round(ic, 4),
                "t_stat":    round(t_stat, 2),
                "p_value":   round(pval, 4),
                "sig":       sig,
                "mean_ret":  round(subset[ret_col].mean() * 100, 2),
                "pos_rate":  round((subset[ret_col] > 0).mean() * 100, 1),
            })
    return pd.DataFrame(rows)


def bias_bucket_analysis(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """long 信号按 bias 分档统计"""
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
                "horizon":     f"{h}d",
                "bias_bucket": str(bucket),
                "n":           len(grp),
                "mean_ret_%":  round(grp[ret_col].mean() * 100, 2),
                "win_rate_%":  round((grp[ret_col] > 0).mean() * 100, 1),
            })
    return pd.DataFrame(rows)


def action_analysis(df: pd.DataFrame, horizons: list[int]) -> None:
    """各动作（long / no_trade / short）的前向收益统计"""
    print(f"\n  [各动作前向收益]")
    print(f"  {'动作':<12} {'样本':>5}", end="")
    for h in horizons:
        print(f"  {h}d均收益".rjust(10), end="")
    print()

    for action in ["long", "no_trade", "short"]:
        sub = df[df["action"] == action]
        if sub.empty:
            continue
        print(f"  {action:<12} {len(sub):>5}", end="")
        for h in horizons:
            ret_col = f"ret_{h}d"
            v = sub[ret_col].dropna()
            if v.empty:
                print("       N/A", end="")
            else:
                print(f"  {v.mean()*100:>+7.2f}%", end="")
        print()


def regime_analysis(df: pd.DataFrame, horizons: list[int]) -> None:
    """制度（regime）× 前向收益"""
    regimes = df["regime"].dropna().unique().tolist()
    if not regimes:
        return
    print(f"\n  [制度×前向收益]  (long信号)")
    longs = df[df["action"] == "long"]
    if longs.empty:
        return

    print(f"  {'制度':<20} {'样本':>5}", end="")
    for h in horizons:
        print(f"  {h}d均收益".rjust(10), end="")
    print()

    for regime in sorted(regimes):
        sub = longs[longs["regime"] == regime]
        if sub.empty:
            continue
        print(f"  {regime:<20} {len(sub):>5}", end="")
        for h in horizons:
            ret_col = f"ret_{h}d"
            v = sub[ret_col].dropna()
            if v.empty:
                print("       N/A", end="")
            else:
                print(f"  {v.mean()*100:>+7.2f}%", end="")
        print()


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM信号IC分析")
    parser.add_argument("--log-dir",     default="logs",
                        help="日志目录（服务器: /opt/finance-analysis/logs）")
    parser.add_argument("--signals-csv", nargs="+",
                        help="回测产生的 signals.csv（可多个）")
    parser.add_argument("--horizons",    nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--tickers",     nargs="+")
    parser.add_argument("--out-csv",     default="ic_analysis_results.csv")
    args = parser.parse_args()

    print("=" * 60)
    print("  验证1：LLM信号 IC 分析（信息系数）")
    print("=" * 60)

    # ── 数据加载 ─────────────────────────────────────────────────────────────
    df_parts = []

    # 来源A：日志文件
    log_dir = Path(args.log_dir)
    if log_dir.exists():
        print(f"\n步骤1：加载日志文件 ({log_dir})")
        df_log = load_from_logs(log_dir, tickers=args.tickers)
        if not df_log.empty:
            df_parts.append(df_log)
    else:
        print(f"\n  日志目录不存在: {log_dir}（服务器路径: /opt/finance-analysis/logs）")

    # 来源B：回测 signals.csv
    if args.signals_csv:
        print(f"\n步骤1b：加载回测信号 CSV")
        df_bt = load_from_backtest_csv(args.signals_csv)
        if not df_bt.empty:
            df_parts.append(df_bt)

    if not df_parts:
        print("\n❌ 无可用数据。")
        print("\n获取数据的两种方式：")
        print("  A. 服务器日志（推荐）：")
        print("     scp -r root@101.201.171.174:/opt/finance-analysis/logs ./logs_server")
        print("     python3 validate_ic.py --log-dir logs_server")
        print("\n  B. 本地回测（需先在服务器跑完整回测）：")
        print("     python3 validate_ic.py --signals-csv nvda_portfolio_backtest/signals.csv")
        sys.exit(0)

    df = pd.concat(df_parts, ignore_index=True)
    df = df.drop_duplicates(subset=["signal_date", "ticker"], keep="last")
    df = df.sort_values(["ticker", "signal_date"]).reset_index(drop=True)

    print(f"\n  合并后：{len(df)} 条唯一信号  |  {df['ticker'].nunique()} 个标的")
    print(f"  日期范围：{df['signal_date'].min().date()} ~ {df['signal_date'].max().date()}")

    # 样本量预警
    if len(df) < 30:
        print(f"\n  ⚠️  当前样本量 {len(df)} < 30，以下IC结论不具统计显著性！")
        print("     需要累积至少 30 条信号才能得出可靠结论")
        print("     建议：从服务器拉取 logs 目录以获得更多历史数据")

    # ── 获取前向收益 ──────────────────────────────────────────────────────────
    print(f"\n步骤2：获取前向收益数据")
    df = fetch_forward_returns(df, args.horizons)

    # ── IC 分析 ───────────────────────────────────────────────────────────────
    print(f"\n步骤3：IC分析结果")
    print("=" * 60)

    # 检查 scipy
    try:
        import scipy
    except ImportError:
        print("❌ 缺少 scipy，请运行：pip install scipy")
        sys.exit(1)

    ic_df = run_ic_analysis(df, args.horizons)
    if ic_df.empty:
        print("  样本不足，无法计算 IC")
    else:
        print(f"\n  [IC 汇总表]")
        print(f"  {'周期':<8} {'分组':<12} {'样本':>5} {'IC':>7} {'t-stat':>7} "
              f"{'p值':>7} {'均收益%':>8} {'胜率%':>7} {'显著性':<10}")
        print(f"  {'─'*75}")
        for _, row in ic_df.iterrows():
            print(f"  {row['horizon']:<8} {row['group']:<12} {row['n']:>5} "
                  f"{row['IC']:>+6.4f} {row['t_stat']:>7.2f} "
                  f"{row['p_value']:>7.4f} {row['mean_ret']:>+7.2f}% "
                  f"{row['pos_rate']:>6.1f}% {row['sig']}")

        ic_df.to_csv(args.out_csv, index=False)
        print(f"\n  IC汇总已保存 → {args.out_csv}")

    # ── Bias 分档分析 ─────────────────────────────────────────────────────────
    print(f"\n  [Bias分档收益统计] (long信号)")
    bucket_df = bias_bucket_analysis(df, args.horizons)
    if bucket_df.empty:
        print("  无足够的 long 信号")
    else:
        print(f"  {'horizon':<8} {'bias区间':<12} {'样本':>5} "
              f"{'均收益%':>9} {'胜率%':>7}")
        for _, row in bucket_df.iterrows():
            flag = "✅" if row["mean_ret_%"] > 2 else ("⚠️" if row["mean_ret_%"] > 0 else "❌")
            print(f"  {row['horizon']:<8} {row['bias_bucket']:<12} {row['n']:>5} "
                  f"  {row['mean_ret_%']:>+7.2f}%  {row['win_rate_%']:>6.1f}%  {flag}")

    # ── 动作 & 制度分析 ──────────────────────────────────────────────────────
    action_analysis(df, args.horizons)
    regime_analysis(df, args.horizons)

    # ── 信号分布摘要 ──────────────────────────────────────────────────────────
    print(f"\n  [信号分布摘要]")
    print(f"  按 action 分布:")
    print(df["action"].value_counts().to_string())
    print(f"\n  按 bias_score 分布:")
    print(df["bias_score"].describe().round(3).to_string())

    # ── 最终结论 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  结论判断")
    print('='*60)

    if not ic_df.empty:
        # 取 21d 全部信号的 t-stat
        row_21 = ic_df[(ic_df["horizon"] == "21d") & (ic_df["group"] == "全部信号")]
        if not row_21.empty:
            t = row_21.iloc[0]["t_stat"]
            ic = row_21.iloc[0]["IC"]
            n  = row_21.iloc[0]["n"]
            if abs(t) >= 2.0:
                print(f"  ✅ LLM信号有效：IC={ic:.4f}, t-stat={t:.2f} ≥ 2.0")
                print(f"     → 继续优化LLM参数，信号具备预测价值")
            elif abs(t) >= 1.5:
                print(f"  ⚠️  LLM信号边缘有效：IC={ic:.4f}, t-stat={t:.2f}")
                print(f"     → 需积累更多样本（当前{n}条，推荐60+条）")
            else:
                print(f"  ❌ LLM信号无效：IC={ic:.4f}, t-stat={t:.2f} < 1.5")
                print(f"     → 建议：停止优化LLM参数，专注Beta Overlay制度过滤")
        if len(df) < 30:
            print(f"\n  ⚠️  注意：当前样本量{len(df)}条不足以做最终结论")
            print(f"     服务器日志获取方式：")
            print(f"     scp -r root@101.201.171.174:/opt/finance-analysis/logs ./logs_server")
            print(f"     python3 validate_ic.py --log-dir logs_server")
    else:
        print("  数据不足，无法输出结论")


if __name__ == "__main__":
    main()
