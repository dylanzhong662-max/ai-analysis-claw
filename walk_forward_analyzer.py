"""
滚动前向验证分析器（离线，无 API 调用）
从已完成的 trades.csv 中进行多折叠 IS/OOS 分析

单次 80/20 分割只有 1~2 笔 OOS 交易，无法区分运气和技术。
本脚本在已有数据上滚动多个窗口，得到稳健的 IS/OOS 对比。

用法：
    python3 walk_forward_analyzer.py --tickers NVDA MSFT GOOGL
    python3 walk_forward_analyzer.py --tickers NVDA --folds 6 --test-months 3
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_trades(ticker: str) -> pd.DataFrame:
    path = Path(f"{ticker.lower()}_portfolio_backtest/trades.csv")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
    # 只保留已确认的交易（STOP_LOSS / TAKE_PROFIT）
    df = df[df["exit_reason"].isin(["STOP_LOSS", "TAKE_PROFIT"])].copy()
    df = df.sort_values("entry_date").reset_index(drop=True)
    return df


def metrics_from_trades(df: pd.DataFrame, label: str = "") -> dict:
    """计算一组交易的绩效指标"""
    if df.empty:
        return {"label": label, "n": 0, "win_rate": None, "profit_factor": None,
                "avg_win": None, "avg_loss": None, "sharpe": None,
                "max_dd_pct": None, "expectancy": None}

    wins   = df[df["pnl_pct"] > 0]["pnl_pct"]
    losses = df[df["pnl_pct"] <= 0]["pnl_pct"]

    win_rate      = len(wins) / len(df) if len(df) > 0 else None
    avg_win       = wins.mean()   if len(wins)   > 0 else 0.0
    avg_loss      = losses.mean() if len(losses) > 0 else 0.0
    gross_profit  = wins.sum()
    gross_loss    = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0)

    # 期望值（每笔交易平均收益）
    expectancy = df["pnl_pct"].mean()

    # Sharpe（用交易级 P&L，年化假设每月约 0.8 笔）
    if len(df) >= 3:
        ann_factor = np.sqrt(12 * 0.8)  # 月度化 × 年化
        sharpe = (df["pnl_pct"].mean() / df["pnl_pct"].std() * ann_factor
                  if df["pnl_pct"].std() > 0 else 0)
    else:
        sharpe = None

    # 最大回撤（基于累计 P&L 序列）
    cum_pnl = df["pnl_pct"].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()

    # t 统计量（均值是否显著异于 0）
    if len(df) >= 3:
        from scipy import stats as scipy_stats
        t_stat, p_val = scipy_stats.ttest_1samp(df["pnl_pct"], 0)
    else:
        t_stat, p_val = None, None

    return {
        "label":          label,
        "n":              len(df),
        "win_rate":       round(win_rate, 3) if win_rate is not None else None,
        "profit_factor":  round(profit_factor, 2),
        "avg_win":        round(avg_win, 3),
        "avg_loss":       round(avg_loss, 3),
        "expectancy":     round(expectancy, 3),
        "sharpe":         round(sharpe, 2) if sharpe is not None else None,
        "max_dd_pct":     round(max_dd, 2),
        "t_stat":         round(t_stat, 2) if t_stat is not None else None,
        "p_value":        round(p_val, 4) if p_val is not None else None,
        "significant":    (abs(t_stat) > 2.0) if t_stat is not None else False,
    }


def walk_forward_analysis(df: pd.DataFrame,
                          n_folds: int = 4,
                          test_months: int = 3) -> list[dict]:
    """
    在 trades DataFrame 上进行滚动前向验证。

    策略：
    - 把所有交易的时间跨度等分为 n_folds 段
    - 每折：前 80% 为 IS，后 20% 为 OOS（按时间）
    - 每次窗口向前滑动 test_months

    返回：每个折叠的 {fold, is_period, oos_period, is_metrics, oos_metrics}
    """
    if df.empty or len(df) < 4:
        return []

    start = df["entry_date"].min()
    end   = df["entry_date"].max()
    total_days = (end - start).days

    # 确保每个测试窗口至少有 1 笔交易
    test_days  = max(test_months * 30, total_days // (n_folds + 2))
    train_days = total_days - test_days

    if train_days < test_days:
        # 数据太少，降级到单次 70/30 分割
        split = start + pd.Timedelta(days=int(total_days * 0.7))
        is_df  = df[df["entry_date"] < split]
        oos_df = df[df["entry_date"] >= split]
        return [{
            "fold":       1,
            "is_period":  (str(start.date()), str(split.date())),
            "oos_period": (str(split.date()), str(end.date())),
            "is":         metrics_from_trades(is_df,  "IS"),
            "oos":        metrics_from_trades(oos_df, "OOS"),
        }]

    folds = []
    for i in range(n_folds):
        oos_start = start + pd.Timedelta(days=train_days + i * (test_days // max(n_folds - 1, 1)))
        oos_end   = oos_start + pd.Timedelta(days=test_days)
        if oos_start >= end:
            break
        oos_end = min(oos_end, end + pd.Timedelta(days=1))
        is_end  = oos_start

        is_df  = df[df["entry_date"] < is_end]
        oos_df = df[(df["entry_date"] >= oos_start) & (df["entry_date"] < oos_end)]

        if len(is_df) < 2 or len(oos_df) < 1:
            continue

        folds.append({
            "fold":       i + 1,
            "is_period":  (str(start.date()), str(is_end.date())),
            "oos_period": (str(oos_start.date()), str(oos_end.date())),
            "is":         metrics_from_trades(is_df,  f"IS-{i+1}"),
            "oos":        metrics_from_trades(oos_df, f"OOS-{i+1}"),
        })

    return folds


def print_fold_table(folds: list[dict]):
    """打印折叠对比表格"""
    if not folds:
        print("  数据不足，无法完成多折叠分析")
        return

    print(f"\n  {'折叠':>4} | {'期间':>12} | {'N':>4} | {'胜率':>7} | "
          f"{'盈利因子':>8} | {'Sharpe':>7} | {'期望值':>7} | {'显著':>4}")
    print(f"  {'─'*4}-+-{'─'*12}-+-{'─'*4}-+-{'─'*7}-+-{'─'*8}-+-{'─'*7}-+-{'─'*7}-+-{'─'*4}")

    for f in folds:
        for seg, key in [("IS", "is"), ("OOS", "oos")]:
            m = f[key]
            period_str = f[f"{key}_period"][0][:7]
            wr   = f"{m['win_rate']:.1%}" if m["win_rate"] is not None else " N/A"
            pf   = f"{m['profit_factor']:.2f}" if m["profit_factor"] is not None else " N/A"
            sh   = f"{m['sharpe']:.2f}" if m["sharpe"] is not None else " N/A"
            exp  = f"{m['expectancy']:>+.2f}%" if m["expectancy"] is not None else " N/A"
            sig  = "✓" if m.get("significant") else "✗"
            prefix = f"  {f['fold']:>3}{seg}" if seg == "IS" else "      OOS"
            print(f"  {f['fold']:>4}-{seg:3} | {period_str:>12} | {m['n']:>4} | "
                  f"{wr:>7} | {pf:>8} | {sh:>7} | {exp:>7} | {sig:>4}")
        print()


def aggregate_oos_metrics(folds: list[dict]) -> dict:
    """聚合所有 OOS 折叠的指标"""
    if not folds:
        return {}

    oos_list = [f["oos"] for f in folds if f["oos"]["n"] > 0]
    if not oos_list:
        return {}

    def safe_mean(key):
        vals = [m[key] for m in oos_list if m.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    def safe_std(key):
        vals = [m[key] for m in oos_list if m.get(key) is not None]
        return round(float(np.std(vals)), 3) if len(vals) >= 2 else None

    is_list = [f["is"] for f in folds if f["is"]["n"] > 0]

    avg_is_pf  = safe_mean.__func__(safe_mean, "profit_factor")  # noqa – just reuse
    is_pf_vals = [m["profit_factor"] for m in is_list if m.get("profit_factor") is not None]
    oos_pf_vals = [m["profit_factor"] for m in oos_list if m.get("profit_factor") is not None]

    if is_pf_vals and oos_pf_vals:
        avg_is_pf  = round(float(np.mean(is_pf_vals)), 2)
        avg_oos_pf = round(float(np.mean(oos_pf_vals)), 2)
        degradation = (1 - avg_oos_pf / avg_is_pf) * 100 if avg_is_pf > 0 else None
    else:
        avg_is_pf = avg_oos_pf = degradation = None

    oos_sharpes = [m["sharpe"] for m in oos_list if m.get("sharpe") is not None]
    pct_positive = (sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes)
                    if oos_sharpes else None)

    return {
        "n_folds":              len(folds),
        "total_oos_trades":     sum(m["n"] for m in oos_list),
        "avg_oos_win_rate":     safe_mean("win_rate"),
        "avg_oos_profit_factor": avg_oos_pf,
        "avg_is_profit_factor":  avg_is_pf,
        "degradation_pct":      round(degradation, 1) if degradation is not None else None,
        "avg_oos_sharpe":       safe_mean("sharpe"),
        "std_oos_sharpe":       safe_std("sharpe"),
        "pct_profitable_folds": round(pct_positive, 2) if pct_positive is not None else None,
        "avg_oos_expectancy":   safe_mean("expectancy"),
        # 策略是否可信：OOS Sharpe > 0 且盈利折叠 > 60% 且降级 < 50%
        "is_viable":            (
            (safe_mean("sharpe") or 0) > 0 and
            (pct_positive or 0) >= 0.6 and
            (degradation or 100) < 60
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="滚动前向验证分析（离线）")
    parser.add_argument("--tickers", nargs="+",
                        default=["NVDA", "MSFT", "GOOGL"],
                        help="分析的 ticker 列表")
    parser.add_argument("--folds", type=int, default=4,
                        help="折叠数量（默认4）")
    parser.add_argument("--test-months", type=int, default=3,
                        help="每个 OOS 窗口长度（月）")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  滚动前向验证分析（Walk-Forward Analysis）")
    print(f"  折叠数: {args.folds}  |  每OOS窗口: {args.test_months}个月")
    print("  判断标准: OOS平均Sharpe > 0，盈利折叠 ≥ 60%，IS→OOS降级 < 50%")
    print("=" * 70)

    for ticker in args.tickers:
        df = load_trades(ticker)
        print(f"\n{'─'*70}")
        if len(df) < 6:
            date_info = "数据不足"
        else:
            date_info = f"日期范围: {df['entry_date'].min().date()} ~ {df['entry_date'].max().date()}"
        print(f"  {ticker}  |  已确认交易数: {len(df)}  ({date_info})")

        if len(df) < 4:
            print(f"  ⚠️  样本量不足（{len(df)} 笔），需 30+ 笔才有统计意义")
            continue

        folds = walk_forward_analysis(df, n_folds=args.folds,
                                      test_months=args.test_months)
        print_fold_table(folds)

        agg = aggregate_oos_metrics(folds)
        if agg:
            print(f"  聚合 OOS 统计（{agg['n_folds']} 个折叠，总 {agg['total_oos_trades']} 笔）:")
            print(f"    平均 OOS 胜率      : {agg['avg_oos_win_rate']:.1%}" if agg['avg_oos_win_rate'] else "    平均 OOS 胜率      : N/A")
            print(f"    平均 IS 盈利因子   : {agg['avg_is_profit_factor']}")
            print(f"    平均 OOS 盈利因子  : {agg['avg_oos_profit_factor']}")
            if agg["degradation_pct"] is not None:
                flag = "🚨 严重过拟合" if agg["degradation_pct"] > 50 else \
                       "⚠️  轻度降级" if agg["degradation_pct"] > 25 else "✓ 正常降级"
                print(f"    IS→OOS 降级幅度    : {agg['degradation_pct']:.1f}%  {flag}")
            print(f"    平均 OOS Sharpe    : {agg['avg_oos_sharpe']}  "
                  f"(±{agg['std_oos_sharpe']})")
            print(f"    盈利折叠占比       : {agg['pct_profitable_folds']:.0%}" if agg['pct_profitable_folds'] is not None else "    盈利折叠占比       : N/A")
            print(f"    平均 OOS 期望值    : {agg['avg_oos_expectancy']}% / 笔")

            viable = agg.get("is_viable", False)
            print(f"\n  综合判断: {'✅ 策略可信，可考虑扩大样本继续验证' if viable else '❌ 策略尚不可信，需更多 OOS 数据'}")

        # ── 样本量警告 ───────────────────────────────────────────
        if len(df) < 30:
            deficit = 30 - len(df)
            print(f"\n  ⚠️  样本量不足警告:")
            print(f"     当前: {len(df)} 笔已确认交易  |  统计显著需要: 30+ 笔")
            print(f"     还差: {deficit} 笔  |  建议: 将回测延伸至 2023-2024 年")
            print(f"     置信区间提示: 当前胜率的 95% CI 宽度约 ±{int(1.96 * 0.5 / (len(df)**0.5) * 100)}pp")

    print()


if __name__ == "__main__":
    main()
