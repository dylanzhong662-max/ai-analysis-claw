"""
信号信息系数（IC）分析脚本
验证 LLM bias_score 是否对未来收益具有真实预测价值

核心问题：bias_score 与未来 N 天收益的 Spearman 相关系数是多少？
  IC > 0.02 且 t > 2 → 有效信号
  IC 0.01~0.02       → 弱信号，需更多数据
  IC < 0.01          → 无信号，所有参数调整都是拟合噪音

用法：
    python3 signal_ic_analysis.py --tickers NVDA MSFT GOOGL
    python3 signal_ic_analysis.py --tickers NVDA --horizons 10 20 30 60
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_CACHE_DIR = Path("data_cache")


def _load_price(ticker: str, interval: str = "1d") -> pd.DataFrame:
    safe = ticker.replace("^", "").replace("/", "_").replace("=", "")
    p = _CACHE_DIR / f"{safe}_{interval}.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    return pd.DataFrame()


def _nearest_date(index: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp | None:
    future = index[index >= target]
    return future[0] if len(future) > 0 else None


def compute_ic(signals_df: pd.DataFrame, prices: pd.DataFrame,
               horizons: list[int]) -> dict:
    """
    计算不同时间窗口下的 IC（信息系数）。

    参数
    ----
    signals_df : signals.csv 内容，含 date/action/bias_score/regime 列
    prices     : 1d OHLCV 数据
    horizons   : 前向收益计算天数列表，如 [10, 20, 30]

    返回
    ----
    dict: {horizon: {ic, t_stat, p_value, hit_rate, monthly_ir, ...}}
    """
    if prices.empty or "Close" not in prices.columns:
        return {}

    close = prices["Close"].squeeze().dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)

    results = {}
    for horizon in horizons:
        # N 日前向收益（用 shift(-horizon) 避免未来泄漏）
        fwd_ret = close.pct_change(horizon).shift(-horizon)

        # 只取有方向的信号（long / short）
        directional = signals_df[
            signals_df["action"].isin(["long", "short"])
        ].copy()
        if len(directional) < 5:
            continue

        rows = []
        for _, row in directional.iterrows():
            t = _nearest_date(fwd_ret.index, pd.Timestamp(row["date"]))
            if t is None:
                continue
            fwd = fwd_ret.get(t)
            if fwd is None or pd.isna(fwd):
                continue

            bias = float(row["bias_score"] or 0)
            # 做空时方向相反：价格下跌 = 盈利
            signed_fwd = fwd if row["action"] == "long" else -fwd

            rows.append({
                "date":          t,
                "bias_score":    bias,
                "forward_return": signed_fwd,
                "action":        row["action"],
                "regime":        row.get("regime", ""),
            })

        if len(rows) < 5:
            continue

        df = pd.DataFrame(rows)

        # ── Spearman IC ──────────────────────────────────────────────
        ic, p_val = stats.spearmanr(df["bias_score"], df["forward_return"])

        # t 统计量（Fischer 变换近似）
        n = len(df)
        ic_safe = min(max(ic, -0.9999), 0.9999)
        t_stat = ic_safe * np.sqrt(n - 2) / np.sqrt(1 - ic_safe ** 2)

        # ── 方向准确率（所有信号，不按 bias 过滤）──────────────────
        hit_rate = (df["forward_return"] > 0).mean()

        # ── 月度 IC 分解 → Information Ratio ────────────────────────
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
        monthly_ic = df.groupby("month").apply(
            lambda x: (stats.spearmanr(x["bias_score"], x["forward_return"])[0]
                       if len(x) >= 3 else np.nan)
        ).dropna()

        ir = (monthly_ic.mean() / monthly_ic.std()
              if len(monthly_ic) >= 2 and monthly_ic.std() > 0 else 0)
        pct_pos = (monthly_ic > 0).mean() if len(monthly_ic) > 0 else None

        # ── 按制度分层的 IC ─────────────────────────────────────────
        regime_ic = {}
        for reg, grp in df.groupby("regime"):
            if len(grp) >= 4:
                r_ic, r_p = stats.spearmanr(grp["bias_score"], grp["forward_return"])
                regime_ic[reg] = round(r_ic, 4)

        results[horizon] = {
            "ic":                   round(ic, 4),
            "t_stat":               round(t_stat, 2),
            "p_value":              round(float(p_val), 4),
            "n":                    n,
            "hit_rate":             round(float(hit_rate), 3),
            "monthly_ir":           round(float(ir), 2),
            "pct_positive_months":  round(float(pct_pos), 3) if pct_pos is not None else None,
            "regime_ic":            regime_ic,
            "significant":          abs(t_stat) > 2.0 and abs(ic) > 0.02,
        }

    return results


def analyze_filter_quality(signals_df: pd.DataFrame, prices: pd.DataFrame,
                            horizon: int = 20) -> dict:
    """
    分析 no_trade 过滤的质量：
    观望期间市场平均收益 vs 入场期间市场平均收益。
    若入场期间收益 > 观望期间，说明过滤有选择性。
    """
    if prices.empty or "Close" not in prices.columns:
        return {}

    close = prices["Close"].squeeze().dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    fwd_ret = close.pct_change(horizon).shift(-horizon)

    def avg_fwd(subset):
        vals = []
        for _, row in subset.iterrows():
            t = _nearest_date(fwd_ret.index, pd.Timestamp(row["date"]))
            if t is None:
                continue
            v = fwd_ret.get(t)
            if v is not None and not pd.isna(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    no_trade = signals_df[signals_df["action"] == "no_trade"]
    traded   = signals_df[signals_df["action"].isin(["long", "short"])]

    nt_fwd = avg_fwd(no_trade)
    tr_fwd = avg_fwd(traded)

    # 用 t-test 检验两组的均值差异是否显著
    nt_vals, tr_vals = [], []
    for df_sub, container in [(no_trade, nt_vals), (traded, tr_vals)]:
        for _, row in df_sub.iterrows():
            t = _nearest_date(fwd_ret.index, pd.Timestamp(row["date"]))
            if t is None:
                continue
            v = fwd_ret.get(t)
            if v is not None and not pd.isna(v):
                container.append(v)

    t_stat, p_val = (stats.ttest_ind(tr_vals, nt_vals)
                     if len(tr_vals) >= 3 and len(nt_vals) >= 3
                     else (None, None))

    return {
        "no_trade_avg_fwd":  round(nt_fwd, 4) if nt_fwd else None,
        "traded_avg_fwd":    round(tr_fwd, 4) if tr_fwd else None,
        "filter_adds_value": tr_fwd > nt_fwd if (tr_fwd and nt_fwd) else None,
        "t_stat_diff":       round(t_stat, 2) if t_stat else None,
        "p_value_diff":      round(p_val, 4) if p_val else None,
        "n_no_trade":        len(no_trade),
        "n_traded":          len(traded),
    }


def main():
    parser = argparse.ArgumentParser(description="LLM 信号 IC 分析（离线，无 API 调用）")
    parser.add_argument("--tickers", nargs="+",
                        default=["NVDA", "MSFT", "GOOGL"],
                        help="分析的 ticker 列表")
    parser.add_argument("--horizons", nargs="+", type=int,
                        default=[10, 20, 30, 60],
                        help="前向收益窗口天数")
    parser.add_argument("--signals-dir", type=str, default=None,
                        help="自定义 signals.csv 所在目录（覆盖默认的 {ticker}_portfolio_backtest/）")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  LLM 交易信号  信息系数（IC）分析")
    print("  判断 bias_score 是否真的预测未来收益，还是随机噪音")
    print("=" * 70)

    all_results = {}

    for ticker in args.tickers:
        if args.signals_dir:
            signals_path = Path(args.signals_dir) / "signals.csv"
        else:
            signals_path = Path(f"{ticker.lower()}_portfolio_backtest/signals.csv")

        if not signals_path.exists():
            print(f"\n[{ticker}] ✗ 未找到 {signals_path}，跳过")
            continue

        signals_df = pd.read_csv(signals_path)
        # 去掉 blocked_death_cross 等非标准 action
        signals_df = signals_df[signals_df["action"].isin(
            ["long", "short", "no_trade"])]

        prices = _load_price(ticker, "1d")
        if prices.empty:
            print(f"\n[{ticker}] ✗ 价格缓存为空，跳过（请先运行 prefetch_data.py）")
            continue

        n_dir = len(signals_df[signals_df["action"].isin(["long", "short"])])
        n_nt  = len(signals_df[signals_df["action"] == "no_trade"])

        print(f"\n{'─'*70}")
        print(f"  {ticker}  |  总信号: {len(signals_df)}  "
              f"(方向信号: {n_dir}, 观望: {n_nt})")
        print(f"{'─'*70}")

        if n_dir < 5:
            print(f"  ⚠️  方向信号仅 {n_dir} 笔，样本量不足，结果不可靠")

        ic_results = compute_ic(signals_df, prices, horizons=args.horizons)

        if not ic_results:
            print("  样本量不足，无法计算 IC")
            continue

        # ── IC 表格 ────────────────────────────────────────────────
        print(f"\n  IC 摘要（IC 是 bias_score 与未来收益的 Spearman 相关系数）")
        print(f"  {'窗口':>5} | {'IC':>7} | {'t统计量':>8} | {'p值':>6} | "
              f"{'方向准确率':>8} | {'月度IR':>7} | {'结论':>14}")
        print(f"  {'─'*5}-+-{'─'*7}-+-{'─'*8}-+-{'─'*6}-+-{'─'*8}-+-{'─'*7}-+-{'─'*14}")

        for h, r in ic_results.items():
            if r["significant"] and r["ic"] > 0.02:
                verdict = "✅ 有效信号"
            elif abs(r["ic"]) > 0.01:
                verdict = "⚠️  弱信号"
            else:
                verdict = "❌ 无信号/噪音"
            print(f"  {h:>4}天 | {r['ic']:>+7.4f} | {r['t_stat']:>8.2f} | "
                  f"{r['p_value']:>6.4f} | {r['hit_rate']:>7.1%}  | "
                  f"{r['monthly_ir']:>7.2f} | {verdict}")

        # ── 按制度分层的 IC ────────────────────────────────────────
        best_horizon = max(ic_results, key=lambda h: abs(ic_results[h]["ic"]))
        regime_ic = ic_results[best_horizon].get("regime_ic", {})
        if regime_ic:
            print(f"\n  制度分层 IC（{best_horizon}日窗口）：")
            for reg, ric in sorted(regime_ic.items(), key=lambda x: -abs(x[1])):
                bar = "█" * int(abs(ric) * 50)
                print(f"    {reg:<20} IC={ric:>+6.4f}  {bar}")

        # ── no_trade 过滤质量 ──────────────────────────────────────
        fq = analyze_filter_quality(signals_df, prices, horizon=20)
        print(f"\n  no_trade 过滤价值分析（20日前向收益）：")
        if fq.get("no_trade_avg_fwd") is not None:
            nt_pct = fq["no_trade_avg_fwd"] * 100
            tr_pct = fq["traded_avg_fwd"] * 100 if fq["traded_avg_fwd"] else 0
            adds   = "✅ 有效（入场期收益更高）" if fq.get("filter_adds_value") else "❌ 无效（应该更多入场）"
            print(f"    观望期间市场均涨跌: {nt_pct:>+6.2f}%  (N={fq['n_no_trade']})")
            print(f"    入场期间市场均涨跌: {tr_pct:>+6.2f}%  (N={fq['n_traded']})")
            if fq.get("t_stat_diff") is not None:
                print(f"    差异显著性: t={fq['t_stat_diff']:.2f}, p={fq['p_value_diff']:.4f}")
            print(f"    过滤效果: {adds}")

        all_results[ticker] = ic_results

    # ── 最终汇总 ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  最终结论")
    print("=" * 70)
    print(f"  IC 解读标准（来自 Two Sigma / DE Shaw 内部准则）:")
    print(f"    IC > 0.02 且 t > 2  → 信号有效，LLM bias_score 有预测价值")
    print(f"    IC 0.01~0.02        → 微弱信号，需更多数据（≥50笔）确认")
    print(f"    IC < 0.01 或 t < 2  → 无显著信号，优化参数是在拟合噪音")
    print()

    has_any_signal = False
    for ticker, results in all_results.items():
        if not results:
            print(f"  {ticker}: 样本不足，无法判断")
            continue
        best_h = max(results, key=lambda h: abs(results[h]["ic"]))
        r = results[best_h]
        if r["significant"] and r["ic"] > 0.02:
            verdict = "✅ LLM 信号有预测价值"
            has_any_signal = True
        elif abs(r["ic"]) > 0.01:
            verdict = "⚠️  信号微弱，样本量不足以确认"
        else:
            verdict = "❌ 无显著信号 — 考虑改用规则策略"
        print(f"  {ticker}: 最佳IC({best_h}天窗口)={r['ic']:>+.4f}, "
              f"t={r['t_stat']:.2f} → {verdict}")

    if not has_any_signal:
        print("\n  ⚠️  警告: 未发现显著 IC。")
        print("  建议: 运行 baseline_strategy.py 检验纯规则策略是否能达到")
        print("  相同或更高 Sharpe，若能，LLM API 费用可能没有必要。")
    print()


if __name__ == "__main__":
    main()
