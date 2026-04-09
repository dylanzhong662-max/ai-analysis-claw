"""
backtest_beta_overlay.py — 验证3：Beta Overlay 策略历史回测

策略逻辑（与 portfolio_tracker.py --beta-overlay 完全一致）：
  Step 1  制度过滤
            价格 < 周线EMA200  → 目标仓位 0%（熊市清仓）
            价格 < 周线EMA50 或 距52周高点回撤>15% → 目标仓位 0%（快速出场）
  Step 2  波动率目标制定仓
            target_pct = min(TARGET_VOL / realized_vol_20w, MAX_PCT)
            TARGET_VOL = 16%，MAX_PCT = 80%
  Step 3  无LLM叠加（历史无LLM信号，纯规则基准）

回测细节：
  - 每周末再平衡（周五收盘）
  - 再平衡触发条件：仓位偏差 > REBAL_THRESHOLD（5%）
  - 交易成本：0.1% 佣金 + 0.1% 滑点 = 0.2%/边（往返0.4%）
  - 初始资金：$100,000
  - 已实现波动率：过去20周收益率的年化标准差

运行：
  python3 backtest_beta_overlay.py
  python3 backtest_beta_overlay.py --tickers NVDA MSFT GOOGL --start 2019-01-01
  python3 backtest_beta_overlay.py --target-vol 0.12  # 更保守的波动率目标
"""

import argparse
import os
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

# ── 策略参数（与 portfolio_tracker.py 保持一致）────────────────────────────────
TARGET_ANNUAL_VOL  = 0.16     # 目标年化波动率
MAX_POSITION_PCT   = 0.80     # 最大仓位上限
VOL_FLOOR_PCT      = 0.30     # 动量修正下限：牛市制度（价格>EMA200且>EMA50）时
                              # 仓位不低于30%，防止高波动牛市严重踏空
DRAWDOWN_EXIT_PCT  = 0.15     # 快速出场回撤阈值
REBAL_THRESHOLD    = 0.05     # 触发再平衡的最小偏差
COMMISSION         = 0.001    # 单边佣金
SLIPPAGE           = 0.001    # 单边滑点
VOL_WINDOW         = 20       # 已实现波动率计算窗口（周）
EMA200_PERIOD      = 200
EMA50_PERIOD       = 50
INITIAL_CAPITAL    = 100_000

DEFAULT_TICKERS = ["NVDA", "MSFT", "GOOGL"]
DEFAULT_START   = "2019-01-01"
DEFAULT_END     = "2026-01-01"


# ── 工具函数 ────────────────────────────────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def download_weekly(ticker: str, start: str, end: str) -> pd.DataFrame:
    # 多下载4年用于EMA200预热（EMA200需要至少200周历史）
    pre_start = (pd.Timestamp(start) - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    session = _make_session()
    df = yf.download(ticker, start=pre_start, end=end,
                     interval="1wk", auto_adjust=True, progress=False,
                     session=session)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def realized_vol_annual(returns: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """过去 window 周收益率的年化标准差"""
    return returns.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(52)


# ── 单资产回测 ───────────────────────────────────────────────────────────────

def backtest_single(ticker: str, start: str, end: str,
                    target_vol: float = TARGET_ANNUAL_VOL) -> dict:
    print(f"\n  [{ticker}] 下载周线数据...", end=" ")
    df = download_weekly(ticker, start, end)
    print(f"{len(df)} 周")

    close  = df["Close"].squeeze()
    weekly_ret = close.pct_change()

    ema200  = calc_ema(close, EMA200_PERIOD)
    ema50   = calc_ema(close, EMA50_PERIOD)
    high_52w = close.rolling(52, min_periods=20).max()
    drawdown = (high_52w - close) / high_52w
    rvol     = realized_vol_annual(weekly_ret, VOL_WINDOW)

    # 过滤到目标区间
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    idx  = df.index[mask]
    if len(idx) < 10:
        print(f"  [{ticker}] 有效数据不足，跳过")
        return {}

    # ── 模拟 ────────────────────────────────────────────────────────────────
    cash           = float(INITIAL_CAPITAL)
    shares         = 0.0
    trade_log      = []
    equity_curve   = []
    rebal_count    = 0
    cost_total     = 0.0

    for date in idx:
        price  = float(close.loc[date])
        e200   = float(ema200.loc[date])
        e50    = float(ema50.loc[date])
        dd     = float(drawdown.loc[date]) if not np.isnan(drawdown.loc[date]) else 0.0
        rv     = float(rvol.loc[date])     if not np.isnan(rvol.loc[date])     else 0.20

        # ── 制度判断 ──────────────────────────────────────────────────────────
        bear_regime = price <= e200
        fast_exit   = (not bear_regime) and (price <= e50 or dd > DRAWDOWN_EXIT_PCT)

        if bear_regime or fast_exit:
            target_pct = 0.0
            reason = "BEAR" if bear_regime else "FAST_EXIT"
        else:
            # 波动率目标制
            vol_target = min(target_vol / rv if rv > 0 else MAX_POSITION_PCT, MAX_POSITION_PCT)
            # 动量修正下限：牛市趋势中（价格>EMA200且>EMA50），仓位不低于 VOL_FLOOR_PCT
            if price > e50 and vol_target < VOL_FLOOR_PCT:
                vol_target = VOL_FLOOR_PCT
            target_pct = vol_target
            reason = "HOLD"

        # ── 当前仓位 ─────────────────────────────────────────────────────────
        portfolio_value = cash + shares * price
        current_pct     = (shares * price) / portfolio_value if portfolio_value > 0 else 0.0

        # ── 再平衡触发 ───────────────────────────────────────────────────────
        pct_diff = abs(target_pct - current_pct)
        if pct_diff > REBAL_THRESHOLD:
            target_value  = portfolio_value * target_pct
            target_shares = target_value / price
            delta_shares  = target_shares - shares
            trade_value   = abs(delta_shares) * price
            cost          = trade_value * (COMMISSION + SLIPPAGE)

            if delta_shares > 0:
                # 买入
                total_cost = delta_shares * price * (1 + COMMISSION + SLIPPAGE)
                if cash >= total_cost:
                    cash   -= total_cost
                    shares += delta_shares
                    cost_total += cost
                    rebal_count += 1
                    trade_log.append({
                        "date": date, "action": "BUY",
                        "shares": round(delta_shares, 2),
                        "price": price, "cost": round(cost, 2),
                        "target_pct": round(target_pct * 100, 1),
                        "reason": reason,
                        "realized_vol": round(rv * 100, 1),
                    })
            elif delta_shares < 0:
                # 卖出
                sell_shares = min(abs(delta_shares), shares)
                proceeds = sell_shares * price * (1 - COMMISSION - SLIPPAGE)
                cash     += proceeds
                shares   -= sell_shares
                cost_total += cost
                rebal_count += 1
                trade_log.append({
                    "date": date, "action": "SELL",
                    "shares": round(sell_shares, 2),
                    "price": price, "cost": round(cost, 2),
                    "target_pct": round(target_pct * 100, 1),
                    "reason": reason,
                    "realized_vol": round(rv * 100, 1),
                })

        # ── 记录净值 ──────────────────────────────────────────────────────────
        portfolio_value = cash + shares * price
        equity_curve.append({
            "date":           date,
            "portfolio_value": round(portfolio_value, 2),
            "shares":         round(shares, 4),
            "price":          round(price, 4),
            "pct":            round((shares * price / portfolio_value * 100) if portfolio_value > 0 else 0, 1),
            "regime":         reason,
            "realized_vol":   round(rv * 100, 1),
            "target_pct":     round(target_pct * 100, 1),
        })

    eq_df = pd.DataFrame(equity_curve).set_index("date")
    tr_df = pd.DataFrame(trade_log)

    # ── B&H 基准 ──────────────────────────────────────────────────────────────
    start_price = float(close.loc[idx[0]])
    end_price   = float(close.loc[idx[-1]])
    bh_shares   = INITIAL_CAPITAL / start_price
    bh_final    = bh_shares * end_price
    bh_return   = (bh_final / INITIAL_CAPITAL - 1) * 100
    years        = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25

    # ── 绩效指标 ──────────────────────────────────────────────────────────────
    pv        = eq_df["portfolio_value"]
    total_ret = (pv.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    cagr      = ((pv.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0

    weekly_returns = pv.pct_change().dropna()
    sharpe  = (weekly_returns.mean() / weekly_returns.std() * np.sqrt(52)
               if weekly_returns.std() > 0 else 0)

    # 最大回撤
    roll_max  = pv.cummax()
    dd_series = (pv - roll_max) / roll_max
    max_dd    = dd_series.min() * 100

    # B&H 最大回撤
    bh_curve  = close.loc[idx] * bh_shares
    bh_roll   = bh_curve.cummax()
    bh_dd     = ((bh_curve - bh_roll) / bh_roll).min() * 100

    return {
        "ticker":         ticker,
        "total_return":   round(total_ret, 2),
        "bh_return":      round(bh_return, 2),
        "alpha":          round(total_ret - bh_return, 2),
        "cagr":           round(cagr, 2),
        "sharpe":         round(sharpe, 2),
        "max_drawdown":   round(max_dd, 2),
        "bh_max_drawdown":round(bh_dd, 2),
        "rebal_count":    rebal_count,
        "cost_total":     round(cost_total, 2),
        "years":          round(years, 1),
        "equity_df":      eq_df,
        "trade_df":       tr_df,
    }


# ── 打印报告 ──────────────────────────────────────────────────────────────────

def print_report(res: dict):
    t = res["ticker"]
    print(f"\n  {'─'*55}")
    print(f"  {t}  {res['years']:.0f}年回测结果")
    print(f"  {'─'*55}")
    print(f"  {'指标':<20}  {'Beta Overlay':>14}  {'B&H':>10}")
    print(f"  {'─'*46}")
    print(f"  {'总收益':<20}  {res['total_return']:>+13.1f}%  {res['bh_return']:>+9.1f}%")
    print(f"  {'CAGR':<20}  {res['cagr']:>+13.1f}%")
    print(f"  {'Sharpe':<20}  {res['sharpe']:>14.2f}")
    print(f"  {'最大回撤':<20}  {res['max_drawdown']:>13.1f}%  {res['bh_max_drawdown']:>+9.1f}%")
    print(f"  {'超额收益(Alpha)':<20}  {res['alpha']:>+13.1f}%")
    print(f"  {'再平衡次数':<20}  {res['rebal_count']:>14}")
    print(f"  {'交易成本合计':<20}  ${res['cost_total']:>13,.0f}")

    # 制度分布
    eq = res["equity_df"]
    regime_dist = eq["regime"].value_counts()
    total_weeks = len(eq)
    print(f"\n  制度分布：")
    for r, n in regime_dist.items():
        bar = "█" * int(n / total_weeks * 30)
        print(f"    {r:<12}: {n:3d}周 ({n/total_weeks*100:4.0f}%)  {bar}")

    # 分年净值
    print(f"\n  分年净值表现：")
    print(f"  {'年份':<6}  {'策略年收益':>10}  {'B&H年收益':>10}  "
          f"{'平均仓位':>8}  {'主要制度':<12}")
    eq["year"] = eq.index.year
    # B&H 年收益
    close_year = eq["price"].copy()
    for yr, grp in eq.groupby("year"):
        strat_ret = (grp["portfolio_value"].iloc[-1] /
                     grp["portfolio_value"].iloc[0] - 1) * 100 if len(grp) > 1 else 0
        bh_ret_yr = (grp["price"].iloc[-1] /
                     grp["price"].iloc[0] - 1) * 100 if len(grp) > 1 else 0
        avg_pct = grp["pct"].mean()
        main_regime = grp["regime"].mode().iloc[0] if not grp.empty else "-"
        flag = "✅" if strat_ret > bh_ret_yr else ("⚠️" if strat_ret > 0 else "❌")
        print(f"  {yr:<6}  {strat_ret:>+9.1f}%  {bh_ret_yr:>+9.1f}%  "
              f"{avg_pct:>7.0f}%  {main_regime:<12}  {flag}")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Beta Overlay 策略历史回测")
    parser.add_argument("--tickers",    nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start",      default=DEFAULT_START)
    parser.add_argument("--end",        default=DEFAULT_END)
    parser.add_argument("--target-vol", type=float, default=TARGET_ANNUAL_VOL,
                        help="目标年化波动率（默认0.16=16%%）")
    parser.add_argument("--save-csv",   action="store_true",
                        help="保存净值曲线CSV到 beta_overlay_results/")
    args = parser.parse_args()

    print("=" * 60)
    print("  验证3：Beta Overlay 策略历史回测")
    print(f"  标的: {args.tickers}")
    print(f"  区间: {args.start} ~ {args.end}")
    print(f"  目标波动率: {args.target_vol*100:.0f}%  上限仓位: {MAX_POSITION_PCT*100:.0f}%")
    print(f"  初始资金: ${INITIAL_CAPITAL:,}  再平衡阈值: {REBAL_THRESHOLD*100:.0f}%")
    print(f"  交易成本: {(COMMISSION+SLIPPAGE)*100:.1f}%/边（往返{(COMMISSION+SLIPPAGE)*200:.1f}bps）")
    print("=" * 60)

    all_results = {}
    for ticker in args.tickers:
        try:
            res = backtest_single(ticker, args.start, args.end, args.target_vol)
            if res:
                all_results[ticker] = res
                print_report(res)
        except Exception as e:
            print(f"  {ticker} 回测失败: {e}")
            import traceback; traceback.print_exc()

    if args.save_csv:
        out_dir = Path("beta_overlay_results")
        out_dir.mkdir(exist_ok=True)
        for ticker, res in all_results.items():
            res["equity_df"].to_csv(out_dir / f"{ticker}_equity.csv")
            if not res["trade_df"].empty:
                res["trade_df"].to_csv(out_dir / f"{ticker}_trades.csv", index=False)
        print(f"\n结果已保存 → {out_dir}/")

    # ── 跨资产汇总对比 ────────────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n\n{'='*60}")
        print("  跨资产汇总：Beta Overlay vs B&H")
        print('='*60)
        print(f"  {'资产':<8}  {'策略总收益':>10}  {'B&H总收益':>10}  "
              f"{'Alpha':>8}  {'Sharpe':>7}  {'最大回撤':>8}  {'B&H最大回撤':>10}")
        print(f"  {'─'*70}")

        total_strat = 0
        total_bh    = 0
        for ticker, res in all_results.items():
            print(f"  {ticker:<8}  {res['total_return']:>+9.1f}%  "
                  f"{res['bh_return']:>+9.1f}%  "
                  f"{res['alpha']:>+7.1f}%  "
                  f"{res['sharpe']:>7.2f}  "
                  f"{res['max_drawdown']:>7.1f}%  "
                  f"{res['bh_max_drawdown']:>9.1f}%")
            total_strat += res["total_return"]
            total_bh    += res["bh_return"]

        avg_strat = total_strat / len(all_results)
        avg_bh    = total_bh / len(all_results)
        print(f"  {'─'*70}")
        print(f"  {'平均':<8}  {avg_strat:>+9.1f}%  {avg_bh:>+9.1f}%  "
              f"{avg_strat-avg_bh:>+7.1f}%")

        print(f"\n  关键结论：")
        winners = [t for t, r in all_results.items() if r["alpha"] > 0]
        losers  = [t for t, r in all_results.items() if r["alpha"] <= 0]
        if winners:
            print(f"  ✅ 跑赢B&H: {winners}")
        if losers:
            print(f"  ❌ 跑输B&H: {losers}")

        dd_saved = {t: r["bh_max_drawdown"] - r["max_drawdown"]
                    for t, r in all_results.items()}
        print(f"  📉 最大回撤改善（相对B&H）:")
        for t, saved in dd_saved.items():
            print(f"     {t}: 减少 {saved:+.1f}pp（策略{all_results[t]['max_drawdown']:.1f}% vs B&H{all_results[t]['bh_max_drawdown']:.1f}%）")


if __name__ == "__main__":
    main()
