"""
规则基准策略（Rule-Based Baseline）
用纯技术规则模拟与 LLM 策略完全相同的执行框架，验证 LLM 是否真的增加价值

信号规则（无 LLM，无 API 调用）：
  做多条件：价格 > 周线EMA200  AND  周线MACD > 0  AND  ADX > 20
  → 说明：股票处于多头结构且趋势有力
  止损：2.5×周线ATR14 以下
  目标：5.0×周线ATR14 以上
  仓位：和 LLM 策略一致（风险 2%/笔）

评估方式：
  在相同时间段跑完整的组合回测，与 LLM 策略的 Sharpe / 胜率 / 盈利因子对比。
  若规则策略 ≈ LLM 策略 → LLM 没有添加增量价值（但 API 费用 + 等待时间更高）
  若 LLM 明显更好 → LLM 信号有真实 Alpha

用法：
    python3 baseline_strategy.py --ticker NVDA --start 2025-01-01 --end 2025-12-31
    python3 baseline_strategy.py --ticker NVDA --start 2022-01-01 --end 2022-12-31
    python3 baseline_strategy.py --ticker NVDA MSFT GOOGL --start 2025-01-01 --end 2025-12-31
"""

import argparse
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import urllib3
import yfinance as yf
from curl_cffi import requests as curl_requests

from gold_analysis import calc_ema, calc_macd, calc_atr, calc_adx

yf.set_tz_cache_location(tempfile.mkdtemp())
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_CACHE_DIR = Path("data_cache")

# ─────────────────────────────────────────────
# 数据加载（复用已有 parquet 缓存）
# ─────────────────────────────────────────────

def _load_cached(ticker: str, interval: str) -> pd.DataFrame:
    safe = ticker.replace("^", "").replace("/", "_").replace("=", "")
    p = _CACHE_DIR / f"{safe}_{interval}.parquet"
    if p.exists():
        try:
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except Exception:
            pass
    return pd.DataFrame()


def fetch_weekly_up_to(ticker: str, ref_date: str, lookback: int = 200) -> pd.DataFrame:
    end_dt    = pd.Timestamp(ref_date)
    weekly    = _load_cached(ticker, "1wk")
    if weekly.empty:
        return pd.DataFrame()
    mask = weekly.index <= end_dt
    return weekly.loc[mask].tail(lookback).copy()


def fetch_daily_up_to(ticker: str, ref_date: str, lookback: int = 60) -> pd.DataFrame:
    end_dt = pd.Timestamp(ref_date)
    daily  = _load_cached(ticker, "1d")
    if daily.empty:
        return pd.DataFrame()
    mask = daily.index <= end_dt
    return daily.loc[mask].tail(lookback).copy()


def get_all_trading_days(ticker: str, start: str, end: str) -> list[str]:
    daily = _load_cached(ticker, "1d")
    if daily.empty:
        return []
    mask = (daily.index >= pd.Timestamp(start)) & (daily.index <= pd.Timestamp(end))
    return [d.strftime("%Y-%m-%d") for d in daily.loc[mask].index]


# ─────────────────────────────────────────────
# 规则信号生成
# ─────────────────────────────────────────────

def generate_rule_signal(ticker: str, ref_date: str) -> dict:
    """
    基于纯规则生成交易信号（无 LLM）。

    返回：
      action     : long / no_trade
      bias_score : 模拟的置信度（用规则强度代替 LLM bias）
      stop_loss  : 2.5×周线 ATR14 以下
      profit_target: 5.0×周线 ATR14 以上
      regime     : 规则判断的制度
    """
    weekly = fetch_weekly_up_to(ticker, ref_date, lookback=250)
    if weekly.empty or len(weekly) < 50:
        return {"action": "no_trade", "reason": "数据不足"}

    close = weekly["Close"].squeeze().dropna()

    if len(close) < 50:
        return {"action": "no_trade", "reason": "数据不足"}

    # ── 技术指标 ──────────────────────────────────────────────
    ema20  = calc_ema(close, 20)
    ema50  = calc_ema(close, 50)
    ema200 = calc_ema(close, 200)
    macd_line, signal_line, histogram = calc_macd(close)
    atr14 = calc_atr(weekly, 14)
    adx_series, plus_di, minus_di = calc_adx(weekly, 14)

    # 取最新值
    def last(s):
        s = s.dropna()
        return float(s.iloc[-1]) if len(s) > 0 else None

    price   = last(close)
    e20     = last(ema20)
    e50     = last(ema50)
    e200    = last(ema200)
    macd_h  = last(histogram)
    atr     = last(atr14)
    adx_val = last(adx_series)

    if any(v is None for v in [price, e50, e200, atr]):
        return {"action": "no_trade", "reason": "指标计算失败"}

    # ── 死叉硬过滤（与 LLM 策略保持一致）─────────────────────
    if e50 < e200:
        return {"action": "no_trade", "reason": "周线死叉(EMA50<EMA200)",
                "regime": "Trending-Down"}

    # ── 趋势判断 ──────────────────────────────────────────────
    is_uptrend = (price > e50 > e200) and (e20 is not None and price > e20)
    macd_bullish = macd_h is not None and macd_h > 0
    adx_trending = adx_val is not None and adx_val > 20

    if is_uptrend:
        regime = "Trending-Up"
    elif price > e200:
        regime = "Consolidation"
    else:
        regime = "Trending-Down"

    # ── 做多信号：趋势 + MACD 多头 + ADX > 20 ─────────────────
    conditions_met = sum([is_uptrend, macd_bullish, adx_trending])
    if conditions_met < 2:
        return {"action": "no_trade", "regime": regime,
                "reason": f"条件不足({conditions_met}/3)"}

    # ── 止损 / 目标 ────────────────────────────────────────────
    stop_loss      = round(price - 1.5 * atr, 2)   # 与v10 LLM策略一致（原2.5×）
    profit_target  = round(price + 3.0 * atr, 2)   # 与v10 LLM策略一致（原5.0×）

    risk   = price - stop_loss
    reward = profit_target - price
    if risk <= 0 or reward / risk < 1.5:
        return {"action": "no_trade", "regime": regime, "reason": "R:R不足"}

    # ── bias_score 模拟（用规则强度替代 LLM）─────────────────
    bias = 0.50
    if conditions_met == 3:
        bias += 0.10
    if adx_val and adx_val > 30:
        bias += 0.05
    if e20 and abs(price - e20) / e20 < 0.01:  # 价格靠近 EMA20（回调入场）
        bias += 0.05
    bias = min(bias, 0.85)

    return {
        "action":        "long",
        "bias_score":    round(bias, 2),
        "stop_loss":     stop_loss,
        "profit_target": profit_target,
        "regime":        regime,
        "conditions_met": conditions_met,
        "rr":            round(reward / risk, 2),
    }


# ─────────────────────────────────────────────
# 组合回测（复用与 LLM 策略相同的执行逻辑）
# ─────────────────────────────────────────────

def run_baseline_backtest(
    ticker: str,
    start: str,
    end: str,
    initial_capital: float = 100_000,
    risk_per_trade: float  = 0.03,   # 与v10 LLM策略一致（原0.02）
    commission_pct: float  = 0.001,
    slippage_pct: float    = 0.001,
    eval_days: int         = 65,
    stop_cooldown: int     = 5,
    min_rr: float          = 1.5,
    trail_breakeven_pct: float = 8.0,   # 与 v6 LLM 策略一致
    trail_lock_pct: float  = 10.0,
) -> dict:
    """
    纯规则组合回测，执行框架与 LLM 策略完全相同。
    信号改为规则生成，无 API 调用，运行速度快。
    """
    all_days   = get_all_trading_days(ticker, start, end)
    daily_data = _load_cached(ticker, "1d")

    if not all_days or daily_data.empty:
        print(f"  [{ticker}] 数据不可用")
        return {}

    daily_data.index = pd.to_datetime(daily_data.index).tz_localize(None)

    print(f"\n{'='*60}")
    print(f"[规则基准] {ticker}  {start} ~ {end}")
    print(f"  初始资金=${initial_capital:,.0f}  风险/笔={risk_per_trade*100:.0f}%")
    print(f"  佣金={commission_pct*100:.2f}%/边  滑点={slippage_pct*100:.2f}%/边")
    print(f"  规则: EMA200上方 + MACD多头 + ADX>20 → 做多")
    print(f"{'='*60}")

    cash         = float(initial_capital)
    position     = None
    cooldown     = 0
    pending      = None
    trade_records = []
    equity_curve  = []
    signal_count  = 0
    entry_count   = 0

    for idx, today_str in enumerate(all_days):
        today_dt = pd.Timestamp(today_str)
        row_idx  = daily_data.index.get_indexer([today_dt], method="nearest")[0]
        if row_idx < 0 or row_idx >= len(daily_data):
            continue

        row       = daily_data.iloc[row_idx]
        today_open  = float(row["Open"].squeeze())
        today_high  = float(row["High"].squeeze())
        today_low   = float(row["Low"].squeeze())
        today_close = float(row["Close"].squeeze())

        # ── A. 次日入场 ──────────────────────────────────────────
        if pending and position is None and cooldown == 0:
            ep = today_open * (1 + slippage_pct)
            fee = ep * commission_pct

            risk_amt = cash * risk_per_trade
            price_risk = ep - pending["stop"]
            if price_risk > 0:
                qty = int(risk_amt / (price_risk + fee))
            else:
                qty = 0

            if qty > 0 and ep * qty <= cash * 0.5:  # 最多用50%资金
                cash -= ep * qty + ep * qty * commission_pct
                position = {
                    "entry_price": ep,
                    "entry_date":  today_str,
                    "stop":        pending["stop"],
                    "target":      pending["target"],
                    "quantity":    qty,
                    "action":      "long",
                    "hold_days":   0,
                    "trailing":    False,
                    "bias":        pending["bias"],
                    "regime":      pending["regime"],
                }
                entry_count += 1
                print(f"  [ENTER] {today_str} @ {ep:.2f}  ×{qty}  "
                      f"stop={pending['stop']:.2f}  target={pending['target']:.2f}")
        pending = None

        # ── B. 移动止损 ───────────────────────────────────────────
        if position:
            position["hold_days"] += 1
            unrealized = (today_close - position["entry_price"]) / position["entry_price"] * 100

            if unrealized >= trail_lock_pct:
                new_trail = position["entry_price"] + (today_close - position["entry_price"]) * 0.5
                if new_trail > position["stop"]:
                    position["stop"]     = round(new_trail, 2)
                    position["trailing"] = True
            elif unrealized >= trail_breakeven_pct and not position["trailing"]:
                if position["entry_price"] > position["stop"]:
                    position["stop"]     = round(position["entry_price"], 2)
                    position["trailing"] = True

            # ── 止损 / 止盈 / 超时 ───────────────────────────────
            exit_price  = None
            exit_reason = None
            if today_low <= position["stop"]:
                exit_price, exit_reason = position["stop"], "STOP_LOSS"
            elif today_high >= position["target"]:
                exit_price, exit_reason = position["target"], "TAKE_PROFIT"
            elif position["hold_days"] >= eval_days and not position["trailing"]:
                exit_price, exit_reason = today_close, "TIMEOUT"
            elif position["hold_days"] >= eval_days * 2:
                exit_price, exit_reason = today_close, "TIMEOUT_EXTENDED"

            if exit_price:
                ep2  = exit_price * (1 - slippage_pct)
                fee2 = position["quantity"] * ep2 * commission_pct
                cash += position["quantity"] * ep2 - fee2
                pnl  = (ep2 - position["entry_price"]) / position["entry_price"] * 100

                trade_records.append({
                    "entry_date":     position["entry_date"],
                    "exit_date":      today_str,
                    "entry_price":    round(position["entry_price"], 4),
                    "exit_price":     round(ep2, 4),
                    "stop":           position["stop"],
                    "target":         position["target"],
                    "exit_reason":    exit_reason,
                    "quantity":       position["quantity"],
                    "pnl_pct":        round(pnl, 4),
                    "win":            pnl > 0,
                    "hold_days":      position["hold_days"],
                    "portfolio_value": round(cash, 2),
                    "trailing":       position["trailing"],
                })
                print(f"  [EXIT]  {today_str} @ {ep2:.2f}  {exit_reason:<16}"
                      f"  pnl={pnl:+.2f}%  portfolio=${cash:,.0f}")

                if exit_reason == "STOP_LOSS" and not position["trailing"]:
                    cooldown = stop_cooldown + 1
                else:
                    cooldown = 0
                position = None

        # ── C. 资金曲线 ───────────────────────────────────────────
        pos_val = position["quantity"] * today_close if position else 0
        equity_curve.append({"date": today_str, "portfolio_value": round(cash + pos_val, 2)})

        if cooldown > 0:
            cooldown -= 1
            continue

        if position:
            continue

        # ── D. 生成规则信号（每日）────────────────────────────────
        signal_count += 1
        sig = generate_rule_signal(ticker, today_str)

        action = sig.get("action", "no_trade")
        if action != "long":
            continue

        sl = sig.get("stop_loss")
        pt = sig.get("profit_target")
        if sl is None or pt is None:
            continue

        risk_chk   = today_close - sl
        reward_chk = pt - today_close
        if risk_chk <= 0 or reward_chk <= 0 or reward_chk / risk_chk < min_rr:
            continue

        pending = {"stop": sl, "target": pt, "bias": sig.get("bias_score", 0.55),
                   "regime": sig.get("regime", "")}

    # ── 强平 ─────────────────────────────────────────────────────
    if position and all_days:
        last_row = daily_data.loc[daily_data.index <= pd.Timestamp(all_days[-1])].iloc[-1]
        ep = float(last_row["Close"].squeeze()) * (1 - slippage_pct)
        fee = position["quantity"] * ep * commission_pct
        cash += position["quantity"] * ep - fee
        pnl = (ep - position["entry_price"]) / position["entry_price"] * 100
        trade_records.append({
            "entry_date": position["entry_date"], "exit_date": all_days[-1],
            "entry_price": round(position["entry_price"], 4), "exit_price": round(ep, 4),
            "stop": position["stop"], "target": position["target"],
            "exit_reason": "END_OF_BACKTEST", "quantity": position["quantity"],
            "pnl_pct": round(pnl, 4), "win": pnl > 0,
            "hold_days": position["hold_days"], "portfolio_value": round(cash, 2),
            "trailing": position["trailing"],
        })

    # ── 绩效统计 ──────────────────────────────────────────────────
    trades_df = pd.DataFrame(trade_records)
    equity_df = pd.DataFrame(equity_curve)

    total_return = (cash - initial_capital) / initial_capital * 100
    print(f"\n{'='*60}")
    print(f"[规则基准结果] {ticker}")
    print("=" * 60)
    print(f"  最终资金     : ${cash:>12,.2f}")
    print(f"  总收益       : {total_return:>+10.2f}%")
    print(f"  总信号次数   : {signal_count}")
    print(f"  实际入场笔数 : {entry_count}")

    completed = trades_df[trades_df["exit_reason"].isin(["STOP_LOSS", "TAKE_PROFIT"])]
    if not completed.empty:
        wins = completed[completed["pnl_pct"] > 0]["pnl_pct"]
        losses = completed[completed["pnl_pct"] <= 0]["pnl_pct"]
        win_rate = len(wins) / len(completed)
        pf = (wins.sum() / abs(losses.sum()) if losses.sum() != 0
              else float("inf") if wins.sum() > 0 else 0)

        print(f"  已确认交易   : {len(completed)}")
        print(f"  胜率         : {win_rate:.1%}")
        print(f"  盈利因子     : {pf:.2f}")
        print(f"  平均盈利     : {wins.mean():>+.2f}%" if len(wins) > 0 else "  平均盈利     : N/A")
        print(f"  平均亏损     : {losses.mean():>+.2f}%" if len(losses) > 0 else "  平均亏损     : N/A")

        # Sharpe
        eq = equity_df["portfolio_value"].pct_change().dropna()
        sharpe = eq.mean() / eq.std() * np.sqrt(252) if eq.std() > 0 else 0
        dd = (equity_df["portfolio_value"] / equity_df["portfolio_value"].cummax() - 1).min()
        print(f"  Sharpe       : {sharpe:.2f}")
        print(f"  最大回撤     : {dd*100:.2f}%")

        # 保存
        out_dir = Path(f"{ticker.lower()}_baseline_backtest")
        out_dir.mkdir(exist_ok=True)
        trades_df.to_csv(out_dir / "trades.csv", index=False)
        equity_df.to_csv(out_dir / "equity.csv", index=False)
        print(f"\n  结果已保存 → {out_dir}/")

        return {
            "ticker": ticker, "period": f"{start}~{end}",
            "total_return": round(total_return, 2),
            "n_trades": len(completed), "win_rate": round(win_rate, 3),
            "profit_factor": round(pf, 2),
            "sharpe": round(sharpe, 2), "max_drawdown": round(dd * 100, 2),
        }

    print("  无已确认交易")
    return {"ticker": ticker, "total_return": round(total_return, 2)}


def main():
    parser = argparse.ArgumentParser(description="规则基准策略回测（无 LLM）")
    parser.add_argument("--ticker", nargs="+",
                        default=["NVDA"],
                        help="股票 ticker 列表")
    parser.add_argument("--start", type=str, required=True,
                        help="回测起始日期，如 2025-01-01")
    parser.add_argument("--end", type=str, required=True,
                        help="回测结束日期，如 2025-12-31")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--risk-per-trade", type=float, default=0.02)
    args = parser.parse_args()

    all_results = []
    for ticker in args.ticker:
        result = run_baseline_backtest(
            ticker=ticker, start=args.start, end=args.end,
            initial_capital=args.capital,
            risk_per_trade=args.risk_per_trade,
        )
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  多 Ticker 汇总对比")
        print("=" * 60)
        print(f"  {'Ticker':>6} | {'总收益':>8} | {'N':>4} | {'胜率':>7} | "
              f"{'盈利因子':>8} | {'Sharpe':>7} | {'最大回撤':>8}")
        print(f"  {'─'*6}-+-{'─'*8}-+-{'─'*4}-+-{'─'*7}-+-{'─'*8}-+-{'─'*7}-+-{'─'*8}")
        for r in all_results:
            print(f"  {r['ticker']:>6} | {r.get('total_return', 0):>+7.2f}% | "
                  f"{r.get('n_trades', 0):>4} | "
                  f"{r.get('win_rate', 0):.1%} | "
                  f"{r.get('profit_factor', 0):>8.2f} | "
                  f"{r.get('sharpe', 0):>7.2f} | "
                  f"{r.get('max_drawdown', 0):>7.2f}%")

    print("\n  使用方法: 将上述指标与 LLM 策略结果对比")
    print("  若规则策略 Sharpe ≈ LLM 策略 → LLM 无增量价值")
    print("  若 LLM 策略 Sharpe 明显更高 → LLM 信号有真实 Alpha")


if __name__ == "__main__":
    main()
