"""
持仓跟踪器
=========
读取 portfolio.json 中的持仓，结合最新 LLM 信号和实时价格，
输出每个持仓的当前状态和具体操作建议，并保存到 portfolio_status.json。

运行方式：
    python portfolio_tracker.py               # 查看持仓状态（不调用 LLM）
    python portfolio_tracker.py --update-signals  # 先刷新各资产信号再评估
    python portfolio_tracker.py --export-orders   # 额外导出 orders.json（交易接口格式）

portfolio_status.json 格式可直接被交易接口读取：
  - action: HOLD / ADD / REDUCE / EXIT / STOP_TRIGGERED / TARGET_REACHED / SIGNAL_REVERSED
  - order:  null 或 {"side": "buy"|"sell", "quantity": X, "order_type": "market"|"limit", "price": X}
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib3
from datetime import datetime, date
from pathlib import Path

import yfinance as yf
import pandas as pd
from curl_cffi import requests as curl_requests

from assets_config import ASSET_UNIVERSE

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PORTFOLIO_FILE       = "portfolio.json"
STATUS_OUTPUT_FILE   = "portfolio_status.json"
ORDERS_OUTPUT_FILE   = "orders.json"

# 阈值配置
STOP_WARNING_PCT  = 0.03   # 距止损 3% 以内触发警告
TARGET_ZONE_PCT   = 0.03   # 距目标 3% 以内触发"接近目标"提示


# ─────────────────────────────────────────────
# 价格获取
# ─────────────────────────────────────────────

def _make_session():
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    return curl_requests.Session(impersonate="chrome", verify=False, proxy=proxy)


def fetch_current_prices(asset_keys: list[str]) -> dict[str, float | None]:
    """批量获取资产最新价格"""
    prices = {}
    session = _make_session()
    print("正在获取实时价格...")
    for key in asset_keys:
        cfg = ASSET_UNIVERSE.get(key)
        if not cfg:
            prices[key] = None
            continue
        ticker = cfg["ticker"]
        try:
            df = yf.download(
                ticker, period="2d", interval="1d",
                auto_adjust=True, progress=False, session=session
            )
            if not df.empty:
                price = float(df["Close"].iloc[-1])
                prices[key] = round(price, 4)
                print(f"  {key:12s} ({ticker:12s}): {price:.4f}")
            else:
                prices[key] = None
                print(f"  {key:12s} ({ticker:12s}): 获取失败")
        except Exception as e:
            prices[key] = None
            print(f"  {key:12s} ({ticker:12s}): 异常 {e}")
    return prices


# ─────────────────────────────────────────────
# 信号读取与解析
# ─────────────────────────────────────────────

def parse_signal_from_file(output_file: str) -> dict | None:
    """读取并解析资产的最新 LLM 信号 JSON"""
    path = Path(__file__).parent / output_file
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8")
    return _parse_json(raw)


def _parse_json(raw: str) -> dict | None:
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    m = re.search(r"```json\s*(.*?)```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    depth, start = 0, -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    start = -1
    return None


def extract_asset_signal(sig_json: dict | None, asset_key: str) -> dict | None:
    """从完整 JSON 中提取 asset_analysis 里对应资产的信号"""
    if not sig_json:
        return None
    analyses = sig_json.get("asset_analysis", [])
    for a in analyses:
        asset_name = a.get("asset", "").upper()
        # 模糊匹配：GOLD / GC=F / PAXG 都能对上 "GOLD"
        if asset_key in asset_name or asset_name in asset_key:
            return a
    # 若只有一个，直接返回
    if len(analyses) == 1:
        return analyses[0]
    return None


# ─────────────────────────────────────────────
# 持仓评估核心逻辑
# ─────────────────────────────────────────────

def evaluate_position(pos: dict, current_price: float | None, latest_signal: dict | None) -> dict:
    """
    对单个持仓进行综合评估，返回状态和建议。

    返回结构：
      asset, type, entry_price, current_price, unrealized_pnl_pct,
      status, action, reason, stop_loss, profit_target,
      distance_to_stop_pct, distance_to_target_pct,
      latest_signal_action, latest_signal_bias,
      order (null 或 trading interface 格式)
    """
    asset      = pos["asset"]
    pos_type   = pos.get("type", "long")         # long / short
    entry      = float(pos.get("entry_price", 0))
    quantity   = float(pos.get("quantity", 0))
    stop_loss  = pos.get("stop_loss")
    target     = pos.get("profit_target")

    # ── P&L 计算 ──
    if current_price is not None and entry > 0:
        if pos_type == "long":
            pnl_pct = (current_price - entry) / entry * 100
        else:
            pnl_pct = (entry - current_price) / entry * 100
    else:
        pnl_pct = None

    # ── 距止损/目标距离 ──
    if current_price and stop_loss:
        if pos_type == "long":
            dist_stop   = (current_price - float(stop_loss)) / current_price
            stop_breach = current_price <= float(stop_loss)
        else:
            dist_stop   = (float(stop_loss) - current_price) / current_price
            stop_breach = current_price >= float(stop_loss)
    else:
        dist_stop, stop_breach = None, False

    if current_price and target:
        if pos_type == "long":
            dist_target  = (float(target) - current_price) / current_price
            target_reach = current_price >= float(target)
        else:
            dist_target  = (current_price - float(target)) / current_price
            target_reach = current_price <= float(target)
    else:
        dist_target, target_reach = None, False

    # ── 提取最新信号 ──
    sig_action = None
    sig_bias   = None
    if latest_signal:
        sig_action = latest_signal.get("action", "").lower()
        sig_bias   = latest_signal.get("bias_score")

    # ── 决策逻辑 ──
    action = "HOLD"
    reason_parts = []

    # 优先级 1：价格已触达止损
    if stop_breach:
        action = "STOP_TRIGGERED"
        reason_parts.append(f"价格 {current_price} 已触及/跌破止损 {stop_loss}")

    # 优先级 2：价格已触达目标
    elif target_reach:
        action = "TARGET_REACHED"
        reason_parts.append(f"价格 {current_price} 已到达/超过目标 {target}")

    # 优先级 3：信号反转
    elif sig_action and (
        (pos_type == "long"  and sig_action == "short") or
        (pos_type == "short" and sig_action == "long")
    ):
        action = "SIGNAL_REVERSED"
        reason_parts.append(f"LLM 最新信号已反转为 {sig_action}（bias={sig_bias}），建议减仓/平仓")

    # 优先级 4：信号变为 no_trade 且 bias 低
    elif sig_action == "no_trade" and (sig_bias is None or sig_bias < 0.5):
        action = "REDUCE"
        reason_parts.append(f"LLM 最新信号为 no_trade（bias={sig_bias}），建议减仓观望")

    # 优先级 5：接近止损（未触发但需警惕）
    elif dist_stop is not None and 0 < dist_stop < STOP_WARNING_PCT:
        action = "HOLD"
        reason_parts.append(f"⚠️ 警告：距止损仅 {dist_stop*100:.1f}%，密切关注")

    # 优先级 6：接近目标
    elif dist_target is not None and 0 < dist_target < TARGET_ZONE_PCT:
        action = "HOLD"
        reason_parts.append(f"✅ 接近目标价（距目标 {dist_target*100:.1f}%），考虑锁利")

    # 默认：信号一致，持仓
    else:
        if sig_action in ("long", "short"):
            reason_parts.append(
                f"LLM 信号与持仓方向一致（{sig_action}，bias={sig_bias}），维持持仓"
            )
        elif not sig_action:
            reason_parts.append("无最新信号文件，维持持仓，建议运行 --update-signals")
        else:
            reason_parts.append(f"信号 {sig_action}，维持观察")

    reason = "；".join(reason_parts) or "无异常"

    # ── 构建 order 字段（供交易接口读取）──
    order = None
    if action in ("STOP_TRIGGERED", "SIGNAL_REVERSED"):
        order = {
            "side":       "sell" if pos_type == "long" else "buy",
            "quantity":   quantity,
            "order_type": "market",
            "price":      current_price,
            "note":       action,
        }
    elif action == "TARGET_REACHED":
        order = {
            "side":       "sell" if pos_type == "long" else "buy",
            "quantity":   quantity,
            "order_type": "limit",
            "price":      target,
            "note":       "TARGET_REACHED - limit close",
        }
    elif action == "REDUCE":
        order = {
            "side":       "sell" if pos_type == "long" else "buy",
            "quantity":   round(quantity * 0.5, 8),  # 减半仓
            "order_type": "market",
            "price":      current_price,
            "note":       "REDUCE 50%",
        }

    return {
        "asset":                    asset,
        "type":                     pos_type,
        "entry_price":              entry,
        "current_price":            current_price,
        "quantity":                 quantity,
        "unrealized_pnl_pct":       round(pnl_pct, 2) if pnl_pct is not None else None,
        "stop_loss":                stop_loss,
        "profit_target":            target,
        "distance_to_stop_pct":     round(dist_stop * 100, 2) if dist_stop is not None else None,
        "distance_to_target_pct":   round(dist_target * 100, 2) if dist_target is not None else None,
        "status":                   action,
        "action":                   action,
        "reason":                   reason,
        "latest_signal_action":     sig_action,
        "latest_signal_bias_score": sig_bias,
        "order":                    order,
        "evaluated_at":             datetime.now().isoformat(timespec="seconds"),
    }


# ─────────────────────────────────────────────
# 持仓汇总报告
# ─────────────────────────────────────────────

def print_report(evaluations: list[dict]):
    print("\n" + "=" * 70)
    print(f"  持仓状态报告  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    action_emoji = {
        "HOLD":             "✅",
        "ADD":              "🟢",
        "REDUCE":           "🟡",
        "EXIT":             "🔴",
        "STOP_TRIGGERED":   "🛑",
        "TARGET_REACHED":   "🎯",
        "SIGNAL_REVERSED":  "↩️",
    }

    for e in evaluations:
        emoji = action_emoji.get(e["action"], "❓")
        pnl_str = f"{e['unrealized_pnl_pct']:+.2f}%" if e["unrealized_pnl_pct"] is not None else "N/A"
        price_str = f"{e['current_price']}" if e["current_price"] else "N/A"

        print(f"\n{emoji} {e['asset']:8s}  [{e['type']:5s}]  "
              f"入场={e['entry_price']}  当前={price_str}  P&L={pnl_str}")
        print(f"   止损={e['stop_loss']}  目标={e['profit_target']}")
        if e["distance_to_stop_pct"] is not None:
            print(f"   距止损={e['distance_to_stop_pct']:+.1f}%  距目标={e['distance_to_target_pct']:+.1f}%" if e["distance_to_target_pct"] is not None else f"   距止损={e['distance_to_stop_pct']:+.1f}%")
        print(f"   建议操作: {e['action']}")
        print(f"   原因: {e['reason']}")
        if e["order"]:
            o = e["order"]
            print(f"   📋 待执行订单: {o['side'].upper()} {o['quantity']} @ {o['order_type']} "
                  f"({'$'+str(o['price']) if o['price'] else 'market'})")

    print("\n" + "=" * 70)
    needs_action = [e for e in evaluations if e["action"] not in ("HOLD",)]
    print(f"  需要操作的持仓: {len(needs_action)}/{len(evaluations)}")
    print("=" * 70 + "\n")




# ─────────────────────────────────────────────
# Beta Overlay 模式
# ─────────────────────────────────────────────

BETA_BASE_PCT    = 0.50   # EMA200 上方底仓比例
LLM_OVERLAY_PCT  = 0.20   # LLM long 信号叠加比例（总 70%）
REBAL_THRESHOLD  = 0.05   # 当前仓位偏离目标 >5% 才触发再平衡


def _calc_weekly_ema200(ticker_sym: str) -> float | None:
    """获取周线 EMA200（用于 Beta 底仓判断）"""
    try:
        session = _make_session()
        df = yf.download(ticker_sym, period="5y", interval="1wk",
                         auto_adjust=True, progress=False, session=session)
        if df.empty or len(df) < 50:
            return None
        closes = df["Close"].squeeze().dropna()
        # 指数移动平均
        ema = closes.ewm(span=200, adjust=False).mean()
        return float(ema.iloc[-1])
    except Exception as e:
        print(f"  EMA200 计算失败 ({ticker_sym}): {e}")
        return None


def _parse_llm_signal_for_ticker(asset_key: str) -> dict | None:
    """读取最新 LLM 信号文件，提取该资产的 action 和 bias_score"""
    cfg = ASSET_UNIVERSE.get(asset_key)
    if not cfg:
        return None
    sig_json = parse_signal_from_file(cfg["output_file"])
    return extract_asset_signal(sig_json, asset_key)


def evaluate_overlay(
    asset_key: str,
    ticker_sym: str,
    current_price: float | None,
    current_shares: float,        # 当前持有股数（来自 portfolio.json）
    allocated_capital: float,     # 分配给这个 ticker 的总资金
) -> dict:
    """
    Beta Overlay 单资产评估：

    目标仓位逻辑：
      price > EMA200 + LLM long  → 70% (底仓 50% + 叠加 20%)
      price > EMA200 + LLM other → 50% (纯底仓)
      price < EMA200             → 0%  (全出)

    输出 target_pct / target_shares / delta_shares / action / order
    """
    result = {
        "asset":             asset_key,
        "ticker":            ticker_sym,
        "current_price":     current_price,
        "current_shares":    current_shares,
        "allocated_capital": allocated_capital,
        "ema200":            None,
        "above_ema200":      None,
        "llm_action":        None,
        "llm_bias":          None,
        "target_pct":        0.0,
        "target_shares":     0,
        "current_pct":       0.0,
        "delta_shares":      0,
        "action":            "HOLD",
        "reason":            "",
        "order":             None,
        "evaluated_at":      datetime.now().isoformat(timespec="seconds"),
    }

    if current_price is None or current_price <= 0:
        result["reason"] = "价格获取失败，跳过"
        return result

    # ── 当前持仓市值比例 ──
    current_value = current_shares * current_price
    result["current_pct"] = round(current_value / allocated_capital, 4) if allocated_capital > 0 else 0.0

    # ── 1. EMA200 状态 ──
    print(f"  [{asset_key}] 计算周线 EMA200...", end=" ", flush=True)
    ema200 = _calc_weekly_ema200(ticker_sym)
    result["ema200"] = round(ema200, 2) if ema200 else None
    above_ema200 = ema200 is not None and current_price > ema200
    result["above_ema200"] = above_ema200
    ema_str = f"{round(ema200,2):.2f}" if ema200 else "N/A"
    print(f"EMA200={ema_str}  {chr(8593)+chr(19978)+chr(26041) if above_ema200 else chr(8595)+chr(19979)+chr(26041)}")

    # ── 2. LLM 信号 ──
    sig = _parse_llm_signal_for_ticker(asset_key)
    if sig:
        result["llm_action"] = sig.get("action", "no_trade")
        result["llm_bias"]   = sig.get("bias_score")

    llm_long = (result["llm_action"] == "long" and
                (result["llm_bias"] is None or float(result["llm_bias"]) >= 0.50))

    # ── 3. 目标仓位 ──
    if not above_ema200:
        target_pct = 0.0
        _e = f'{ema200:.2f}' if ema200 else 'N/A'
        ema_note   = f"价格{current_price:.2f} < EMA200({_e})，清空底仓"
    elif llm_long:
        target_pct = BETA_BASE_PCT + LLM_OVERLAY_PCT   # 70%
        ema_note   = f"EMA200上方 + LLM long(bias={result['llm_bias']}) → 70%仓位"
    else:
        target_pct = BETA_BASE_PCT                      # 50%
        ema_note   = f"EMA200上方 + LLM {result['llm_action']} → 50%底仓"

    result["target_pct"]    = target_pct
    target_value            = allocated_capital * target_pct
    target_shares           = int(target_value / current_price) if current_price > 0 else 0
    result["target_shares"] = target_shares
    delta                   = target_shares - int(current_shares)
    result["delta_shares"]  = delta

    # ── 4. 决策 ──
    current_pct_val = result["current_pct"]
    pct_diff        = abs(target_pct - current_pct_val)

    if pct_diff <= REBAL_THRESHOLD:
        result["action"] = "HOLD"
        result["reason"] = f"仓位偏差 {pct_diff*100:.1f}% ≤ 阈值 {REBAL_THRESHOLD*100:.0f}%，无需再平衡"
    elif not above_ema200 and current_shares > 0:
        result["action"] = "EXIT_ALL"
        result["reason"] = ema_note
        result["order"]  = {
            "side": "sell", "quantity": int(current_shares),
            "order_type": "market", "price": current_price,
            "note": f"EXIT_ALL — {ema_note}",
        }
    elif target_pct > current_pct_val:
        if current_shares == 0:
            result["action"] = "ENTER_BASE" if not llm_long else "ENTER_FULL"
        else:
            result["action"] = "ADD_OVERLAY" if llm_long else "REBALANCE_UP"
        result["reason"] = ema_note
        if delta > 0:
            result["order"] = {
                "side": "buy", "quantity": delta,
                "order_type": "market", "price": current_price,
                "note": f"{result['action']} — {ema_note}",
            }
    else:
        result["action"] = "REMOVE_OVERLAY" if current_pct_val > BETA_BASE_PCT else "REBALANCE_DOWN"
        result["reason"] = ema_note
        if delta < 0:
            result["order"] = {
                "side": "sell", "quantity": abs(delta),
                "order_type": "market", "price": current_price,
                "note": f"{result['action']} — {ema_note}",
            }

    return result


def print_overlay_report(results: list[dict]):
    action_emoji = {
        "HOLD":          "➖",
        "ENTER_BASE":    "🟢",
        "ENTER_FULL":    "🚀",
        "ADD_OVERLAY":   "➕",
        "REMOVE_OVERLAY":"➖",
        "REBALANCE_UP":  "🔼",
        "REBALANCE_DOWN":"🔽",
        "EXIT_ALL":      "🔴",
    }
    print("\n" + "=" * 70)
    print(f"  Beta Overlay 仓位报告  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  底仓={BETA_BASE_PCT:.0%}(EMA200上方)  叠加={LLM_OVERLAY_PCT:.0%}(LLM long)  上限={BETA_BASE_PCT+LLM_OVERLAY_PCT:.0%}")
    print("=" * 70)

    total_alloc  = sum(r["allocated_capital"] for r in results)
    total_cur    = sum(r["current_shares"] * (r["current_price"] or 0) for r in results)
    total_target = sum(r["target_shares"]  * (r["current_price"] or 0) for r in results)

    for r in results:
        emoji = action_emoji.get(r["action"], "❓")
        cp    = r["current_price"] or 0
        print(f"\n{emoji} {r['asset']:8s} ({r['ticker']:8s})  价格=${cp:.2f}  "
              f"EMA200=${r['ema200'] or 0:.2f}  {'↑' if r['above_ema200'] else '↓'}")
        print(f"   LLM信号: {r['llm_action'] or 'N/A':12s}  bias={r['llm_bias'] or 'N/A'}")
        print(f"   当前: {r['current_shares']:.0f}股 ({r['current_pct']*100:.1f}%)  →  "
              f"目标: {r['target_shares']}股 ({r['target_pct']*100:.0f}%)  "
              f"差额: {r['delta_shares']:+d}股")
        print(f"   建议: {r['action']}  |  {r['reason']}")
        if r["order"]:
            o = r["order"]
            val = o["quantity"] * cp
            print(f"   📋 订单: {o['side'].upper()} {o['quantity']}股  ≈ ${val:,.0f}  ({o['order_type']})")

    print("\n" + "-" * 70)
    print(f"  总分配资金: ${total_alloc:,.0f}  |  当前市值: ${total_cur:,.0f}  |  目标市值: ${total_target:,.0f}")
    needs = [r for r in results if r["action"] != "HOLD"]
    print(f"  需要操作: {len(needs)}/{len(results)} 个资产")
    print("=" * 70 + "\n")


def run_beta_overlay_mode(args):
    """Beta Overlay 主流程"""
    portfolio_path = Path(__file__).parent / PORTFOLIO_FILE
    if not portfolio_path.exists():
        print(f"找不到 {portfolio_path}")
        sys.exit(1)

    with open(portfolio_path, encoding="utf-8") as f:
        portfolio = json.load(f)

    # 读取 beta_overlay_config 配置段
    overlay_cfg = portfolio.get("beta_overlay_config", {})
    if not overlay_cfg:
        print("portfolio.json 中没有 beta_overlay_config 段，请添加配置。")
        print("""示例：
{
  "beta_overlay_config": {
    "NVDA": {"allocated_capital": 50000, "current_shares": 0},
    "MSFT": {"allocated_capital": 30000, "current_shares": 0},
    "GOOGL": {"allocated_capital": 20000, "current_shares": 0}
  }
}""")
        sys.exit(1)

    # 可选：先刷新信号
    if args.update_signals:
        print("\n--- 刷新 LLM 信号 ---")
        for key in overlay_cfg:
            cfg = ASSET_UNIVERSE.get(key)
            if not cfg:
                continue
            cmd = [sys.executable, cfg["script"]] + cfg.get("script_args", []) + [
                "--api", "--model", args.model,
                "--dual-model", "--second-model", args.second_model,
            ]
            print(f"  [{key}] {' '.join(cmd)}")
            try:
                subprocess.run(cmd, timeout=300, cwd=Path(__file__).parent)
            except Exception as e:
                print(f"  [{key}] 刷新失败: {e}")

    # 获取实时价格
    price_map = fetch_current_prices(list(overlay_cfg.keys()))

    # 逐资产评估
    print("\n--- Beta Overlay 评估 ---")
    results = []
    for key, cfg_vals in overlay_cfg.items():
        asset_cfg = ASSET_UNIVERSE.get(key)
        if not asset_cfg:
            print(f"  [{key}] 未在 ASSET_UNIVERSE 中找到，跳过")
            continue
        ticker_sym      = asset_cfg["ticker"]
        allocated       = float(cfg_vals.get("allocated_capital", 0))
        current_shares  = float(cfg_vals.get("current_shares", 0))
        current_price   = price_map.get(key)

        print(f"\n[{key}] 分配资金=${allocated:,.0f}  当前{current_shares:.0f}股  现价={current_price}")
        r = evaluate_overlay(key, ticker_sym, current_price, current_shares, allocated)
        results.append(r)

    print_overlay_report(results)

    # 保存状态
    out = {
        "mode": "beta_overlay",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "beta_base_pct":   BETA_BASE_PCT,
            "llm_overlay_pct": LLM_OVERLAY_PCT,
            "rebal_threshold": REBAL_THRESHOLD,
        },
        "positions": results,
        "orders": [r["order"] for r in results if r["order"]],
    }
    status_path = Path(__file__).parent / "overlay_status.json"
    status_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"状态已保存: {status_path}")

    if args.export_orders:
        orders = [{"asset": r["asset"], **r["order"]} for r in results if r["order"]]
        if orders:
            op = Path(__file__).parent / ORDERS_OUTPUT_FILE
            op.write_text(json.dumps(
                {"generated_at": out["generated_at"], "mode": "beta_overlay", "orders": orders},
                ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"订单已导出: {op}  ({len(orders)} 笔)")
        else:
            print("当前无需执行任何订单")

# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="持仓跟踪器：实时评估持仓状态并给出操作建议")
    parser.add_argument(
        "--update-signals", action="store_true",
        help="在评估前先运行各资产的分析脚本（--api 模式）刷新信号"
    )
    parser.add_argument(
        "--export-orders", action="store_true",
        help="额外导出 orders.json（仅包含需要执行的订单）"
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="刷新信号时使用的模型（需配合 --update-signals）"
    )
    parser.add_argument(
        "--beta-overlay", action="store_true",
        help="Beta Overlay 模式：底仓50pct(EMA200上方)+LLM叠加20pct，替代纯信号择时"
    )
    parser.add_argument(
        "--second-model", default="claude-sonnet-4-6",
        help="刷新信号时双模型确认用的第二模型"
    )
    args = parser.parse_args()

    # ── Beta Overlay 模式 ──
    if args.beta_overlay:
        run_beta_overlay_mode(args)
        return

    # ── 读取持仓文件 ──
    portfolio_path = Path(__file__).parent / PORTFOLIO_FILE
    if not portfolio_path.exists():
        print(f"持仓文件不存在: {portfolio_path}")
        print("请先创建 portfolio.json（参考模板），或运行:\n  cp portfolio.json.example portfolio.json")
        sys.exit(1)

    with open(portfolio_path, encoding="utf-8") as f:
        portfolio = json.load(f)

    positions = [p for p in portfolio.get("positions", []) if not p.get("_example")]
    if not positions:
        print("portfolio.json 中没有实际持仓（只有示例条目），请先添加真实持仓。")
        return

    print(f"已读取 {len(positions)} 个持仓")
    asset_keys = list({p["asset"] for p in positions if p["asset"] in ASSET_UNIVERSE})

    # ── 可选：刷新各资产信号 ──
    if args.update_signals:
        print("\n--- 刷新资产信号 ---")
        for key in asset_keys:
            cfg = ASSET_UNIVERSE.get(key)
            if not cfg:
                print(f"  [{key}] 未在 assets_config 中找到，跳过")
                continue
            script    = cfg["script"]
            base_args = cfg.get("script_args", [])
            cmd = [sys.executable, script] + base_args + ["--api", "--model", args.model]
            print(f"  [{key}] 运行: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, timeout=300, cwd=Path(__file__).parent)
            except Exception as e:
                print(f"  [{key}] 刷新失败: {e}")

    # ── 获取实时价格 ──
    prices = fetch_current_prices(asset_keys)

    # ── 评估每个持仓 ──
    print("\n--- 评估持仓 ---")
    evaluations = []
    for pos in positions:
        asset = pos["asset"]
        cfg   = ASSET_UNIVERSE.get(asset)

        current_price  = prices.get(asset)
        latest_signal  = None

        if cfg:
            sig_json = parse_signal_from_file(cfg["output_file"])
            latest_signal = extract_asset_signal(sig_json, asset)

        result = evaluate_position(pos, current_price, latest_signal)
        evaluations.append(result)

    # ── 打印报告 ──
    print_report(evaluations)

    # ── 保存状态文件 ──
    status_path = Path(__file__).parent / STATUS_OUTPUT_FILE
    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "total":        len(evaluations),
            "needs_action": sum(1 for e in evaluations if e["action"] not in ("HOLD",)),
            "stop_alerts":  sum(1 for e in evaluations if e["action"] == "STOP_TRIGGERED"),
            "target_hits":  sum(1 for e in evaluations if e["action"] == "TARGET_REACHED"),
        },
        "positions": evaluations,
    }
    status_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"持仓状态已保存: {status_path}")

    # ── 可选：导出订单文件 ──
    if args.export_orders:
        orders = [e["order"] for e in evaluations if e["order"] is not None]
        if orders:
            orders_with_meta = [
                {"asset": e["asset"], "action": e["action"], **e["order"]}
                for e in evaluations if e["order"]
            ]
            orders_path = Path(__file__).parent / ORDERS_OUTPUT_FILE
            orders_path.write_text(
                json.dumps(
                    {"generated_at": datetime.now().isoformat(timespec="seconds"), "orders": orders_with_meta},
                    ensure_ascii=False, indent=2
                ),
                encoding="utf-8"
            )
            print(f"订单文件已导出: {orders_path}（{len(orders)} 笔待执行）")
        else:
            print("当前无需执行任何订单")


if __name__ == "__main__":
    main()
