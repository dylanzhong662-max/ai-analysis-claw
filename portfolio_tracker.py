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
                ticker, period="5d", interval="1d",
                auto_adjust=True, progress=False, session=session
            )
            if not df.empty:
                # squeeze() 兼容 MultiIndex；dropna() 跳过当天盘中 NaN
                close_series = df["Close"].squeeze()
                valid = close_series.dropna()
                if valid.empty:
                    prices[key] = None
                    print(f"  {key:12s} ({ticker:12s}): 无有效价格")
                    continue
                price = float(valid.iloc[-1])
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
# Beta Overlay 模式 — 波动率目标制架构
# ─────────────────────────────────────────────

TARGET_ANNUAL_VOL   = 0.16   # 目标年化波动率 16%（仓位随波动率自动缩放）
MAX_POSITION_PCT    = 0.80   # 硬性上限：不超过分配资金 80%
VOL_FLOOR_PCT       = 0.30   # 动量修正下限：牛市制度（价格>EMA200且>EMA50）时，
                             # 无论波动率多高，仓位不低于此值，防止高波动牛市严重欠配
LLM_OVERLAY_PCT     = 0.10   # LLM 确认信号叠加（bias≥0.60 时额外 +10%）
REBAL_THRESHOLD     = 0.05   # 当前仓位偏离目标 >5% 才触发再平衡
VOL_LOOKBACK_WEEKS  = 20     # 实现波动率计算窗口（周）
DRAWDOWN_EXIT_PCT   = 0.15   # 距近期52周高点回撤 >15% 触发快速出场


def _calc_regime_data(ticker_sym: str, asset_key: str) -> dict:
    """
    计算制度判断所需的全套数据：
      - weekly EMA200（牛熊制度主过滤）
      - weekly EMA50（快速出场信号，比 EMA200 快 4-6 周）
      - 200WMA —— 200周简单移动均值（BTC 专用，经三轮牛熊验证）
      - 近20周年化实现波动率（波动率目标制定仓核心）
      - 近52周最高价（计算回撤用于快速出场）

    返回 dict，键：ema200 / ema50 / wma200 / realized_vol_annual / high_52w
    任何字段失败返回 None，调用方需做 None 检查。
    """
    try:
        session = _make_session()
        df = yf.download(ticker_sym, period="6y", interval="1wk",
                         auto_adjust=True, progress=False, session=session)
        if df.empty or len(df) < 52:
            return {}
        closes = df["Close"].squeeze().dropna()

        ema200 = float(closes.ewm(span=200, adjust=False).mean().iloc[-1])
        ema50  = float(closes.ewm(span=50,  adjust=False).mean().iloc[-1])

        # BTC 专用：200WMA（简单移动均线，非 EMA）
        wma200 = float(closes.rolling(200).mean().iloc[-1]) if len(closes) >= 200 else None

        # 近20周年化实现波动率
        weekly_rets = closes.pct_change().dropna().tail(VOL_LOOKBACK_WEEKS)
        realized_vol_annual = float(weekly_rets.std() * (52 ** 0.5)) if len(weekly_rets) >= 4 else 0.20

        # 近52周最高价（用于回撤判断）
        high_52w = float(closes.tail(52).max())

        return {
            "ema200":              ema200,
            "ema50":               ema50,
            "wma200":              wma200,
            "realized_vol_annual": realized_vol_annual,
            "high_52w":            high_52w,
        }
    except Exception as e:
        print(f"  制度数据计算失败 ({ticker_sym}): {e}")
        return {}


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
    Beta Overlay 单资产评估 — 波动率目标制架构

    仓位计算逻辑：
      Step 1  制度过滤（出场优先于加仓）
                价格 < 周线EMA200                    → 清仓（熊市制度）
                BTC 额外：价格 < 200WMA（周线SMA200） → 清仓
                价格 < 周线EMA50 OR 距52周高点回撤>15% → 快速出场

      Step 2  波动率目标制定仓（含动量修正下限）
                vol_target  = TARGET_ANNUAL_VOL(16%) / realized_vol_annual
                上限 MAX_POSITION_PCT(80%)
                低波动牛市（vol=10%）→ 自动加至 80%
                高波动市 （vol=40%）→ 自动降至 40%
                动量修正：价格>EMA200 且 >EMA50（趋势上行） →
                  target_pct = max(vol_target, VOL_FLOOR_PCT=30%)
                  防止高波动牛市（如 NVDA 2019-2021）仓位过低而严重踏空

      Step 3  LLM 叠加（辅助信号）
                action=long AND bias≥0.60 → +LLM_OVERLAY_PCT(10%)
                否则不叠加

    输出新增字段：ema50 / above_ema50 / wma200 / realized_vol_annual /
                 high_52w / drawdown_from_high / fast_exit_triggered /
                 vol_target_pct / llm_overlay_pct
    """
    result = {
        "asset":                 asset_key,
        "ticker":                ticker_sym,
        "current_price":         current_price,
        "current_shares":        current_shares,
        "allocated_capital":     allocated_capital,
        # 制度字段
        "ema200":                None,
        "ema50":                 None,
        "wma200":                None,
        "above_ema200":          None,
        "above_ema50":           None,
        "high_52w":              None,
        "drawdown_from_high":    None,
        "fast_exit_triggered":   False,
        # 波动率字段
        "realized_vol_annual":   None,
        "vol_target_pct":        None,
        # LLM 字段
        "llm_action":            None,
        "llm_bias":              None,
        "llm_overlay_pct":       0.0,
        # 仓位字段
        "target_pct":            0.0,
        "target_shares":         0,
        "current_pct":           0.0,
        "delta_shares":          0,
        "action":                "HOLD",
        "reason":                "",
        "order":                 None,
        "evaluated_at":          datetime.now().isoformat(timespec="seconds"),
    }

    if current_price is None or current_price <= 0:
        result["reason"] = "价格获取失败，跳过"
        return result

    # ── 当前持仓市值比例 ──
    current_value = current_shares * current_price
    result["current_pct"] = round(current_value / allocated_capital, 4) if allocated_capital > 0 else 0.0

    # ── 1. 计算制度数据 ──
    print(f"  [{asset_key}] 计算制度数据（EMA50/200 + 波动率）...", end=" ", flush=True)
    rd = _calc_regime_data(ticker_sym, asset_key)

    ema200              = rd.get("ema200")
    ema50               = rd.get("ema50")
    wma200              = rd.get("wma200")
    realized_vol        = rd.get("realized_vol_annual", 0.20)
    high_52w            = rd.get("high_52w", current_price)

    result["ema200"]             = round(ema200, 2)             if ema200    else None
    result["ema50"]              = round(ema50,  2)             if ema50     else None
    result["wma200"]             = round(wma200, 2)             if wma200    else None
    result["realized_vol_annual"]= round(realized_vol, 4)
    result["high_52w"]           = round(high_52w, 2)           if high_52w  else None

    # EMA200 缓冲带（2%）：减少 V 形反转时的 whipsaw，
    # 价格需跌破 EMA200 的 2% 以下才判定为确认性熊市，避免震荡区间频繁触发/解除
    EMA200_BUFFER = 0.02
    above_ema200 = ema200 is not None and current_price > ema200 * (1 - EMA200_BUFFER)
    above_ema50  = ema50  is not None and current_price > ema50
    result["above_ema200"] = above_ema200
    result["above_ema50"]  = above_ema50

    # 回撤计算
    drawdown = (high_52w - current_price) / high_52w if high_52w and high_52w > 0 else 0.0
    result["drawdown_from_high"] = round(drawdown, 4)

    # BTC 额外检查 200WMA（BTC 波动大，同样加 2% 缓冲）
    btc_wma_ok = True
    if asset_key == "BTC" and wma200 is not None:
        btc_wma_ok = current_price > wma200 * (1 - EMA200_BUFFER)

    # 快速出场：EMA50 跌破 OR 回撤超限
    fast_exit = (not above_ema50) or (drawdown > DRAWDOWN_EXIT_PCT)
    result["fast_exit_triggered"] = fast_exit

    vol_str = f"{realized_vol*100:.0f}%"
    ema_str = f"EMA200={'↑' if above_ema200 else '↓'}{round(ema200,1) if ema200 else 'N/A'}"
    ema50_str = f"EMA50={'↑' if above_ema50 else '↓'}{round(ema50,1) if ema50 else 'N/A'}"
    dd_str  = f"回撤={drawdown*100:.1f}%"
    print(f"{ema_str}  {ema50_str}  {dd_str}  vol={vol_str}")

    # ── 2. 制度判断：是否出场 ──
    bear_regime  = not above_ema200 or not btc_wma_ok
    exit_reason  = ""

    if bear_regime:
        target_pct = 0.0
        parts = []
        if not above_ema200:
            bear_threshold = round(ema200 * (1 - EMA200_BUFFER), 2) if ema200 else 'N/A'
            parts.append(f"价格{current_price:.2f} < EMA200×{1-EMA200_BUFFER:.2f}（{bear_threshold}）")
        if not btc_wma_ok:
            parts.append(f"BTC价格 < 200WMA×{1-EMA200_BUFFER:.2f}（{round(wma200*(1-EMA200_BUFFER),2)}）")
        exit_reason = "熊市制度 — " + " & ".join(parts)
    elif fast_exit:
        target_pct = 0.0
        parts = []
        if not above_ema50:
            parts.append(f"价格{current_price:.2f} < EMA50({round(ema50,2) if ema50 else 'N/A'})（快速出场）")
        if drawdown > DRAWDOWN_EXIT_PCT:
            parts.append(f"距52周高点回撤{drawdown*100:.1f}% > {DRAWDOWN_EXIT_PCT*100:.0f}%触发")
        exit_reason = " & ".join(parts)
    else:
        # ── 3. 波动率目标制定仓（含动量修正下限）──
        if realized_vol > 0:
            vol_target_pct = TARGET_ANNUAL_VOL / realized_vol
        else:
            vol_target_pct = 0.50  # fallback
        vol_target_pct = min(vol_target_pct, MAX_POSITION_PCT)

        # 动量修正下限：牛市趋势中（价格>EMA200且>EMA50），仓位不低于 VOL_FLOOR_PCT
        # 解决问题：高波动牛市（如 NVDA 2019-2021）vol目标制会把仓位压到10%，严重踏空
        # 激活条件：不触发快速出场（above_ema50=True）且在牛市制度中
        floor_active = above_ema50  # 仅在 EMA50 以上才激活（fast_exit时已在上面处理）
        if floor_active and vol_target_pct < VOL_FLOOR_PCT:
            vol_target_pct = VOL_FLOOR_PCT
        result["vol_target_pct"]  = round(vol_target_pct, 4)
        result["vol_floor_active"] = floor_active and (TARGET_ANNUAL_VOL / max(realized_vol, 0.001) < VOL_FLOOR_PCT)

        # ── 4. LLM 叠加（辅助信号，+10%）──
        sig = _parse_llm_signal_for_ticker(asset_key)
        if sig:
            result["llm_action"] = sig.get("action", "no_trade")
            result["llm_bias"]   = sig.get("bias_score")

        llm_long    = (result["llm_action"] == "long" and
                       result["llm_bias"] is not None and
                       float(result["llm_bias"]) >= 0.60)
        llm_overlay = LLM_OVERLAY_PCT if llm_long else 0.0
        result["llm_overlay_pct"] = llm_overlay

        target_pct = min(vol_target_pct + llm_overlay, MAX_POSITION_PCT)

    result["target_pct"] = round(target_pct, 4)
    target_value          = allocated_capital * target_pct
    target_shares         = int(target_value / current_price) if current_price > 0 else 0
    result["target_shares"] = target_shares
    delta                   = target_shares - int(current_shares)
    result["delta_shares"]  = delta

    # ── 5. 决策 & 生成订单 ──
    current_pct_val = result["current_pct"]
    pct_diff        = abs(target_pct - current_pct_val)

    # 出场逻辑（bear_regime 或 fast_exit）
    if (bear_regime or fast_exit) and current_shares > 0:
        result["action"] = "EXIT_BEAR_REGIME" if bear_regime else "EXIT_FAST"
        result["reason"] = exit_reason
        result["order"]  = {
            "side": "sell", "quantity": int(current_shares),
            "order_type": "market", "price": current_price,
            "note": f"{result['action']} — {exit_reason}",
        }
    elif (bear_regime or fast_exit) and current_shares == 0:
        result["action"] = "HOLD"
        result["reason"] = f"已空仓 — {exit_reason}"
    elif pct_diff <= REBAL_THRESHOLD:
        result["action"] = "HOLD"
        vt = result["vol_target_pct"]
        lo = result["llm_overlay_pct"]
        result["reason"] = (
            f"仓位偏差{pct_diff*100:.1f}%≤阈值{REBAL_THRESHOLD*100:.0f}%，无需再平衡  "
            f"[vol目标={vt*100:.0f}% llm叠加={lo*100:.0f}% 目标={target_pct*100:.0f}%]"
        )
    elif target_pct > current_pct_val:
        result["action"] = "ENTER" if current_shares == 0 else "REBALANCE_UP"
        vt = result["vol_target_pct"]
        lo = result["llm_overlay_pct"]
        result["reason"] = (
            f"vol目标制={vt*100:.0f}% + LLM叠加={lo*100:.0f}% → 目标{target_pct*100:.0f}%  "
            f"vol={realized_vol*100:.0f}%"
        )
        if delta > 0:
            result["order"] = {
                "side": "buy", "quantity": delta,
                "order_type": "market", "price": current_price,
                "note": f"{result['action']} — {result['reason']}",
            }
    else:
        result["action"] = "REBALANCE_DOWN"
        result["reason"] = (
            f"波动率上升/LLM信号减弱，目标仓位从{current_pct_val*100:.0f}%降至{target_pct*100:.0f}%  "
            f"vol={realized_vol*100:.0f}%"
        )
        if delta < 0:
            result["order"] = {
                "side": "sell", "quantity": abs(delta),
                "order_type": "market", "price": current_price,
                "note": f"{result['action']} — {result['reason']}",
            }

    return result


def print_overlay_report(results: list[dict]):
    action_emoji = {
        "HOLD":              "➖",
        "ENTER":             "🟢",
        "REBALANCE_UP":      "🔼",
        "REBALANCE_DOWN":    "🔽",
        "EXIT_BEAR_REGIME":  "🔴",
        "EXIT_FAST":         "🟠",
    }
    print("\n" + "=" * 70)
    print(f"  Beta Overlay 仓位报告（波动率目标制）  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  目标年化波动率={TARGET_ANNUAL_VOL:.0%}  LLM叠加={LLM_OVERLAY_PCT:.0%}(bias≥0.60)  上限={MAX_POSITION_PCT:.0%}")
    print(f"  快速出场触发：EMA50跌破 OR 52周回撤>{DRAWDOWN_EXIT_PCT:.0%}")
    print("=" * 70)

    total_alloc  = sum(r["allocated_capital"] for r in results)
    total_cur    = sum(r["current_shares"] * (r["current_price"] or 0) for r in results)
    total_target = sum(r["target_shares"]  * (r["current_price"] or 0) for r in results)

    for r in results:
        emoji = action_emoji.get(r["action"], "❓")
        cp    = r["current_price"] or 0
        vol   = r["realized_vol_annual"]
        dd    = r["drawdown_from_high"]
        vt    = r["vol_target_pct"]
        lo    = r["llm_overlay_pct"]

        print(f"\n{emoji} {r['asset']:8s} ({r['ticker']:8s})  价格=${cp:.2f}")
        print(f"   制度: EMA200={'↑' if r['above_ema200'] else '↓'}${r['ema200'] or 0:.1f}  "
              f"EMA50={'↑' if r['above_ema50'] else '↓'}${r['ema50'] or 0:.1f}  "
              f"回撤={dd*100:.1f}% 快速出场={'是' if r['fast_exit_triggered'] else '否'}"
              + (f"  200WMA=${r['wma200']:.1f}" if r.get("wma200") and r["asset"] == "BTC" else ""))
        floor_tag = "  [动量下限激活✓]" if r.get("vol_floor_active") else ""
        print(f"   波动率: 年化={vol*100:.0f}%  →  vol目标仓={vt*100:.0f}%{floor_tag} + LLM={lo*100:.0f}% = 目标{r['target_pct']*100:.0f}%"
              if vt is not None else f"   目标仓位: {r['target_pct']*100:.0f}%（出场）")
        print(f"   LLM信号: {r['llm_action'] or 'N/A':12s}  bias={r['llm_bias'] or 'N/A'}")
        print(f"   当前: {r['current_shares']:.0f}股 ({r['current_pct']*100:.1f}%)  →  "
              f"目标: {r['target_shares']}股 ({r['target_pct']*100:.0f}%)  "
              f"差额: {r['delta_shares']:+d}股")
        print(f"   建议: {r['action']}  |  {r['reason']}")
        if r["order"]:
            o = r["order"]
            val = o["quantity"] * cp
            print(f"   订单: {o['side'].upper()} {o['quantity']}股  ≈ ${val:,.0f}  ({o['order_type']})")

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
        "mode": "beta_overlay_vol_target",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "target_annual_vol":   TARGET_ANNUAL_VOL,
            "max_position_pct":    MAX_POSITION_PCT,
            "vol_floor_pct":       VOL_FLOOR_PCT,
            "llm_overlay_pct":     LLM_OVERLAY_PCT,
            "llm_bias_threshold":  0.60,
            "rebal_threshold":     REBAL_THRESHOLD,
            "drawdown_exit_pct":   DRAWDOWN_EXIT_PCT,
            "vol_lookback_weeks":  VOL_LOOKBACK_WEEKS,
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
        help="Beta Overlay 模式（波动率目标制）：vol_target=16pct/实现波动率 + LLM叠加10pct(bias>=0.60)，EMA50或15pct回撤触发快速出场"
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
