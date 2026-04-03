"""
lumibot_nvda_backtest.py — v5 全面修复版

修复点（v5，相比 v4）：
  1. [修复] Bracket 平仓盈亏判断：改用开仓时保存的止盈/止损价比对，
            不再依赖平仓后的 current_price（跳空时 current_price 与成交价偏差大）
  2. [修复] LLM 幻觉价格字段：所有 profit_target / stop_loss 统一走
            _safe_float() 解析，兼容 "145.5 (approx)" / "~145" 等非标格式，
            解析失败视为 MISSING_LEVELS 拒绝入场，不崩溃
  3. [优化] ATR 计算：只取 daily_df 最后 28 行，避免对全量历史做无用 rolling
  4. [修复] 双模型价格取舍：以 Claude 确认模型的 profit_target / stop_loss 为准，
            R:R 用 Claude 报价验证，避免 DeepSeek 激进目标被盲目执行
  5. [修复] 冷却期 off-by-one：止损当天检测后立刻就扣了 1 天；
            改用 STOP_COOLDOWN + 1 写入，保证完整 STOP_COOLDOWN 个交易日不入场
  6. [改进] 入场门槛分层：
            高置信（DeepSeek bias ≥ 0.70）→ 快速通道，跳过 Claude 确认
            中等置信（bias 0.50~0.69）→ 仍需 Claude 确认（门槛降至 bias ≥ 0.40）
            R:R 门槛从 2.0 降至 1.8，ATR 止损校验从 1.0×ATR 降至 0.8×ATR

运行方式：
  python3 lumibot_nvda_backtest.py

可选环境变量：
  BACKTEST_MODEL=deepseek-reasoner   # 主模型（默认）
  CONFIRM_MODEL=claude-sonnet-4-6    # 确认模型（默认）
  ALIYUN_API_KEY=sk-xxx
  ANTHROPIC_API_KEY=sk-xxx
"""

import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from typing import Optional

import pandas as pd
from openai import OpenAI

from lumibot.strategies import Strategy
from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities.trading_fee import TradingFee

from tech_backtest_engine import (
    build_blind_prompt,
    call_api,
    _build_system_prompt,
    fetch_data_up_to,
    fetch_macro_for_date,
    prefetch_all_data,
    parse_signal,
)

# ─────────────────────────────────────────────
# 回测配置
# ─────────────────────────────────────────────

TICKER            = "NVDA"
PRIMARY_MODEL     = os.environ.get("BACKTEST_MODEL", "deepseek-reasoner")
CONFIRM_MODEL     = os.environ.get("CONFIRM_MODEL",  "claude-sonnet-4-6")
EVAL_DAYS         = 22       # 最长持仓天数
COMMISSION_PCT    = 0.001    # 0.1% / 边
STOP_COOLDOWN     = 5        # 止损后完整冷却天数（off-by-one 已修复，内部写 +1）
EVAL_INTERVAL     = 2        # 每隔几个交易日触发一次 LLM（1=每天，2=隔天，平衡速度与机会）
MAX_POSITION_SIZE = 0.50     # 仓位上限（ATR 定仓的兜底 cap）
RISK_PER_TRADE    = 0.02     # 每笔最大亏损占组合比例
MIN_ATR_MULT      = 0.8      # 止损距离 ≥ 0.8×ATR-14（原 1.0，略放宽）
MIN_RR            = 1.8      # R:R 最低门槛（原 2.0，略放宽）
HIGH_CONVICTION_BIAS = 0.70  # ≥ 此阈值跳过 Claude 确认（快速通道）
CONFIRM_MIN_BIAS  = 0.40     # Claude 确认最低 bias（原 0.50，放宽）

BACKTEST_START = datetime(2025, 1, 1)
BACKTEST_END   = datetime(2025, 12, 31)

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY",
                                    "sk-6BV9Xfa9AJ09pkt0AHFPQtZUtlM28pCOnon6ArdIJW1fVyDP")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL",
                                    "https://api.openai-proxy.org/anthropic")


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    """
    鲁棒价格解析，兼容 LLM 幻觉格式：
      "145.5 (approx)"  →  145.5
      "~145"            →  145.0
      None / ""         →  None
    返回 None 代表解析失败（调用方按 MISSING_LEVELS 处理）。
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # 去除注释文字：取第一个数字序列（包含小数点和负号）
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def _compute_atr14(daily_df: pd.DataFrame) -> float:
    """
    计算 ATR-14，只用最近 28 行，避免全量 rolling 浪费。
    """
    tail = daily_df.tail(28)
    high  = tail["High"].squeeze().astype(float)
    low   = tail["Low"].squeeze().astype(float)
    close = tail["Close"].squeeze().astype(float)
    prev  = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev).abs(),
                    (low  - prev).abs()], axis=1).max(axis=1)
    val = tr.rolling(14).mean().iloc[-1]
    return float(val) if not pd.isna(val) else 0.0


def _call_claude_confirm(prompt: str, system_prompt: str) -> dict:
    """
    Claude 二次确认信号。走 openai-proxy.org OpenAI 兼容接口。
    失败返回 {} → 视为确认失败，不入场。
    """
    base = ANTHROPIC_BASE_URL.rstrip("/")
    if base.endswith("/anthropic"):
        base = base[: -len("/anthropic")] + "/v1"

    client = OpenAI(api_key=ANTHROPIC_API_KEY, base_url=base)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=CONFIRM_MODEL,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt},
                ],
            )
            return parse_signal(resp.choices[0].message.content)
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = 60 * (attempt + 1)
                print(f"  [限速/Claude] 等待 {wait}s (attempt {attempt+1}/3)...")
                time.sleep(wait)
            else:
                print(f"  [API错误/Claude] attempt {attempt+1}/3: {e}")
                if attempt < 2:
                    time.sleep(10)
    return {}


# ─────────────────────────────────────────────
# Lumibot 策略类
# ─────────────────────────────────────────────

class LLMNVDAStrategy(Strategy):
    """
    Lumibot NVDA LLM 摆动交易策略 v5。

    入场分层：
    - DeepSeek bias ≥ 0.70  → 快速通道（跳过 Claude）
    - DeepSeek bias 0.50~0.69 → 需要 Claude bias ≥ 0.40 确认
    价格以 Claude 确认模型的目标价/止损为准；快速通道使用 DeepSeek 价格。
    """

    def initialize(self):
        self.sleeptime = "1D"
        self._cooldown         = 0     # 剩余冷却天数（0 = 可入场）
        self._eval_counter     = 0     # 隔天评估计数器
        self._hold_counter     = 0     # 当前持仓天数
        self._was_in_position  = False
        self._entry_price      = None
        self._tp_price         = None  # 修复1：开仓时记录止盈价
        self._sl_price         = None  # 修复1：开仓时记录止损价

        self.set_market("NYSE")
        self.asset         = self.create_asset(TICKER, asset_type="stock")
        self.system_prompt = _build_system_prompt(TICKER)

        print(f"\n[初始化] 预加载 {TICKER} 历史数据缓存...")
        prefetch_all_data(
            TICKER,
            BACKTEST_START.strftime("%Y-%m-%d"),
            BACKTEST_END.strftime("%Y-%m-%d"),
            lookback=200,
            eval_days=EVAL_DAYS,
        )
        print(
            f"[初始化] {TICKER} 策略就绪\n"
            f"  主模型={PRIMARY_MODEL}  确认模型={CONFIRM_MODEL}\n"
            f"  高置信快速通道 bias ≥ {HIGH_CONVICTION_BIAS}\n"
            f"  确认模型最低 bias = {CONFIRM_MIN_BIAS}\n"
            f"  区间={BACKTEST_START.date()}~{BACKTEST_END.date()}\n"
        )

    # ------------------------------------------------------------------
    def on_trading_iteration(self):

        position = self.get_position(self.asset)
        currently_in_position = position is not None and abs(position.quantity) > 0

        # ── A. 持仓中：递增天数，检查 EVAL_DAYS 超时 ───────────────────
        if currently_in_position:
            self._hold_counter += 1

            if self._hold_counter >= EVAL_DAYS:
                current_price = self.get_last_price(self.asset) or 0.0
                pnl_pct = (
                    (current_price - self._entry_price) / self._entry_price * 100
                    if self._entry_price else 0.0
                )
                self.log_message(
                    f"[TIME_EXIT] 持仓 {self._hold_counter}天 达到上限 {EVAL_DAYS}，"
                    f"强制平仓  价={current_price:.2f}  pnl={pnl_pct:+.1f}%"
                )
                self.sell_all(cancel_open_orders=True)
                self._reset_position_state()
                # 超时盈利 → 立即重评；超时亏损 → 冷却（+1 修正 off-by-one）
                self._cooldown = 0 if pnl_pct >= 0 else (STOP_COOLDOWN + 1)
                return

        # ── B. 检测 bracket 自动平仓（止损 or 止盈）────────────────────
        if self._was_in_position and not currently_in_position:
            # 修复1：用开仓时记录的 TP/SL 判断，而非跳空后的 current_price
            current_price = self.get_last_price(self.asset) or 0.0
            exit_is_stop  = self._infer_exit_is_stop(current_price)

            if exit_is_stop:
                # 修复5：STOP_COOLDOWN + 1，保证完整 N 个交易日冷却
                self._cooldown = STOP_COOLDOWN + 1
                self._eval_counter = 0
                self.log_message(
                    f"[止损] bracket 平仓（推断）  "
                    f"entry={self._entry_price}  sl={self._sl_price}  tp={self._tp_price}  "
                    f"current={current_price:.2f}  → 冷却 {STOP_COOLDOWN} 天"
                )
            else:
                self._cooldown = 0
                self._eval_counter = 0   # 止盈后下一个评估日立即触发
                self.log_message(
                    f"[止盈/平] bracket 平仓（推断）  "
                    f"entry={self._entry_price}  tp={self._tp_price}  "
                    f"current={current_price:.2f}  → 立即可重评"
                )
            self._reset_position_state()

        self._was_in_position = currently_in_position

        # ── C. 持仓中跳过 ─────────────────────────────────────────────
        if currently_in_position:
            self.log_message(
                f"[SKIP 持仓中] hold={self._hold_counter}天/{EVAL_DAYS}天"
            )
            return

        # ── D. 冷却期倒数（修复5：先减再判，当天不交易）──────────────
        if self._cooldown > 0:
            self._cooldown -= 1
            self._eval_counter = 0   # 冷却结束后立即重评
            self.log_message(f"[SKIP 冷却中] 剩余 {self._cooldown} 天")
            return

        # ── D2. 隔天评估门控 ─────────────────────────────────────────
        self._eval_counter += 1
        if self._eval_counter % EVAL_INTERVAL != 1:
            return   # 非评估日，静默跳过

        # ── E. 获取数据 ──────────────────────────────────────────────
        ref_date = self.get_datetime().strftime("%Y-%m-%d")

        daily, weekly = fetch_data_up_to(TICKER, ref_date)
        if daily.empty or len(daily) < 30:
            self.log_message(f"[SKIP DATA] 数据不足  {ref_date}")
            return

        macro  = fetch_macro_for_date(ref_date, TICKER)
        prompt = build_blind_prompt(TICKER, daily, weekly, macro)
        if not prompt:
            self.log_message(f"[SKIP PROMPT] Prompt 构建失败  {ref_date}")
            return

        # 修复3：只用 tail(28) 计算 ATR-14
        atr14 = _compute_atr14(daily)

        # ── F. 主模型（DeepSeek R1）──────────────────────────────────
        primary_signal = call_api(prompt, PRIMARY_MODEL, self.system_prompt)
        if not primary_signal:
            self.log_message(f"[SKIP API_FAIL/主模型] {ref_date}")
            return

        primary_sig = self._extract_ticker_sig(primary_signal)
        if not primary_sig:
            self.log_message(f"[SKIP PARSE/主模型]  {ref_date}")
            return

        p_action = primary_sig.get("action", "no_trade")
        p_bias   = float(primary_sig.get("bias_score") or 0)

        self.log_message(
            f"[主模型] {ref_date}  action={p_action}  bias={p_bias:.2f}  "
            f"regime={primary_sig.get('regime')}"
        )

        if p_action != "long" or p_bias < 0.50:
            self.log_message(
                f"[SKIP 主模型no_trade] action={p_action}  bias={p_bias:.2f}  {ref_date}"
            )
            return

        # ── G. 分层入场：高置信快速通道 or Claude 确认 ────────────────
        use_fast_path = (p_bias >= HIGH_CONVICTION_BIAS)
        price_sig     = primary_sig   # 默认用主模型价格（快速通道）

        if use_fast_path:
            self.log_message(
                f"[快速通道] bias={p_bias:.2f} ≥ {HIGH_CONVICTION_BIAS}，跳过 Claude 确认  {ref_date}"
            )
            confirm_note = f"快速通道(bias={p_bias:.2f})"
        else:
            # 中等置信：需要 Claude 确认
            confirm_signal = _call_claude_confirm(prompt, self.system_prompt)
            if not confirm_signal:
                self.log_message(f"[SKIP API_FAIL/确认模型]  {ref_date}")
                return

            confirm_sig = self._extract_ticker_sig(confirm_signal)
            if not confirm_sig:
                self.log_message(f"[SKIP PARSE/确认模型]  {ref_date}")
                return

            c_action = confirm_sig.get("action", "no_trade")
            c_bias   = float(confirm_sig.get("bias_score") or 0)

            self.log_message(
                f"[确认模型] {ref_date}  action={c_action}  bias={c_bias:.2f}  "
                f"regime={confirm_sig.get('regime')}"
            )

            if c_action != "long" or c_bias < CONFIRM_MIN_BIAS:
                self.log_message(
                    f"[SKIP 确认模型拒绝] action={c_action}  bias={c_bias:.2f}  {ref_date}"
                )
                return

            # 修复4：以 Claude 的价格为准
            price_sig    = confirm_sig
            confirm_note = f"双模型确认(R1 bias={p_bias:.2f}, Claude bias={c_bias:.2f})"

        # ── H. 价格解析（修复2：_safe_float 防崩溃）────────────────────
        profit_target = _safe_float(price_sig.get("profit_target"))
        stop_loss     = _safe_float(price_sig.get("stop_loss"))

        if profit_target is None or stop_loss is None:
            self.log_message(
                f"[SKIP MISSING_LEVELS] target={price_sig.get('profit_target')}  "
                f"stop={price_sig.get('stop_loss')}  {ref_date}"
            )
            return

        if p_action == "short":
            self.log_message(f"[SKIP SHORT_FILTERED]  {ref_date}")
            return

        # ── I. 实时价格 + R:R 验证 ──────────────────────────────────
        current_price = self.get_last_price(self.asset)
        if current_price is None:
            self.log_message(f"[SKIP NO_PRICE]  {ref_date}")
            return

        risk   = current_price - stop_loss
        reward = profit_target - current_price

        if risk <= 0 or reward <= 0 or (reward / risk) < MIN_RR:
            rr = reward / risk if risk > 0 else 0
            self.log_message(
                f"[SKIP INVALID_RR] rr={rr:.2f} < {MIN_RR}  "
                f"price={current_price:.2f}  stop={stop_loss:.2f}  target={profit_target:.2f}  {ref_date}"
            )
            return

        # ── J. 止损 ATR 校验 ────────────────────────────────────────
        if atr14 > 0 and risk < MIN_ATR_MULT * atr14:
            self.log_message(
                f"[SKIP STOP_TOO_TIGHT] risk={risk:.2f} < {MIN_ATR_MULT}×ATR14={atr14:.2f}  {ref_date}"
            )
            return

        # ── K. ATR 风险定仓 ─────────────────────────────────────────
        risk_budget   = self.portfolio_value * RISK_PER_TRADE
        quantity_risk = int(risk_budget / risk) if risk > 0 else 0
        quantity_cap  = int(self.portfolio_value * MAX_POSITION_SIZE / current_price)
        quantity      = min(quantity_risk, quantity_cap)

        if quantity <= 0:
            self.log_message(
                f"[SKIP QTY=0] budget={risk_budget:.0f}  risk/sh={risk:.2f}  {ref_date}"
            )
            return

        actual_pct = quantity * current_price / self.portfolio_value

        # ── L. 提交 bracket 订单，保存止盈/止损价（修复1）─────────────
        order = self.create_order(
            self.asset,
            quantity,
            "buy",
            order_class="bracket",
            take_profit_price=round(profit_target, 2),
            stop_loss_price=round(stop_loss, 2),
        )
        self.submit_order(order)

        self._entry_price = current_price
        self._tp_price    = round(profit_target, 2)
        self._sl_price    = round(stop_loss, 2)
        self._hold_counter = 0

        self.log_message(
            f"[LONG ✓] x{quantity} @ {current_price:.2f}  "
            f"tp={self._tp_price}  sl={self._sl_price}  "
            f"rr={reward/risk:.2f}  atr14={atr14:.2f}  size={actual_pct:.0%}  "
            f"({ref_date})  [{confirm_note}]"
        )

    # ------------------------------------------------------------------
    def _infer_exit_is_stop(self, current_price: float) -> bool:
        """
        修复1：通过开仓时记录的 TP/SL 推断是止损还是止盈平仓。
        策略：current_price 距 SL 更近 → 判断为止损；距 TP 更近 → 判断为止盈。
        """
        if self._tp_price is None or self._sl_price is None:
            # 没有记录就退回 P&L 比较
            if self._entry_price:
                return current_price < self._entry_price
            return False

        dist_sl = abs(current_price - self._sl_price)
        dist_tp = abs(current_price - self._tp_price)
        return dist_sl <= dist_tp

    def _extract_ticker_sig(self, signal: dict) -> Optional[dict]:
        asset_list = signal.get("asset_analysis", [])
        return next(
            (x for x in asset_list if x.get("asset", "").upper() == TICKER),
            None,
        )

    def _reset_position_state(self):
        self._hold_counter    = 0
        self._entry_price     = None
        self._tp_price        = None
        self._sl_price        = None
        self._was_in_position = False


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Lumibot NVDA LLM 回测  v5 全面修复版")
    print(f"  区间         : {BACKTEST_START.date()} ~ {BACKTEST_END.date()}")
    print(f"  主模型       : {PRIMARY_MODEL}")
    print(f"  确认模型     : {CONFIRM_MODEL}")
    print(f"  高置信快通道 : bias ≥ {HIGH_CONVICTION_BIAS}（跳过 Claude）")
    print(f"  确认门槛     : Claude bias ≥ {CONFIRM_MIN_BIAS}")
    print(f"  持仓上限     : {EVAL_DAYS} 交易日")
    print(f"  评估间隔     : 每 {EVAL_INTERVAL} 个交易日（止盈/止损后立即重评）")
    print(f"  止损冷却     : {STOP_COOLDOWN} 天（完整天数，off-by-one 已修复）")
    print(f"  R:R 门槛     : {MIN_RR}")
    print(f"  ATR 止损校验 : ≥ {MIN_ATR_MULT}×ATR-14")
    print(f"  风险定仓     : {RISK_PER_TRADE:.0%}/笔 × 组合")
    print(f"  仓位上限     : {MAX_POSITION_SIZE:.0%}")
    print(f"  佣金         : {COMMISSION_PCT*100:.1f}% / 边")
    print("=" * 70 + "\n")

    commission = TradingFee(percent_fee=COMMISSION_PCT)

    results = LLMNVDAStrategy.backtest(
        YahooDataBacktesting,
        BACKTEST_START,
        BACKTEST_END,
        benchmark_asset="SPY",
        parameters={},
        show_plot=False,
        show_tearsheet=False,
        save_tearsheet=True,
        tearsheet_file="lumibot_nvda_tearsheet.html",
        buy_trading_fees=[commission],
        sell_trading_fees=[commission],
        quiet_logs=False,
        show_progress_bar=True,
    )

    print("\n" + "=" * 70)
    print("Lumibot NVDA LLM 回测 v5 完成")
    print("=" * 70)

    if results:
        key_metrics = [
            "total_return", "cagr", "max_drawdown",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "win_rate", "profit_factor", "number_of_trades",
        ]
        print()
        for key in key_metrics:
            print(f"  {key:<30}: {results.get(key, 'N/A')}")
        print(f"\n详细报告 → lumibot_nvda_tearsheet.html")
    else:
        print("  未返回结果（可能无信号触发或初始化失败）")
