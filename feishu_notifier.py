"""
飞书通知器 - 解析分析输出文件并发送到飞书群机器人 Webhook

运行方式:
    python feishu_notifier.py --mode daily          # 当日汇总（所有 daily_scan=True 资产）
    python feishu_notifier.py --mode weekly         # 每周汇总（更详细）
    python feishu_notifier.py --mode test           # 发送测试消息，验证 Webhook 连通性

环境变量:
    FEISHU_WEBHOOK_URL  飞书群机器人 Webhook 地址（必填）
    或通过 --webhook 参数传入

依赖:
    pip install requests
"""

import json
import os
import re
import time
import argparse
from datetime import datetime

import requests

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
FEISHU_WEBHOOK_URL = os.environ.get("FEISHU_WEBHOOK_URL", "")

# 动态读取资产注册表，无需手动维护路径
try:
    from assets_config import ASSET_UNIVERSE, get_daily_assets
    _ASSET_CONFIG_AVAILABLE = True
except ImportError:
    _ASSET_CONFIG_AVAILABLE = False
    ASSET_UNIVERSE = {}

# 推送分组顺序（按资产类别划分，每批发一条飞书消息）
# 顺序影响推送顺序，可随意调整
PUSH_GROUPS = [
    ("大宗商品 & 加密货币", ["GOLD", "SILVER", "COPPER", "RARE_EARTH", "OIL", "BTC"]),
    ("Mag-7 科技股",        ["GOOGL", "MSFT", "NVDA", "AAPL", "META", "AMZN", "TSLA"]),
    ("半导体 & 能源",       ["AMD", "QCOM", "INTC", "DELL", "XOM", "HYNIX"]),
]

# ─────────────────────────────────────────────
# JSON 解析（兼容 markdown code block 格式）
# ─────────────────────────────────────────────

def _repair_json(text: str) -> str:
    """修复 LLM 输出中常见的 JSON 错误：字符串内的裸换行符。"""
    in_string = False
    escaped = False
    out = []
    for ch in text:
        if escaped:
            escaped = False
            out.append(ch)
        elif ch == "\\":
            escaped = True
            out.append(ch)
        elif ch == '"':
            in_string = not in_string
            out.append(ch)
        elif in_string and ch == "\n":
            out.append("\\n")
        elif in_string and ch == "\r":
            pass
        else:
            out.append(ch)
    return "".join(out)


def parse_json_from_file(filepath: str) -> dict | None:
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    content = "".join(lines).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    candidates = []
    in_block = False
    block_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not in_block and stripped == "```json":
            in_block = True
            block_lines = []
        elif in_block:
            if stripped == "```":
                candidates.append("".join(block_lines))
                in_block = False
                block_lines = []
            else:
                block_lines.append(line)

    if in_block and block_lines:
        candidates.append("".join(block_lines))

    for candidate in reversed(candidates):
        text = candidate.strip()
        if not text:
            continue
        for attempt in (text, _repair_json(text)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                pass
        start = text.find("{")
        if start == -1:
            continue
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start: i + 1]
                    try:
                        return json.loads(_repair_json(chunk))
                    except json.JSONDecodeError:
                        break

    result = _regex_extract(content)
    if result:
        return result

    return None


def _regex_extract(text: str) -> dict | None:
    def _str(key: str) -> str | None:
        m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
        return m.group(1) if m else None

    def _num(key: str):
        m = re.search(rf'"{key}"\s*:\s*([0-9.]+)', text)
        return float(m.group(1)) if m else None

    def _bool(key: str) -> bool:
        m = re.search(rf'"{key}"\s*:\s*(true|false)', text, re.IGNORECASE)
        return m.group(1).lower() == "true" if m else False

    action     = _str("action")
    bias_score = _num("bias_score")
    if not action or not bias_score:
        return None

    result: dict = {
        "period": _str("period") or "Unknown",
        "overall_market_sentiment": _str("overall_market_sentiment") or "N/A",
        "dxy_assessment": _str("dxy_assessment") or "",
        "btc_cycle_regime": _str("btc_cycle_regime") or "",
        "macro_environment": _str("macro_environment") or "",
        "stock_ticker": _str("stock_ticker") or "",
        "qqq_assessment": _str("qqq_assessment") or "",
        "macro_rate_environment": _str("macro_rate_environment") or "",
        "earnings_risk_flag": _bool("earnings_risk_flag"),
        "earnings_days_away": int(_num("earnings_days_away") or 0),
        "sentiment_summary": {
            "fear_greed_index": _num("fear_greed_index") or "N/A",
            "fear_greed_classification": _str("fear_greed_classification") or "N/A",
            "funding_rate_current": _num("funding_rate_current"),
        },
        "asset_analysis": [{
            "asset":                _str("asset") or "N/A",
            "regime":               _str("regime") or "N/A",
            "action":               action,
            "bias_score":           bias_score,
            "entry_zone":           _str("entry_zone") or "N/A",
            "profit_target":        _num("profit_target"),
            "stop_loss":            _num("stop_loss"),
            "risk_reward_ratio":    _num("risk_reward_ratio"),
            "position_sizing":      _str("position_sizing") or "N/A",
            "estimated_holding_weeks": _num("estimated_holding_weeks"),
            "justification":        _str("justification") or "(正则降级，详情查看输出文件)",
            "cycle_context":        _str("cycle_context") or "",
            "macro_catalyst":       _str("macro_catalyst") or "",
            "technical_setup":      _str("technical_setup") or "",
            "intelligence_analysis": {
                "analyst_consensus": _str("analyst_consensus") or "",
                "valuation_context": _str("valuation_context") or "",
            },
        }],
    }
    return result


# ─────────────────────────────────────────────
# 格式化工具
# ─────────────────────────────────────────────

def _action_tag(action: str) -> str:
    return {"long": "做多 ▲", "short": "做空 ▼", "no_trade": "观望 —"}.get(action, action)


def _action_emoji(action: str) -> str:
    return {"long": "▲做多", "short": "▼做空", "no_trade": "— 观望"}.get(action, action)


def _trunc(s, n=120) -> str:
    if not s:
        return "N/A"
    s = str(s)
    return s[:n] + "..." if len(s) > n else s


def _row(label: str, value) -> list:
    return [
        {"tag": "text", "un_escape": True, "text": f"{label}: "},
        {"tag": "text", "text": str(value) if value is not None else "N/A"},
    ]


def _divider(title: str) -> list:
    return [{"tag": "text", "text": f"\n{'─' * 18} {title} {'─' * 18}"}]


def _get_asset_output_file(asset_key: str) -> str:
    """从 assets_config 获取资产输出文件路径，找不到则按规则猜测。"""
    if _ASSET_CONFIG_AVAILABLE and asset_key in ASSET_UNIVERSE:
        return ASSET_UNIVERSE[asset_key].get("output_file", f"outputs/{asset_key.lower()}_api_output.txt")
    return f"outputs/{asset_key.lower()}_api_output.txt"


def _get_asset_type(asset_key: str) -> str:
    if _ASSET_CONFIG_AVAILABLE and asset_key in ASSET_UNIVERSE:
        return ASSET_UNIVERSE[asset_key].get("type", "equity")
    return "equity"


def _get_asset_description(asset_key: str) -> str:
    if _ASSET_CONFIG_AVAILABLE and asset_key in ASSET_UNIVERSE:
        cfg = ASSET_UNIVERSE[asset_key]
        return cfg.get("description", cfg.get("ticker", asset_key))
    return asset_key


# ─────────────────────────────────────────────
# 黄金信号格式化
# ─────────────────────────────────────────────

def format_gold_block(data: dict, mode: str = "daily") -> list[list]:
    a   = (data.get("asset_analysis") or [{}])[0]
    ms  = data.get("overall_market_sentiment", "N/A")
    dxy = data.get("dxy_assessment", "N/A")

    action  = a.get("action", "N/A")
    bias    = a.get("bias_score", "N/A")
    entry   = a.get("entry_zone", "N/A")
    target  = a.get("profit_target", "N/A")
    stop    = a.get("stop_loss", "N/A")
    rr      = a.get("risk_reward_ratio", "N/A")
    regime  = a.get("regime", "N/A")
    macro_c = a.get("macro_catalyst", "")
    tech    = a.get("technical_setup", "")
    just    = a.get("justification", "")

    lines = [
        _divider("GOLD 黄金"),
        _row("市场情绪", ms),
        _row("市场制度", regime),
        _row("操作建议", _action_tag(action)),
        _row("偏向得分", bias),
        _row("入场区间", entry),
        _row("止盈目标", target),
        _row("止损位置", stop),
        _row("风险回报", f"R:R = {rr}"),
    ]

    if mode == "weekly":
        lines += [
            _row("DXY 解读", _trunc(dxy, 100)),
            _row("宏观逻辑", _trunc(macro_c, 150)),
            _row("技术面", _trunc(tech, 150)),
        ]

    lines.append(_row("综合判断", _trunc(just, 200)))
    return lines


# ─────────────────────────────────────────────
# BTC 信号格式化
# ─────────────────────────────────────────────

def format_btc_block(data: dict, mode: str = "daily") -> list[list]:
    a       = (data.get("asset_analysis") or [{}])[0]
    macro_e = data.get("macro_environment", "N/A")
    cycle   = data.get("btc_cycle_regime", "N/A")
    sent    = data.get("sentiment_summary", {})
    fg      = sent.get("fear_greed_index", "N/A")
    fg_cls  = sent.get("fear_greed_classification", "N/A")
    funding = sent.get("funding_rate_current", "N/A")

    action   = a.get("action", "N/A")
    bias     = a.get("bias_score", "N/A")
    entry    = a.get("entry_zone", "N/A")
    target   = a.get("profit_target", "N/A")
    stop     = a.get("stop_loss", "N/A")
    rr       = a.get("risk_reward_ratio", "N/A")
    position = a.get("position_sizing", "N/A")
    cycle_c  = a.get("cycle_context", "")
    just     = a.get("justification", "")

    risks = a.get("key_bearish_risks", [])
    high_risks = [r for r in risks if r.get("severity") in ("High", "Medium")][:3]
    risk_str = " | ".join(
        f"{r.get('risk_type','?')}({r.get('severity','')})" for r in high_risks
    ) or "无重大利空"

    lines = [
        _divider("BTC 比特币"),
        _row("减半周期", cycle),
        _row("宏观环境", macro_e),
        _row("恐惧贪婪", f"{fg} - {fg_cls}"),
        _row("资金费率", funding),
        _row("操作建议", _action_tag(action)),
        _row("仓位建议", position),
        _row("偏向得分", bias),
        _row("入场区间", entry),
        _row("止盈目标", target),
        _row("止损位置", stop),
        _row("风险回报", f"R:R = {rr}"),
        _row("主要利空", risk_str),
    ]

    if mode == "weekly":
        lines.append(_row("周期背景", _trunc(cycle_c, 200)))

    lines.append(_row("综合判断", _trunc(just, 200)))
    return lines


# ─────────────────────────────────────────────
# 通用资产信号格式化（股票 / ETF / 大宗商品 ETF）
# ─────────────────────────────────────────────

def format_asset_block(asset_key: str, data: dict, mode: str = "daily") -> list[list]:
    ticker    = data.get("stock_ticker", "") or data.get("asset_ticker", "") or asset_key
    ms        = data.get("overall_market_sentiment", "N/A")
    qqq       = data.get("qqq_assessment", "")
    rates     = data.get("macro_rate_environment", "")
    earn_flag = data.get("earnings_risk_flag", False)
    earn_days = data.get("earnings_days_away", None)

    a        = (data.get("asset_analysis") or [{}])[0]
    action   = a.get("action", "N/A")
    bias     = a.get("bias_score", "N/A")
    entry    = a.get("entry_zone", "N/A")
    target   = a.get("profit_target", "N/A")
    stop     = a.get("stop_loss", "N/A")
    rr       = a.get("risk_reward_ratio", "N/A")
    regime   = a.get("regime", "N/A")
    hold_wks = a.get("estimated_holding_weeks", "N/A")
    pos      = a.get("position_size_pct", a.get("position_sizing", "N/A"))
    just     = a.get("justification", "")

    intel     = a.get("intelligence_analysis", {})
    analyst   = intel.get("analyst_consensus", "")
    valuation = intel.get("valuation_context", "")

    desc = _get_asset_description(asset_key)
    label = f"{ticker}" if ticker == desc else f"{ticker} {desc}"

    lines = [
        _divider(label),
        _row("市场情绪", ms),
        _row("市场制度", regime),
    ]

    if earn_days is not None:
        earn_str = f"距财报 {earn_days} 天  {'[高风险]' if earn_flag else '[平静期]'}"
        lines.append(_row("财报风险", earn_str))

    lines += [
        _row("操作建议", _action_tag(action)),
        _row("偏向得分", bias),
        _row("仓位建议", pos),
        _row("预计持仓", f"{hold_wks} 周"),
        _row("入场区间", entry),
        _row("止盈目标", target),
        _row("止损位置", stop),
        _row("风险回报", f"R:R = {rr}"),
    ]

    if mode == "weekly":
        lines += [
            _row("QQQ 背景", _trunc(qqq, 120)),
            _row("利率环境", _trunc(rates, 100)),
            _row("分析师共识", _trunc(analyst, 120)),
            _row("估值背景", _trunc(valuation, 120)),
        ]

    lines.append(_row("综合判断", _trunc(just, 200)))
    return lines


# ─────────────────────────────────────────────
# 汇总信号表（第一条消息）
# ─────────────────────────────────────────────

def build_summary_message(mode: str, all_results: list[dict]) -> dict:
    """构造第一条消息：信号汇总表"""
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    title_map = {
        "daily":  f"每日金融分析  {datetime.now().strftime('%Y-%m-%d')}",
        "weekly": f"每周金融汇总  {datetime.now().strftime('%Y 第%W周')}",
    }
    title = title_map.get(mode, "金融分析报告")

    long_list, watch_list, short_list, fail_list = [], [], [], []
    for item in all_results:
        key    = item["key"]
        action = item.get("action", "?")
        bias   = item.get("bias", "?")
        rr     = item.get("rr", "—")
        regime = item.get("regime", "?")
        if item.get("failed"):
            fail_list.append(key)
            continue
        rr_str = f"R:R={rr}" if rr not in (None, "N/A", "?", "—") else "—"
        bias_str = f"{bias:.2f}" if isinstance(bias, float) else str(bias)
        line = f"{key}({bias_str},{rr_str})"
        if action == "long":
            long_list.append(line)
        elif action == "short":
            short_list.append(line)
        else:
            watch_list.append(line)

    content: list[list] = [
        [{"tag": "text", "text": f"生成时间: {today}  |  资产总数: {len(all_results)}"}],
        [{"tag": "text", "text": f"\n▲ 做多 ({len(long_list)}): " + ("  ".join(long_list) if long_list else "无")}],
        [{"tag": "text", "text": f"▼ 做空 ({len(short_list)}): " + ("  ".join(short_list) if short_list else "无")}],
        [{"tag": "text", "text": f"— 观望 ({len(watch_list)}): " + ("  ".join(watch_list) if watch_list else "无")}],
    ]
    if fail_list:
        content.append([{"tag": "text", "text": f"⚠ 解析失败: {', '.join(fail_list)}"}])

    content.append([{"tag": "text", "text": "\n详情见后续消息 ↓"}])

    return {
        "msg_type": "post",
        "content": {"post": {"zh_cn": {"title": title, "content": content}}},
    }


# ─────────────────────────────────────────────
# 分组详情消息
# ─────────────────────────────────────────────

def build_group_message(group_name: str, asset_keys: list[str],
                        mode: str, date_str: str) -> dict:
    """构造某一分组的详情消息"""
    content: list[list] = []

    for asset_key in asset_keys:
        fpath = _get_asset_output_file(asset_key)
        data  = parse_json_from_file(fpath)

        if data is None:
            content.append([{"tag": "text", "text": f"\n[{asset_key}] 数据文件缺失或解析失败: {fpath}"}])
            continue

        asset_type = _get_asset_type(asset_key)
        if asset_key == "GOLD":
            content.extend(format_gold_block(data, mode))
        elif asset_key == "BTC":
            content.extend(format_btc_block(data, mode))
        else:
            content.extend(format_asset_block(asset_key, data, mode))

    if not content:
        return None

    content.append([{"tag": "text", "text": "\n以上为 LLM 生成分析，仅供参考，不构成投资建议。"}])

    return {
        "msg_type": "post",
        "content": {"post": {"zh_cn": {
            "title": f"{group_name}  {date_str}",
            "content": content,
        }}},
    }


# ─────────────────────────────────────────────
# 收集所有资产结果（用于汇总表）
# ─────────────────────────────────────────────

def collect_all_results() -> list[dict]:
    """遍历所有 daily_scan=True 资产，提取核心字段用于汇总表。"""
    results = []
    if _ASSET_CONFIG_AVAILABLE:
        asset_keys = [k for k, cfg in ASSET_UNIVERSE.items() if cfg.get("daily_scan", False)]
    else:
        # 兜底：遍历所有已知分组
        asset_keys = []
        for _, keys in PUSH_GROUPS:
            asset_keys.extend(keys)

    for key in asset_keys:
        fpath = _get_asset_output_file(key)
        data  = parse_json_from_file(fpath)
        if data is None:
            results.append({"key": key, "failed": True})
            continue
        aa     = (data.get("asset_analysis") or [{}])[0]
        action = aa.get("action", "?")
        bias   = aa.get("bias_score", "?")
        rr     = aa.get("risk_reward_ratio", None)
        regime = aa.get("regime", "?")
        results.append({"key": key, "action": action, "bias": bias, "rr": rr, "regime": regime})
    return results


# ─────────────────────────────────────────────
# 发送
# ─────────────────────────────────────────────

def send_to_feishu(message: dict, webhook_url: str) -> bool:
    try:
        resp = requests.post(
            webhook_url,
            json=message,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        result = resp.json()
        if result.get("StatusCode") == 0 or result.get("code") == 0:
            print("  飞书消息发送成功")
            return True
        else:
            print(f"  飞书消息发送失败: {result}")
            return False
    except Exception as e:
        print(f"  飞书发送异常: {e}")
        return False


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="飞书金融分析通知器")
    parser.add_argument(
        "--mode",
        choices=["daily", "weekly", "test"],
        default="daily",
        help="消息模式: daily=每日简报  weekly=每周汇总  test=连通性测试",
    )
    parser.add_argument(
        "--webhook",
        default="",
        help="飞书 Webhook URL（优先级高于环境变量 FEISHU_WEBHOOK_URL）",
    )
    args = parser.parse_args()

    webhook_url = args.webhook or FEISHU_WEBHOOK_URL
    if not webhook_url:
        print("错误: 请设置环境变量 FEISHU_WEBHOOK_URL 或通过 --webhook 参数传入")
        print("示例: export FEISHU_WEBHOOK_URL='https://open.feishu.cn/open-apis/bot/v2/hook/xxx'")
        return

    # ── 连通性测试 ──
    if args.mode == "test":
        msg = {
            "msg_type": "text",
            "content": {"text": f"金融分析系统连接测试 OK\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"},
        }
        send_to_feishu(msg, webhook_url)
        return

    # ── 收集所有资产结果 ──
    print("正在收集所有资产信号...")
    all_results = collect_all_results()

    date_str = datetime.now().strftime("%Y-%m-%d")

    # ── 第1条：汇总表 ──
    print("[1/N] 发送信号汇总表...")
    summary_msg = build_summary_message(args.mode, all_results)
    send_to_feishu(summary_msg, webhook_url)
    time.sleep(1)  # 避免飞书频率限制

    # ── 后续：按分组发详情 ──
    # 只推送有资产输出文件的分组（过滤掉所有资产都未跑的组）
    total = len(PUSH_GROUPS)
    for idx, (group_name, asset_keys) in enumerate(PUSH_GROUPS, start=2):
        # 只保留实际存在输出文件的资产
        available = [k for k in asset_keys if os.path.exists(_get_asset_output_file(k))]
        if not available:
            print(f"[{idx}/{total+1}] {group_name} — 无输出文件，跳过")
            continue

        print(f"[{idx}/{total+1}] 发送 {group_name} ({len(available)} 个资产)...")
        msg = build_group_message(group_name, available, args.mode, date_str)
        if msg:
            send_to_feishu(msg, webhook_url)
            time.sleep(1.5)  # 飞书 Webhook 频率限制约 5条/秒

    print("飞书推送全部完成。")


if __name__ == "__main__":
    main()
