"""
飞书通知器 - 解析分析输出文件并发送到飞书群机器人 Webhook

运行方式:
    python feishu_notifier.py --mode daily          # 当日汇总（黄金 + BTC）
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
import argparse
from datetime import datetime

import requests

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
FEISHU_WEBHOOK_URL = os.environ.get("FEISHU_WEBHOOK_URL", "")

OUTPUT_FILES = {
    "gold":  "gold_api_output.txt",
    "btc":   "btc_api_output.txt",
    "googl": "googl_api_output.txt",
    "nvda":  "nvda_api_output.txt",
    "amzn":  "amzn_api_output.txt",
}

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
            out.append("\\n")   # 转义换行
        elif in_string and ch == "\r":
            pass                # 丢弃回车
        else:
            out.append(ch)
    return "".join(out)


def parse_json_from_file(filepath: str) -> dict | None:
    if not os.path.exists(filepath):
        print(f"  [警告] 文件不存在: {filepath}")
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    content = "".join(lines).strip()

    # 1. 直接解析整个文件
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. 逐行扫描，找所有 ```json ... ``` 代码块（兼容截断文件）
    #    收集所有候选段落，优先从文件末尾往前试
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
                # 代码块正常结束
                candidates.append("".join(block_lines))
                in_block = False
                block_lines = []
            else:
                block_lines.append(line)

    # 文件截断时 block_lines 里有未关闭的块
    if in_block and block_lines:
        candidates.append("".join(block_lines))

    # 从最后一个候选往前，取第一个能解析成功的
    for candidate in reversed(candidates):
        text = candidate.strip()
        if not text:
            continue
        # 先直接解析，失败则尝试修复换行符，再失败则大括号计数
        for attempt in (text, _repair_json(text)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                pass
        # 截断的 JSON：大括号计数提取最完整部分后再修复
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

    # 最终降级：正则提取关键字段（应对 LLM 输出含未转义引号或文件截断）
    result = _regex_extract(content)
    if result:
        return result

    print(f"  [警告] 无法解析 JSON: {filepath}")
    return None


def _regex_extract(text: str) -> dict | None:
    """当 JSON 解析彻底失败时，用正则从原始文本提取关键字段，构造最小可用 dict。"""
    def _str(key: str) -> str | None:
        m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
        return m.group(1) if m else None

    def _num(key: str):
        m = re.search(rf'"{key}"\s*:\s*([0-9.]+)', text)
        return float(m.group(1)) if m else None

    def _bool(key: str) -> bool:
        m = re.search(rf'"{key}"\s*:\s*(true|false)', text, re.IGNORECASE)
        return m.group(1).lower() == "true" if m else False

    # 尝试提取公共字段
    action     = _str("action")
    bias_score = _num("bias_score")
    if not action or not bias_score:
        return None   # 连最基础字段都没有，放弃

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
    return {"long": "做多 [多]", "short": "做空 [空]", "no_trade": "观望 [空仓]"}.get(action, action)


def _trunc(s, n=120) -> str:
    if not s:
        return "N/A"
    s = str(s)
    return s[:n] + "..." if len(s) > n else s


def _row(label: str, value) -> list:
    """飞书 rich-text 单行：加粗 label + 普通 value"""
    return [
        {"tag": "text", "un_escape": True, "text": f"{label}: "},
        {"tag": "text", "text": str(value) if value is not None else "N/A"},
    ]


def _divider(title: str) -> list:
    return [{"tag": "text", "text": f"\n{'─' * 20} {title} {'─' * 20}"}]


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
        _divider("黄金 GOLD / PAXG"),
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

    lines.append(_row("综合判断", _trunc(just, 250)))
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

    # 利空风险汇总（最多3条高/中风险）
    risks = a.get("key_bearish_risks", [])
    high_risks = [r for r in risks if r.get("severity") in ("High", "Medium")][:3]
    risk_str = " | ".join(
        f"{r.get('risk_type','?')}({r.get('severity','')})" for r in high_risks
    ) or "无重大利空"

    lines = [
        _divider("比特币 BTC"),
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

    lines.append(_row("综合判断", _trunc(just, 250)))
    return lines


# ─────────────────────────────────────────────
# 科技股信号格式化
# ─────────────────────────────────────────────

def format_tech_block(data: dict, mode: str = "daily") -> list[list]:
    ticker  = data.get("stock_ticker", "STOCK")
    ms      = data.get("overall_market_sentiment", "N/A")
    qqq     = data.get("qqq_assessment", "")
    rates   = data.get("macro_rate_environment", "")
    earn_flag = data.get("earnings_risk_flag", False)
    earn_days = data.get("earnings_days_away", "N/A")

    a        = (data.get("asset_analysis") or [{}])[0]
    action   = a.get("action", "N/A")
    bias     = a.get("bias_score", "N/A")
    entry    = a.get("entry_zone", "N/A")
    target   = a.get("profit_target", "N/A")
    stop     = a.get("stop_loss", "N/A")
    rr       = a.get("risk_reward_ratio", "N/A")
    regime   = a.get("regime", "N/A")
    hold_wks = a.get("estimated_holding_weeks", "N/A")
    just     = a.get("justification", "")

    intel    = a.get("intelligence_analysis", {})
    analyst  = intel.get("analyst_consensus", "")
    valuation = intel.get("valuation_context", "")

    earn_str = f"距财报 {earn_days} 天  {'[高风险]' if earn_flag else '[平静期]'}"

    lines = [
        _divider(f"科技股 {ticker}"),
        _row("市场情绪", ms),
        _row("市场制度", regime),
        _row("财报风险", earn_str),
        _row("操作建议", _action_tag(action)),
        _row("偏向得分", bias),
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

    lines.append(_row("综合判断", _trunc(just, 250)))
    return lines


# ─────────────────────────────────────────────
# 组装飞书消息体
# ─────────────────────────────────────────────

def build_message(mode: str) -> dict:
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    title_map = {
        "daily":  f"每日金融分析  {datetime.now().strftime('%Y-%m-%d')}",
        "weekly": f"每周金融汇总  {datetime.now().strftime('%Y 第%W周')}",
    }
    title = title_map.get(mode, "金融分析报告")

    content: list[list] = [
        [{"tag": "text", "text": f"生成时间: {today}"}],
    ]

    # 黄金
    gold_data = parse_json_from_file(OUTPUT_FILES["gold"])
    if gold_data:
        content.extend(format_gold_block(gold_data, mode))
    else:
        content.append([{"tag": "text", "text": "\n[黄金] 数据文件缺失，请检查 gold_api_output.txt"}])

    # BTC
    btc_data = parse_json_from_file(OUTPUT_FILES["btc"])
    if btc_data:
        content.extend(format_btc_block(btc_data, mode))
    else:
        content.append([{"tag": "text", "text": "\n[BTC] 数据文件缺失，请检查 btc_api_output.txt"}])

    # 科技股（GOOGL / NVDA / AMZN）
    for key, label in [("googl", "GOOGL"), ("nvda", "NVDA"), ("amzn", "AMZN")]:
        tech_data = parse_json_from_file(OUTPUT_FILES[key])
        if tech_data:
            content.extend(format_tech_block(tech_data, mode))
        else:
            content.append([{"tag": "text", "text": f"\n[{label}] 数据文件缺失，请检查 {OUTPUT_FILES[key]}"}])

    # 尾部
    content.append([{"tag": "text", "text": "\n以上为 LLM 生成的分析建议，仅供参考，不构成投资建议。"}])

    return {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": title,
                    "content": content,
                }
            }
        },
    }


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
        # 飞书成功码：StatusCode=0 或 code=0
        if result.get("StatusCode") == 0 or result.get("code") == 0:
            print("飞书消息发送成功")
            return True
        else:
            print(f"飞书消息发送失败: {result}")
            return False
    except Exception as e:
        print(f"飞书发送异常: {e}")
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

    if args.mode == "test":
        msg = {
            "msg_type": "text",
            "content": {"text": f"金融分析系统连接测试 OK\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"},
        }
    else:
        msg = build_message(args.mode)

    send_to_feishu(msg, webhook_url)


if __name__ == "__main__":
    main()
