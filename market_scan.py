"""
多资产横向扫描器
=============
两阶段分析流程：
  Stage 1 — 对配置中的每个资产分别调用对应的分析脚本，获取单资产 JSON 信号
  Stage 2 — 把所有信号汇总，再做一次 LLM 调用，输出：
              ① 宏观主题识别
              ② 板块排名（哪个板块/行业最强）
              ③ 个资产机会排名（Top N）
              ④ 相关性风险提示

运行方式：
    python market_scan.py                         # 扫描默认 quick 分组，只生成提示词
    python market_scan.py --group tech --api      # 扫描科技股并调用 API
    python market_scan.py --group all --api       # 全资产扫描（耗时较长）
    python market_scan.py --assets GOLD NVDA BTC --api  # 自定义资产列表
    python market_scan.py --group metals --api --model deepseek-reasoner

依赖：
    assets_config.py（本目录）
    各资产对应的分析脚本（gold_analysis.py / tech_stock_analysis.py / btc_analysis.py）
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import httpx
import urllib3
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from openai import OpenAI

from assets_config import ASSET_UNIVERSE, SCAN_GROUPS

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── API 配置（优先读取环境变量）──
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.openai-proxy.org/anthropic")
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY",  "sk-6BV9Xfa9AJ09pkt0AHFPQtZUtlM28pCOnon6ArdIJW1fVyDP")
ANTHROPIC_MODEL    = "claude-sonnet-4-6"

DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "sk-9574b3366dfd41178a5493d0f6af33c0")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

CLAUDE_MODELS   = {"claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"}
DEEPSEEK_MODELS = {"deepseek-reasoner", "deepseek-chat"}

SCAN_OUTPUT_FILE = "market_scan_output.json"
SCAN_REPORT_FILE = "market_scan_report.txt"


# ─────────────────────────────────────────────
# Stage 1：运行单个资产分析脚本
# ─────────────────────────────────────────────

def run_asset_script(asset_key: str, cfg: dict, use_api: bool, model: str) -> bool:
    """
    调用对应资产的分析脚本。
    use_api=True 时会追加 --api（以及 --model），让脚本直接写 output_file。
    返回 True 表示脚本正常退出。
    """
    script = cfg["script"]
    base_args = cfg.get("script_args", [])

    cmd = [sys.executable, script] + base_args
    if use_api:
        cmd += ["--api", "--model", model]

    print(f"\n[{asset_key}] 运行: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,   # 让子进程的 print 直接输出到终端
            timeout=300,
            cwd=Path(__file__).parent,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  [{asset_key}] 超时（300s），跳过")
        return False
    except Exception as e:
        print(f"  ⚠️  [{asset_key}] 执行失败: {e}")
        return False


def read_output_file(asset_key: str, cfg: dict) -> str:
    """读取该资产最新的 API 输出文件"""
    path = Path(__file__).parent / cfg["output_file"]
    if not path.exists():
        print(f"  ⚠️  [{asset_key}] 输出文件不存在: {path}")
        return ""
    return path.read_text(encoding="utf-8")


def parse_signal_from_output(raw_text: str) -> dict | None:
    """从资产分析输出中提取第一个完整 JSON 对象"""
    if not raw_text:
        return None

    # 1. 尝试直接解析
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    # 2. 提取 ```json … ``` 代码块
    m = re.search(r"```json\s*(.*?)```", raw_text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. 大括号匹配提取
    depth, start = 0, -1
    for i, ch in enumerate(raw_text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(raw_text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1

    return None


# ─────────────────────────────────────────────
# Stage 2：跨资产汇总 + 机会排名
# ─────────────────────────────────────────────

def build_scan_prompt(collected: list[dict]) -> str:
    """
    collected 是每个资产的汇总信息列表，每项包含：
      asset_key, description, sector, signal（解析后的 JSON，可能为 None）, raw_text
    """

    assets_section = ""
    for item in collected:
        key   = item["asset_key"]
        desc  = item["description"]
        sec   = item["sector"]
        sig   = item["signal"]

        if sig is None:
            assets_section += f"\n### {key}（{desc} | {sec}）\n> 信号解析失败或数据不可用\n"
            continue

        # 提取核心字段（兼容 gold / equity / btc 不同 schema）
        asset_analyses = sig.get("asset_analysis", [])
        first = asset_analyses[0] if asset_analyses else {}

        action      = first.get("action", "N/A")
        bias        = first.get("bias_score", "N/A")
        entry       = first.get("entry_zone", "N/A")
        sl          = first.get("stop_loss", "N/A")
        tp          = first.get("profit_target", "N/A")
        rr          = first.get("risk_reward_ratio", "N/A")
        regime      = first.get("regime", "N/A")
        macro_cat   = first.get("macro_catalyst", first.get("justification", "")[:200])
        sentiment   = sig.get("overall_market_sentiment", "N/A")

        assets_section += f"""
### {key}（{desc} | 板块: {sec}）
- **信号**: `{action}`  |  **bias_score**: {bias}  |  **市场状态**: {regime}
- **整体市场情绪**: {sentiment}
- **entry_zone**: {entry}  |  **stop_loss**: {sl}  |  **profit_target**: {tp}  |  **R:R**: {rr}
- **核心逻辑**: {macro_cat}
"""

    prompt = f"""你是一名资深宏观量化分析师，现在需要对以下 {len(collected)} 个资产的个股分析信号进行**横向比较和综合排名**。

今日日期：{datetime.now().strftime('%Y-%m-%d')}

---

## 各资产信号摘要

{assets_section}

---

## 分析任务

请基于上述所有信号，完成以下分析：

### 1. 宏观主题识别
当前市场的 1-2 个核心驱动主题是什么？（如：AI 算力超级周期、贵金属避险、大宗商品价格回升等）

### 2. 板块强弱排名
把上述资产按板块（Commodities/PreciousMetals、Technology/AI、Crypto 等）分组，
给出各板块的综合评分（强 / 中 / 弱），并说明核心逻辑。

### 3. 个资产机会排名（Top 5）
从所有 `action != no_trade` 的资产中，选出最强的 5 个机会，按机会质量从高到低排列，说明选择依据。

### 4. 相关性风险提示
识别哪些资产之间存在高度相关性（如 GOLD 和 SILVER），若同时持仓需注意集中度风险。

### 5. 观望资产
列出当前信号为 no_trade 或数据不可用的资产，简要说明等待的触发条件。

---

请严格按照以下 JSON 格式输出（不要输出 markdown 代码块以外的内容）：

```json
{{
  "scan_date": "{datetime.now().strftime('%Y-%m-%d')}",
  "macro_themes": [
    {{
      "theme": "<主题名称>",
      "description": "<100字以内描述>",
      "beneficiary_assets": ["<资产key列表>"]
    }}
  ],
  "sector_ranking": [
    {{
      "sector": "<板块名>",
      "strength": "Strong | Neutral | Weak",
      "assets": ["<资产key列表>"],
      "rationale": "<50字以内>"
    }}
  ],
  "top_opportunities": [
    {{
      "rank": 1,
      "asset": "<资产key>",
      "action": "long | short",
      "bias_score": <0.0-1.0>,
      "entry_zone": "<价格区间>",
      "stop_loss": <数字或null>,
      "profit_target": <数字或null>,
      "risk_reward_ratio": <数字或null>,
      "rationale": "<为什么排第X，核心逻辑，≤100字>"
    }}
  ],
  "correlation_risks": [
    {{
      "assets": ["<资产A>", "<资产B>"],
      "correlation_type": "<正相关/负相关/同受DXY影响等>",
      "risk_note": "<持仓风险说明>"
    }}
  ],
  "watchlist": [
    {{
      "asset": "<资产key>",
      "reason": "<no_trade原因>",
      "trigger_condition": "<什么情况下可以入场>"
    }}
  ]
}}
```
"""
    return prompt.strip()


def call_claude_api(prompt: str, model: str = ANTHROPIC_MODEL) -> str:
    print(f"\n[扫描汇总] 调用 Claude API（{model}）...")
    client = Anthropic(
        base_url=ANTHROPIC_BASE_URL,
        api_key=ANTHROPIC_API_KEY,
        http_client=httpx.Client(verify=False, timeout=180.0),
    )
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=8096,
                messages=[{"role": "user", "content": prompt}],
            )
            result = ""
            for block in msg.content:
                if hasattr(block, "text"):
                    result += block.text
            return result
        except Exception as e:
            if attempt < 2:
                wait = 20 * (attempt + 1)
                print(f"  [重试 {attempt+1}/3] {e}，{wait}s 后重试...")
                time.sleep(wait)
            else:
                raise


def call_deepseek_api(prompt: str, model: str) -> str:
    print(f"\n[扫描汇总] 调用 DeepSeek API（{model}）...")
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content or ""
            return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        except Exception as e:
            print(f"  第 {attempt+1} 次失败: {e}")
            if attempt < 2:
                time.sleep(5)
    return ""


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="多资产横向扫描器：批量运行分析 + 跨资产机会排名"
    )
    parser.add_argument(
        "--group", default="quick",
        help=f"预定义资产分组（{', '.join(SCAN_GROUPS.keys())}），默认 quick"
    )
    parser.add_argument(
        "--assets", nargs="+",
        help="自定义资产列表（覆盖 --group），例如 --assets GOLD NVDA BTC"
    )
    parser.add_argument(
        "--api", action="store_true",
        help="在运行单资产脚本和汇总分析时都调用 API（默认：仅生成提示词）"
    )
    parser.add_argument(
        "--skip-individual", action="store_true",
        help="跳过 Stage 1（单资产脚本），直接读取已有输出文件做汇总分析"
    )
    parser.add_argument(
        "--model", default=ANTHROPIC_MODEL,
        help=f"汇总分析使用的模型，默认 {ANTHROPIC_MODEL}"
    )
    args = parser.parse_args()

    # 确定要扫描的资产列表
    if args.assets:
        asset_keys = [k.upper() for k in args.assets]
        invalid = [k for k in asset_keys if k not in ASSET_UNIVERSE]
        if invalid:
            print(f"未知资产: {invalid}，可选: {list(ASSET_UNIVERSE.keys())}")
            sys.exit(1)
    else:
        asset_keys = SCAN_GROUPS.get(args.group, SCAN_GROUPS["quick"])

    print("=" * 60)
    print(f"市场扫描  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  模式: {'API' if args.api else '提示词'}")
    print(f"扫描资产 ({len(asset_keys)}): {', '.join(asset_keys)}")
    print("=" * 60)

    # ── Stage 1：运行各资产分析脚本 ──
    if not args.skip_individual:
        for key in asset_keys:
            cfg = ASSET_UNIVERSE[key]
            success = run_asset_script(key, cfg, use_api=args.api, model=args.model)
            if not success:
                print(f"  [{key}] 脚本未正常退出，将尝试读取已有输出文件")
    else:
        print("\n[跳过 Stage 1] 直接读取已有输出文件")

    # ── 收集各资产信号 ──
    collected = []
    for key in asset_keys:
        cfg = ASSET_UNIVERSE[key]
        raw = read_output_file(key, cfg)
        sig = parse_signal_from_output(raw) if raw else None
        collected.append({
            "asset_key":   key,
            "description": cfg["description"],
            "sector":      cfg["sector"],
            "signal":      sig,
            "raw_text":    raw,
        })
        status = "✅ 信号解析成功" if sig else "❌ 无法解析信号"
        print(f"  [{key}] {status}")

    valid_count = sum(1 for c in collected if c["signal"] is not None)
    print(f"\n有效信号: {valid_count}/{len(asset_keys)}")

    # 如果没有 API 模式，保存汇总提示词后退出
    scan_prompt = build_scan_prompt(collected)
    prompt_out = Path(__file__).parent / "market_scan_prompt.txt"
    prompt_out.write_text(scan_prompt, encoding="utf-8")
    print(f"\n[汇总提示词已保存] {prompt_out}")

    if not args.api:
        print("\n提示：使用 --api 参数可直接调用模型获取综合排名结果。")
        return

    # ── Stage 2：汇总分析 LLM 调用 ──
    model = args.model
    if model in DEEPSEEK_MODELS:
        scan_result = call_deepseek_api(scan_prompt, model)
    else:
        scan_result = call_claude_api(scan_prompt, model)

    # 保存原始输出
    report_path = Path(__file__).parent / SCAN_REPORT_FILE
    report_path.write_text(scan_result, encoding="utf-8")
    print(f"\n[扫描报告已保存] {report_path}")

    # 尝试解析 JSON 并保存结构化结果
    scan_json = parse_signal_from_output(scan_result)
    if scan_json:
        output_path = Path(__file__).parent / SCAN_OUTPUT_FILE
        output_path.write_text(
            json.dumps(scan_json, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"[结构化结果已保存] {output_path}")

        # 打印 Top 机会简报
        print("\n" + "=" * 60)
        print("  TOP 交易机会排名")
        print("=" * 60)
        for opp in scan_json.get("top_opportunities", []):
            print(
                f"  #{opp.get('rank')} {opp.get('asset'):8s} "
                f"| {opp.get('action'):5s} "
                f"| bias={opp.get('bias_score', 'N/A')} "
                f"| {opp.get('rationale', '')[:80]}"
            )

        print("\n  板块强弱")
        print("-" * 40)
        for sec in scan_json.get("sector_ranking", []):
            print(f"  {sec.get('strength'):8s} | {sec.get('sector')}")

    else:
        print("\n[警告] 无法解析汇总 JSON，原始报告已保存到文件，请手动查看。")
        print(scan_result[:2000])


if __name__ == "__main__":
    main()
