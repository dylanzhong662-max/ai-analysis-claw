# 大模型金融分析 — 项目说明文档

## 项目概述

本项目是一个**以大语言模型（LLM）为核心决策引擎的黄金期货交易信号系统**，包含两大功能：

1. **实时信号生成**（`gold_analysis.py`）：采集黄金期货最新行情与宏观数据，构建结构化提示词，通过 Claude 或 DeepSeek API 生成当日交易建议。
2. **历史回测验证**（`backtest_engine.py`）：对 LLM 的信号质量做历史模拟验证，输出胜率、盈亏比等绩效指标，并将结果反馈到下一次实时分析的提示词中，形成闭环。

---

## 核心设计理念

- **LLM 作为推理引擎**：不依赖传统量化模型，改由 LLM 读取结构化技术指标 + 宏观数据，模拟对冲基金分析师的推理过程，输出标准化 JSON 交易信号。
- **防时间泄漏（Anti-Leakage）**：回测时提示词不包含具体日期，避免模型利用训练集中的未来知识作弊。
- **风险约束硬编码**：所有信号必须满足 R:R ≥ 2.0、止损 ≥ 0.8×ATR-14，否则系统自动降级为 `no_trade`。
- **性能反馈闭环**：`gold_analysis.py` 会读取 `backtest_results/performance.csv`，将最近的胜率、连续亏损次数等指标注入提示词，动态调整模型决策阈值。

---

## 目录结构

```
大模型金融分析/
├── gold_analysis.py          # 实时信号生成脚本（主入口）
├── backtest_engine.py        # 历史回测引擎
├── backtest_prompts/         # 回测用盲化提示词（每个交易日一个 .txt）
│   └── YYYY-MM-DD.txt
├── backtest_responses/       # 手动回测时保存 LLM JSON 响应（用户自行维护）
│   └── YYYY-MM-DD.json
├── backtest_results/         # 回测输出（自动生成）
│   ├── signals.csv           # 逐笔信号与交易记录
│   └── performance.csv       # 汇总绩效指标
├── gold_prompt_output.txt    # 最新生成的实时提示词（自动覆盖）
└── gold_api_output.txt       # 最新 API 调用返回的分析结果（自动覆盖）
```

---

## 两个主脚本详解

### 1. `gold_analysis.py` — 实时信号生成

#### 数据流

```
yfinance (GC=F)
  ├── 日线 6个月 (1d)  ──┐
  └── 周线 1年   (1wk) ──┤
                          ├──> compute_indicators() ──> 技术指标
yfinance (宏观)            │
  ├── DX-Y.NYB (DXY)    ──┤
  ├── ^TNX (10Y收益率)   ──┤──> summarize_macro()   ──> 宏观摘要
  ├── ^VIX               ──┤
  └── SI=F (白银)        ──┘
                          │
backtest_results/          │
  └── performance.csv  ────┤──> load_perf_metrics() ──> 历史绩效反馈
                          │
                          └──> build_prompt() ──> 结构化提示词
                                                    │
                               ┌────────────────────┤
                               │                    │
                          保存为文件              call_claude_api()
                     gold_prompt_output.txt       call_deepseek_api()
                         (默认模式)                     │
                                               gold_api_output.txt
```

#### 技术指标（`compute_indicators`）

| 类别 | 指标 |
|------|------|
| 趋势 | EMA-20/50/200, MACD (12/26/9) |
| 震荡 | RSI-7, RSI-14, Stochastic %K/%D (14/3) |
| 波动 | ATR-3, ATR-14, Bollinger Bands (20, 2σ), %B, 带宽 |
| 趋势强度 | ADX, +DI, -DI (14) |
| 量价动量 | OBV, ROC-10, ROC-20 |

#### 运行方式

```bash
# 仅生成提示词文件（默认）
python gold_analysis.py

# 直接调用 Claude API 获取分析结果
python gold_analysis.py --api

# 使用 DeepSeek 模型
python gold_analysis.py --api --model deepseek-reasoner
python gold_analysis.py --api --model deepseek-chat
```

---

### 2. `backtest_engine.py` — 历史回测引擎

#### 三种运行模式

**模式一：生成提示词文件（无需 API Key）**
```bash
python backtest_engine.py --generate --start 2024-01-01 --end 2024-12-31 --step 5
```
- 按指定步长（默认每 5 个交易日）遍历历史日期
- 每个日期生成盲化提示词（不含日期）保存到 `backtest_prompts/`
- 用户手动粘贴到 Claude.ai，把 JSON 响应保存为 `backtest_responses/YYYY-MM-DD.json`

**模式二：评估已有响应（无需 API Key）**
```bash
python backtest_engine.py --evaluate
```
- 读取 `backtest_responses/` 下所有 JSON 文件
- 获取对应日期之后的真实价格，模拟交易，计算 P&L
- 输出绩效汇总到 `backtest_results/`

**模式三：全自动回测（需要 DeepSeek API Key）**
```bash
python backtest_engine.py --start 2024-01-01 --end 2024-12-31 --step 5
python backtest_engine.py --start 2025-01-01 --end 2025-12-31 --resume  # 断点续跑
```

#### 交易模拟规则（`simulate_trade`）

- 入场：信号日次日**开盘价**入场
- 持仓期：最长 `EVAL_DAYS = 15` 个交易日
- 出场逻辑（按优先级）：
  1. 当日 Low ≤ stop_loss → `STOP_LOSS`
  2. 当日 High ≥ profit_target → `TAKE_PROFIT`
  3. 超过 15 天 → `TIMEOUT`，按最后收盘价结算
- 入场有效性校验：实际入场价下 R:R < 1.5 → `INVALID_RR`，不计入成交

#### 绩效统计（`compute_performance`）

输出指标：总信号数、实际入场次数、no_trade 次数、胜率、平均盈利/亏损、盈利因子、最大回撤、总收益、逐月胜率。

---

## API 配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ANTHROPIC_API_KEY` | 见代码 | Claude 代理 API Key |
| `ANTHROPIC_BASE_URL` | `https://api.openai-proxy.org/anthropic` | Claude 代理地址 |
| `DEEPSEEK_API_KEY` | 见代码 | DeepSeek API Key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | 默认 Claude 模型 |

支持通过环境变量覆盖所有配置值。

---

## LLM 输出格式（JSON Schema）

系统要求 LLM 严格输出如下 JSON：

```json
{
  "period": "Daily",
  "overall_market_sentiment": "Risk-On | Risk-Off | Neutral",
  "dxy_assessment": "<DXY 趋势描述>",
  "asset_analysis": [
    {
      "asset": "GOLD",
      "regime": "Trending | Mean-Reverting | Choppy",
      "action": "long | short | no_trade",
      "bias_score": 0.0-1.0,
      "entry_zone": "<价格区间>",
      "profit_target": <float | null>,
      "stop_loss": <float | null>,
      "risk_reward_ratio": <float | null>,
      "invalidation_condition": "<失效条件>",
      "macro_catalyst": "<宏观逻辑>",
      "technical_setup": "<指标信号描述>",
      "justification": "<综合判断，≤300字>"
    }
  ]
}
```

解析逻辑（`parse_signal`）按优先级尝试：直接 JSON 解析 → 提取 markdown 代码块 → 大括号计数匹配，兼容 DeepSeek R1 的 `<think>` 推理标签。

---

## 信号质量过滤规则

以下规则在提示词中硬编码，由 LLM 自行执行：

- `bias_score < 0.50` → 强制 `no_trade`
- 日线 MACD < 0 时，Trending 制度禁止做多
- RSI-7 > 75 → 做多 bias_score 上限 0.55
- 价格偏离 EMA-20 超过 3% → bias_score 上限 0.55
- DXY 高于 EMA20（美元强势）→ 做多 bias_score 降低 0.05–0.10
- ADX < 20 → 市场振荡，Trending 信号降级为 Choppy
- OBV 与价格背离 → bias_score 降低 0.10

性能反馈调整（来自 `performance.csv`）：
- 胜率 < 40% → bias_score 门槛提升至 ≥ 0.65
- 连续亏损 ≥ 2 次 → 需 bias_score ≥ 0.75 才入场

---

## 当前回测绩效（2025 全年，截至最新数据）

| 指标 | 数值 |
|------|------|
| 总信号数 | 45 |
| 实际入场 | 33 |
| no_trade 率 | 26.7% |
| 胜率 | 48.5% |
| 平均盈利 | +2.88% |
| 平均亏损 | -1.04% |
| 盈利因子 | 5.54 |
| 最大回撤 | -2.01% |
| 总收益 | +37.71% |

---

## 依赖安装

```bash
pip install yfinance pandas numpy anthropic openai curl_cffi urllib3 httpx
```

---

## 常见注意事项

- **SSL 证书**：`urllib3.disable_warnings` + `curl_cffi` 的 `verify=False` 用于企业 VPN/代理环境，生产环境建议启用证书验证。
- **yfinance SQLite 冲突**：`backtest_engine.py` 用 `tempfile.mkdtemp()` 为每次运行创建独立缓存目录，避免并发冲突。
- **数据重试**：`_download_with_retry` 最多重试 3 次，间隔递增，应对网络抖动。
- **`backtest_responses/` 目录**：需用户手动创建并填充，或通过全自动模式（模式三）由程序自动完成。
