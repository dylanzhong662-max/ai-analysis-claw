# 大模型金融分析 — 项目说明文档

## 项目概述

本项目是一个**以大语言模型（LLM）为核心决策引擎的多资产交易信号系统**，覆盖三类资产共 12 个标的：

1. **贵金属 / 大宗商品**（`gold_analysis.py` + `tech_stock_analysis.py`）：黄金、白银（SLV）、铜（COPX）、稀土/钨（REMX）、原油（USO）。
2. **加密货币**（`btc_analysis.py`）：BTC 战略周期分析。
3. **纳斯达克科技股**（`tech_stock_analysis.py`）：GOOGL、MSFT、NVDA、AAPL、META、AMZN。

在单资产分析之上，新增两个横向模块：
- **多资产扫描**（`market_scan.py`）：批量分析所有资产，LLM 额外输出板块排名、Top 5 机会、相关性风险。
- **持仓跟踪**（`portfolio_tracker.py`）：读取 `portfolio.json` 持仓，结合实时价格和最新信号，输出 HOLD / STOP_TRIGGERED / TARGET_REACHED 等具体操作建议，并生成 `orders.json` 供交易接口读取。

---

## 核心设计理念

- **LLM 作为推理引擎**：不依赖传统量化模型，由 LLM 读取结构化技术指标 + 宏观数据，模拟对冲基金分析师推理过程，输出标准化 JSON 交易信号。
- **防时间泄漏（Anti-Leakage）**：回测时提示词不包含具体日期，避免模型利用训练集中的未来知识作弊。
- **风险约束硬编码**：所有信号必须满足 R:R ≥ 2.0（科技股 ≥ 2.0）、止损 ≥ 1.5×ATR-14（周线），否则系统自动降级为 `no_trade`。
- **性能反馈闭环**：实时分析脚本读取 `backtest_results/performance.csv`，将胜率、连续亏损次数注入提示词，动态调整模型决策阈值。
- **资产注册表集中管理**：所有资产的 ticker、分析脚本、输出文件路径统一在 `assets_config.py` 中注册，新增资产只需修改一处。

---

## 目录结构

```
大模型金融分析/
├── gold_analysis.py          # 黄金实时信号生成 + PAXG 自动下单
├── btc_analysis.py           # BTC 战略周期分析
├── tech_stock_analysis.py    # 科技股 + 大宗商品 ETF 分析（通用）
├── market_scan.py            # 多资产横向扫描 + 跨资产机会排名（新）
├── portfolio_tracker.py      # 持仓跟踪 + 操作建议生成器（新）
├── assets_config.py          # 资产注册表 + 扫描分组配置（新）
├── backtest_engine.py        # 黄金历史回测引擎（EVAL_DAYS=20，1.5×ATR，Mean-Reverting 过滤）
├── tech_backtest_engine.py   # 科技股/ETF 回测引擎（EVAL_DAYS=22，仓位管理，SHORT_FILTERED）
├── btc_backtest_engine.py    # BTC 专用回测引擎（EVAL_DAYS=60，长周期评估）
├── google_backtest.py        # GOOGL 科技股回测引擎（旧版，逐步迁移至 tech_backtest_engine）
├── run_all_backtests.py      # 多资产批量回测编排脚本
├── calibrate_models.py       # 双模型校准脚本（R1 vs Claude bias 系数对齐）
├── feishu_notifier.py        # 飞书推送器
│
├── portfolio.json            # 当前持仓文件（用户维护）
├── gold_prompt_output.txt    # 最新黄金提示词（自动覆盖）
├── gold_api_output.txt       # 最新黄金分析结果
├── {ticker}_api_output.txt   # 各资产最新分析结果
├── market_scan_output.json   # 最新多资产扫描结果（结构化）
├── market_scan_report.txt    # 最新多资产扫描报告（原始文本）
├── portfolio_status.json     # 最新持仓状态评估（自动生成）
├── orders.json               # 待执行订单列表（--export-orders 时生成）
│
├── backtest_prompts/         # 黄金回测盲化提示词
├── backtest_responses/       # 手动回测 LLM 响应
├── backtest_results/         # 黄金回测输出（signals.csv / performance.csv）
└── googl_backtest_results/   # GOOGL 回测输出
```

---

## 脚本详解

### 1. `gold_analysis.py` — 黄金实时信号生成

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
                                                       │
                                               gold_api_output.txt
                                                       │
                                               execute_trade() ──> Binance PAXG/USDT
```

#### 技术指标

| 类别 | 指标 |
|------|------|
| 趋势 | EMA-20/50/200, MACD (12/26/9) |
| 震荡 | RSI-7, RSI-14, Stochastic %K/%D (14/3) |
| 波动 | ATR-3, ATR-14, Bollinger Bands (20, 2σ), %B, 带宽 |
| 趋势强度 | ADX, +DI, -DI (14) |
| 量价动量 | OBV, ROC-10, ROC-20 |

#### 运行方式

```bash
python3 gold_analysis.py                              # 只生成提示词文件
python3 gold_analysis.py --api                        # 调用 Claude 分析
python3 gold_analysis.py --api --model deepseek-reasoner
python3 gold_analysis.py --api --trade                # 分析 + 自动下单
python3 gold_analysis.py --api --trade --dry-run      # 模拟下单
```

---

### 2. `tech_stock_analysis.py` — 科技股 + 大宗商品 ETF 分析

支持任意 yfinance 可查询的 ticker，内置以下资产的专属行业上下文（`_INDUSTRY_CONTEXT`）：

| Ticker | 资产 | 专属分析维度 |
|--------|------|------------|
| GOOGL | Alphabet | 搜索广告 vs AI、Google Cloud、YouTube |
| MSFT | Microsoft | Azure 增速、Copilot 货币化、OpenAI 押注 |
| NVDA | NVIDIA | Blackwell 出货、AI CapEx 周期、出口管制 |
| AAPL | Apple | 换机周期、服务收入、印度市场 |
| META | Meta | 广告 CPM、Reels 货币化、Reality Labs |
| AMZN | Amazon | AWS、零售利润率、广告业务 |
| SLV | 白银 ETF | 金银比、工业需求、DXY 负相关 |
| COPX | 铜矿 ETF | 全球 PMI、能源转型需求、中国敞口 |
| REMX | 稀土/钨 ETF | 中国出口管制、EV 永磁体需求、去中国化 |
| USO | 原油 ETF | OPEC+、EIA 库存、页岩油成本 |

宏观数据：QQQ、XLK、SPY、^TNX、^VIX、DX-Y.NYB
基本面情报：财报日期、EPS 预估、分析师评级、估值指标、盈利惊喜历史

```bash
python3 tech_stock_analysis.py --ticker NVDA --api
python3 tech_stock_analysis.py --ticker SLV  --api   # 白银
python3 tech_stock_analysis.py --ticker REMX --api   # 稀土/钨
```

---

### 3. `market_scan.py` — 多资产横向扫描（新）

两阶段流程：
1. **Stage 1**：依次调用各资产对应的分析脚本，写入 `{ticker}_api_output.txt`
2. **Stage 2**：汇总所有信号，额外发起一次 LLM 调用，输出跨资产排名

```bash
python3 market_scan.py --api                          # 快速扫描（GOLD+BTC+NVDA+MSFT）
python3 market_scan.py --group tech --api             # 6 只科技股
python3 market_scan.py --group metals --api           # 贵金属/大宗商品
python3 market_scan.py --group all --api              # 全部资产
python3 market_scan.py --assets GOLD NVDA SLV --api   # 自定义列表
python3 market_scan.py --group tech --skip-individual --api  # 跳过重新分析，直接汇总
```

Stage 2 LLM 输出字段：`macro_themes`、`sector_ranking`、`top_opportunities`、`correlation_risks`、`watchlist`

---

### 4. `portfolio_tracker.py` — 持仓跟踪（新）

读取 `portfolio.json`，结合实时价格和最新 LLM 信号，按以下优先级输出操作建议：

| 优先级 | 状态 | 触发条件 | 生成订单 |
|--------|------|---------|---------|
| 1 | `STOP_TRIGGERED` | 当前价触及/跌破止损 | 市价平仓 |
| 2 | `TARGET_REACHED` | 当前价触及/超过目标价 | 限价锁利 |
| 3 | `SIGNAL_REVERSED` | LLM 信号方向与持仓相反 | 市价平仓 |
| 4 | `REDUCE` | 信号变为 no_trade（bias < 0.5） | 市价减半仓 |
| 5 | `HOLD` | 一切正常 | 无 |

```bash
python3 portfolio_tracker.py                         # 查看持仓状态
python3 portfolio_tracker.py --update-signals        # 先刷新信号再评估
python3 portfolio_tracker.py --export-orders         # 额外导出 orders.json
```

`orders.json` 字段（对齐 Binance API）：`side`、`quantity`、`order_type`、`price`、`note`

---

### 5. `assets_config.py` — 资产注册表（新）

所有资产的路由信息集中于此，`market_scan.py` 和 `portfolio_tracker.py` 共用：

```python
ASSET_UNIVERSE = {
    "GOLD":  {"ticker": "GC=F",  "script": "gold_analysis.py",       "output_file": "gold_api_output.txt",  ...},
    "NVDA":  {"ticker": "NVDA",  "script": "tech_stock_analysis.py", "output_file": "nvda_api_output.txt",  ...},
    "REMX":  {"ticker": "REMX",  "script": "tech_stock_analysis.py", "output_file": "remx_api_output.txt",  ...},
    ...
}

SCAN_GROUPS = {
    "quick":      ["GOLD", "BTC", "NVDA", "MSFT"],
    "tech":       ["GOOGL", "MSFT", "NVDA", "AAPL", "META", "AMZN"],
    "metals":     ["GOLD", "SILVER", "COPPER", "RARE_EARTH"],
    "commodities":["GOLD", "SILVER", "COPPER", "RARE_EARTH", "OIL"],
    "all":        [...所有资产...],
}
```

新增资产只需在此注册，无需改动其他脚本。

---

### 6. `backtest_engine.py` — 黄金历史回测引擎

**三种模式：**

```bash
# 生成盲化提示词（无需 API Key）
python3 backtest_engine.py --generate --start 2024-01-01 --end 2024-12-31 --step 5

# 评估已有手动响应
python3 backtest_engine.py --evaluate

# 全自动回测（需要 DeepSeek API Key）
python3 backtest_engine.py --start 2025-01-01 --end 2025-12-31
python3 backtest_engine.py --start 2025-01-01 --end 2025-12-31 --resume  # 断点续跑
```

**交易模拟规则：**
- 入场：信号日次日开盘价
- 持仓期：最长 `EVAL_DAYS = 15` 个交易日
- 出场：Low ≤ stop_loss → `STOP_LOSS`；High ≥ profit_target → `TAKE_PROFIT`；超时 → `TIMEOUT`
- 有效性校验：入场价下 R:R < 1.5 → `INVALID_RR`，不计入统计

---

## API 配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ANTHROPIC_API_KEY` | 见代码 | Claude 代理 API Key |
| `ANTHROPIC_BASE_URL` | `https://api.openai-proxy.org/anthropic` | Claude 代理地址 |
| `DEEPSEEK_API_KEY` | 见代码 | DeepSeek API Key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | 默认 Claude 模型 |

所有变量支持通过环境变量覆盖。

---

## LLM 输出格式

### 黄金 / 大宗商品（Daily）

```json
{
  "period": "Daily",
  "overall_market_sentiment": "Risk-On | Risk-Off | Neutral",
  "dxy_assessment": "<DXY 趋势描述>",
  "asset_analysis": [{
    "asset": "GOLD",
    "regime": "Trending | Mean-Reverting | Choppy",
    "action": "long | short | no_trade",
    "bias_score": 0.0,
    "entry_zone": "<价格区间>",
    "profit_target": null,
    "stop_loss": null,
    "risk_reward_ratio": null,
    "estimated_holding_weeks": null,
    "position_size_pct": 0.0,
    "invalidation_condition": "<失效条件>",
    "macro_catalyst": "<宏观逻辑>",
    "technical_setup": "<指标信号>",
    "justification": "<≤300字综合判断>"
  }]
}
```

### 科技股 / ETF（Weekly，中长线）

```json
{
  "period": "Weekly",
  "stock_ticker": "NVDA",
  "overall_market_sentiment": "Risk-On | Risk-Off | Neutral",
  "qqq_assessment": "<QQQ 对该股的方向性影响>",
  "sector_assessment": "<XLK 板块轮动信号>",
  "macro_rate_environment": "<10Y 收益率对成长股估值影响>",
  "earnings_risk_flag": false,
  "asset_analysis": [{
    "asset": "NVDA",
    "regime": "Trending-Up | Trending-Down | Consolidation | ...",
    "action": "long | short | no_trade",
    "bias_score": 0.0,
    "entry_zone": "<价格区间>",
    "profit_target": null,
    "stop_loss": null,
    "risk_reward_ratio": null,
    "estimated_holding_weeks": 8,
    "position_size_pct": 0.0,
    "price_action_analysis": {},
    "structured_analysis": {},
    "intelligence_analysis": {},
    "justification": "<≤300字>"
  }]
}
```

### 多资产扫描汇总（market_scan.py Stage 2）

```json
{
  "scan_date": "2026-03-30",
  "macro_themes": [{"theme": "", "description": "", "beneficiary_assets": []}],
  "sector_ranking": [{"sector": "", "strength": "Strong | Neutral | Weak", "assets": [], "rationale": ""}],
  "top_opportunities": [{"rank": 1, "asset": "", "action": "", "bias_score": 0.0, "rationale": ""}],
  "correlation_risks": [{"assets": [], "correlation_type": "", "risk_note": ""}],
  "watchlist": [{"asset": "", "reason": "", "trigger_condition": ""}]
}
```

解析逻辑（`parse_signal`）：直接 JSON → markdown 代码块提取 → 大括号计数匹配，兼容 DeepSeek R1 的 `<think>` 标签。

---

## 信号质量过滤规则

**通用（所有资产）：**
- `bias_score < 0.50` → 强制 `no_trade`
- R:R < 2.0 → 强制 `no_trade`
- **regime = Mean-Reverting → 强制 `no_trade`**（所有资产，历史回测胜率接近 0%）
- ADX < 20 → 制度降级为 Choppy，bias_score 上限 0.45
- OBV 与价格背离 → bias_score 降低 0.10
- 止损最小距离：**1.5×周线 ATR-14**（中长线持仓需容纳周线波动噪音）

**仓位管理（所有资产 position_size_pct）：**
- bias 0.50–0.59 / Choppy 制度 → 0.1–0.2
- bias 0.60–0.69 / Trending 制度 → 0.3–0.5
- bias 0.70–0.79 / 多时间框架共振 → 0.5–0.7
- bias ≥ 0.80 / 趋势强劲+量价配合 → 0.7–1.0
- 周线 RSI-7 > 75 追高入场 → 上限 0.3（防追高亏损）
- 价格偏离 EMA-20 > 5–10% → 上限 0.3

**黄金专属：**
- 周线 MACD < 0 且 Trending 制度 → 禁止做多
- RSI-7 > 75 → 做多 bias_score 上限 0.55
- DXY 高于 EMA20 → 做多 bias_score 降低 0.05–0.10

**科技股专属：**
- QQQ 死叉（EMA50 < EMA200）→ 禁止做多
- **做空需同时满足：QQQ EMA50 < EMA200 AND VIX > 20 AND 股价低于周线 EMA-200**
- 财报日 ≤ 5 天 → 强制 `no_trade`；6–15 天 → bias_score 上限 0.55，仓位上限 0.3
- 周线 RSI-7 > 75 → 做多 bias_score 上限 0.55
- EPS 连续 2 季不及预期 → bias_score 降低 0.10

**BTC 专属：**
- 价格低于 200WMA → 做多 bias_score ≤ 0.50
- 月线 RSI > 85 → 做多 bias_score ≤ 0.50
- 减半周期 > 45% 且月 RSI > 70 → 顶部预警，bias_score 额外 -0.10

**性能反馈（来自 `performance.csv`）：**
- 胜率 < 40% → bias_score 门槛提升至 ≥ 0.65
- 连续亏损 ≥ 2 次 → 需 bias_score ≥ 0.75 才入场

---

## 当前回测绩效

> 以下为优化后（v3）规则下的回测结果：止损 1.5×ATR、Mean-Reverting 过滤、仓位管理、SHORT_FILTERED、eval_days 扩展。

### NVDA（2025-02 ~ 2025-12，新规则）

| 指标 | 数值 |
|------|------|
| 总信号数 | 45 |
| 有效入场 | 16 |
| no_trade 率 | 62.2% |
| 胜率 | 50.0% |
| 平均盈利 | +10.12% |
| 平均亏损 | -3.93% |
| 盈利因子 | 2.57 |
| 最大回撤 | -22.35% |
| 总收益 | +49.49% |
| 弱点 | 7–8月高位震荡连续止损 |

### MSFT（2025-02 ~ 2025-12，新规则）

| 指标 | 数值 |
|------|------|
| 总信号数 | 46 |
| 有效入场 | 11 |
| no_trade 率 | ~76% |
| 胜率 | 91% |
| 平均盈利 | +5.72% |
| 平均亏损 | -3.03% |
| 盈利因子 | 18.80 |
| 最大回撤 | -3.03% |
| 总收益 | +54.00% |

### GOLD（2025-01 ~ 2025-12，新规则）

| 指标 | 数值 |
|------|------|
| 总信号数 | 34（17节点 NO_DATA，Yahoo 限流导致评估缺失） |
| 有效入场 | 19 |
| no_trade 率 | 35.3% |
| 胜率 | 52.6% |
| 平均盈利 | +4.85% |
| 平均亏损 | -2.20% |
| 盈利因子 | 7.36 |
| 最大回撤 | -4.50% |
| 总收益 | +41.95% |

### GOOGL（2025-02 ~ 2025-12，新规则 v3）

| 指标 | 数值 | vs 旧规则 |
|------|------|---------|
| 总信号数 | 46 | 93 |
| 有效入场 | 19 | 60 |
| no_trade 率 | 56.5% | 32.7% |
| 胜率 | 52.6% | 35% → 大幅提升 |
| 平均盈利 | +7.74% | +5.95% |
| 平均亏损 | -3.60% | -2.99% |
| 盈利因子 | 2.39 | 1.07 → 大幅提升 |
| 最大回撤 | -13.85% | -52% → 大幅收窄 |
| 总收益 | +45.01% | +8.48% |

**科技股 vs 黄金系统对比：**

| 维度 | 黄金 | 科技股 |
|------|------|-------|
| 信号频率 | 周线 | 周线 |
| 宏观驱动 | DXY、白银、10Y | QQQ、VIX、10Y |
| 跳空风险 | 低 | 高（财报/AI 新闻） |
| R:R 目标 | ≥ 2.0 | ≥ 2.0 |
| 止损 | 1.5×周线 ATR | 1.5×周线 ATR |
| 持仓周期 | 最长 20 交易日（EVAL_DAYS） | 最长 22 交易日（EVAL_DAYS） |
| BTC 持仓周期 | — | 最长 60 交易日（EVAL_DAYS） |
| 性能反馈 | 已实现 | 已实现（v3） |

---

## 依赖安装

```bash
pip install yfinance pandas numpy anthropic openai curl_cffi urllib3 httpx
```

---

## 常见注意事项

- **SSL 证书**：`urllib3.disable_warnings` + `curl_cffi` 的 `verify=False` 用于企业 VPN/代理环境，生产环境建议启用验证。
- **yfinance SQLite 冲突**：`backtest_engine.py` 用 `tempfile.mkdtemp()` 为每次运行创建独立缓存目录，避免并发冲突。
- **数据重试**：`_download_with_retry` 最多重试 5 次，间隔递增，应对网络抖动。
- **Parquet 持久化缓存**：`tech_backtest_engine.py` 将 yfinance 数据缓存至 `data_cache/*.parquet`，重跑回测无需重新下载，避免 Yahoo Finance 429 限流。
- **Yahoo Finance 限流**：密集回测可能触发 429，等待 2–4 小时自动解封；缓存覆盖范围不足时才会触发下载。
- **代理配置排查清单（每次部署后必查）**：
  1. **SS 节点是否过期**：节点配置（server/port/password）会定期更换。每次遇到代理异常，先执行 `curl -s "https://ss.wawaapp.net/t/520fa9d967e39ce4b19a54c88312e52d2991ecf63894998e00f031308"` 获取最新配置，与 `/etc/shadowsocks-libev/local.json` 对比，不一致则更新并 `systemctl restart ss-local`。
  2. **privoxy 是否跟随重启**：ss-local 重启后需同时 `systemctl restart privoxy`。
  3. **`.env` 代理变量必须带 `export`**：`.env` 中 `HTTPS_PROXY` / `HTTP_PROXY` / `NO_PROXY` 若没有 `export` 关键字，Python 子进程拿不到这些环境变量（`os.environ` 返回 None），代理形同虚设。确认格式为 `export HTTPS_PROXY=http://127.0.0.1:8118`。
  4. **验证完整链路**：`curl --socks5 127.0.0.1:1080 http://httpbin.org/ip -s`（验证 SS），`curl --proxy http://127.0.0.1:8118 http://httpbin.org/ip -s`（验证 privoxy），出口 IP 应与 SS 节点出口一致。
- **NO_PROXY 配置**：以下 API 端点直连不走代理，`.env` 中 `NO_PROXY` 须包含所有这些域名：
  ```
  export NO_PROXY=open.feishu.cn,feishu.cn,api.deepseek.com,dashscope.aliyuncs.com,api.openai-proxy.org
  ```
  `api.openai-proxy.org`（Claude + GPT 聚合平台）直连 ECS 可达，**不能**走 SS 代理，否则 TLS 握手报 `SSL: UNEXPECTED_EOF_WHILE_READING` 错误。
- **`portfolio.json` 维护**：每次开仓/平仓后需手动更新，或在交易接口对接完成后自动同步。
- **`backtest_responses/` 目录**：需用户手动创建并填充，或通过全自动模式由程序自动完成。
- **新增资产**：在 `assets_config.py` 注册 + 在 `tech_stock_analysis.py` 的 `_INDUSTRY_CONTEXT` 添加专属上下文（可选）。
