# 大模型金融分析 — 项目说明文档

## 执行环境约定

**所有脚本默认在阿里云 ECS（`101.201.171.174`）上运行**，不在本地 Mac 执行。本地网络存在 TLS 拦截，Python API 调用会报 `SSL: CERTIFICATE_VERIFY_FAILED`。

需要执行脚本时，直接给出服务器命令：

```bash
ssh root@101.201.171.174 "cd /opt/finance-analysis && source .env && .venv/bin/python3 <script> <args>"
```

---

## 项目概述

本项目是一个**以大语言模型（LLM）为核心决策引擎的多资产交易信号系统**，覆盖三类资产共 12 个标的：

1. **贵金属 / 大宗商品**（`gold_analysis.py` + `tech_stock_analysis.py`）：黄金、白银（SLV）、铜（COPX）、稀土/钨（REMX）、原油（USO）。
2. **加密货币**（`btc_analysis.py`）：BTC 战略周期分析。
3. **纳斯达克科技股**（`tech_stock_analysis.py`）：GOOGL、MSFT、NVDA、AAPL、META、AMZN。

在单资产分析之上，两个横向模块：
- **多资产扫描**（`market_scan.py`）：批量分析所有资产，LLM 额外输出板块排名、Top 5 机会、相关性风险。
- **持仓跟踪**（`portfolio_tracker.py`）：读取 `portfolio.json` 持仓，结合实时价格和最新信号，输出 HOLD / STOP_TRIGGERED / TARGET_REACHED 等具体操作建议，并生成 `orders.json` 供交易接口读取。

---

## 核心设计理念

- **LLM 作为推理引擎**：不依赖传统量化模型，由 LLM 读取结构化技术指标 + 宏观数据，模拟对冲基金分析师推理过程，输出标准化 JSON 交易信号。
- **防时间泄漏（Anti-Leakage）**：回测时提示词不包含具体日期，避免模型利用训练集中的未来知识作弊。
- **风险约束硬编码**：所有信号必须满足 R:R ≥ 2.0（科技股 ≥ 2.0）、止损 ≥ **2.5×ATR-14**（周线），否则系统自动降级为 `no_trade`。
- **Beta Overlay 架构（波动率目标制）**：`portfolio_tracker.py --beta-overlay` 模式。核心是**波动率目标制**：目标仓位 = 16% 目标年化波动率 / 资产实现波动率（20周），低波动牛市自动加仓，高波动市自动减仓，上限 80%。**动量修正下限（VOL_FLOOR=30%）**：牛市趋势中（价格>EMA200且>EMA50），无论波动率多高，仓位不低于 30%，防止高波动牛市（如 NVDA 2019-2021）严重踏空。LLM 仅作辅助叠加层：bias≥0.60 的 long 信号额外 +10%。出场使用**双层触发**：EMA200 跌破（制度出场）或 EMA50 跌破 / 距52周高点回撤>15%（快速出场），比纯 EMA200 快 4-6 周。BTC 额外检查 200WMA（200周简单均线）。
- **资产注册表集中管理**：所有资产的 ticker、分析脚本、输出文件路径统一在 `assets_config.py` 中注册，新增资产只需修改一处。

---

## 目录结构

```
大模型金融分析/
├── gold_analysis.py          # 黄金实时信号生成 + PAXG 自动下单
├── btc_analysis.py           # BTC 战略周期分析
├── tech_stock_analysis.py    # 科技股 + 大宗商品 ETF 分析（通用）
├── market_scan.py            # 多资产横向扫描 + 跨资产机会排名
├── portfolio_tracker.py      # 持仓跟踪 + Beta Overlay 仓位管理
├── assets_config.py          # 资产注册表 + 扫描分组配置
├── tech_backtest_engine.py   # 科技股/ETF 回测引擎（EVAL_DAYS=22，仓位管理，SHORT_FILTERED）
├── baseline_strategy.py      # 5规则极简策略（对照组，供 run_historical_backtest.sh --simplified 调用）
├── backtest_beta_overlay.py  # Beta Overlay 策略历史回测（含 vol_floor，对比 B&H）
├── feishu_notifier.py        # 飞书推送器
├── signal_logger.py          # 信号持久化（追加到 live_signal_log.csv）
├── validate_regime.py        # 验证：EMA200/EMA50 制度过滤准确率分析（2018-2026）
├── validate_ic.py            # 验证：LLM 信号 IC 分析（需服务器日志，本地可演示）
├── freeze_holdout.py         # 封存 holdout 测试集（防数据泄漏，运行一次）
├── news_signal_bridge.py     # RAG 新闻信号桥接（语义检索 + LLM 情绪标注）
├── run_daily.sh              # 每日定时任务脚本（08:00 CST）
├── run_weekly.sh             # 每周汇总脚本（周一 08:30 CST）
├── run_historical_backtest.sh # 2018-2021 历史泛化验证脚本
├── setup_cron.sh             # 服务器 crontab 配置脚本
├── deploy.sh                 # 一键部署到阿里云 ECS
├── start.sh                  # 本地启动后端服务
├── requirements.txt          # Python 依赖列表
│
├── 纳斯达克科技股分析.markdown  # 科技股分析系统提示词（LLM system prompt）
├── 大宗商品分析.markdown        # 大宗商品分析系统提示词
├── 加密货币分析交易提示词.markdown # BTC 分析系统提示词
├── 使用流程.markdown            # 操作手册
│
├── portfolio.json            # 当前持仓文件（用户维护）
│
├── nvda_portfolio_backtest/  # NVDA 回测参考数据（equity.csv / signals.csv / trades.csv）
├── data_cache/               # Parquet 价格缓存（tech_backtest_engine.py 使用，.gitignore）
├── logs/                     # 运行日志（.gitignore）
│
├── backend/                  # FastAPI REST API（持仓/信号/仪表盘接口）
└── frontend/                 # Vue 3 + Vite 前端仪表盘
```

> **运行时输出文件**（不提交 git，每次运行自动覆盖）：
> `{ticker}_api_output.txt` / `{ticker}_prompt_output.txt` / `trading.db` /
> `market_scan_output.json` / `overlay_status.json` / `orders.json` / `live_signal_log.csv`

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
**近期新闻情绪**：优先使用 `news_signal_bridge.py`（RAG 语义检索 + LLM 情绪标注），降级到 yfinance.news（仅实盘；回测禁用）

System Prompt：运行时自动从 `纳斯达克科技股分析.markdown` 加载，作为 `system` 字段传给模型。

```bash
python3 tech_stock_analysis.py --ticker NVDA --api
python3 tech_stock_analysis.py --ticker SLV  --api   # 白银
python3 tech_stock_analysis.py --ticker REMX --api   # 稀土/钨
```

---

### 3. `market_scan.py` — 多资产横向扫描

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

### 4. `portfolio_tracker.py` — 持仓跟踪 + Beta Overlay 模式

两种运行模式：

#### 模式A：传统持仓跟踪

读取 `portfolio.json` 的 `positions` 段，按以下优先级输出操作建议：

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

#### 模式B：Beta Overlay（推荐）— 波动率目标制架构

读取 `portfolio.json` 的 `beta_overlay_config` 段，三步定仓：

**Step 1 — 制度过滤（优先出场）**

| 触发条件 | 目标仓位 | 动作 |
|---------|---------|------|
| 价格 < 周线 EMA200 | **0%** | `EXIT_BEAR_REGIME` |
| BTC 额外：价格 < 200WMA（200周SMA） | **0%** | `EXIT_BEAR_REGIME` |
| 价格 < 周线 EMA50 OR 距52周高点回撤>15% | **0%** | `EXIT_FAST`（快 4-6 周） |

**Step 2 — 波动率目标制定仓（主逻辑，含动量修正下限）**

```
vol_target  = min(16% / 资产年化实现波动率, 80%)
# 动量修正：牛市趋势中（价格>EMA200且>EMA50）仓位不低于30%
target_pct  = max(vol_target, 30%)  ← VOL_FLOOR，仅在 above_ema200 AND above_ema50 时激活

示例：
  NVDA（vol=40%）→ 16%/40% = 40%  → max(40%, 30%) = 40%（floor未触发）
  NVDA（vol=80%）→ 16%/80% = 20%  → max(20%, 30%) = 30%（floor激活，✓ 防踏空）
  MSFT（vol=20%）→ 16%/20% = 80%（触发上限）
  黄金（vol=12%）→ 16%/12% = 80%（触发上限）
  BTC （vol=70%）→ 16%/70% = 23%  → max(23%, 30%) = 30%（floor激活）
```

**Step 3 — LLM 辅助叠加（次要信号）**

| LLM 信号 | 叠加量 | 说明 |
|---------|-------|------|
| action=long AND bias≥0.60 | **+10%** | 确认信号，小幅增仓 |
| 其他 | 0% | 不影响基础仓位 |

```bash
python3 portfolio_tracker.py --beta-overlay                          # 查看仓位建议
python3 portfolio_tracker.py --beta-overlay --update-signals         # 先刷新 LLM 信号
python3 portfolio_tracker.py --beta-overlay --update-signals --export-orders  # 额外导出 orders.json
```

`portfolio.json` 中的 `beta_overlay_config` 配置：
```json
"beta_overlay_config": {
  "NVDA": {"allocated_capital": 50000, "current_shares": 0},
  "MSFT": {"allocated_capital": 30000, "current_shares": 0},
  "GOOGL": {"allocated_capital": 20000, "current_shares": 0}
}
```
每次实际买卖后手动更新 `current_shares`。输出保存到 `overlay_status.json`。

`overlay_status.json` 新增字段：`realized_vol_annual`（年化波动率）、`vol_target_pct`（波动率目标仓位，已含floor修正）、`vol_floor_active`（动量下限是否触发）、`llm_overlay_pct`（LLM叠加量）、`ema50`/`above_ema50`（快速出场信号）、`wma200`（BTC专用）、`drawdown_from_high`/`fast_exit_triggered`（回撤出场状态）

`orders.json` 字段（对齐 Binance API）：`side`、`quantity`、`order_type`、`price`、`note`

**动作类型说明：**

| 动作 | 含义 |
|------|------|
| `ENTER` | 空仓进场（波动率目标制计算仓位） |
| `REBALANCE_UP` | 波动率下降或LLM信号，增仓至目标 |
| `REBALANCE_DOWN` | 波动率上升，减仓至目标 |
| `EXIT_BEAR_REGIME` | EMA200跌破（或BTC 200WMA），清仓 |
| `EXIT_FAST` | EMA50跌破 或 回撤>15%，快速清仓 |
| `HOLD` | 仓位偏差≤5%，无需操作 |

---

### 5. `assets_config.py` — 资产注册表

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

## 策略验证工具

三个本地可独立运行的验证脚本，用于评估策略各模块的有效性。

### 验证1：LLM信号IC分析（`validate_ic.py`）

检验 `bias_score` 对未来收益是否有统计显著的预测力（需 30+ 条信号样本）。

```bash
# 本地演示（样本量不足，仅展示框架）
python3 validate_ic.py

# 完整版：先从服务器拉取日志
scp -r root@101.201.171.174:/opt/finance-analysis/logs ./logs_server
python3 validate_ic.py --log-dir logs_server

# 指定回测 signals.csv
python3 validate_ic.py --signals-csv nvda_portfolio_backtest/signals.csv
```

解读：`t-stat > 2.0` 表示LLM信号有预测力；`t-stat < 1.5` 表示LLM择时无效，应专注制度过滤。

### 验证2：制度过滤准确率（`validate_regime.py`）

检验 EMA200/EMA50 制度过滤在历史上是否能区分牛熊、规避回撤。

```bash
python3 validate_regime.py                            # 默认：NVDA MSFT GOOGL，2018-2026
python3 validate_regime.py --tickers NVDA --start 2020-01-01
```

关键输出：各制度（bull/fast_exit/bear）下的前向N周收益、胜率、EMA50相对EMA200的提前出场周数。

### 验证3：Beta Overlay历史回测（`backtest_beta_overlay.py`）

**所有 Beta Overlay 策略的标准回测工具**（含 `VOL_FLOOR_PCT=30%` 动量修正）。

```bash
python3 backtest_beta_overlay.py                      # 默认：NVDA MSFT GOOGL，2019-2026
python3 backtest_beta_overlay.py --tickers NVDA --start 2019-01-01 --save-csv
python3 backtest_beta_overlay.py --target-vol 0.12    # 更保守：12%目标波动率
```

参数与 `portfolio_tracker.py` 保持一致：`TARGET_ANNUAL_VOL=16%`、`VOL_FLOOR_PCT=30%`、`MAX_POSITION_PCT=80%`、再平衡阈值5%、双边成本0.2%/边。

---

## 部署与运维

### 阿里云 ECS 服务器

| 项目 | 内容 |
|------|------|
| 公网 IP | `101.201.171.174` |
| 登录用户 | `root` |
| 项目目录 | `/opt/finance-analysis` |
| Python 环境 | `/opt/finance-analysis/.venv` |
| 日志目录 | `/opt/finance-analysis/logs` |

```bash
# SSH 登录
ssh root@101.201.171.174

# 代码更新后一键部署（本地 Mac 执行）
cd ~/Desktop/大模型金融分析
bash deploy.sh 101.201.171.174 root

# 或手动 scp 单文件
scp portfolio_tracker.py root@101.201.171.174:/opt/finance-analysis/
```

### 服务器常用操作

```bash
cd /opt/finance-analysis && source .env

# 手动触发每日分析
bash run_daily.sh

# 手动触发周报
bash run_weekly.sh

# 查看今日运行日志
tail -f logs/daily_$(date +%Y%m%d).log

# 查看 crontab 日志
tail -f logs/cron.log
```

### crontab 定时任务

```
# 阿里云 ECS 默认时区 CST（UTC+8），crontab 按本地时间执行，无需换算 UTC

# 每天 10:00 CST：早盘日报
0 10 * * * bash /opt/finance-analysis/run_daily.sh >> /opt/finance-analysis/logs/cron.log 2>&1

# 每天 19:00 CST：晚盘日报
0 19 * * * bash /opt/finance-analysis/run_daily.sh >> /opt/finance-analysis/logs/cron.log 2>&1

# 每周一 08:30 CST：周报
30 8 * * 1 bash /opt/finance-analysis/run_weekly.sh >> /opt/finance-analysis/logs/cron.log 2>&1
```

### Mac 本地定时任务（LaunchAgents）

```bash
# 注册定时任务（首次）
launchctl load ~/Library/LaunchAgents/com.finance.daily.plist
launchctl load ~/Library/LaunchAgents/com.finance.weekly.plist

# 取消注册
launchctl unload ~/Library/LaunchAgents/com.finance.daily.plist

# 查看状态
launchctl list | grep finance

# 立即触发一次（测试）
launchctl start com.finance.daily
```

---

## API 配置与 .env 说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ANTHROPIC_API_KEY` | 见代码 | Claude 代理 API Key |
| `ANTHROPIC_BASE_URL` | `https://api.openai-proxy.org/anthropic` | Claude 代理地址 |
| `DEEPSEEK_API_KEY` | 见代码 | DeepSeek API Key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | 默认 Claude 模型 |

所有变量支持通过环境变量覆盖。完整 `.env` 格式：

```bash
# Claude API（通过 openai-proxy.org 代理，直连不走 SS）
export ANTHROPIC_API_KEY=sk-...
export ANTHROPIC_BASE_URL=https://api.openai-proxy.org/anthropic

# DeepSeek API
export DEEPSEEK_API_KEY=sk-...

# 飞书群机器人 Webhook
export FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/...

# HTTP 代理（服务器需要，本地 Mac 不需要）
export HTTPS_PROXY=http://127.0.0.1:8118
export HTTP_PROXY=http://127.0.0.1:8118

# 直连域名（不走 SS 代理）
export NO_PROXY=open.feishu.cn,feishu.cn,api.deepseek.com,dashscope.aliyuncs.com,api.openai-proxy.org
```

**注意**：`.env` 中所有变量必须带 `export`，否则 Python 子进程无法继承（`os.environ` 返回 None）。

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

---

## 回测模式说明

**所有回测一律使用 `--portfolio` 模式 + `--second-model claude-sonnet-4-6` 双模型确认**，禁止用默认的 `run_backtest` 模式或单模型汇报绩效。

原因：
- 默认模式输出的"总收益"是每笔交易 P&L 百分比的简单累加（非复利、不考虑仓位大小），会严重高估真实收益。
- 单模型容易在趋势转折期连续止损，双模型确认（DeepSeek R1 初筛 → Claude 确认）可过滤分歧信号，减少假突破入场。

`--portfolio` 模式才是真实资金曲线：串行持仓、复利计算、双边佣金+滑点从资金里实际扣除、按风险比例定仓。

### 模型分层说明

| 环境 | 模式 | 说明 |
|------|------|------|
| **回测（tech_backtest_engine.py）** | 双模型确认 | DeepSeek R1 初筛 → Claude Sonnet 确认；一致入场，分歧强制 no_trade |
| **实盘分析（gold/tech_stock/btc_analysis.py）** | 三模型投票 | DeepSeek R1 + Claude Sonnet + GPT-4o；多数票（2/3）决定信号，无共识强制 no_trade |

三模型投票覆盖范围（阿里云已部署）：
- **黄金/大宗商品**：`gold_analysis.py --dual-model --third-model gpt-4o`
- **科技股/ETF**：`tech_stock_analysis.py --dual-model --third-model gpt-4o`
- **BTC**：`btc_analysis.py --dual-model --third-model gpt-4o`

### 标准回测命令（阿里云服务器，双模型）

```bash
cd /opt/finance-analysis && source .env

# 标准模式（完整规则链）
nohup .venv/bin/python3 -u tech_backtest_engine.py \
    --ticker NVDA \
    --portfolio \
    --start 2025-02-01 --end 2025-12-31 \
    --model deepseek-reasoner \
    --second-model claude-sonnet-4-6 \
    --capital 100000 \
    --commission 0.001 \
    --slippage 0.001 \
    --risk-per-trade 0.03 \
    --step 1 \
    --oos-split 0.2 \
    --reproducible \
    --rate-limit 15 \
    > logs/portfolio_nvda_$(date +%Y%m%d_%H%M).log 2>&1 &

# 5规则极简模式（bias≥0.60/无财报黑名单/无死叉软限/无冷却期）
nohup .venv/bin/python3 -u tech_backtest_engine.py \
    --ticker NVDA \
    --start 2025-02-01 --end 2025-12-31 \
    --model deepseek-reasoner --second-model claude-sonnet-4-6 \
    --capital 100000 --commission 0.001 --slippage 0.001 \
    --risk-per-trade 0.02 --step 1 --reproducible --simplified \
    > logs/simplified_nvda_$(date +%Y%m%d_%H%M).log 2>&1 &

# Beta底仓模式（EMA200上方持50%底仓 + LLM叠加+30%）
nohup .venv/bin/python3 -u tech_backtest_engine.py \
    --ticker NVDA \
    --start 2025-02-01 --end 2025-12-31 \
    --model deepseek-reasoner --second-model claude-sonnet-4-6 \
    --capital 100000 --commission 0.001 --slippage 0.001 \
    --beta-floor --floor-pct 0.50 --overlay-pct 0.30 \
    --step 5 --reproducible \
    > logs/beta_floor_nvda_$(date +%Y%m%d_%H%M).log 2>&1 &

# 封存 Holdout（只运行一次！）
python3 freeze_holdout.py             # 执行封存
python3 freeze_holdout.py --status    # 查看封存状态

# 2018-2021 历史泛化验证
bash run_historical_backtest.sh                # 标准模式（NVDA+MSFT+GOOGL）
bash run_historical_backtest.sh simplified     # 5规则极简模式
bash run_historical_backtest.sh NVDA           # 只跑 NVDA
```

### 标准实盘分析命令（阿里云服务器，三模型）

```bash
# 科技股三模型分析
python3 tech_stock_analysis.py --ticker NVDA --api \
    --dual-model --third-model gpt-4o \
    --model deepseek-reasoner --confirm-model claude-sonnet-4-6

# 黄金三模型分析
python3 gold_analysis.py --api \
    --dual-model --third-model gpt-4o \
    --model deepseek-reasoner --confirm-model claude-sonnet-4-6

# BTC 三模型分析
python3 btc_analysis.py --api \
    --dual-model --third-model gpt-4o \
    --model deepseek-reasoner --confirm-model claude-sonnet-4-6
```

**已优化的回测参数：**
- 主模型：DeepSeek R1（初筛），确认模型：Claude Sonnet（双模型确认）
- 连续止损熔断：连续 2 次止损 → 暂停入场 15 个交易日
- 移动止损：浮盈 ≥5% 移至入场价（保本），≥10% 跟踪至锁定 50% 盈利，不触发硬超时
- 滑点：0.1%/边（买卖价差 + 市场冲击）
- 佣金：0.1%/边（往返总摩擦约 0.4%）
- 风险/笔：账户净值的 2%
- OOS 分割：后 20% 为样本外验证
- 复现模式：强制使用磁盘缓存，确保结果可复现

---

## 当前回测绩效

> 数据来源：阿里云服务器 `/opt/finance-analysis/` 上的实际回测结果（2026-04-05 更新）。  
> 引擎版本：`tech_backtest_engine.py`（含熔断、ATR trailing stop、TIMEOUT_EXTENDED、双模型确认）。  
> 参数：初始资金 $100,000 | 佣金 0.1%/边 | 滑点 0.1%/边 | 风险/笔 2-3% | `--portfolio` 模式（真实资金曲线）

---

### 一、LLM 策略 — Portfolio 回测结果

#### NVDA 2025（2025-02-03 ~ 2025-12-05）

| 指标 | 数值 |
|------|------|
| 信号节点数 | 42 |
| no_trade 率 | **92.9%**（39/42） |
| 有效入场交易 | 3（含 1 笔 EOB） |
| 完成交易 | 2 |
| 胜率 | 1/1（完成交易中 1胜/1负） |
| 平均盈利 | +36.19% |
| 平均亏损 | -26.12% |
| 盈利因子 | 1.39 |
| 最终净值 | $101,482（+1.48%） |
| **B&H 收益** | **~+80%**（NVDA $103→$185） |
| **Alpha** | **-78.5%** ⚠️ 严重跑输 |

**逐笔明细：**
```
2025-02-18 ~ 2025-03-31  STOP_LOSS        -26.12%  Trending-Up    bias=0.58  hold=30d
2025-05-19 ~ 2025-11-20  TIMEOUT_EXTENDED +36.19%  Trending-Recovery bias=0.57 hold=130d [移动止损]
2025-12-08 ~ 2025-12-30  END_OF_BACKTEST  +2.48%   Trending-Up    bias=0.55  hold=16d
```

#### MSFT 2025（2025-01-02 ~ 2025-12-30）

| 指标 | 数值 |
|------|------|
| 信号节点数 | 102 |
| no_trade 率 | **96.1%**（98/102） |
| 有效入场交易 | 3 |
| 胜率 | 1/3（33.3%） |
| 平均盈利 | +22.75% |
| 平均亏损 | -8.26% |
| 盈利因子 | 1.38 |
| 最终净值 | $99,986（**-0.01%**） |
| **B&H 收益** | **+17.32%** |
| **Alpha** | **-17.3%** ⚠️ 跑输 |

**逐笔明细：**
```
2025-05-19 ~ 2025-07-31  TAKE_PROFIT  +22.75%  Trending-Up  bias=0.60  hold=51d [移动止损]
2025-08-05 ~ 2025-09-05  STOP_LOSS    -8.36%   Trending-Up  bias=0.55  hold=23d
2025-09-16 ~ 2025-11-21  STOP_LOSS    -8.17%   Trending-Up  bias=0.58  hold=49d
[熔断触发] 连续止损 2 次 → 暂停入场至 2025-12-15
```

#### GOOGL 2022（2022-01-03 ~ 2022-12-30）熊市测试

| 指标 | 数值 |
|------|------|
| 信号节点数 | 125 |
| no_trade 率 | **96.0%**（120/125） |
| 有效入场交易 | 4（含 1 笔 short） |
| 胜率 | 0/3（完成交易 0%，TIMEOUT 中 1胜） |
| 最终净值 | $99,833（**-0.17%**） |
| 最大回撤 | -3.29% |
| **B&H 收益** | **-39.15%** |
| **Alpha vs B&H** | **+39.0%** ✅ 熊市资本保全有效 |

**逐笔明细：**
```
2022-01-04 ~ 2022-01-07  STOP_LOSS  -6.81%   Trending-Up       bias=0.50  hold=4d
2022-03-10 ~ 2022-06-10  TIMEOUT   +15.15%   Trending-Down     bias=0.45  hold=65d (short)
2022-07-29 ~ 2022-08-24  STOP_LOSS  -0.10%   Trending-Recovery bias=0.50  hold=19d [移动止损]
2022-11-14 ~ 2022-11-30  STOP_LOSS  -0.11%   Trending-Recovery bias=0.50  hold=12d [移动止损]
```

---

### 二、压力测试结果（熊市 2022）

| 标的 | 策略收益 | B&H收益 | Alpha | 交易数 | 最大回撤 |
|------|---------|---------|-------|--------|---------|
| NVDA 2022 | **-7.45%** | -51.44% | **+44.0%** ✅ | 7 | -8.12% |
| MSFT 2022 | **-6.31%** | -27.69% | **+21.4%** ✅ | 5 | -6.56% |
| GOOGL 2022 | **-0.17%** | -39.15% | **+39.0%** ✅ | 4 | -3.29% |

> **结论：熊市下策略有效，大幅减少最大回撤，资本保全能力强。**

---

### 三、规则基准对比（EMA200 + MACD + ADX > 20 纯规则策略）

| 标的/期间 | LLM策略 | 规则基准 | B&H | LLM vs 规则 |
|----------|---------|---------|-----|------------|
| NVDA 2025 | +1.48% | +0.07%（Sharpe 0.04） | ~+80% | LLM略好，两者均远跑输B&H |
| MSFT 2025 | -0.01% | **+8.56%**（Sharpe 1.27） | +17.32% | **规则策略碾压LLM** ⚠️ |
| GOOGL 2025 | 未测试2025 | **+6.29%**（Sharpe 1.67） | — | — |
| NVDA 2022 | -7.45% | -0.81%（Sharpe -0.42） | -51% | LLM更激进，规则更保守 |
| MSFT 2022 | -6.31% | -1.08%（Sharpe -0.40） | -27.7% | 两者均保本 |
| GOOGL 2022 | -0.17% | -1.01%（Sharpe -0.47） | -39.2% | LLM略好 |

> **MSFT 2025 警示**：规则基准 +8.56%，LLM 策略 -0.01%。LLM 信号在 MSFT 2025 牛市中**完全失效**，无增量价值。

---

### 四、量化研究员诊断（截至 2026-04-05）

#### 核心问题

**1. 样本量严重不足 — 所有结论均不具统计显著性**
- NVDA/MSFT/GOOGL portfolio 回测每个只有 2-4 笔完成交易
- 统计显著性最低要求：30 笔；推荐：100 笔以上
- 当前数据量只能描述「发生了什么」，无法验证「是否有 Alpha」

**2. 信号过滤过于激进（92-96% no_trade 率）**
- 理想 no_trade 率：50-70%（保留足够交易频率）
- 当前问题：大量有效 Trending-Up 信号被 `BAD_RR` 拦截（信号日收盘价计算止损，次日开盘入场导致 RR 降级）
- MSFT 2025-02-04：信号生成但 `ENTRY_SKIP rr=1.35` → 错过后续 +22% 行情

**3. 牛市 Alpha 漏出（最致命的结构性问题）**
- 策略在牛市中过于保守，大量 no_trade → 跑输 B&H 数十个百分点
- 本质是：信号过滤成本 > 信号准确率带来的收益
- 与 B&H 对比：NVDA +1.48% vs +80%，MSFT -0.01% vs +17%

**4. 多版本迭代等于多次假设检验（过拟合风险）**
- 已迭代多个版本，每次看结果再改规则 = 对历史路径的隐式拟合
- 当前没有真正封存的 holdout 集（`freeze_holdout.py` 可解决）

**5. 制度依赖性强**
- 熊市（2022）：有效，+39-44% Alpha vs B&H ✅
- 牛市（2025）：无效，-17 至 -78% Alpha vs B&H ❌
- 这不是 Alpha，是「不对称 beta 暴露」= 熊市保本 + 牛市踩空

#### 待解决的技术问题

| 优先级 | 问题 | 根因 | 建议 |
|--------|------|------|------|
| 🔴 P0 | BAD_RR 大量拦截有效信号 | stop/target 用信号日收盘算，入场用次日开盘 | 改用「next_open + 2.5×ATR」重算 RR |
| 🔴 P0 | 样本量不足 | 每年只有 2-4 笔交易 | 多资产多年并跑，目标每资产 60+ 笔 |
| 🟡 P1 | IC 分析样本量不足 | 本地信号记录少 | 从服务器拉日志后运行 `validate_ic.py --log-dir logs_server` |
| 🟡 P1 | 无真实 OOS holdout | 所有期间均参与了参数调整 | 运行 `freeze_holdout.py` 封存 2024 全年数据 |

#### 下一步行动（按优先级）

```bash
# 1. 安装缺失依赖
pip install scipy

# 2. 运行 IC 分析（最直接的信号质量检验）
scp -r root@101.201.171.174:/opt/finance-analysis/logs ./logs_server
python3 validate_ic.py --log-dir logs_server

# 3. 扩大样本：多资产多年并跑
python3 tech_backtest_engine.py --ticker NVDA --portfolio --start 2022-01-01 --end 2025-12-31

# 4. 验证 BAD_RR 修复效果（改完后先用 2023 年数据测）
```

---

## 依赖安装

```bash
# 基础依赖
pip install yfinance pandas numpy anthropic openai curl_cffi urllib3 httpx

# 量化分析依赖（IC 分析所需）
pip install scipy statsmodels

# 或直接从 requirements.txt 安装
pip install -r requirements.txt
```

---

## 费用参考

| 项目 | 费用 |
|------|------|
| 阿里云 ECS（2核2G） | ~¥24/月 |
| Claude API（单次分析） | ~$0.01–0.05 |
| Claude API（全资产扫描，~12个资产） | ~$0.15–0.60 |
| DeepSeek API（单次） | ~¥0.001（极便宜，推荐批量回测使用） |
| yfinance 行情数据 | 免费 |
| CoinGecko PAXG 价格 | 免费 |
| 飞书机器人 | 免费 |

---

## 常见注意事项

- **SSL 证书**：`urllib3.disable_warnings` + `curl_cffi` 的 `verify=False` 用于企业 VPN/代理环境，生产环境建议启用验证。
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
- **新增资产**：见下方「新增资产完整操作链路」章节，只需改 `assets_config.py` 一处，其余脚本自动感知。

---

## 新增 / 删除资产完整操作链路

> **设计原则**：`assets_config.py` 是唯一配置来源（Single Source of Truth）。  
> `news_signal_bridge.py`、`run_daily.sh`、RAG 采集器三个模块在启动时动态读取，  
> **新增或删除资产只需改一个文件**。

---

### 新增资产：3 步操作

#### Step 1 — `assets_config.py`（必改，唯一入口）

在 `ASSET_UNIVERSE` 里追加一条记录，填写所有字段：

```python
"TSLA": {
    # ── 基础信息 ──
    "ticker":       "TSLA",              # yfinance ticker
    "type":         "equity",            # equity / etf / commodity / crypto
    "script":       "tech_stock_analysis.py",
    "script_args":  ["--ticker", "TSLA"],
    "output_file":  "outputs/tsla_api_output.txt",
    "prompt_file":  "outputs/tsla_prompt_output.txt",
    "backtest_dir": None,
    "sector":       "Technology/EV",
    "ccy":          "USD",
    "description":  "Tesla",
    # ── 每日定时任务 ──
    "daily_scan":       True,            # True = 纳入 run_daily.sh 每日分析
    "daily_extra_args": [],              # 传给脚本的额外参数，如 ["--trade"]
    # ── 新闻 RAG（腾讯云 news-rag-system）──
    "rag_weight":        0.10,           # 0.0 = 禁用 RAG 监控
    "news_keywords":     ["TSLA", "Tesla", "Elon Musk", "Cybertruck",
                          "Full Self-Driving", "FSD", "EV", "Giga"],
    "news_primary_terms":["TSLA", "Tesla"],   # 整词匹配，防止 "Tesla" ≠ "Teslara"
    # ── SEC / 财报采集（equity 类型填写；ETF/commodity/crypto 留 None/False）──
    "sec_cik":           "0001318605",   # SEC EDGAR CIK（equity 必填）
    "earnings_tracking": True,           # True = Polygon 季报财务采集
    "insider_tracking":  True,           # True = Form 4 内部人交易监控
},
```

如需纳入某个扫描分组，同时更新 `SCAN_GROUPS`：

```python
SCAN_GROUPS = {
    "tech": ["GOOGL", "MSFT", "NVDA", "AAPL", "META", "AMZN", "TSLA"],  # 加在此处
    ...
}
```

**`type` 字段对应关系：**

| type | 适用 | sec_cik | earnings_tracking | insider_tracking |
|------|------|---------|-------------------|-----------------|
| `equity` | 个股（TSLA、NVDA…） | 必填 | 视需要 | 视需要 |
| `etf` | ETF（SLV、COPX…） | None | False | False |
| `commodity` | 期货（GC=F…） | None | False | False |
| `crypto` | 加密货币（BTC-USD…） | None | False | False |

---

#### Step 2 — `tech_stock_analysis.py` → `_INDUSTRY_CONTEXT`（可选，提升分析质量）

不加也能运行（使用通用 prompt），但加了 LLM 分析质量更高：

```python
"TSLA": """
Tesla-specific analysis dimensions:
- Quarterly delivery numbers vs. Wall Street estimates (key market mover)
- Full Self-Driving (FSD) regulatory progress and subscription attach rate
- Energy storage (Megapack) revenue as gross margin diversifier
- Elon Musk political/brand risk and distraction factor
- China market share vs. BYD / NIO competition
- Gross margin trajectory (vehicle + software mix shift)
""",
```

---

#### Step 3 — RAG 系统同步（腾讯云，`43.139.5.125`）

```bash
# 本地 Mac 执行：同步 portfolio.json（由 assets_config 自动生成，无需手动编辑）
python3 - <<'EOF'
import json, sys
sys.path.insert(0, "/Users/zhongsongzhi/Desktop/大模型金融分析")
from assets_config import get_rag_portfolio
print(json.dumps(get_rag_portfolio(), ensure_ascii=False, indent=2))
EOF

# 或直接 rsync 整个 config 目录
rsync -avz /Users/zhongsongzhi/Documents/news-rag-system/config/ \
    ubuntu@43.139.5.125:/home/ubuntu/news-rag-system/config/

# 服务器上重启 preprocessor 使配置生效（5 分钟内开始收录新资产新闻）
ssh ubuntu@43.139.5.125 "sudo systemctl restart preprocessor"
```

> **注**：`polygon.py` / `sec_edgar.py` / `form4.py` 三个采集器在启动时自动从 `assets_config.py`
> 读取 `EARNINGS_TICKERS` / `COMPANY_8K` / `TICKERS`，无需手动修改。
> 如果 `ASSETS_CONFIG_DIR` 环境变量未设置，采集器会使用硬编码的兜底列表并打印警告。

---

### 删除资产：2 步操作

1. **`assets_config.py`**：从 `ASSET_UNIVERSE` 删除对应条目，同时从 `SCAN_GROUPS` 的各分组中移除。
2. **RAG 系统同步**：重新 rsync + restart preprocessor（旧数据保留在数据库中，只是不再采集新数据）。

不需要改其他任何文件。

---

### 修改每日分析范围（不增删资产）

只需修改 `assets_config.py` 中的 `daily_scan` 字段：

```python
# 暂停某资产的每日分析（如 AMZN 财报期静默）
"daily_scan": False,

# 恢复
"daily_scan": True,
```

`run_daily.sh` 下次运行时自动感知，无需重启任何服务。

---

### 各模块读取 `assets_config` 的方式

| 模块 | 调用函数 | 读取内容 |
|------|---------|---------|
| `run_daily.sh` | `get_daily_assets()` | `daily_scan=True` 的资产列表 + `daily_extra_args` |
| `news_signal_bridge.py` | `get_news_keywords(ticker)` | RAG 检索扩展词 |
| `news_signal_bridge.py` | `get_news_primary_terms(ticker)` | 主标识符（整词匹配） |
| `polygon.py` | `get_earnings_tracked_tickers()` | Polygon 季报采集列表 |
| `sec_edgar.py` | `get_sec_tracked_assets()` | SEC 8-K 监控 `{ticker: CIK}` |
| `form4.py` | `get_sec_tracked_assets()` | Form 4 监控 `{ticker: CIK}` |
| `market_scan.py` | `ASSET_UNIVERSE` / `SCAN_GROUPS` | 全量资产路由 + 分组 |
| `portfolio_tracker.py` | `ASSET_UNIVERSE` | 持仓评估 + Beta Overlay |

---

### 新增数据源（非资产，例如新闻 RSS / 新 API）

在 RAG 系统（腾讯云）中操作：

1. 在 `preprocessor/collectors/` 新建采集器文件，实现 `@collector("name")` 装饰的异步函数
2. 在 `preprocessor/scheduler.py` 注册定时任务（interval 或 cron）
3. 在 `preprocessor/main.py` 的 `STALE_DAYS` 和 `DATA_TYPE_MAP` 里添加对应配置
4. Rsync + `systemctl restart preprocessor`

不影响金融分析系统（阿里云）任何代码。
