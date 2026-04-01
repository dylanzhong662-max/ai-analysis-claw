# 大模型金融分析系统 — 使用手册

---

## 一、项目概览

本系统以 Claude / DeepSeek 大语言模型为核心决策引擎，自动采集多资产行情与宏观数据，生成结构化交易建议，并支持持仓跟踪和自动下单。

### 支持的资产

| 分组 | 资产 | Ticker | 持仓周期 | 分析脚本 |
|------|------|--------|----------|----------|
| **贵金属** | 黄金 | GC=F | 短线（日线） | `gold_analysis.py` |
| | 白银 ETF | SLV | 中线 | `tech_stock_analysis.py` |
| **工业金属** | 铜矿 ETF | COPX | 中线 | `tech_stock_analysis.py` |
| | 稀土/钨 ETF | REMX | 中线 | `tech_stock_analysis.py` |
| **能源** | 原油 ETF | USO | 中线 | `tech_stock_analysis.py` |
| **加密货币** | 比特币 | BTC-USD | 长线（6个月~3年） | `btc_analysis.py` |
| **科技股** | Alphabet | GOOGL | 中线（1~6个月） | `tech_stock_analysis.py` |
| | Microsoft | MSFT | 中线 | `tech_stock_analysis.py` |
| | NVIDIA | NVDA | 中线 | `tech_stock_analysis.py` |
| | Apple | AAPL | 中线 | `tech_stock_analysis.py` |
| | Meta | META | 中线 | `tech_stock_analysis.py` |
| | Amazon | AMZN | 中线 | `tech_stock_analysis.py` |

所有资产配置集中在 `assets_config.py`，新增资产只需在此注册。

---

## 二、目录结构

```
大模型金融分析/
├── gold_analysis.py          # 黄金实时分析（含 PAXG 链上价格 + 自动下单）
├── btc_analysis.py           # BTC 战略分析
├── tech_stock_analysis.py    # 科技股 + 大宗商品 ETF 分析
├── market_scan.py            # 多资产横向扫描 + 跨资产机会排名（新）
├── portfolio_tracker.py      # 持仓跟踪 + 操作建议生成器（新）
├── assets_config.py          # 资产注册表 + 扫描分组配置（新）
├── portfolio.json            # 当前持仓文件（用户维护）（新）
├── backtest_engine.py        # 历史回测引擎（黄金）
├── google_backtest.py        # 历史回测引擎（科技股）
├── feishu_notifier.py        # 飞书推送器
├── run_daily.sh              # 每日定时任务脚本
├── run_weekly.sh             # 每周汇总脚本
├── setup_cron.sh             # 服务器 crontab 配置
├── deploy.sh                 # 一键部署到阿里云 ECS
├── requirements.txt          # Python 依赖列表
│
├── gold_prompt_output.txt    # 最新黄金提示词（自动覆盖）
├── gold_api_output.txt       # 最新黄金分析结果
├── btc_api_output.txt        # 最新 BTC 分析结果
├── {ticker}_api_output.txt   # 各股票/ETF 最新分析结果
├── market_scan_output.json   # 最新多资产扫描结果（结构化）
├── market_scan_report.txt    # 最新多资产扫描报告（原始文本）
├── portfolio_status.json     # 最新持仓状态评估结果（新）
├── orders.json               # 待执行订单列表（--export-orders 时生成）（新）
│
├── logs/                     # 运行日志
├── backtest_prompts/         # 黄金回测用盲化提示词
├── backtest_responses/       # 手动回测 LLM 响应
├── backtest_results/         # 黄金回测输出（signals.csv / performance.csv）
└── googl_backtest_results/   # GOOGL 回测输出
```

---

## 三、本地 Mac 运行

### 3.1 单资产分析

```bash
cd ~/Desktop/大模型金融分析
source .env

# 黄金
python3 gold_analysis.py                              # 只生成提示词文件
python3 gold_analysis.py --api                        # 调用 Claude 直接分析
python3 gold_analysis.py --api --model deepseek-reasoner
python3 gold_analysis.py --api --trade                # 分析 + 自动下单 PAXG/USDT
python3 gold_analysis.py --api --trade --dry-run      # 模拟下单（不实际成交）

# BTC
python3 btc_analysis.py --api
python3 btc_analysis.py --api --model deepseek-reasoner

# 科技股
python3 tech_stock_analysis.py --ticker GOOGL --api
python3 tech_stock_analysis.py --ticker NVDA  --api
python3 tech_stock_analysis.py --ticker MSFT  --api
python3 tech_stock_analysis.py --ticker AAPL  --api
python3 tech_stock_analysis.py --ticker META  --api
python3 tech_stock_analysis.py --ticker AMZN  --api

# 大宗商品 ETF
python3 tech_stock_analysis.py --ticker SLV  --api   # 白银
python3 tech_stock_analysis.py --ticker COPX --api   # 铜矿
python3 tech_stock_analysis.py --ticker REMX --api   # 稀土/钨
python3 tech_stock_analysis.py --ticker USO  --api   # 原油
```

### 3.2 多资产横向扫描（新功能）

一次性分析多个资产，LLM 额外输出：
- 当前宏观主题识别（AI 算力周期、贵金属避险等）
- 板块强弱排名（Strong / Neutral / Weak）
- Top 5 交易机会排名
- 持仓相关性风险提示

```bash
# 快速扫描（GOLD + BTC + NVDA + MSFT，日常推荐）
python3 market_scan.py --api

# 按预定义分组扫描
python3 market_scan.py --group tech     --api   # 6 只科技股
python3 market_scan.py --group metals   --api   # 黄金/白银/铜/稀土
python3 market_scan.py --group commodities --api
python3 market_scan.py --group all      --api   # 全部资产（耗时约 15 分钟）

# 自定义资产列表
python3 market_scan.py --assets GOLD NVDA AAPL SLV --api

# 跳过重新分析，直接对已有输出做汇总排名（节省时间）
python3 market_scan.py --group tech --skip-individual --api

# 使用 DeepSeek 降低 API 成本
python3 market_scan.py --group all --api --model deepseek-chat
```

**输出文件**：
- `market_scan_report.txt` — 完整扫描报告
- `market_scan_output.json` — 结构化 JSON，包含 `top_opportunities`、`sector_ranking`、`correlation_risks`

### 3.3 持仓跟踪（新功能）

#### 第一步：维护 portfolio.json

```json
{
  "positions": [
    {
      "asset":          "GOLD",
      "type":           "long",
      "entry_price":    3200.0,
      "entry_date":     "2026-03-15",
      "quantity":       0.01,
      "stop_loss":      3100.0,
      "profit_target":  3500.0,
      "exchange":       "Binance",
      "symbol":         "PAXGUSDT",
      "notes":          "可选备注"
    }
  ]
}
```

`asset` 字段必须与 `assets_config.py` 中的 key 一致（大写），例如 `GOLD`、`NVDA`、`SLV`。

#### 第二步：运行跟踪器

```bash
# 查看持仓状态（使用已有信号文件，不重新分析）
python3 portfolio_tracker.py

# 先刷新所有持仓资产的 LLM 信号，再评估
python3 portfolio_tracker.py --update-signals

# 导出 orders.json（供后续交易接口读取）
python3 portfolio_tracker.py --export-orders

# 完整流程：刷新信号 + 评估 + 导出订单
python3 portfolio_tracker.py --update-signals --export-orders
```

#### 操作建议说明

| 状态 | 触发条件 | 生成订单 |
|------|---------|---------|
| `HOLD` | 信号方向一致，价格正常 | 无 |
| `STOP_TRIGGERED` | 当前价已触及/跌破止损 | 市价平仓 |
| `TARGET_REACHED` | 当前价已触及/超过目标价 | 限价锁利 |
| `SIGNAL_REVERSED` | LLM 最新信号方向与持仓相反 | 市价平仓 |
| `REDUCE` | LLM 信号变为 no_trade（bias < 0.5） | 市价减半仓 |

#### orders.json 格式（交易接口对接）

```json
{
  "generated_at": "2026-03-30T10:00:00",
  "orders": [
    {
      "asset":      "GOLD",
      "action":     "STOP_TRIGGERED",
      "side":       "sell",
      "quantity":   0.01,
      "order_type": "market",
      "price":      3080.0,
      "note":       "STOP_TRIGGERED"
    }
  ]
}
```

### 3.4 飞书推送

```bash
source .env
python3 feishu_notifier.py --mode daily    # 推送当日报告
python3 feishu_notifier.py --mode weekly   # 推送周报
python3 feishu_notifier.py --mode test     # 测试连通性
```

### 3.5 Mac 定时任务管理

```bash
# 注册定时任务（首次）
launchctl load ~/Library/LaunchAgents/com.finance.daily.plist
launchctl load ~/Library/LaunchAgents/com.finance.weekly.plist

# 取消注册
launchctl unload ~/Library/LaunchAgents/com.finance.daily.plist

# 查看任务状态
launchctl list | grep finance

# 立即触发一次（测试）
launchctl start com.finance.daily
```

---

## 四、服务器（阿里云 ECS）管理

### 服务器基本信息

| 项目 | 内容 |
|------|------|
| 公网 IP | 101.201.171.174 |
| 登录用户 | root |
| 项目目录 | /opt/finance-analysis |
| Python 环境 | /opt/finance-analysis/.venv |
| 日志目录 | /opt/finance-analysis/logs |

### SSH 登录

```bash
ssh root@101.201.171.174
```

### 常用服务器操作

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
# 每天 08:00 CST
0 0 * * * bash /opt/finance-analysis/run_daily.sh >> /opt/finance-analysis/logs/cron.log 2>&1

# 每周一 08:30 CST
30 0 * * 1 bash /opt/finance-analysis/run_weekly.sh >> /opt/finance-analysis/logs/cron.log 2>&1
```

### 代理服务管理（Shadowsocks + Privoxy）

```bash
# 查看代理状态
systemctl status ss-local
systemctl status privoxy

# 重启代理
systemctl restart ss-local && systemctl restart privoxy

# 验证代理链路
curl --socks5 127.0.0.1:1080 http://httpbin.org/ip -s
curl --proxy http://127.0.0.1:8118 http://httpbin.org/ip -s
```

### 更新 Shadowsocks 节点配置

```bash
# 获取最新配置
curl -s "https://ss.wawaapp.net/t/520fa9d967e39ce4b19a54c88312e52d2991ecf63894998e00f031308"

# 更新并重启
nano /etc/shadowsocks-libev/local.json
systemctl restart ss-local
```

### 重新部署（代码更新后）

```bash
cd ~/Desktop/大模型金融分析
bash deploy.sh 101.201.171.174 root
```

---

## 五、配置文件 .env 说明

```bash
# Claude API
ANTHROPIC_API_KEY=sk-...
ANTHROPIC_BASE_URL=https://api.openai-proxy.org/anthropic

# DeepSeek API
DEEPSEEK_API_KEY=sk-...

# 飞书群机器人 Webhook
FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/...

# HTTP 代理（服务器需要，本地 Mac 不需要）
HTTPS_PROXY=http://127.0.0.1:8118
HTTP_PROXY=http://127.0.0.1:8118
NO_PROXY=open.feishu.cn,feishu.cn
```

---

## 六、飞书消息说明

### 每日报告（08:00）
- 黄金：方向、入场区间、止盈止损、R:R、PAXG 链上价差
- BTC：减半周期、情绪指数、方向、仓位建议
- 科技股（GOOGL / NVDA / AMZN）：制度、财报风险、方向

### 每周汇总（周一 08:30）
所有资产的完整深度分析（含宏观逻辑、技术面、估值背景）

### 信号说明

| 字段 | 含义 |
|------|------|
| `action: long` | 建议做多 |
| `action: short` | 建议做空 |
| `action: no_trade` | 信号不足，观望 |
| `bias_score` | 0~1，≥ 0.50 才给出交易建议，越高越确定 |

---

## 七、回测系统

### 回测引擎说明（v3）

| 引擎 | 适用资产 | EVAL_DAYS | 关键特性 |
|------|---------|-----------|---------|
| `backtest_engine.py` | 黄金（GC=F） | 20 | Mean-Reverting 过滤，1.5×ATR 止损 |
| `tech_backtest_engine.py` | 科技股 / ETF | 22 | 仓位管理，SHORT_FILTERED，Parquet 缓存 |
| `btc_backtest_engine.py` | BTC | 60 | 长周期评估，减半周期上下文 |

### 黄金回测

```bash
# 全自动回测（需要 DeepSeek API Key）
python3 backtest_engine.py --start 2025-01-01 --end 2025-12-31

# 断点续跑
python3 backtest_engine.py --start 2025-01-01 --end 2025-12-31 --resume
```

### 科技股回测（推荐使用新引擎）

```bash
# NVDA / MSFT / GOOGL / AAPL / META / AMZN / SLV / COPX / REMX / USO
python3 tech_backtest_engine.py --ticker NVDA --start 2025-01-01 --end 2025-12-31
python3 tech_backtest_engine.py --ticker MSFT --start 2025-01-01 --end 2025-12-31

# 断点续跑
python3 tech_backtest_engine.py --ticker NVDA --start 2025-01-01 --end 2025-12-31 --resume
```

### 批量回测

```bash
# 同时跑多个资产 / 多个年份（后台 nohup）
python3 run_all_backtests.py --assets GOLD NVDA MSFT --period both --model deepseek-reasoner
```

### BTC 回测

```bash
python3 btc_backtest_engine.py --start 2024-01-01 --end 2025-12-31
```

### 回测结果自动反馈

回测完成后 `{ticker}_backtest_results/performance.csv` 自动更新。下次运行实时分析时，脚本自动读取最新胜率并注入提示词，动态调整 LLM 的入场阈值：

- 胜率 < 40% → `bias_score` 门槛提升至 ≥ 0.65
- 连续亏损 ≥ 2 次 → 需 `bias_score` ≥ 0.75 才入场

### 数据缓存说明

`tech_backtest_engine.py` 使用 Parquet 持久化缓存（`data_cache/`），首次运行会下载并保存，后续回测直接读本地文件。若 Yahoo Finance 限流（429），等待 2–4 小时再重跑，缓存不会丢失。

### 新增资产的回测方案

| 资产类型 | 推荐引擎 | 说明 |
|---------|---------|------|
| 黄金（GC=F） | `backtest_engine.py` | 专属提示词和宏观指标 |
| 科技股 / ETF | `tech_backtest_engine.py --ticker XXX` | 通用，内置 `_INDUSTRY_CONTEXT` |
| BTC | `btc_backtest_engine.py` | 专属减半周期逻辑 |

---

## 八、新增资产配置方法

所有资产统一在 `assets_config.py` 中注册：

```python
"TSLA": {
    "ticker":       "TSLA",
    "type":         "equity",
    "script":       "tech_stock_analysis.py",
    "script_args":  ["--ticker", "TSLA"],
    "output_file":  "tsla_api_output.txt",
    "prompt_file":  "tsla_prompt_output.txt",
    "backtest_dir": None,
    "sector":       "Technology/EV",
    "ccy":          "USD",
    "description":  "Tesla",
},
```

注册后：
- `market_scan.py` 自动识别并扫描
- `portfolio_tracker.py` 可跟踪该资产的持仓
- 在 `tech_stock_analysis.py` 的 `_INDUSTRY_CONTEXT` 字典中添加该 ticker 的专属分析维度（可选，不加则使用通用提示词）

---

## 九、常见问题

### Q：yfinance 报 YFRateLimitError

服务器代理链路断了，排查步骤：

```bash
# 检查代理
curl --socks5 127.0.0.1:1080 http://httpbin.org/ip -s

# 更新 SS 节点配置
curl -s "https://ss.wawaapp.net/t/520fa9d967e39ce4b19a54c88312e52d2991ecf63894998e00f031308"
nano /etc/shadowsocks-libev/local.json
systemctl restart ss-local
```

### Q：market_scan 某个资产输出为"信号解析失败"

该资产的 `*_api_output.txt` 文件不存在或格式异常。先单独运行该资产的分析脚本：

```bash
python3 tech_stock_analysis.py --ticker NVDA --api
```

然后再运行 `market_scan.py --skip-individual`。

### Q：portfolio_tracker 提示"无最新信号文件"

运行 `portfolio_tracker.py --update-signals` 刷新信号，或手动运行对应资产的分析脚本。

### Q：orders.json 生成了，如何对接交易接口

`orders.json` 中每个订单的 `side`、`quantity`、`order_type`、`price` 字段已对齐 Binance API 格式。在交易接口中读取此文件并遍历 `orders` 数组，调用 `client.create_order()` 即可。详见 `gold_analysis.py` 中的 `execute_trade()` 函数作为参考实现。

### Q：飞书发送失败

```bash
source .env && python3 feishu_notifier.py --mode test
```

### Q：分析结果为空或 JSON 解析失败

```bash
cat gold_api_output.txt     # 查看原始输出是否有报错信息
```

---

## 十、费用说明

| 项目 | 费用 |
|------|------|
| 阿里云 ECS（2核2G） | ~¥24/月 |
| Claude API（单次分析） | 约 $0.01~0.05 |
| Claude API（全资产扫描，~12个资产） | 约 $0.15~0.60 |
| DeepSeek API（单次） | 约 ¥0.001（极便宜，推荐批量扫描使用） |
| yfinance 数据 | 免费 |
| CoinGecko PAXG | 免费 |
| 飞书机器人 | 免费 |
