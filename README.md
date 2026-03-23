# 大模型金融分析系统 — 使用手册

---

## 一、项目概览

本系统以 Claude / DeepSeek 大语言模型为核心决策引擎，每天自动采集以下资产的行情与宏观数据，生成结构化交易建议，并推送到飞书群：

| 资产   | 品种                      | 持仓周期            | 分析脚本                 |
| ------ | ------------------------- | ------------------- | ------------------------ |
| 黄金   | GC=F 期货 + PAXG 链上代币 | 短线（日线）        | `gold_analysis.py`       |
| 比特币 | BTC-USD                   | 中长线（6个月~3年） | `btc_analysis.py`        |
| 谷歌   | GOOGL                     | 中线（1~6个月）     | `tech_stock_analysis.py` |
| 英伟达 | NVDA                      | 中线（1~6个月）     | `tech_stock_analysis.py` |
| 亚马逊 | AMZN                      | 中线（1~6个月）     | `tech_stock_analysis.py` |

---

## 二、目录结构

```
大模型金融分析/
├── gold_analysis.py          # 黄金实时分析（含 PAXG 链上价格）
├── btc_analysis.py           # BTC 战略分析
├── tech_stock_analysis.py    # 科技股分析（GOOGL / NVDA / AMZN 等）
├── backtest_engine.py        # 历史回测引擎（黄金）
├── feishu_notifier.py        # 飞书推送器
├── run_daily.sh              # 每日定时任务脚本
├── run_weekly.sh             # 每周汇总脚本
├── setup_cron.sh             # 服务器 crontab 配置
├── setup_proxy.sh            # 服务器代理安装（Shadowsocks + Privoxy）
├── deploy.sh                 # 一键部署到阿里云 ECS
├── requirements.txt          # Python 依赖列表
├── .env                      # API Key 和配置（不提交 git）
├── .env.example              # 环境变量模板
├── com.finance.daily.plist   # macOS launchd 每日定时配置
├── com.finance.weekly.plist  # macOS launchd 每周定时配置
├── gold_prompt_output.txt    # 最新黄金提示词（自动覆盖）
├── gold_api_output.txt       # 最新黄金分析结果（自动覆盖）
├── btc_prompt_output.txt     # 最新 BTC 提示词
├── btc_api_output.txt        # 最新 BTC 分析结果
├── googl_api_output.txt      # 最新 GOOGL 分析结果
├── nvda_api_output.txt       # 最新 NVDA 分析结果
├── amzn_api_output.txt       # 最新 AMZN 分析结果
├── logs/                     # 运行日志（自动生成）
├── backtest_prompts/         # 回测用盲化提示词
├── backtest_responses/       # 手动回测 LLM 响应
└── backtest_results/         # 回测输出（signals.csv / performance.csv）
```

---

## 三、本地 Mac 运行

### 手动运行单次分析

```bash
cd ~/Desktop/大模型金融分析
source .env

# 黄金分析（生成提示词文件，不调用 API）
python3 gold_analysis.py

# 黄金分析（直接调用 Claude API）
python3 gold_analysis.py --api

# BTC 分析
python3 btc_analysis.py --api

# 科技股分析
python3 tech_stock_analysis.py --ticker GOOGL --api
python3 tech_stock_analysis.py --ticker NVDA --api
python3 tech_stock_analysis.py --ticker AMZN --api

# 使用 DeepSeek 模型替代 Claude
python3 gold_analysis.py --api --model deepseek-reasoner
```

### 手动推送飞书

```bash
source .env
# 推送当日报告（读取现有输出文件）
python3 feishu_notifier.py --mode daily

# 推送周报
python3 feishu_notifier.py --mode weekly

# 测试飞书连通性
python3 feishu_notifier.py --mode test
```

### Mac 定时任务管理

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

| 项目        | 内容                        |
| ----------- | --------------------------- |
| 公网 IP     | 101.201.171.174             |
| 登录用户    | root                        |
| 项目目录    | /opt/finance-analysis       |
| Python 环境 | /opt/finance-analysis/.venv |
| 日志目录    | /opt/finance-analysis/logs  |

### SSH 登录

```bash
ssh root@101.201.171.174
```

### 常用服务器操作

```bash
# 进入项目目录
cd /opt/finance-analysis

# 加载环境变量
source .env

# 手动触发每日分析
bash run_daily.sh

# 手动触发周报
bash run_weekly.sh

# 查看今日运行日志
tail -f logs/daily_$(date +%Y%m%d).log

# 查看 crontab 日志
tail -f logs/cron.log

# 查看所有日志文件
ls -lh logs/
```

### crontab 定时任务

```bash
# 查看当前定时任务
crontab -l

# 修改定时任务
crontab -e
```

当前配置：

```
# 每天 08:00 CST（北京时间）→ 日报
0 0 * * * bash /opt/finance-analysis/run_daily.sh >> /opt/finance-analysis/logs/cron.log 2>&1

# 每周一 08:30 CST → 周报
30 0 * * 1 bash /opt/finance-analysis/run_weekly.sh >> /opt/finance-analysis/logs/cron.log 2>&1
```

### 代理服务管理（Shadowsocks + Privoxy）

```bash
# 查看代理状态
systemctl status privoxy
pgrep -a ss-local

# 重启代理
pkill -f ss-local
nohup ss-local -c /etc/shadowsocks-libev/local.json > /var/log/ss-local.log 2>&1 &
systemctl restart privoxy

# 验证代理可用
curl --proxy http://127.0.0.1:8118 https://finance.yahoo.com -I -s | head -3

# 查看代理日志
tail -20 /var/log/ss-local.log
journalctl -u privoxy -n 30
```

### 重新部署（代码更新后）

在 Mac 上执行（不带 --init，跳过依赖安装）：

```bash
cd ~/Desktop/大模型金融分析
bash deploy.sh 101.201.171.174 root
```

---

## 五、配置文件 .env 说明

服务器路径：`/opt/finance-analysis/.env`
本地路径：`~/Desktop/大模型金融分析/.env`

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

# 飞书不走代理
NO_PROXY=open.feishu.cn,feishu.cn
```

修改 .env 后不需要重启任何服务，每次 `source .env` 或脚本运行时自动加载。

---

## 六、飞书消息说明

### 每日报告（08:00）包含：

- 黄金：操作方向、入场区间、止盈止损、R:R、PAXG 链上价差
- BTC：减半周期、情绪指数、操作方向、仓位建议
- GOOGL / NVDA / AMZN：市场制度、财报风险、操作方向

### 每周汇总（周一 08:30）包含：

- 以上所有资产的完整深度分析（含宏观逻辑、技术面、估值背景）

### 信号说明：

| 信号        | 含义                                    |
| ----------- | --------------------------------------- |
| 做多 [多]   | 建议做多，有明确入场、止盈、止损价位    |
| 做空 [空]   | 建议做空                                |
| 观望 [空仓] | 信号不足，建议不操作                    |
| bias_score  | 0~1，≥0.50 才会给出交易建议，越高越确定 |

---

## 七、回测系统

```bash
# 生成历史提示词（不需要 API Key）
python3 backtest_engine.py --generate --start 2024-01-01 --end 2024-12-31 --step 5

# 评估已有的手动回测响应
python3 backtest_engine.py --evaluate

# 全自动回测（需要 DeepSeek API Key，便宜）
python3 backtest_engine.py --start 2025-01-01 --end 2025-12-31

# 断点续跑
python3 backtest_engine.py --start 2025-01-01 --end 2025-12-31 --resume
```

回测结果保存在 `backtest_results/`，会被下次实时分析读取用于动态调整决策阈值。

---

## 八、常见问题

### Q：yfinance 报 YFRateLimitError

服务器 IP 被 Yahoo Finance 限速。检查代理是否正常：

```bash
systemctl status privoxy
curl --proxy http://127.0.0.1:8118 https://finance.yahoo.com -I -s | head -3
```

如代理异常，重启：

```bash
pkill -f ss-local && sleep 1
nohup ss-local -c /etc/shadowsocks-libev/local.json > /var/log/ss-local.log 2>&1 &
systemctl restart privoxy
```

### Q：飞书发送失败

检查 Webhook URL 是否正确，以及 NO_PROXY 是否设置：

```bash
source .env && python3 feishu_notifier.py --mode test
```

### Q：分析结果为空或 JSON 解析失败

查看对应的 `*_api_output.txt` 文件内容，看是否有报错信息而非 JSON：

```bash
cat gold_api_output.txt
```

### Q：服务器重启后代理失效

每次重启需重新启动 ss-local（privoxy 开机自启）：

```bash
nohup ss-local -c /etc/shadowsocks-libev/local.json > /var/log/ss-local.log 2>&1 &
```

建议创建 systemd 服务让 ss-local 开机自启（见下方）：

```bash
cat > /etc/systemd/system/ss-local.service << EOF
[Unit]
Description=Shadowsocks Local Client
After=network.target

[Service]
ExecStart=/usr/bin/ss-local -c /etc/shadowsocks-libev/local.json
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl enable ss-local
systemctl start ss-local
```

### Q：想更换 Webhook（飞书群换了）

```bash
nano /opt/finance-analysis/.env
# 修改 FEISHU_WEBHOOK_URL 那行
# 保存后测试：
source .env && python3 feishu_notifier.py --mode test
```

---

## 九、费用说明

| 项目                | 费用                   |
| ------------------- | ---------------------- |
| 阿里云 ECS（2核2G） | ~¥24/月                |
| Claude API          | 约 $0.01~0.05/次分析   |
| DeepSeek API        | 约 ¥0.001/次（极便宜） |
| yfinance 数据       | 免费                   |
| CoinGecko PAXG      | 免费                   |
| Binance 资金费率    | 免费                   |
| 恐惧贪婪指数        | 免费                   |
| 飞书机器人          | 免费                   |
