# Backend — FastAPI 服务说明

## 概览

基于 FastAPI + SQLite 的 REST API 服务，为前端仪表盘提供数据接口。运行于 `8000` 端口。

## 目录结构

```
backend/
├── main.py               # FastAPI 入口，注册路由，创建数据库表
├── database.py           # SQLAlchemy 引擎配置，SQLite 连接
├── models.py             # ORM 模型（4 张表）
├── schemas.py            # Pydantic v2 请求/响应 Schema
├── signal_reader.py      # 读取项目根目录下的 *_api_output.txt 信号文件
├── price_fetcher.py      # yfinance 实时价格抓取（5 分钟内存缓存）
├── sync.py               # SQLite positions → portfolio.json 双向同步
└── routers/
    ├── dashboard.py      # GET /api/dashboard/summary  /macro
    ├── portfolio.py      # CRUD /api/portfolio/positions + /close
    ├── trades.py         # CRUD /api/trades + /stats
    ├── signals.py        # GET/POST /api/signals + /refresh/{asset}
    └── scan.py           # GET/POST /api/scan/latest + /run
```

## 数据库 Schema（trading.db）

数据库文件位于项目根目录 `trading.db`，由 SQLAlchemy 自动创建。

| 表名 | 用途 |
|------|------|
| `positions` | 活跃持仓，status = open / closed |
| `trades` | 已平仓交易历史 |
| `orders` | 待执行订单（对接交易接口用） |
| `signals_cache` | LLM 信号缓存（按 asset + analysis_date 去重） |

关键字段说明：
- `positions.status`：`open` = 持仓中，`closed` = 已平仓（平仓时自动生成 trade 记录）
- `trades.exit_reason`：`stop_loss` / `take_profit` / `manual` / `signal_reversed` / `timeout`
- `positions` 每次写入后自动通过 `sync.py` 更新 `portfolio.json`，保证现有 Python 脚本兼容

## API 端点速查

```
GET  /api/health                          健康检查
GET  /api/dashboard/summary               持仓 KPI + 告警
GET  /api/dashboard/macro                 宏观环境（VIX/DXY/10Y/BTC情绪）

GET  /api/portfolio/positions             所有开仓（含实时价格+P&L，约 5s）
POST /api/portfolio/positions             新建持仓
PUT  /api/portfolio/positions/{id}        更新止损/目标/备注
DELETE /api/portfolio/positions/{id}      删除持仓
POST /api/portfolio/positions/{id}/close  平仓 → 自动生成 trade 记录

GET  /api/trades?asset=&page=&limit=      交易记录列表
POST /api/trades                          手动录入交易
DELETE /api/trades/{id}                   删除记录
GET  /api/trades/stats                    统计（胜率/盈利因子/总盈亏）

GET  /api/signals                         所有资产最新信号摘要
GET  /api/signals/{asset}                 单资产完整信号（含 justification）
POST /api/signals/refresh/{asset}         后台触发重新分析（非阻塞）

GET  /api/scan/latest                     最新 market_scan_output.json
POST /api/scan/run?group=quick            后台触发多资产扫描（非阻塞）
```

完整文档：运行后访问 `http://localhost:8000/docs`（Swagger UI）

## 信号读取逻辑（signal_reader.py）

读取顺序：直接 JSON 解析 → markdown 代码块提取 → 大括号计数匹配，同时剥离 DeepSeek R1 的 `<think>` 标签。文件修改时间作为 `analysis_date` 显示给前端。

资产 → 文件映射：

| 资产 | 文件 | 触发脚本 |
|------|------|---------|
| GOLD | gold_api_output.txt | gold_analysis.py --api |
| BTC | btc_api_output.txt | btc_analysis.py --api |
| GOOGL/MSFT/NVDA/AAPL/META/AMZN | {ticker}_api_output.txt | tech_stock_analysis.py --ticker {TICKER} --api |
| SILVER/COPPER/RARE_EARTH/OIL | {ticker}_api_output.txt | tech_stock_analysis.py --ticker {TICKER} --api |

## 实时价格计算

`price_fetcher.get_current_price(asset)` 通过 yfinance 下载最近 2 日日线数据取最新收盘价，结果缓存 300 秒。`GET /api/portfolio/positions` 响应时间受此影响，12 个持仓最慢约 5–10 秒（首次无缓存）。

宏观价格（VIX/DXY/10Y）同理，由 `get_macro_prices()` 统一批量拉取。

## 运行方式

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload   # 开发模式
uvicorn main:app --host 0.0.0.0 --port 8000             # 生产模式
```

**必须在 `backend/` 目录下运行**，`sys.path` 配置依赖当前工作目录。

## 注意事项

- **Pydantic v2**：所有 schema 使用 `model_config = {"from_attributes": True}` 替代 v1 的 `class Config`
- **旧版 TradeResponse/TradeStats**：如果 schemas.py 里还有 `class Config` 写法需替换
- **sync.py 调用时机**：每次 create / update / delete / close position 后自动调用，保持 portfolio.json 与数据库一致
- **subprocess 非阻塞**：`/refresh` 和 `/scan/run` 使用 `BackgroundTasks + subprocess.Popen`，接口立即返回，脚本在后台运行
- **CORS**：当前 `allow_origins=["*"]`，生产环境建议改为具体前端域名/IP
