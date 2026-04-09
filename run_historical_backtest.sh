#!/bin/bash
# run_historical_backtest.sh
# 目的：在 2018-2021（策略从未见过的数据）上跑历史回测，验证泛化能力
#
# 使用方法（阿里云服务器）：
#   bash run_historical_backtest.sh            # 默认：NVDA + MSFT + GOOGL
#   bash run_historical_backtest.sh NVDA       # 只跑指定标的
#   bash run_historical_backtest.sh simplified # 跑5规则极简版本
#
# 前提：cd /opt/finance-analysis && source .env

set -e
cd "$(dirname "$0")"
source .env 2>/dev/null || true

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M)

# 默认标的
TICKERS=("NVDA" "MSFT" "GOOGL")

# 参数解析
MODE="standard"
if [ "$1" = "simplified" ]; then
    MODE="simplified"
    shift
elif [ -n "$1" ] && [ "$1" != "simplified" ]; then
    TICKERS=("$1")
fi

echo "========================================"
echo "历史回测 2018-2021（策略从未见过的数据）"
echo "模式: $MODE"
echo "标的: ${TICKERS[*]}"
echo "========================================"

for TICKER in "${TICKERS[@]}"; do
    echo ""
    echo "▶ 开始 $TICKER 2018-2021 回测 [模式: $MODE]"
    LOG_FILE="$LOG_DIR/historical_${TICKER}_${MODE}_${TIMESTAMP}.log"

    if [ "$MODE" = "simplified" ]; then
        # P0.1 极简模式：5规则，去除过拟合验证
        nohup .venv/bin/python3 -u tech_backtest_engine.py \
            --ticker "$TICKER" \
            --start 2018-01-01 \
            --end 2021-12-31 \
            --model deepseek-reasoner \
            --second-model claude-sonnet-4-6 \
            --capital 100000 \
            --commission 0.001 \
            --slippage 0.001 \
            --risk-per-trade 0.02 \
            --step 1 \
            --eval-days 65 \
            --reproducible \
            --simplified \
            --rate-limit 15 \
            > "$LOG_FILE" 2>&1 &
        echo "  PID=$!  日志 → $LOG_FILE"
    else
        # 标准模式：完整规则链
        nohup .venv/bin/python3 -u tech_backtest_engine.py \
            --ticker "$TICKER" \
            --start 2018-01-01 \
            --end 2021-12-31 \
            --model deepseek-reasoner \
            --second-model claude-sonnet-4-6 \
            --capital 100000 \
            --commission 0.001 \
            --slippage 0.001 \
            --risk-per-trade 0.02 \
            --step 1 \
            --eval-days 65 \
            --oos-split 0.2 \
            --reproducible \
            --rate-limit 15 \
            > "$LOG_FILE" 2>&1 &
        echo "  PID=$!  日志 → $LOG_FILE"
    fi

    # 避免同时触发 API 限流（两个任务错开 30 秒启动）
    sleep 30
done

echo ""
echo "所有回测已在后台启动。"
echo ""
echo "查看进度："
echo "  tail -f $LOG_DIR/historical_*_${TIMESTAMP}.log"
echo ""
echo "预期结果（与 2022 熊市压力测试对比）："
echo "  2018-2021 含 2018年贸易战、2019年牛市、2020年新冠崩盘+复苏、2021年牛市"
echo "  若策略有真实泛化能力，应在 2020 崩盘段保本，2019+2021 牛市跑赢或接近 B&H"
echo ""
echo "Beta底仓回测（同期）："
echo "  bash run_historical_backtest.sh  然后再运行："
for TICKER in "${TICKERS[@]}"; do
echo "  .venv/bin/python3 tech_backtest_engine.py --ticker $TICKER --beta-floor --start 2018-01-01 --end 2021-12-31 --reproducible"
done
