#!/bin/bash
# ─────────────────────────────────────────────
# run_weekly.sh  —  每周定时汇总任务
# 由 crontab 在每周一 08:30 CST (00:30 UTC Mon) 自动调用
# 在 run_daily.sh 完成后额外发送周汇总飞书消息
# 手动测试: bash run_weekly.sh
# ─────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
DATE=$(date +%Y%m%d)
LOG_FILE="${LOG_DIR}/weekly_${DATE}.log"

mkdir -p "${LOG_DIR}"

# 加载环境变量
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.env"
    set +a
fi

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }

log "===== 每周金融汇总开始 ====="

cd "${SCRIPT_DIR}"

# 每周一同样跑一遍最新日线分析（获取最新数据）
log "--- [1/4] 黄金分析（周） ---"
python3 gold_analysis.py --api  2>&1 | tee -a "${LOG_FILE}" || log "[错误] 黄金分析失败"

log "--- [2/4] BTC 分析（周） ---"
python3 btc_analysis.py --api  2>&1 | tee -a "${LOG_FILE}" || log "[错误] BTC 分析失败"

# 科技股分析
log "--- [3/6] 科技股分析（GOOGL）---"
python3 tech_stock_analysis.py --ticker GOOGL --api  2>&1 | tee -a "${LOG_FILE}" || log "[错误] GOOGL 分析失败"

log "--- [4/6] 科技股分析（NVDA）---"
python3 tech_stock_analysis.py --ticker NVDA --api  2>&1 | tee -a "${LOG_FILE}" || log "[错误] NVDA 分析失败"

log "--- [5/6] 科技股分析（AMZN）---"
python3 tech_stock_analysis.py --ticker AMZN --api  2>&1 | tee -a "${LOG_FILE}" || log "[错误] AMZN 分析失败"

# 发送飞书周报（weekly 模式信息更详尽）
log "--- [6/6] 飞书周报推送 ---"
python3 feishu_notifier.py --mode weekly 2>&1 | tee -a "${LOG_FILE}" || log "[错误] 飞书周报推送失败"

log "===== 每周汇总全部完成 ====="
