#!/bin/bash
# ─────────────────────────────────────────────
# run_daily.sh  —  每日定时分析任务
# 由 crontab 在每天 08:00 CST (00:00 UTC) 自动调用
# 手动测试: bash run_daily.sh
# ─────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
DATE=$(date +%Y%m%d)
LOG_FILE="${LOG_DIR}/daily_${DATE}.log"

mkdir -p "${LOG_DIR}"

# 加载环境变量
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.env"
    set +a
fi

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }

log "===== 每日金融分析开始 ====="

cd "${SCRIPT_DIR}"

# ── 黄金分析 ──
log "--- [1/3] 黄金分析 ---"
if python3 gold_analysis.py --api  2>&1 | tee -a "${LOG_FILE}"; then
    log "黄金分析完成"
else
    log "[错误] 黄金分析失败，继续下一步"
fi

# ── BTC 分析 ──
log "--- [2/3] BTC 分析 ---"
if python3 btc_analysis.py --api  2>&1 | tee -a "${LOG_FILE}"; then
    log "BTC 分析完成"
else
    log "[错误] BTC 分析失败，继续下一步"
fi

# ── 科技股分析 ──
log "--- [3/5] GOOGL 分析 ---"
if python3 tech_stock_analysis.py --ticker GOOGL --api  2>&1 | tee -a "${LOG_FILE}"; then
    log "GOOGL 分析完成"
else
    log "[错误] GOOGL 分析失败，继续下一步"
fi

log "--- [4/5] NVDA 分析 ---"
if python3 tech_stock_analysis.py --ticker NVDA --api  2>&1 | tee -a "${LOG_FILE}"; then
    log "NVDA 分析完成"
else
    log "[错误] NVDA 分析失败，继续下一步"
fi

log "--- [5/5 前] AMZN 分析 ---"
if python3 tech_stock_analysis.py --ticker AMZN --api  2>&1 | tee -a "${LOG_FILE}"; then
    log "AMZN 分析完成"
else
    log "[错误] AMZN 分析失败，继续下一步"
fi

# ── 飞书推送 ──
log "--- [6/6] 飞书推送 ---"
if python3 feishu_notifier.py --mode daily 2>&1 | tee -a "${LOG_FILE}"; then
    log "飞书推送完成"
else
    log "[错误] 飞书推送失败"
fi

log "===== 每日分析全部完成（黄金 + BTC + GOOGL + NVDA + AMZN）====="
