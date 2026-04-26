#!/bin/bash
# ─────────────────────────────────────────────
# run_daily.sh  —  每日定时分析任务
# 由 crontab 在每天 10:00 CST 和 19:00 CST 自动调用
#
# 分析哪些资产由 assets_config.py 中的 daily_scan=True 控制。
# 新增资产只需在 assets_config.py 设 daily_scan=True，无需改此脚本。
#
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

# ── 从 assets_config.py 动态读取每日扫描资产列表 ──
# 输出格式（每行一条）: ASSET_KEY|script|ticker|extra_args
# daily_scan=True 且有 script 的资产自动纳入
ASSET_LIST=$(python3 - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from assets_config import get_daily_assets

for key, cfg in get_daily_assets():
    script     = cfg.get("script", "")
    ticker     = cfg.get("ticker", key)
    extra_args = " ".join(cfg.get("daily_extra_args", []))
    print(f"{key}|{script}|{ticker}|{extra_args}")
PYEOF
)

if [ -z "$ASSET_LIST" ]; then
    log "[警告] assets_config.py 中没有 daily_scan=True 的资产，退出"
    exit 0
fi

# ── 按顺序逐个分析 ──
TOTAL=$(echo "$ASSET_LIST" | wc -l | tr -d ' ')
IDX=0

while IFS='|' read -r ASSET_KEY SCRIPT TICKER EXTRA_ARGS; do
    IDX=$((IDX + 1))
    log "--- [${IDX}/${TOTAL}] ${ASSET_KEY} (${TICKER}) ---"

    # 根据脚本类型组装命令
    case "$SCRIPT" in
        gold_analysis.py)
            CMD="python3 gold_analysis.py --api --model deepseek-reasoner ${EXTRA_ARGS}"
            ;;
        btc_analysis.py)
            CMD="python3 btc_analysis.py --api --model deepseek-reasoner ${EXTRA_ARGS}"
            ;;
        tech_stock_analysis.py)
            CMD="python3 tech_stock_analysis.py --ticker ${TICKER} --api --model deepseek-reasoner ${EXTRA_ARGS}"
            ;;
        *)
            log "[跳过] 未知脚本类型: ${SCRIPT}"
            continue
            ;;
    esac

    if eval "$CMD" 2>&1 | tee -a "${LOG_FILE}"; then
        log "${ASSET_KEY} 分析完成"
    else
        log "[错误] ${ASSET_KEY} 分析失败，继续下一步"
    fi

done <<< "$ASSET_LIST"

# ── 飞书推送 ──
log "--- 飞书推送 ---"
if python3 feishu_notifier.py --mode daily 2>&1 | tee -a "${LOG_FILE}"; then
    log "飞书推送完成"
else
    log "[错误] 飞书推送失败"
fi

# ── 信号持久化 ──
log "--- 信号持久化 ---"
if python3 signal_logger.py 2>&1 | tee -a "${LOG_FILE}"; then
    log "信号持久化完成"
else
    log "[错误] 信号持久化失败（不影响分析）"
fi

SCANNED=$(echo "$ASSET_LIST" | awk -F'|' '{print $1}' | tr '\n' '+' | sed 's/+$//')
log "===== 每日分析全部完成（${SCANNED}）====="
