#!/bin/bash
# ─────────────────────────────────────────────
# setup_cron.sh  —  在服务器上配置 crontab
# 由 deploy.sh 远程调用，也可单独执行: bash setup_cron.sh
# ─────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 确保脚本有执行权限
chmod +x "${SCRIPT_DIR}/run_daily.sh"
chmod +x "${SCRIPT_DIR}/run_weekly.sh"

# 获取 Python3 路径（兼容 venv 和系统 Python）
PYTHON3=$(which python3)
echo "Python3 路径: ${PYTHON3}"

# 备份现有 crontab
crontab -l 2>/dev/null > /tmp/crontab_backup.txt || true
echo "已备份现有 crontab 到 /tmp/crontab_backup.txt"

# 移除旧的金融分析相关条目（幂等操作）
grep -v "run_daily.sh\|run_weekly.sh\|finance-analysis" /tmp/crontab_backup.txt > /tmp/crontab_new.txt || true

# 添加新的定时任务
# 说明：
#   每天 08:00 CST = 00:00 UTC  → 美股前一交易日收盘数据已完整
#   每周一 08:30 CST = 00:30 UTC → 比每日任务晚 30 分钟，避免冲突
cat >> /tmp/crontab_new.txt << EOF

# ── 金融分析定时任务 ──
# 每天 08:00 CST (00:00 UTC)：日报
0 0 * * * bash ${SCRIPT_DIR}/run_daily.sh >> ${SCRIPT_DIR}/logs/cron.log 2>&1
# 每周一 08:30 CST (00:30 UTC Mon)：周报
30 0 * * 1 bash ${SCRIPT_DIR}/run_weekly.sh >> ${SCRIPT_DIR}/logs/cron.log 2>&1

EOF

# 应用新 crontab
crontab /tmp/crontab_new.txt
echo ""
echo "crontab 配置完成，当前任务列表："
crontab -l
