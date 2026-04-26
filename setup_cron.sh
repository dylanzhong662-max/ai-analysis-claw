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
#   阿里云 ECS 默认时区为 CST（UTC+8），crontab 按服务器本地时间执行，无需换算 UTC
#   每天 10:00 CST → 早盘前，亚盘数据已完整
#   每天 19:00 CST → 美股开盘前，全球主要市场收盘数据已完整
#   每周一 08:30 CST → 周报，早于早盘日报
cat >> /tmp/crontab_new.txt << EOF

# ── 金融分析定时任务（时间均为服务器本地时间 CST/UTC+8）──
# 每天 10:00 CST：早盘日报
0 10 * * * bash ${SCRIPT_DIR}/run_daily.sh >> ${SCRIPT_DIR}/logs/cron.log 2>&1
# 每天 19:00 CST：晚盘日报
0 19 * * * bash ${SCRIPT_DIR}/run_daily.sh >> ${SCRIPT_DIR}/logs/cron.log 2>&1
# 每周一 08:30 CST：周报
30 8 * * 1 bash ${SCRIPT_DIR}/run_weekly.sh >> ${SCRIPT_DIR}/logs/cron.log 2>&1

EOF

# 应用新 crontab
crontab /tmp/crontab_new.txt
echo ""
echo "crontab 配置完成，当前任务列表："
crontab -l
