#!/bin/bash
# ─────────────────────────────────────────────
# deploy.sh  —  一键部署到阿里云 ECS
#
# 用法:
#   bash deploy.sh <服务器公网IP>                # 默认用户 ubuntu
#   bash deploy.sh <服务器公网IP> root           # 指定用户 root
#   bash deploy.sh <服务器公网IP> ubuntu --init  # 首次部署（安装依赖）
#
# 前提条件:
#   1. 本机已配置 SSH 密钥免密登录服务器
#      生成密钥: ssh-keygen -t rsa -b 4096
#      上传公钥: ssh-copy-id ubuntu@<服务器IP>
#   2. 本地已安装 rsync
# ─────────────────────────────────────────────

set -euo pipefail

SERVER_IP="${1:?请提供服务器IP，例如: bash deploy.sh 47.xxx.xxx.xxx}"
USERNAME="${2:-ubuntu}"
INIT_MODE="${3:-}"
REMOTE_DIR="/opt/finance-analysis"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "========================================"
echo "  部署目标: ${USERNAME}@${SERVER_IP}"
echo "  远程目录: ${REMOTE_DIR}"
echo "  本地目录: ${LOCAL_DIR}"
echo "========================================"
echo ""

# ── Step 1: 创建远程目录并设置权限 ──
echo "[1/5] 创建远程目录..."
ssh "${USERNAME}@${SERVER_IP}" "
    sudo mkdir -p ${REMOTE_DIR}/logs
    sudo mkdir -p ${REMOTE_DIR}/backtest_results
    sudo mkdir -p ${REMOTE_DIR}/backtest_responses
    sudo chown -R ${USERNAME}:${USERNAME} ${REMOTE_DIR}
    echo '目录创建完成'
"

# ── Step 2: 同步代码文件 ──
echo "[2/5] 同步代码文件..."
rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.log' \
    --exclude='backtest_prompts' \
    --exclude='backtest_responses' \
    --exclude='backtest_results' \
    --exclude='.env' \
    "${LOCAL_DIR}/" "${USERNAME}@${SERVER_IP}:${REMOTE_DIR}/"

echo "代码同步完成"

# ── Step 3: 首次部署时安装系统依赖和 Python 包 ──
if [ "${INIT_MODE}" == "--init" ]; then
    echo "[3/5] 安装系统依赖（首次部署）..."
    ssh "${USERNAME}@${SERVER_IP}" "
        sudo apt-get update -qq
        sudo apt-get install -y python3 python3-pip python3-venv rsync
        cd ${REMOTE_DIR}
        python3 -m venv .venv
        .venv/bin/pip install --upgrade pip -q
        .venv/bin/pip install -r requirements.txt -q
        echo 'Python 依赖安装完成'
    "

    # 上传 .env 文件（如果本地存在）
    if [ -f "${LOCAL_DIR}/.env" ]; then
        echo "  上传 .env 文件..."
        scp "${LOCAL_DIR}/.env" "${USERNAME}@${SERVER_IP}:${REMOTE_DIR}/.env"
        ssh "${USERNAME}@${SERVER_IP}" "chmod 600 ${REMOTE_DIR}/.env"
    else
        echo "  [提示] 未找到 .env 文件，请手动上传或在服务器上创建 ${REMOTE_DIR}/.env"
    fi
else
    echo "[3/5] 跳过依赖安装（非首次部署）"
    echo "      首次部署请使用: bash deploy.sh ${SERVER_IP} ${USERNAME} --init"
fi

# ── Step 4: 配置 crontab ──
echo "[4/5] 配置 crontab..."
ssh "${USERNAME}@${SERVER_IP}" "
    # 如果使用 venv，更新脚本中的 python3 路径
    sed -i 's|python3 |${REMOTE_DIR}/.venv/bin/python3 |g' ${REMOTE_DIR}/run_daily.sh
    sed -i 's|python3 |${REMOTE_DIR}/.venv/bin/python3 |g' ${REMOTE_DIR}/run_weekly.sh
    bash ${REMOTE_DIR}/setup_cron.sh
"

# ── Step 5: 验证部署 ──
echo "[5/5] 验证部署..."
ssh "${USERNAME}@${SERVER_IP}" "
    echo '--- 文件列表 ---'
    ls -la ${REMOTE_DIR}/*.py ${REMOTE_DIR}/*.sh
    echo ''
    echo '--- crontab 任务 ---'
    crontab -l
    echo ''
    echo '--- Python 版本 ---'
    ${REMOTE_DIR}/.venv/bin/python3 --version 2>/dev/null || python3 --version
"

echo ""
echo "========================================"
echo "  部署完成！"
echo ""
echo "  下一步:"
echo "  1. 在服务器上编辑 .env 文件："
echo "     ssh ${USERNAME}@${SERVER_IP}"
echo "     nano ${REMOTE_DIR}/.env"
echo ""
echo "  2. 手动测试飞书推送:"
echo "     ssh ${USERNAME}@${SERVER_IP}"
echo "     cd ${REMOTE_DIR} && .venv/bin/python3 feishu_notifier.py --mode test"
echo ""
echo "  3. 手动触发一次完整分析:"
echo "     ssh ${USERNAME}@${SERVER_IP}"
echo "     bash ${REMOTE_DIR}/run_daily.sh"
echo "========================================"
