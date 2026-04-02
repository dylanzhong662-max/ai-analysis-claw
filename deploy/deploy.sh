#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  deploy.sh — 部署 / 更新前端仪表盘到 Ubuntu 服务器
#
#  用法：
#    首次部署：  bash deploy/deploy.sh <服务器IP> <用户名>
#    更新部署：  bash deploy/deploy.sh <服务器IP> <用户名> --update
#
#  示例：
#    bash deploy/deploy.sh 101.201.171.174 root
#    bash deploy/deploy.sh 101.201.171.174 root --update
# ─────────────────────────────────────────────────────────────────

set -e

SERVER_IP="${1:?请提供服务器 IP，例如: bash deploy.sh 101.201.171.174 root}"
SERVER_USER="${2:-root}"
UPDATE_ONLY="${3}"
PROJECT_DIR="/opt/finance-analysis"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================"
echo "  部署目标: ${SERVER_USER}@${SERVER_IP}"
echo "  本地目录: ${LOCAL_DIR}"
echo "  服务器目录: ${PROJECT_DIR}"
echo "========================================"

# ── Step 1: 构建前端 ──────────────────────────────────────────────
echo ""
echo "▶ Step 1/5  构建前端..."
cd "${LOCAL_DIR}/frontend"

# 写入生产环境变量（指向服务器自身）
cat > .env << EOF
VITE_API_URL=http://${SERVER_IP}
EOF

npm run build
echo "  ✅ 前端构建完成 → dist/"

# ── Step 2: 同步代码到服务器 ─────────────────────────────────────
echo ""
echo "▶ Step 2/5  同步文件到服务器..."
ssh "${SERVER_USER}@${SERVER_IP}" "mkdir -p ${PROJECT_DIR}/backend ${PROJECT_DIR}/frontend/dist"

# 同步后端
rsync -az --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'trading.db' \
  "${LOCAL_DIR}/backend/" \
  "${SERVER_USER}@${SERVER_IP}:${PROJECT_DIR}/backend/"

# 同步前端构建产物
rsync -az --delete \
  "${LOCAL_DIR}/frontend/dist/" \
  "${SERVER_USER}@${SERVER_IP}:${PROJECT_DIR}/frontend/dist/"

# 同步现有 Python 分析脚本（*.py, *.json, *.txt 排除大文件）
rsync -az \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'backend/' \
  --exclude 'frontend/' \
  --exclude 'deploy/' \
  --exclude '.git/' \
  --exclude 'trading.db' \
  --exclude 'backtest_results/' \
  --exclude 'data_cache/' \
  "${LOCAL_DIR}/" \
  "${SERVER_USER}@${SERVER_IP}:${PROJECT_DIR}/"

echo "  ✅ 文件同步完成"

# ── Step 3: 首次部署 — 安装系统依赖 ─────────────────────────────
if [ "${UPDATE_ONLY}" != "--update" ]; then
  echo ""
  echo "▶ Step 3/5  安装系统依赖（首次）..."
  ssh "${SERVER_USER}@${SERVER_IP}" bash << 'REMOTE'
    set -e
    export DEBIAN_FRONTEND=noninteractive

    # 系统包
    apt-get update -q
    apt-get install -y -q python3 python3-venv python3-pip nginx curl

    # Python 虚拟环境
    cd /opt/finance-analysis
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install \
      fastapi "uvicorn[standard]" sqlalchemy "pydantic>=2" \
      python-multipart yfinance pandas numpy -q

    echo "  ✅ 依赖安装完成"
REMOTE
else
  echo ""
  echo "▶ Step 3/5  跳过依赖安装（--update 模式）"
fi

# ── Step 4: 配置并启动 Nginx + systemd ───────────────────────────
echo ""
echo "▶ Step 4/5  配置 Nginx + systemd 服务..."
scp "${LOCAL_DIR}/deploy/nginx.conf" \
    "${SERVER_USER}@${SERVER_IP}:/etc/nginx/sites-available/finance"
scp "${LOCAL_DIR}/deploy/backend.service" \
    "${SERVER_USER}@${SERVER_IP}:/etc/systemd/system/finance-backend.service"

ssh "${SERVER_USER}@${SERVER_IP}" bash << REMOTE
  set -e

  # 替换 nginx.conf 中的占位 IP（如有域名可手动替换）
  sed -i 's/server_name _;/server_name _ ${SERVER_IP};/' /etc/nginx/sites-available/finance

  # 启用站点
  ln -sf /etc/nginx/sites-available/finance /etc/nginx/sites-enabled/finance
  rm -f /etc/nginx/sites-enabled/default

  # 测试 nginx 配置
  nginx -t

  # 启动/重载服务
  systemctl daemon-reload
  systemctl enable finance-backend
  systemctl restart finance-backend
  systemctl reload nginx || systemctl restart nginx

  echo "  ✅ Nginx + systemd 配置完成"
REMOTE

# ── Step 5: 验证 ──────────────────────────────────────────────────
echo ""
echo "▶ Step 5/5  验证部署..."
sleep 3

HEALTH=$(curl -s --max-time 5 "http://${SERVER_IP}/api/health" 2>/dev/null || echo "failed")
if echo "${HEALTH}" | grep -q '"ok"'; then
  echo "  ✅ 后端 API 正常: ${HEALTH}"
else
  echo "  ⚠️  后端健康检查失败，请登录服务器查看日志:"
  echo "     ssh ${SERVER_USER}@${SERVER_IP}"
  echo "     journalctl -u finance-backend -n 30"
fi

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://${SERVER_IP}" 2>/dev/null || echo "000")
if [ "${HTTP_CODE}" = "200" ]; then
  echo "  ✅ 前端页面正常: HTTP ${HTTP_CODE}"
else
  echo "  ⚠️  前端返回 HTTP ${HTTP_CODE}"
fi

echo ""
echo "========================================"
echo "  部署完成！"
echo "  访问地址: http://${SERVER_IP}"
echo "  API 文档: http://${SERVER_IP}/api/docs"
echo "========================================"
