#!/bin/bash
# ─────────────────────────────────────────────
# setup_proxy.sh  —  在服务器上安装 Shadowsocks 代理
# 将 ssconf:// 订阅转换为本地 HTTP 代理（127.0.0.1:8118）
# 用法: bash setup_proxy.sh
# ─────────────────────────────────────────────

set -euo pipefail

SSCONF_URL="https://ss.wawaapp.net/t/520fa9d967e39ce4b19a54c88312e52d2991ecf63894998e00f031308"
SS_CONFIG="/etc/shadowsocks-libev/local.json"
LOCAL_PORT=1080
HTTP_PROXY_PORT=8118

echo "=== [1/5] 安装依赖 ==="
apt-get install -y shadowsocks-libev privoxy jq -q

echo "=== [2/5] 获取 SS 配置 ==="
CONFIG_JSON=$(curl -s "${SSCONF_URL}")
echo "配置获取成功"

SERVER=$(echo "${CONFIG_JSON}"      | jq -r '.server')
SERVER_PORT=$(echo "${CONFIG_JSON}" | jq -r '.server_port')
PASSWORD=$(echo "${CONFIG_JSON}"    | jq -r '.password')
METHOD=$(echo "${CONFIG_JSON}"      | jq -r '.method')

echo "  服务器: ${SERVER}:${SERVER_PORT}  加密: ${METHOD}"

echo "=== [3/5] 写入 ss-local 配置 ==="
cat > "${SS_CONFIG}" << EOF
{
    "server":        "${SERVER}",
    "server_port":   ${SERVER_PORT},
    "local_address": "127.0.0.1",
    "local_port":    ${LOCAL_PORT},
    "password":      "${PASSWORD}",
    "method":        "${METHOD}",
    "timeout":       300,
    "fast_open":     false
}
EOF

echo "=== [4/5] 启动 ss-local（SOCKS5 代理 → 端口 ${LOCAL_PORT}）==="
# 停止可能存在的旧进程
pkill -f "ss-local" 2>/dev/null || true
sleep 1
nohup ss-local -c "${SS_CONFIG}" > /var/log/ss-local.log 2>&1 &
sleep 2

# 验证是否启动
if pgrep -f "ss-local" > /dev/null; then
    echo "  ss-local 启动成功（PID: $(pgrep -f ss-local)）"
else
    echo "  [错误] ss-local 启动失败，查看日志: cat /var/log/ss-local.log"
    exit 1
fi

echo "=== [5/5] 配置 privoxy（HTTP 代理 → 端口 ${HTTP_PROXY_PORT}）==="
# 清除旧的转发规则（避免重复）
sed -i '/^forward-socks5/d' /etc/privoxy/config

# 添加转发规则：所有 HTTP 请求 → SOCKS5 本地代理
echo "forward-socks5 / 127.0.0.1:${LOCAL_PORT} ." >> /etc/privoxy/config

# 确保监听地址正确
sed -i "s/^listen-address.*/listen-address  127.0.0.1:${HTTP_PROXY_PORT}/" /etc/privoxy/config

systemctl enable privoxy
systemctl restart privoxy
sleep 1

echo ""
echo "=== 验证代理是否工作 ==="
if curl --silent --proxy "http://127.0.0.1:${HTTP_PROXY_PORT}" \
        --max-time 10 "https://www.google.com" -o /dev/null -w "%{http_code}" | grep -q "200\|301\|302"; then
    echo "  代理验证成功！可以访问外网"
else
    echo "  [警告] 代理验证失败，可能是 VPN 节点问题，请手动检查"
fi

echo ""
echo "========================================"
echo "  代理安装完成！"
echo "  HTTP 代理地址: http://127.0.0.1:${HTTP_PROXY_PORT}"
echo ""
echo "  下一步：在 .env 文件中添加："
echo "  HTTPS_PROXY=http://127.0.0.1:${HTTP_PROXY_PORT}"
echo "  HTTP_PROXY=http://127.0.0.1:${HTTP_PROXY_PORT}"
echo "========================================"
