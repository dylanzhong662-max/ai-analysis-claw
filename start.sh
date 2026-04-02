#!/bin/bash
# 启动后端
cd "$(dirname "$0")/backend"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "✅ 后端启动 PID=$BACKEND_PID  http://localhost:8000"

# 启动前端开发服务（开发模式）
cd ../frontend
npm run dev -- --host 0.0.0.0 --port 3000 &
FRONTEND_PID=$!
echo "✅ 前端启动 PID=$FRONTEND_PID  http://localhost:3000"

echo ""
echo "按 Ctrl+C 停止所有服务"
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
