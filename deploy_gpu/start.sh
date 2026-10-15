#!/usr/bin/env bash
# GPU 服务器（AutoDL）一键部署：装依赖 → 启动 → 自检
# 用法：本文件夹所有文件放到 /root/autodl-tmp/crop_cultivation/ 后执行：bash start.sh
set -e
cd "$(dirname "$0")"
PORT=${PORT:-8000}
PY=${PYTHON:-python3}; "$PY" -c "" >/dev/null 2>&1 || PY=python
"$PY" -c "" >/dev/null 2>&1 || { echo "❌ 找不到可用的 python，请先安装 Python 3"; exit 1; }

[ -f server.pid ] && kill "$(cat server.pid)" 2>/dev/null || true   # 重跑先停旧进程
sleep 1

echo "== 1/3 安装依赖（已装则秒过）=="
$PY -m pip install -q fastapi uvicorn neo4j -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "== 2/3 启动服务（端口 $PORT；首次加载模型约 1-3 分钟）=="
nohup $PY server.py --port $PORT > server.log 2>&1 &
echo $! > server.pid
for i in $(seq 1 150); do
  sleep 2
  curl -s -o /dev/null "http://127.0.0.1:$PORT/health" && break
  if ! kill -0 "$(cat server.pid)" 2>/dev/null; then
    echo "❌ 启动失败，末尾日志："; tail -20 server.log; exit 1
  fi
done

echo "== 3/3 自检 =="
$PY selftest.py "$PORT"

echo "
🎉 部署完成！
· 日志：tail -f server.log ｜ 停止：kill \$(cat server.pid) ｜ 重启：再跑 bash start.sh
· 对外访问（AutoDL）：控制台 → 该实例 → 自定义服务 → 端口填 $PORT → 复制链接 → 本地浏览器打开"
