#!/usr/bin/env bash
# CPU 服务器一键部署：检查文件 → 装依赖（llama-cpp-python 要现场编译）→ 启动 → 自检
# 前提：内存 ≥8GB；本目录里应有 server.py、rag.py、static/、graph_snapshot.json、
#       bge_small/、agri_qwen3_4b_q8_0.gguf（后两样从 AutoDL 拷来，生成方法见下）
set -e
cd "$(dirname "$0")"
PORT=${PORT:-8000}
PY=${PYTHON:-python3}; "$PY" -c "" >/dev/null 2>&1 || PY=python
"$PY" -c "" >/dev/null 2>&1 || { echo "❌ 找不到可用的 python，请先安装 Python 3"; exit 1; }
GGUF=agri_qwen3_4b_q8_0.gguf

if [ ! -f "$GGUF" ]; then
  cat <<'EOF'
❌ 缺 agri_qwen3_4b_q8_0.gguf（约 4.3GB，须先在有模型的机器上转换）。在 AutoDL 上执行：
   cd /root/autodl-tmp && git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
   pip install -r requirements.txt
   python convert_hf_to_gguf.py /root/autodl-tmp/crop_cultivation/output/qwen3_4b_agri_merged \
          --outfile /root/autodl-tmp/crop_cultivation/agri_qwen3_4b_q8_0.gguf --outtype q8_0
然后把生成的 agri_qwen3_4b_q8_0.gguf 传到本目录（和 server.py 放一起），再重跑 bash start.sh
详见 部署说明-CPU.md
EOF
  exit 1
fi
if [ ! -d bge_small ]; then
  echo "❌ 缺 bge_small/（95MB）：从 AutoDL 的 /root/autodl-tmp/crop_cultivation/bge_small 整个拷到本目录"
  exit 1
fi

[ -f server.pid ] && kill "$(cat server.pid)" 2>/dev/null || true
sleep 1

echo "== 1/4 系统依赖（编译需要 gcc/cmake；无 root 权限请让管理员装 build-essential cmake）=="
(apt update -qq && apt install -y -qq build-essential cmake) \
  || sudo apt install -y build-essential cmake \
  || echo "⚠ apt 安装跳过——若系统已有 gcc/cmake 可忽略"
command -v gcc >/dev/null 2>&1 || command -v cc >/dev/null 2>&1 \
  || echo "⚠ 没检测到 C 编译器，llama-cpp-python 大概率装不上"

echo "== 2/4 安装 Python 依赖（llama-cpp-python 现场编译，约 5-15 分钟，请耐心）=="
$PY -m pip install -q fastapi uvicorn neo4j llama-cpp-python torch transformers \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "== 3/4 启动服务（端口 $PORT；加载 GGUF 约 1-2 分钟）=="
nohup $PY server.py --port $PORT > server.log 2>&1 &
echo $! > server.pid
for i in $(seq 1 120); do
  sleep 2
  curl -s -o /dev/null "http://127.0.0.1:$PORT/health" && break
  if ! kill -0 "$(cat server.pid)" 2>/dev/null; then
    echo "❌ 启动失败，末尾日志："; tail -20 server.log; exit 1
  fi
done

echo "== 4/4 自检（含一次真实问答，CPU 上约 30-90 秒，请等）=="
$PY selftest.py "$PORT"

echo "
🎉 部署完成！
· 日志：tail -f server.log ｜ 停止：kill \$(cat server.pid) ｜ 重启：再跑 bash start.sh
· 有公网 IP：浏览器开 http://服务器IP:$PORT（防火墙/安全组放行 $PORT）
· 内网服务器：本地执行 ssh -L $PORT:localhost:$PORT 用户@服务器，再开 http://localhost:$PORT"
