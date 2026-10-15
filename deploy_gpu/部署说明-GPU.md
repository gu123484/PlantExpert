# GPU 版部署（AutoDL 显卡服务器，3 步）

**第 1 步：上传**
本文件夹（deploy_gpu）里所有文件，传到 AutoDL 的 `/root/autodl-tmp/crop_cultivation/`：
- `server.py`、`rag.py`、`graph_snapshot.json`、`selftest.py`
- `static/index.html`、`static/echarts.min.js`
- `start.sh`（本说明）

模型（`output/qwen3_4b_agri_merged/`）和向量模型（`bge_small/`）实例里已有，**不用传**。
（若是部署到 AutoDL 以外的 GPU 服务器：把这两样也拷到 `/root/autodl-tmp/crop_cultivation/`，
或改 `server.py` 顶部的 `BASE`。）

**第 2 步：启动 + 自动测试（一条命令）**
```bash
cd /root/autodl-tmp/crop_cultivation
# 本实例实际使用的启动命令（端口 6006 对应自定义服务映射；rag 是带 torch 的 conda 环境）：
PORT=6006 PYTHON=/root/miniconda3/envs/rag/bin/python bash start.sh
```
脚本会自动装依赖 → 启动服务 → 逐项体检，最后打出 9 个 ✅ + `🎉 全部通过，部署成功！`
即为 GPU 部署测试成功。任何 ❌ 会附带原因，照着提示处理。

**第 3 步：本地访问**
AutoDL 控制台 → 该实例 → **自定义服务** → 端口填 `6006` → 复制生成的链接 → 本地浏览器打开。
（实例关机后服务停止；开机后用上面第 2 步的命令重启即可。）

**日常**
| 操作 | 命令 |
|---|---|
| 看日志 | `tail -f server.log` |
| 停止 | `kill $(cat server.pid)` |
| 重启 | 再跑 `bash start.sh` |
| 换端口 | `PORT=8010 bash start.sh`（自定义服务里端口同步改） |
