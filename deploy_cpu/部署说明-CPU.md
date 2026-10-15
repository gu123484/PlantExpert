# 无显卡版部署（普通 CPU 服务器，内存 ≥8GB）

**第 1 步：量化模型已在 AutoDL 上转好（无需再做）**

`agri_qwen3_4b_q8_0.gguf`（Q8_0，4.28GB，精度几乎无损）已在
`/root/autodl-tmp/crop_cultivation/agri_qwen3_4b_q8_0.gguf` 转换完成，直接执行第 2 步拉取即可。

> 仅当需要重新转换时才执行（**注意**：llama.cpp 的 `pip install -r requirements.txt`
> 会把 transformers 升到 4.57+ 导致 AutoDL 上模型加载报错，转完务必
> `pip install transformers==4.51.3` 降回来再重启服务）：
> ```bash
> cd /root/autodl-tmp && git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
> pip install -r requirements.txt
> python convert_hf_to_gguf.py /root/autodl-tmp/crop_cultivation/output/qwen3_4b_agri_merged \
>        --outfile /root/autodl-tmp/crop_cultivation/agri_qwen3_4b_q8_0.gguf --outtype q8_0
> ```
> **不要**把 8GB 原模型丢给 CPU 服务器——内存和速度都扛不住。

**第 2 步：把 3 样东西传到 CPU 服务器，放成一个目录**
```bash
# 在 CPU 服务器上执行（下方是本项目的 AutoDL 实例地址，密码见实例卡片）
mkdir -p ~/agri_web && cd ~/agri_web
# ① 本文件夹（deploy_cpu）全部文件 —— 从本地电脑上传
# ② 量化模型 + ③ 向量模型 —— 直接从 AutoDL 拉：
scp -P 38231 root@connect.bjb2.seetacloud.com:/root/autodl-tmp/crop_cultivation/agri_qwen3_4b_q8_0.gguf .
scp -rP 38231 root@connect.bjb2.seetacloud.com:/root/autodl-tmp/crop_cultivation/bge_small .
```
最终目录内容：
```
~/agri_web/  server.py  rag.py  graph_snapshot.json  selftest.py  start.sh
             static/(index.html, echarts.min.js)  部署说明-CPU.md
             agri_qwen3_4b_q8_0.gguf  bge_small/
```
（放哪个目录都行——量化模型和向量模型会自动在 server.py 同级找；Neo4j 不需要，快照已带全图谱。）

**第 3 步：一键部署 + 自动测试**
```bash
cd ~/agri_web
bash start.sh
```
自动：检查文件 → 装 gcc/cmake → 编译安装 llama-cpp-python（5-15 分钟，最久的一步）→
启动 → 逐项体检（含一次真实问答，CPU 上 30-90 秒）。打出 `🎉 全部通过，部署成功！` 即成。

**第 4 步：本地访问**
- 服务器有公网 IP：浏览器开 `http://服务器IP:8000`（云安全组/防火墙放行 8000）
- 内网：本地终端 `ssh -L 8000:localhost:8000 用户@服务器` 保持连接，浏览器开 `http://localhost:8000`

**日常**：`tail -f server.log` 看日志｜`kill $(cat server.pid)` 停止｜`bash start.sh` 重启。
回答速度约 15-60 秒/条（CPU 正常水平）；GPU 版约 3-10 秒/条。
