# -*- coding: utf-8 -*-
"""部署自检：服务启动后跑一遍，全 ✅ 即部署成功
用法：python selftest.py [端口，默认 8000]
检查项：服务在线 / 图谱数据 / 语义检索 / 模型加载 / 图谱总览 / 邻域搜索 /
        节点详情 / 超范围拒答 / GraphRAG 真实问答
"""
import json, sys, io, urllib.request, urllib.parse

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")   # Windows GBK 控制台兜底

PORT = sys.argv[1] if len(sys.argv) > 1 else "8000"
BASE = f"http://127.0.0.1:{PORT}"
ok = fail = 0

def check(name, cond, detail=""):
    global ok, fail
    print(("✅ " if cond else "❌ ") + name + (f"（{detail}）" if detail else ""))
    if cond: ok += 1
    else:    fail += 1

def ask(q, timeout=300):
    req = urllib.request.Request(BASE + "/ask",
        data=json.dumps({"question": q}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    meta, text = {}, ""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8")
    for line in raw.split("\n"):
        if line.startswith("data: "):
            ev = json.loads(line[6:])
            if ev["type"] == "meta":   meta = ev
            elif ev["type"] == "delta": text += ev.get("t", "")
    return meta, text

print(f"—— 部署自检（{BASE}）——")

h = json.load(urllib.request.urlopen(BASE + "/health", timeout=15))
check("服务在线", True, f"实体 {h['entities']}｜节点 {h['nodes']}｜边 {h['edges']}")
check("图谱数据完整", h["nodes"] > 1000 and h["edges"] > 1500,
      f"{h['nodes']} 节点 / {h['edges']} 边｜来源：{h['graph']}")
check("语义检索就绪（bge）", h["dense"])
model_ready = h["llm"] != "未加载"
check("模型已加载", model_ready, h["llm"])

g = json.load(urllib.request.urlopen(BASE + "/graph", timeout=30))
check("图谱·总览", g["mode"] == "overview" and len(g["nodes"]) > 100,
      f"{len(g['nodes'])} 节点 / {len(g['edges'])} 边")
g2 = json.load(urllib.request.urlopen(
    BASE + "/graph?q=" + urllib.parse.quote("条锈病"), timeout=30))
check("图谱·邻域搜索", g2["mode"] == "ego" and len(g2["nodes"]) >= 20,
      f"{len(g2['nodes'])} 节点 / {len(g2['edges'])} 边")
nid = next((n["id"] for n in g2["nodes"] if n["lb"] == "Entity"), None)
if nid:
    d = json.load(urllib.request.urlopen(
        BASE + "/node?id=" + urllib.parse.quote(nid, safe=""), timeout=30))
    check("图谱·节点详情", len(d["facts"]) > 0, f"{d['name']}：{len(d['facts'])} 条事实")

meta, text = ask("种棉花怎么选品种？")
check("对话·超范围拒答", meta.get("mode") == "refusal" and "超出" in text)

meta, text = ask("玉米收完怎么存才不会霉？")
if model_ready:
    check("对话·GraphRAG 真实问答",
          meta.get("mode") == "rag" and len(text) > 50
          and "未加载" not in text and "生成出错" not in text,
          f"模式 {meta.get('mode')}｜命中 {len(meta.get('entities', []))} 实体｜回答 {len(text)} 字")
else:
    print(f"⏸ 对话·GraphRAG 真实问答：跳过（模型未加载：{h['llm']}）——在 GPU/CPU 服务器上此项必须 ✅")

print(f"\n结果：{ok} 通过 / {fail} 失败" +
      ("　🎉 全部通过，部署成功！" if fail == 0 else "　↑↑ 修复失败项后重跑"))
sys.exit(1 if fail else 0)
