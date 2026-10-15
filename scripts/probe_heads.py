# -*- coding: utf-8 -*-
"""离线探针：不加载模型，只看 retrieve→rag_prompt 构造出的头与上下文前几行。
用本地 graph_snapshot.json（ents 结构与服务端一致）。纯词法（无 bge）——
词法分和头构造逻辑与服务端相同；仅 sim 逃逸分支服务端才可能触发。"""
import json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import rag

with open(os.path.join(HERE, "graph_snapshot.json"), encoding="utf-8") as f:
    d = json.load(f)
ents = d["ents"]
# facts 反序列化成 tuple（与服务端一致）
for e in ents.values():
    e["facts"] = [tuple(f) for f in e["facts"]]

QS = ["玉米空秆怎么回事？",
      "玉米大喇叭口期怎么管理？",
      "小麦返青期怎么管理？",
      "玉米大喇叭口期怎么追肥？",
      "小麦播种前怎么拌种？",
      "玉米大斑病怎么防治？用药量多少？",
      "俺家玉米叶子上一条一条的黄斑，咋回事？",
      "小麦什么时候浇水？",
      "黄淮海小麦什么时候播种？亩基本苗多少？"]

for q in QS:
    hits, top_sim = rag.retrieve(ents, q)
    print("=" * 70)
    print("问:", q)
    if not hits:
        print("  [无命中] top_sim =", top_sim)
        continue
    print("命中:", "、".join(f"{rag.short_name(n)}({lx:g})" for _, n, _, lx, _ in hits[:4]))
    prompt, has_ctx, head = rag.rag_prompt(hits, q)
    print("模式:", "rag" if has_ctx else "plain")
    print("HEAD:", head)
    ctx = rag.build_context(hits[:1] if head.startswith(rag.HEAD_SYM) else hits, q)
    for ln in ctx.split("\n")[:6]:
        print("  CTX|", ln)
