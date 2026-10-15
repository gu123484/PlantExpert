# -*- coding: utf-8 -*-
"""活图检索调试 v2：内联 retrieve 全流程，逐步打印每步之后的 hits 前四"""
import sys
sys.path.insert(0, "/root/autodl-tmp/crop_cultivation")
import rag

ents = rag.load_graph()
rag.init_dense(ents)

def stage(name, hits):
    print(f"  [{name}] " + "、".join(f"{n[:20]}" for _, n, *_ in hits[:4]))

for q in sys.argv[1:] or ["小麦叶片上有一层白粉一样的东西，是啥病？咋治？",
                          "小麦赤霉病什么时候打药？打多少？"]:
    print("=" * 70)
    print("问:", q)
    in_q = next((c for c in rag.CROPS if c in q), None)
    topk = rag.TOP_ENTITIES
    if rag._BROAD_RE.search(q):
        topk = max(topk, 6)
    noun0 = rag._residue_noun(q, in_q)
    print("  noun =", noun0, "｜psyns =", {s for s in rag._op_syns(q) if s != "管理"})
    lex = rag._lexical(ents, q, noun=noun0)[:15]
    dense = []
    sims = (rag.ENC.encode([q])[0] @ rag.ENT_MAT.T).tolist()
    order = sorted(range(len(rag.ENT_NAMES)), key=lambda i: -sims[i])
    dense = [(sims[i], rag.ENT_NAMES[i], ents[rag.ENT_NAMES[i]]) for i in order[:15]]
    pool = {}
    for lst in (lex, dense):
        for r, (s, name, e) in enumerate(lst):
            p = pool.setdefault(name, {"rrf": 0.0, "e": e, "lex": 0.0, "sim": 0.0})
            p["rrf"] += 1.0 / (60 + r + 1)
            p["lex"] = max(p["lex"], s if (s, name, e) in lex else 0.0)
            p["sim"] = max(p["sim"], s if (s, name, e) in dense else 0.0)
    hits = [(v["rrf"], n, v["e"], v["lex"], v["sim"]) for n, v in pool.items()
            if v["sim"] >= rag.DENSE_MIN or v["lex"] >= rag.LEX_MIN]
    hits.sort(key=lambda x: -x[0])
    stage("门槛过滤+RRF", hits)
    strong = max((h for h in hits if h[3] >= 6.0), key=lambda h: h[3], default=None)
    if strong and strong is not hits[0] and hits[0][3] < strong[3] - 1.0:
        hits.remove(strong); hits.insert(0, strong)
        stage("词面强提升", hits)
    noun = rag._residue_noun(q, in_q)
    if noun and len(noun) >= 2 and hits:
        for i, h in enumerate(hits):
            if any(noun in f[0] for f in h[2]["facts"]):
                if i:
                    hits.insert(0, hits.pop(i))
                    stage(f"名词提升({noun})", hits)
                break
    psyns = {s for s in rag._op_syns(q) if s != "管理"}
    if psyns and not rag._BROAD_RE.search(q) and hits:
        def _does_op(h):
            return (any(s in h[1] for s in psyns)
                    or any(any(s in f[0] for f in h[2]["facts"]) for s in psyns))
        print("  _does_op(hits[0]) =", _does_op(hits[0]))
        if not _does_op(hits[0]):
            for i, h in enumerate(hits):
                if i and _does_op(h):
                    hits.insert(0, hits.pop(i))
                    stage("操作实体提升", hits)
                    break
    if in_q:
        same = [h for h in hits if h[2]["crop"] in (in_q, None, "")]
        print("  同作物保留:", len(same), "/", len(hits),
              "｜被剔:", [f"{n[:16]}(crop={e['crop']})" for _, n, e, *_ in hits if e["crop"] not in (in_q, None, "")][:4])
        hits = same or hits[:2]
    hits = hits[:topk]
    hits.sort(key=lambda h: -(h[0] + (0.02 if rag._curated(h[2]) else 0)))
    stage("最终", hits)

# 实体作物/环节属性核对
from neo4j import GraphDatabase
drv = GraphDatabase.driver(rag.NEO4J_URI, auth=(rag.NEO4J_USER, rag.NEO4J_PASSWORD))
with drv.session(database=rag.NEO4J_DATABASE) as s:
    print("=" * 70)
    for nm in ("小麦白粉病", "长江中下游/黄淮海南片小麦赤霉病", "小麦病害防治", "小麦春季病虫草害的防治药剂"):
        rows = s.run("MATCH (h:Entity {name:$n}) OPTIONAL MATCH (h)-[:属于作物]->(c:Crop) "
                     "OPTIONAL MATCH (h)-[:属于环节]->(st:Stage) RETURN c.name AS c, st.name AS st", n=nm).data()
        print(nm, "→", rows[:3])
drv.close()
