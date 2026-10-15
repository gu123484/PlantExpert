# -*- coding: utf-8 -*-
"""农业智能问答网站后端（FastAPI）：GraphRAG 对话 + 知识图谱可视化
================================================================
浏览器打开 http://服务器IP:8000 即可使用（左：对话问答；右：知识图谱）。

模型后端自动二选一（谁存在用谁）：
  A. GPU 服务器（如 AutoDL）：transformers 加载合并后的微调模型
  B. CPU 服务器：llama-cpp-python 加载 GGUF 量化模型（内存 ≥6GB，Q8_0）

安装：
  GPU: pip install fastapi uvicorn neo4j
  CPU: pip install fastapi uvicorn neo4j llama-cpp-python transformers torch
运行：
  python server.py [--port 8000]
  python server.py --dump     # 连 Aura 导出本地快照 graph_snapshot.json
                              #（之后 Aura 休眠/服务器没网也能启动：自动改用快照）
本地访问：
  服务器有公网 IP → http://IP:8000；内网服务器 → ssh -L 8000:localhost:8000 用户@服务器
"""
import json, os, sys, threading, time

HERE  = os.path.dirname(os.path.abspath(__file__))
BASE  = "/root/autodl-tmp/crop_cultivation"                      # 本机调试可改成实际路径
MERGED = os.path.join(BASE, "output", "qwen3_4b_agri_merged")    # GPU：合并模型目录
GGUF  = next((p for p in (os.path.join(BASE, "agri_qwen3_4b_q8_0.gguf"),
                          os.path.join(HERE,  "agri_qwen3_4b_q8_0.gguf"))
              if os.path.exists(p)),
             os.path.join(BASE, "agri_qwen3_4b_q8_0.gguf"))      # CPU：量化模型（BASE 或 server.py 同级）
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "0"))          # llama.cpp 想 offload 到 GPU 时设 999

sys.path.insert(0, HERE)
import rag   # 复用检索逻辑（load_graph/retrieve/build_context/rag_text/chat_text/REFUSAL）

SNAPSHOT = os.path.join(HERE, "graph_snapshot.json")   # Aura 连不上时的本地快照

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="小麦/水稻/玉米 栽培智能问答")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")),
          name="static")          # echarts.min.js 等本地静态资源（不依赖外部 CDN）
LOCK = threading.Lock()                       # 单模型实例不支持并发推理，生成互斥
STATE = {"ents": {}, "llm": None, "llm_mode": "未加载", "graph_src": ""}
GRAPH = {"nodes": {}, "edges": [], "adj": {}}  # 全量图谱，启动时拉一次（重建图谱后重启服务）

# ================= 模型后端 =================
class _StopFlag:
    """跨线程停机旗：整行去重闸门发现复读时置位，generate 立即停"""
    stop = False

def init_llm():
    if os.path.exists(MERGED):              # 两个模型都在（如 GPU 实例转过 GGUF）→ 优先合并模型（GPU 快）
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MERGED, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kw = dict(torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(MERGED, **kw)
        except TypeError:                       # 新版 transformers 参数改名 dtype
            kw["dtype"] = kw.pop("torch_dtype")
            model = AutoModelForCausalLM.from_pretrained(MERGED, **kw)
        model.eval()
        STATE["llm"], STATE["llm_mode"] = ("hf", model, tok), "GPU·合并模型"
        return
    if os.path.exists(GGUF):
        try:
            from llama_cpp import Llama
            llm = Llama(model_path=GGUF, n_ctx=4096,
                        n_gpu_layers=N_GPU_LAYERS, verbose=False)
            STATE["llm"], STATE["llm_mode"] = llm, f"CPU·GGUF（{os.path.basename(GGUF)}）"
            return
        except Exception as ex:
            print(f"[LLM] GGUF 加载失败：{ex}")
    print(f"[LLM] 警告：未找到模型（{MERGED} 与 {GGUF} 都不存在），/ask 将返回提示")

def generate_iter(prompt, stop=None):
    """统一流式生成器：逐段 yield 文本（GPU=TextIteratorStreamer，CPU=llama.cpp stream）"""
    llm = STATE["llm"]
    if llm is None:
        yield "[模型未加载：请把合并模型放到 MERGED 路径（GPU）或 GGUF 放到 GGUF 路径（CPU）]"
        return
    with LOCK:
        if isinstance(llm, tuple) and llm[0] == "hf":
            from threading import Thread
            from transformers import (TextIteratorStreamer, LogitsProcessorList,
                                      StoppingCriteria)
            _, model, tok = llm
            eos = tok.convert_tokens_to_ids(rag.IM_E)
            streamer = TextIteratorStreamer(tok, skip_prompt=True,
                                            skip_special_tokens=True)
            ids = tok(prompt, add_special_tokens=False,
                      return_tensors="pt").input_ids.to(model.device)

            class _GenRepPen:   # 只罚生成段内部的复读（引用参考资料不罚——防答非所问）
                def __init__(self, n, p=1.12):
                    self.n, self.p = n, p
                def __call__(self, input_ids, scores):
                    for b in range(scores.shape[0]):
                        for t in input_ids[b, self.n:].unique().tolist():
                            scores[b, t] = (scores[b, t] / self.p
                                            if scores[b, t] > 0
                                            else scores[b, t] * self.p)
                    return scores

            class _FlagStop(StoppingCriteria):    # 去重闸门触发 → generate 立即收工
                def __call__(self, input_ids, scores, **kw):
                    return bool(stop and stop.stop)

            # 受控采样（temperature 0.7）：同一问题措辞有变化；关键数值由预写头（照抄图谱）
            # 保证不漂移，复读由 _GenRepPen + 整行去重闸门兜底
            th = Thread(target=model.generate, kwargs=dict(
                input_ids=ids, max_new_tokens=500, do_sample=True,
                temperature=0.7, top_p=0.9,
                logits_processor=LogitsProcessorList([_GenRepPen(ids.shape[1])]),
                stopping_criteria=[_FlagStop()],
                pad_token_id=tok.pad_token_id or eos, eos_token_id=eos,
                streamer=streamer))
            th.start()
            for t in streamer:
                if stop and stop.stop:
                    break
                yield t
            th.join()
        else:                                    # llama-cpp-python（按需拉取，break 即停算）
            for ch in llm(prompt, max_tokens=500, temperature=0.01,
                          stop=[rag.IM_E, rag.IM_S], stream=True):
                if stop and stop.stop:
                    break
                c = ch["choices"][0]
                t = c.get("text") or c.get("delta", {}).get("content") or ""
                if t:
                    yield t

# ================= 全量图谱（可视化用） =================
STRUCT_LB = ("Crop", "Stage", "Region", "药剂", "生育期", "标准")

def load_full_graph():
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(rag.NEO4J_URI,
                                  auth=(rag.NEO4J_USER, rag.NEO4J_PASSWORD))
    nodes, edges = {}, []
    with driver.session(database=rag.NEO4J_DATABASE) as s:
        for r in s.run("""MATCH (n) RETURN elementId(n) AS id, labels(n)[0] AS lb,
                          coalesce(n.name, n.编号) AS name, n.作物 AS crop"""):
            nodes[r["id"]] = {"id": r["id"], "lb": r["lb"], "name": r["name"],
                              "crop": r["crop"] or "", "stage": ""}
        for r in s.run("""MATCH (a)-[r]->(b) RETURN elementId(a) AS s, elementId(b) AS t,
                          type(r) AS rel, r.生育期 AS g, r.条件 AS c, r.用量 AS dose"""):
            edges.append({"s": r["s"], "t": r["t"], "rel": r["rel"],
                          "g": r["g"] or "", "c": r["c"] or "", "dose": r["dose"] or ""})
    driver.close()
    for n in nodes.values():                     # Entity 补上作物/环节维度（rag 已拉好）
        if n["lb"] == "Entity" and n["name"] in STATE["ents"]:
            e = STATE["ents"][n["name"]]
            n["crop"], n["stage"] = e["crop"] or "", e["stage"] or ""
    adj = {}
    for i, e in enumerate(edges):
        adj.setdefault(e["s"], []).append((e["t"], i))
        adj.setdefault(e["t"], []).append((e["s"], i))
    GRAPH.update(nodes=nodes, edges=edges, adj=adj)
    print(f"[图谱] 全量载入：{len(nodes)} 节点 / {len(edges)} 边")

def _build_adj(edges):
    adj = {}
    for i, e in enumerate(edges):
        adj.setdefault(e["s"], []).append((e["t"], i))
        adj.setdefault(e["t"], []).append((e["s"], i))
    return adj

def load_data():
    """图谱数据：优先 Neo4j Aura（最新），连不上自动改用本地快照 graph_snapshot.json。
    返回数据来源。运行期间不再访问 Neo4j——对话/可视化全部走内存。"""
    try:
        STATE["ents"] = rag.load_graph()
        rag.init_dense(STATE["ents"])
        load_full_graph()
        STATE["graph_src"] = "Neo4j Aura（云端）"
    except Exception as ex:
        if not os.path.exists(SNAPSHOT):
            raise RuntimeError(
                f"Neo4j Aura 连不上（{ex}），且没有本地快照 {SNAPSHOT}。\n"
                f"先在网络可达的机器上执行：python server.py --dump   生成快照")
        print(f"[图谱] Aura 连接失败（{ex}）→ 改用本地快照")
        with open(SNAPSHOT, encoding="utf-8") as f:
            d = json.load(f)
        STATE["ents"] = d["ents"]
        GRAPH.update(nodes={n["id"]: n for n in d["nodes"]},
                     edges=d["edges"], adj=_build_adj(d["edges"]))
        rag.init_dense(STATE["ents"])
        STATE["graph_src"] = "本地快照 graph_snapshot.json（未连 Neo4j）"
    return STATE["graph_src"]

def dump_snapshot():
    """连 Aura 拉全量图谱，存成本地 JSON 快照——之后服务器不连网/ Aura 休眠也能启动"""
    STATE["ents"] = rag.load_graph()
    load_full_graph()
    with open(SNAPSHOT, "w", encoding="utf-8") as f:
        json.dump({"ents": STATE["ents"], "nodes": list(GRAPH["nodes"].values()),
                   "edges": GRAPH["edges"]}, f, ensure_ascii=False, indent=1)
    print(f"[快照] 已保存 {os.path.basename(SNAPSHOT)}"
          f"（{len(STATE['ents'])} 实体 / {len(GRAPH['nodes'])} 节点 / {len(GRAPH['edges'])} 边）"
          f"——把它和 server.py 放一起，Aura 连不上时自动启用")

# ================= 接口 =================
@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))

@app.get("/health")
def health():
    return {"llm": STATE["llm_mode"], "dense": rag.ENC is not None,
            "graph": STATE["graph_src"],
            "entities": len(STATE["ents"]),
            "nodes": len(GRAPH["nodes"]), "edges": len(GRAPH["edges"])}

class Ask(BaseModel):
    question: str
    history: list = []          # 多轮：[{role: "user"|"assistant", content: "..."}]，最新在最后

def _clean_hist(hist):
    """只留最近 4 条合法消息，单条截 300 字——前端传什么都兜得住"""
    out = []
    for m in hist or []:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") \
           and isinstance(m.get("content"), str) and m["content"].strip():
            out.append({"role": m["role"], "content": m["content"][:300]})
    return out[-4:]

def log_miss(q, top_sim, n_hits):
    """坏例收集：无命中/低置信/拒答的问题记账到 misses.log，攒着扩同义词表"""
    try:
        with open(os.path.join(HERE, "misses.log"), "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{top_sim:.2f}\t{n_hits}\t{q}\n")
    except Exception:
        pass

@app.post("/ask")
def ask(a: Ask):
    """GraphRAG 问答（SSE 流式）：data: {"type": "meta"|"status"|"delta"|"done", ...}"""
    q = a.question.strip()
    if not q:
        return JSONResponse({"error": "问题为空"}, status_code=400)
    hist = _clean_hist(a.history)
    # 追问增强：当前问没点名作物时，拼上一问再做检索（"那打多少？"→"上一问 那打多少？"）
    rq = q
    last_user = next((m["content"] for m in reversed(hist) if m["role"] == "user"), "")
    if last_user and not any(c in q for c in rag.CROPS):
        rq = last_user + " " + q
    hits, top_sim = rag.retrieve(STATE["ents"], rq)
    if top_sim < 0.65 or not hits:
        log_miss(q, top_sim, len(hits))

    def sse(obj):
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    if top_sim < 0:                              # 领域门外（棉花等）/市场政策题 → 固定拒答，不经模型
        refusal_txt = rag.REFUSAL_MKT if top_sim < -1.5 else rag.REFUSAL

        def gen_refusal():
            yield sse({"type": "meta", "mode": "refusal", "entities": []})
            yield sse({"type": "delta", "t": refusal_txt})
            yield sse({"type": "done"})
        return StreamingResponse(gen_refusal(), media_type="text/event-stream")

    prompt, has_ctx, head = rag.rag_prompt(hits, q, cls_q=rq, hist=hist)
    meta = {"type": "meta", "mode": "rag" if has_ctx else "plain",
            "entities": [{"name": rag.short_name(n), "crop": e["crop"],
                          "stage": e["stage"]} for _, n, e, *_ in hits]}

    import re as _re
    _NUM = _re.compile(r"^[0-9①-⑩]+[.、．)）]?\s*")   # 行首序号（②/3./（一））
    _SEG = _re.compile(r"[；;。]")                     # 段分隔（模型爱用"；"连排事实）
    _LEAD = _re.compile(r"^(具体如下|如下|答)[:：]\s*")  # 微调训练格式的死开头（模型无视系统提示）
    _QMARK = "【问题】"                                # 答完没停自己编下一问（幻觉泄漏），砍掉
    _ECHO = _re.compile(r"来源为参考资料|相关要求[：:]|相关参数来源|参考资料中的内容")
                                                       # 提示词回声（模型把要求复述成答案尾巴
                                                       # '相关参数来源为参考资料…'），静默丢弃
    # 生育期一致性闸门：问某期管理时，模型自由续写的段落若①不含该期②却提别的生育期
    # ③又与参考资料文本对不上（问返青却补"冬前追氮8—10公斤"——语料记忆不是图谱）
    # → 从该段起截断。对齐源用提示词里的参考资料原文：模型从上下文抄的段（哪怕换了
    # 措辞/带了（生育期）注记）bigram 覆盖率必然高，只有上下文外的内容才受闸门约束
    st0 = next((t for t in rag.STAGE_TOKENS if t in rq), "")
    st0 = st0[:-1] if st0.endswith("期") else st0
    _OTHER_ST = (_re.compile("|".join(t for t in rag.STAGE_TOKENS
                                      if t != st0 and t not in st0 and st0 not in t))
                 if st0 else None)
    _ref = prompt.partition("【要求】")[0] if "【要求】" in prompt else ""
    _KNOWN = rag.bigrams(_ref) if (st0 and _ref) else set()

    seen = {}                                   # 本问已放行的事实段（查重账本：段→字符bigram集）

    def emit_line(line):
        """一行放行前按 ；/。 分段查重（段≥12字才记账，忽略序号）。
        精确双向包含 + bigram 模糊匹配（≥0.72 判复读——模型爱换措辞复述同一段）。
        返回 (放行文本, 是否复读)——复读时只放到复读段之前。"""
        out, pos = [], 0
        cuts = list(_SEG.finditer(line)) + [None]
        for m in cuts:
            end = m.end() if m else len(line)
            seg = line[pos:end]
            if _ECHO.search(seg):                 # 提示词回声段：跳过不输出（不算复读，不停机）
                pos = end
                continue
            key = _NUM.sub("", seg.strip())
            if len(key) >= 12:
                kg = set(zip(key, key[1:]))     # 字符 bigram：措辞微调（—/至、℃写法）不影响
                for s, sg in seen.items():
                    if key in s or s in key:
                        return "".join(out), True
                    ov = len(kg & sg)
                    if ov and ov / min(len(kg), len(sg)) >= 0.72:
                        return "".join(out), True
                if _KNOWN and st0 and st0 not in key and _OTHER_ST.search(key) \
                   and len(kg & _KNOWN) < 0.5 * len(kg):               # 越期自由段：截断
                    return "".join(out), True
                seen[key] = kg
            out.append(seg)
            pos = end
        return "".join(out), False

    def gen():
        yield sse(meta)
        if LOCK.locked():                        # 单模型串行推理：有人正在生成时告知排队
            yield sse({"type": "status", "t": "模型正在回答其他问题，排队中…"})
        if head:                                 # 预写的结论开头先吐给前端
            emit_line(head)                      # 先记进查重账本：预填的药方被模型复读时砍得掉
            yield sse({"type": "delta", "t": head})
        stop = _StopFlag()
        buf = ""
        started = False                          # 还没放过生成正文（首行剥死开头）
        try:
            for t in generate_iter(prompt, stop):
                buf += t
                dup = False
                while "\n" in buf:               # 按行放行；行内段复读（忽略序号）→ 截断
                    line, buf = buf.split("\n", 1)
                    if not started:              # 首个生成行剥"具体如下："类训练格式死开头
                        line = _LEAD.sub("", line)
                        started = True
                    if _QMARK in line:           # 答完没停、自己编下一问（幻觉泄漏）→ 砍掉
                        line = line.partition(_QMARK)[0]
                        dup = True
                    text, dup2 = emit_line(line)
                    if text:
                        yield sse({"type": "delta", "t": text + "\n"})
                    if dup or dup2:
                        stop.stop = True
                        break                    # 复读/泄漏段及之后不输出
                if dup or stop.stop:
                    break
            if not stop.stop and buf.strip():    # 收尾残行同样处理（过查重账本——
                if _QMARK in buf:                # 模型答完常不带换行，复读尾行会从这里漏出去）
                    buf = buf.partition(_QMARK)[0]
                if not started:
                    buf = _LEAD.sub("", buf)
                if buf.strip():
                    text, _ = emit_line(buf)
                    if text:
                        yield sse({"type": "delta", "t": text})
        except Exception as ex:
            yield sse({"type": "delta", "t": f"\n[生成出错：{ex}]"})
        yield sse({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})

@app.get("/graph")
def graph(q: str = "", crop: str = ""):
    """总览（q 为空）：非 Value 节点的结构视图，可按作物过滤；
    搜索（q 非空）：以名称含 q 的节点为焦点的邻域子图（含参数值叶子）"""
    nodes, edges, adj = GRAPH["nodes"], GRAPH["edges"], GRAPH["adj"]

    if not q:
        keep = {i for i, n in nodes.items() if n["lb"] != "Value"}
        if crop:
            keep = {i for i, n in nodes.items()
                    if n["lb"] != "Value" and (n["crop"] == crop or
                        (n["lb"] == "Crop" and n["name"] == crop) or
                        n["lb"] in ("Stage", "Region", "药剂", "标准"))}
        mode = "overview"
    else:
        matched = [i for i, n in nodes.items()
                   if n["lb"] != "Value" and q in n["name"]][:8]
        if not matched:                        # 实在没有，才用参数值兜底匹配
            matched = [i for i, n in nodes.items() if q in n["name"]][:8]
        chosen = set(matched)
        for i in matched:                        # 一跳邻居（含事实值）
            chosen.update(nb for nb, _ in adj.get(i, []))
        for i in list(chosen):                   # 结构类节点再扩一跳（药剂治哪些病/生育期农事）
            if nodes[i]["lb"] in ("药剂", "生育期", "标准"):
                chosen.update(nb for nb, _ in adj.get(i, []))
        keep, mode = chosen, "ego"

    out_e = [e for e in edges if e["s"] in keep and e["t"] in keep]
    deg = {}
    for e in out_e:
        deg[e["s"]] = deg.get(e["s"], 0) + 1
        deg[e["t"]] = deg.get(e["t"], 0) + 1
    out_n = [{"id": i, "name": nodes[i]["name"], "lb": nodes[i]["lb"],
              "crop": nodes[i]["crop"], "deg": deg.get(i, 0),
              "match": bool(q and q in nodes[i]["name"])}
             for i in keep]
    return {"mode": mode, "nodes": out_n, "edges": out_e,
            "totals": {"nodes": len(nodes), "edges": len(edges)}}

@app.get("/node")
def node_detail(id: str):
    """节点详情：事实（实体）+ 邻居关系（点击图谱节点时调用）"""
    n = GRAPH["nodes"].get(id)
    if not n:
        return JSONResponse({"error": "节点不存在"}, status_code=400)
    facts = []
    if n["lb"] == "Entity":
        e = STATE["ents"].get(n["name"])
        if e:
            for rel, tail, g, c, src in e["facts"]:
                facts.append({"rel": rel, "value": tail, "g": g, "c": c, "src": src})
    neighbors = []
    for other, ei in GRAPH["adj"].get(id, []):
        m, e = GRAPH["nodes"][other], GRAPH["edges"][ei]
        if m["lb"] == "Value" and facts:         # 事实已在 facts 里，不重复
            continue
        neighbors.append({"name": m["name"], "lb": m["lb"], "rel": e["rel"],
                          "dir": "→" if e["s"] == id else "←",
                          "dose": e["dose"], "g": e["g"]})
    neighbors.sort(key=lambda x: (x["lb"] == "Value", x["lb"]))
    seen, uniq = set(), []
    for x in neighbors:                         # 双向边(易混淆/可防治)只显示一次
        key = (x["name"], x["rel"], x["dose"])
        if key not in seen:
            seen.add(key)
            uniq.append(x)
    return {"name": n["name"], "lb": n["lb"], "crop": n.get("crop", ""),
            "stage": n.get("stage", ""), "facts": facts, "neighbors": uniq[:80]}

# ================= 启动 =================
def startup():
    print("加载图谱数据（优先 Neo4j Aura，连不上用本地快照）…")
    src = load_data()
    print(f"[图谱] 实体 {len(STATE['ents'])} / "
          f"事实 {sum(len(e['facts']) for e in STATE['ents'].values())}｜来源：{src}")
    print("加载模型 …")
    init_llm()
    print(f"[就绪] LLM 后端：{STATE['llm_mode']}")

if __name__ == "__main__":
    if "--dump" in sys.argv:                     # 只导出快照，不起服务
        dump_snapshot()
        sys.exit(0)
    port = 8000
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    startup()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
