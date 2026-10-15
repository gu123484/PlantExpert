# -*- coding: utf-8 -*-
"""完全复刻 server.py 的生成+过滤流水线，逐步打印 emit_line 判定，定位截断点"""
import sys
sys.path.insert(0, "/root/autodl-tmp/crop_cultivation")
import rag, torch
from threading import Thread
from transformers import (AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer,
                          LogitsProcessorList, StoppingCriteria)
import re as _re

ents = rag.load_graph()
rag.init_dense(ents)
tok = AutoTokenizer.from_pretrained(rag.MERGED, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    rag.MERGED, torch_dtype=torch.bfloat16, device_map="cuda",
    trust_remote_code=True).eval()

_NUM = _re.compile(r"^[0-9①-⑩]+[.、．)）]?\s*")
_SEG = _re.compile(r"[；;。]")
_LEAD = _re.compile(r"^(具体如下|如下|答)[:：]\s*")

class RepPen:
    def __init__(self, n, p=1.12):
        self.n, self.p = n, p
    def __call__(self, input_ids, scores):
        for b in range(scores.shape[0]):
            for t in input_ids[b, self.n:].unique().tolist():
                scores[b, t] = (scores[b, t] / self.p if scores[b, t] > 0
                                else scores[b, t] * self.p)
        return scores

for q in sys.argv[1:]:
    hits, ts = rag.retrieve(ents, q)
    rq = q
    prompt, has_ctx, head = rag.rag_prompt(hits, q, cls_q=rq, hist=[])
    st0 = next((t for t in rag.STAGE_TOKENS if t in rq), "")
    st0 = st0[:-1] if st0.endswith("期") else st0
    OTHER = (_re.compile("|".join(t for t in rag.STAGE_TOKENS
                                  if t != st0 and t not in st0 and st0 not in t))
             if st0 else None)
    KNOWN = ([(rag.bigrams(f[0] + f[1]), f[0] + f[1])
              for _, _, e, *_ in hits for f in e["facts"]] if st0 else [])
    seen = {}

    def emit_line(line, tag=""):
        out, pos = [], 0
        for m in list(_SEG.finditer(line)) + [None]:
            end = m.end() if m else len(line)
            seg = line[pos:end]
            key = _NUM.sub("", seg.strip())
            if len(key) >= 12:
                kg = set(zip(key, key[1:]))
                for s, sg in seen.items():
                    if key in s or s in key:
                        print(f"    [DUP-包含]{tag}", key[:30]); return "".join(out), True
                    ov = len(kg & sg)
                    if ov and ov / min(len(kg), len(sg)) >= 0.72:
                        print(f"    [DUP-0.72]{tag}", key[:30], "vs", s[:20]); return "".join(out), True
                if st0 and st0 not in key and OTHER.search(key):
                    if not any(len(kg & k) / max(1, min(len(kg), len(k))) >= 0.4
                               for k, _ in KNOWN):
                        print(f"    [越期截断]{tag}", key[:40]); return "".join(out), True
                seen[key] = kg
            out.append(seg); pos = end
        return "".join(out), False

    print("=" * 60)
    print("问:", q, "| st0 =", st0 or "(无)")
    emit_line(head, tag="HEAD")
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    eos = tok.convert_tokens_to_ids(rag.IM_E)
    th = Thread(target=model.generate, kwargs=dict(
        input_ids=ids, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9,
        logits_processor=LogitsProcessorList([RepPen(ids.shape[1])]),
        pad_token_id=tok.pad_token_id or eos, eos_token_id=eos, streamer=streamer))
    th.start()
    raw_all, emitted = "", ""
    for t in streamer:
        raw_all += t
    th.join()
    print("  RAW生成:", raw_all[:220])
    for line in raw_all.split("\n"):
        txt, dup = emit_line(line, tag="GEN")
        emitted += txt + "\n"
        if dup:
            print("  → 复读停")
            break
    print("  最终放行:", (head + emitted)[:260])
