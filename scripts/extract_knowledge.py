# -*- coding: utf-8 -*-
"""模型驱动的知识抽取管线：问答语料 → 图谱知识块（JSONL）。
============================================================
在 AutoDL GPU 上运行（用合并后的微调模型做抽取）：
  python extract_knowledge.py --src qa/rich_converted_clean.json --out extract/rich.jsonl
  python extract_knowledge.py --src qa/wheat.json --out extract/wheat.jsonl --n 30   # 先试 30 条
断点续跑：已写入 out 的 idx 自动跳过。
产物：每行 {"idx","q","extract":{...}}，后续由 aggregate_extraction.py 合并去重成图谱模块。

设计要点（防幻觉）：
1. 事实值必须照抄答案原文——模型只做"结构化判断"（归属作物/环节/主题、拆键值），不做知识创造；
2. 无具体事实的问答输出"跳过": true；
3. 数值不改写、不换算、不补全。
"""
import argparse, json, os, sys, time

BASE = "/root/autodl-tmp/crop_cultivation"
MERGED = os.path.join(BASE, "output", "qwen3_4b_agri_merged")
IM_S, IM_E = "<|im_start|>", "<|im_end|>"

PROMPT = """从下面这条农业问答中抽取【长期可复用的事实知识】。已给出 JSON 的开头，接着补完剩余字段。

【问题】{q}
【答案】{a}

抽取规则：
1. 只抽答案原文明确写出的事实（品种名、参数、时期、用量、操作要点），数值一律原样照抄，不得改写换算或补充；
2. 问题只用于判断主题归属，不得把问题里的口语当事实；
3. 字段要求：
   "作物": "小麦" / "水稻" / "玉米" / "通用"（以问题问的作物为准）
   "环节": 从 育种选种/育秧移栽/播种/施肥/灌溉排水/病害防治/虫害防治/草害防治/田间管理/逆境应对/收获/储藏运输 中选一个，不得自造
   "地区": 只填一个产区（东北/黄淮海/长江中下游/华南/西南/西北），答案没提就填 ""
   "主题": 4—12 字名词短语（如"浸种消毒""稻瘟病防治""玉米穗期追肥"），不含数字和标点；同一主题不同问答用相同措辞
   "事实": [["键", "值"], ...]  键为简短术语（播期/用量/药剂/防治指标/识别特征/操作要点…）；
           值照抄答案原文且单条不超过 60 字，过长内容拆成多条键值
4. 答案没有具体事实（纯观点、太笼统、答非所问）→ "跳过": true，"事实": []
只输出 JSON 剩余部分，开头 "{"已给出，不要输出 "{"，也不要任何解释文字。"""

def _salvage(s):
    """尾部坏掉的 JSON：逐层剥掉尾部杂字符（逗号/引号/多余括号，浅剥优先保内容），
    每种剥法配一组闭合组合重试，能通就通"""
    cands, c = [s], s
    for _ in range(6):
        c2 = c.rstrip(',， \n\t"\'')
        if c2 == c:                        # 没有杂字符可剥 → 看末位是否多余括号
            if c and c[-1] in "]}":
                c = c[:-1]
                cands.append(c)
                continue
            break
        c = c2
        cands.append(c)
    tails = ("}]}", "]]}", "]}", "}}", '"}]', '"}', "}", "]")
    for cut in cands:
        for t in tails:
            try:
                return json.loads(cut + t)
            except json.JSONDecodeError:
                pass
    raise ValueError("尾部修复失败")

import re as _re

def _repair(s):
    """微调模型爱犯的两类 JSON 笔误：值/数组/布尔后缺逗号直接接下一个键"""
    s = _re.sub(r'\]\s*(?=")', '],', s)              # ]\n"跳过" → ],"跳过"
    s = _re.sub(r'\}\s*(?=")', '},', s)
    s = _re.sub(r'(true|false|null)\s*(?=")', r'\1,', s)
    s = _re.sub(r'(?<=")\s+(?=")', ',', s)           # 数组内 "a" "b" 缺逗号
    return s

def _fallback_fields(s):
    """JSON 救不活时的行式兜底：输出本身行式规整（每行一个 "键": "值"），
    按正则直接抠字段——内容是对的就不因标点残缺丢弃"""
    d = {}
    for k in ("作物", "环节", "地区", "主题"):
        m = _re.search(rf'"{k}"\s*[：:]\s*"([^"\n]{{0,30}})"', s)
        if m:
            d[k] = m.group(1)
    m = _re.search(r'"跳过"\s*[：:]\s*(true|false)', s)
    if m:
        d["跳过"] = m.group(1) == "true"
    facts = []
    for m in _re.finditer(r'\[\s*"([^"\n]{1,15})"\s*[，,]\s*"([^"]{1,90}?)"', s):
        k, v = m.group(1), m.group(2)
        if not k.startswith(("作物", "环节", "地区")):      # 跳过头部字段行被误配
            facts.append([k, v])
    d["事实"] = facts
    if not facts:
        d["跳过"] = True
    return d

def parse_extract(text):
    """总入口：严格解析 → 修复解析 → 行式兜底"""
    try:
        return parse_json_loose(text)
    except Exception:
        d = _fallback_fields(text)
        if d.get("事实"):
            return d
        raise ValueError("行式兜底也无事实")

def parse_json_loose(s):
    """模型输出里抠出 JSON 再 json.loads（assistant 已预填 {，故先补一个）"""
    s = "{" + s.strip()
    if s.startswith("{```"):
        s = "{" + s[4:].strip()
    if s.startswith('{"'):            # { 后先吐杂白，引号可能是残片也可能是键名自己的：
        rest = s[2:].lstrip()         #   {"\n"作物"(残片→删) / {"\n作物"(自己的→留)
        s = ("{" if rest.startswith('"') else '{"') + rest
    s = _repair(s)
    depth = 0
    for j in range(len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[:j + 1])
                except json.JSONDecodeError:      # 括号平衡但内部坏
                    return _salvage(s[:j + 1])
    return _salvage(s)                 # 到头没闭合（截断）→ 自动闭包

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="只处理前 N 条（0=全部）")
    ap.add_argument("--model", default=MERGED)
    args = ap.parse_args()

    # 语料可能是 [{问题,答案}...] 或 {"问题","答案"} 键名变体，统一成列表
    with open(args.src, encoding="utf-8") as f:
        raw = json.load(f)
    pairs = []
    for i, it in enumerate(raw):
        q = it.get("问题") or it.get("q") or it.get("instruction") or ""
        a = it.get("答案") or it.get("a") or it.get("output") or ""
        if q and a:
            pairs.append((i, str(q), str(a)))
    if args.n:
        pairs = pairs[:args.n]

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for ln in f:
                try:
                    done.add(json.loads(ln)["idx"])
                except Exception:
                    pass
    todo = [(i, q, a) for i, q, a in pairs if i not in done]
    print(f"[抽取] 语料 {len(pairs)} 条，已完成 {len(done)}，本次待抽 {len(todo)} 条")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cuda",
            trust_remote_code=True).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda",
            trust_remote_code=True).eval()
    eos = tok.convert_tokens_to_ids(IM_E)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0, n_ok, n_skip, n_bad = time.time(), 0, 0, 0
    with open(args.out, "a", encoding="utf-8") as f:
        for k, (i, q, a) in enumerate(todo):
            body = PROMPT.replace("{q}", q[:200]).replace("{a}", a[:600])
            prompt = (f"{IM_S}system\n你是农业知识图谱构建助手，只输出 JSON。{IM_E}\n"
                      f"{IM_S}user\n{body}{IM_E}\n"
                      f"{IM_S}assistant\n{{")            # 预填 {：锁死 JSON 起点，微调模型的"具体如下"开头就插不进来
            ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                o = model.generate(ids, max_new_tokens=560, do_sample=False,
                                   pad_token_id=tok.pad_token_id or eos, eos_token_id=eos)
            text = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip()
            rec = {"idx": i, "q": q[:120]}
            try:
                ex = parse_extract(text)
                if not isinstance(ex.get("事实"), list) or not ex.get("事实"):
                    ex["跳过"] = True
                rec["extract"] = ex
                if ex.get("跳过"):
                    n_skip += 1
                else:
                    n_ok += 1
            except Exception as exerr:
                rec["extract"] = None
                rec["raw"] = text[:1600]
                rec["err"] = str(exerr)[:80]
                n_bad += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (k + 1) % 20 == 0:
                el = time.time() - t0
                print(f"  进度 {k+1}/{len(todo)}  有效 {n_ok} 跳过 {n_skip} 坏例 {n_bad}  "
                      f"{el/60:.1f} 分钟  预计剩余 {(el/(k+1)*(len(todo)-k-1))/60:.0f} 分钟", flush=True)
    print(f"[完成] 有效 {n_ok} / 跳过 {n_skip} / 坏例 {n_bad} → {args.out}")

if __name__ == "__main__":
    main()
