# -*- coding: utf-8 -*-
"""农业问答对生成器：从知识块程序化合成微调问答对
生成类型（按变体级别递进，直到达到目标条数）：
  L1 块综合型 / 事实直问型(3问法)
  L2 数值多少型 / 判断标准型
  L3 场景包装型(2)
  L4 原理追问型 / 条件反查型
  L5 操作步骤型 / 列举要点型
  L6 对比型(限区域块) / 组合型
去重：归一化精确去重 + 块内近重复(0.72) + 与grain现有数据近重复(0.6)
"""
import glob, importlib.util, json, os, random, re, sys, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from templates import get_key_questions, OPENERS, CLOSERS, CONNECT

random.seed(20260930)

ROOT = r"D:\桌面\桌面总\wew"
KG = os.path.join(ROOT, "agri_dataset")

CROP_CODE = {"小麦": "W", "水稻": "R", "玉米": "C", "通用": "G"}
STAGE_CODE = {
    "育种选种": "SEL", "播种": "SOW", "育秧移栽": "SED", "施肥": "FER", "灌溉排水": "IRR",
    "病害防治": "DIS", "虫害防治": "PES", "草害防治": "WEE", "田间管理": "MGT",
    "时期判断": "STG", "逆境应对": "STR", "收获": "HAR", "储藏运输": "STO",
    "区域综合": "REG", "诊断急救": "DIA",
}

def load_blocks():
    blocks = []
    for f in sorted(glob.glob(os.path.join(KG, "scripts", "data", "*.py"))):
        name = "m_" + os.path.basename(f).replace(".py", "").replace("-", "_")
        spec = importlib.util.spec_from_file_location(name, f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        blocks += mod.BLOCKS
    return blocks

def norm(s):
    return re.sub(r"[，。？?！!、\s""''（）()【】—\-—]", "", s)

def ngrams(s, n=3):
    s = norm(s)
    return set(s[i:i+n] for i in range(max(len(s)-n+1, 1)))

def jaccard(a, b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

# ---------------- 答案组装 ----------------
def compose_answer(block, focus_keys=None, qtype="direct", rng=None):
    """从块事实组装答案；focus_keys 指定核心事实，其余作补充"""
    facts = block["facts"]
    lead_pool = [v for k, v in facts if not focus_keys or k in focus_keys]
    supp_pool = [v for k, v in facts if (not focus_keys or k not in focus_keys)]
    if not lead_pool:
        lead_pool, supp_pool = [facts[0][1]], [v for _, v in facts[1:]]
    lead = lead_pool[0]
    opener = rng.choice(OPENERS)
    topic = block["topic"]
    parts = [opener.format(lead=lead, topic=topic, region=block.get("region", ""))]
    n_supp = 3 if qtype in ("comprehensive", "how", "scenario") else 2
    rng.shuffle(supp_pool)
    for i, sp in enumerate(supp_pool[:n_supp]):
        conn = rng.choice(CONNECT) if i == 0 else ("另外" if i % 2 else "还要注意")
        parts.append(f"{conn}，{sp}。")
    closer = rng.choice(CLOSERS)
    if closer:
        parts.append(closer)
    text = parts[0].rstrip()
    if not text.endswith(("。", "；", "？", "！", "：")):
        text += "。"
    for p in parts[1:]:
        if not p:
            continue
        if p.endswith("。"):
            text += p
        else:
            text += p + "。"
    # 中文句读规整
    text = re.sub(r"。+", "。", text)
    return text

def steps_answer(block, rng):
    facts = block["facts"]
    rng.shuffle(facts)
    picks = facts[:min(5, len(facts))]
    lines = []
    lead = f"{block['crop']}{block['topic']}的操作要点："
    lines.append(lead)
    for i, (k, v) in enumerate(picks, 1):
        lines.append(f"{chr(0x2460+i-1)}{k}：{v}。")
    return "\n".join(lines)

def why_answer(block, key, value, rng):
    rationale = ""
    for k, v in block["facts"]:
        if any(w in k for w in ("作用", "优点", "机理", "意义", "规律", "原因", "优点")):
            rationale = v; break
    if not rationale:
        for k, v in block["facts"]:
            if k not in (key,) and ("提高" in v or "防止" in v or "减少" in v or "增产" in v or "避免" in v or "保证" in v):
                rationale = v; break
    cond = block.get("condition") or block.get("region") or "生产上"
    ans = f"{block['crop']}{block['topic']}中，{key}定为{value}。"
    if rationale:
        ans += f"这样做的原因在于：{rationale}。"
    else:
        ans += "这一指标来自大量栽培试验与生产验证，是兼顾产量、安全与成本的平衡点。"
    extra = [v for k, v in block["facts"] if k != key]
    rng.shuffle(extra)
    if extra:
        ans += f"配套上，{extra[0]}。"
    return ans

def reverse_answer(block, key, value, rng):
    region = block.get("region", "")
    cond = block.get("condition", "")
    ans = f"该指标（{key}：{value}）对应的技术场景是："
    ctx = []
    if region and region not in ("通用", "区域对比", "分区域"):
        ctx.append(f"{region}地区")
    if cond:
        ctx.append(f"适用条件为{cond}")
    ctx.append(f"{block['crop']}的{block['stage']}环节（{block['topic']}）")
    ans += "，".join(ctx) + "。"
    extra = [v for k, v in block["facts"] if k != key]
    rng.shuffle(extra)
    for e in extra[:2]:
        ans += f"配套要点：{e}。"
    return ans

# ---------------- 问题生成 ----------------
SCEN_OS = ["今年准备种{crop}，", "{crop}生产上，", "新手种{crop}，", "合作社规模种植{crop}，"]

def _clean_subject(rctx, crop, topic):
    """拼'地区+作物+主题'并避免词语重复（主题已含地区或作物时不再重复）"""
    parts = []
    if rctx and not topic.startswith(rctx) and rctx not in topic[:6]:
        parts.append(rctx)
    if crop and crop not in topic[:6]:
        parts.append(crop)
    parts.append(topic)
    return "".join(parts)

def _obj_phrase(block):
    """构造块的问题主语：病虫草害块用'作物+病名'，其余返回 None 由各 qtype 自行组织"""
    crop = block["crop"]; topic = block["topic"]; stage = block["stage"]; region = block.get("region", "")
    rctx = region if region and region not in ("通用", "") else ""
    if stage in ("病害防治", "虫害防治", "草害防治"):
        if topic.startswith(crop):
            return f"{rctx}{topic}"
        return f"{rctx}{crop}{topic}"
    return None

def gen_block_qas(block, rng):
    """块级综合型 QA（每块 2 条）"""
    out = []
    region = block.get("region", "")
    crop = block["crop"]
    topic = block["topic"]
    stage = block["stage"]
    rctx = "" if region in ("通用", "") else f"{region}"
    q1 = f"{rctx}{crop}{topic}的技术要点有哪些？"
    q2 = f"{rctx}{crop}{topic}具体怎么做？请给出关键参数。"
    if stage in ("区域综合",):
        q1 = f"{region}{crop}生产有哪些技术特点和农时安排？"
        q2 = f"{region}种{crop}，全年关键技术环节怎么把握？"
    for q, qt in ((q1, "comprehensive"), (q2, "comprehensive")):
        if stage == "区域综合":
            ans = steps_answer(block, rng)
        else:
            ans = compose_answer(block, qtype=qt, rng=rng)
        out.append((q, ans, qt, None))
    return out

def fact_question(block, key, value, qi, qtype, rng):
    """按类型构造事实级问题（主题词保证区分度）"""
    crop = block["crop"]; region = block.get("region", ""); topic = block["topic"]
    stage = block["stage"]; cond = block.get("condition", "")
    base_qs = get_key_questions(key, stage, crop)
    base = base_qs[qi % len(base_qs)]
    rctx = region if region and region not in ("通用", "区域对比", "分区域") else ""
    obj = _obj_phrase(block)  # 病虫草害对象短语（如"玉米大斑病"），否则 None
    subject = f"{rctx}{crop}"
    if stage == "区域综合":
        subject = f"{region}{crop}"
    if stage == "诊断急救":
        subject = topic  # 主题本身即对象（如"小麦黄苗诊断"）
    if obj:
        subject = obj
    else:
        subject_ctx = _clean_subject(rctx, crop, topic)  # 非病虫块：主题随行
        if qtype == "direct":
            return f"{subject_ctx}方面，{base}？"
        subject = subject_ctx
    if qtype == "direct":
        return f"{subject}{base}？"
    if qtype == "howmuch":
        return f"{subject}中，{base}？亩用量请给具体数字。"
    if qtype == "criterion":
        return f"{subject}中，{key}的判断标准是什么？"
    if qtype == "scenario":
        so = rng.choice(SCEN_OS).format(crop=crop)
        scen = cond if cond else (topic if stage != "诊断急救" else "田间出现异常")
        return f"{so}{scen}，{base}？"
    if qtype == "why":
        return f"{subject}中{base}？为什么要这样定？"
    if qtype == "reverse":
        v = value if len(value) <= 28 else value[:26] + "…"
        return f"生产上用到“{v}”，这对应{crop}什么环节、怎么用？"
    if qtype == "how":
        return f"{subject}怎么操作？{base}的具体做法是什么？"
    if qtype == "list":
        return f"{subject}中关于{key}，都要注意什么？"
    if qtype == "compare":
        return f"{rctx}{crop}{topic}和其他地区相比，{base}？"
    if qtype == "wrapped":
        return f"请教一下，{subject}的{base}？"
    if qtype == "detail":
        return f"{subject}中，{base}？请展开说明具体细节。"
    return f"{subject}{base}？"

def fact_answer(block, key, value, qtype, rng):
    if qtype == "why":
        return why_answer(block, key, value, rng)
    if qtype == "reverse":
        return reverse_answer(block, key, value, rng)
    if qtype in ("how", "list"):
        return steps_answer(block, rng)
    if qtype == "howmuch":
        ans = f"{key}：{value}。"
        extra = [v for k, v in block["facts"] if k != key]
        rng.shuffle(extra)
        for e in extra[:2]:
            ans += f"{rng.choice(['配套参数', '同时注意', '相关要求'])}：{e}。"
        return ans
    if qtype == "criterion":
        ans = f"{key}：{value}。"
        extra = [v for k, v in block["facts"] if k != key]
        rng.shuffle(extra)
        for e in extra[:2]:
            ans += f"辅助判断：{e}。"
        return ans
    if qtype == "compare":
        return compose_answer(block, focus_keys=[key], qtype="direct", rng=rng)
    if qtype == "scenario":
        return steps_answer(block, rng)
    return compose_answer(block, focus_keys=[key], qtype="direct", rng=rng)

# 变体级别序列：fact级每级生成的问题类型
LEVEL_TYPES = [
    ["direct", "direct", "howmuch"],
    ["criterion", "how", "direct"],
    ["scenario", "scenario"],
    ["why", "reverse"],
    ["how", "list"],
    ["compare", "direct"],
    ["wrapped", "detail"],
    ["direct", "howmuch"],
]

TIME_HINT = ("期", "时", "天", "时间", "阶段")
AMT_HINT = ("量", "多少", "公斤", "毫升", "厘米", "株", "万", "立方米", "克", "升", "倍液")

def combo_qas(block, rng):
    """每块生成2条'时间+用量'组合型问答"""
    facts = block["facts"]
    tf = [(k, v) for k, v in facts if any(h in k for h in TIME_HINT)]
    af = [(k, v) for k, v in facts if any(h in k for h in AMT_HINT)]
    if not tf or not af:
        return []
    rctx = block.get("region", "")
    rctx = rctx if rctx and rctx not in ("通用", "区域对比", "分区域") else ""
    obj = _obj_phrase(block) or _clean_subject(rctx, block["crop"], block["topic"])
    out = []
    used = set()
    for t, a in [(tf[0], af[0]), (tf[-1], af[-1])]:
        if (t[0], a[0]) in used or t[0] == a[0]:
            continue
        used.add((t[0], a[0]))
        q = f"{obj}从时间到用量怎么安排？重点说{t[0]}和{a[0]}。"
        others = [v for k, v in facts if k not in (t[0], a[0])]
        rng.shuffle(others)
        ans = f"{t[0]}：{t[1]}。{a[0]}：{a[1]}。"
        for o in others[:2]:
            ans += f"配套要点：{o}。"
        out.append((q, ans, "combo", None))
    return out

def main():
    rng = random.Random(4711)
    blocks = load_blocks()
    print(f"知识块 {len(blocks)}，事实 {sum(len(b['facts']) for b in blocks)}")

    # 外部对照问题集（grain 现有数据）
    ext_qs = []
    try:
        for fn, keys in [("grain/train_data.json", None), ("grain/val_data.json", None)]:
            with open(os.path.join(ROOT, fn), encoding="utf-8") as f:
                ext_qs += [list(it.values())[0] for it in json.loads(f.read().replace("}\uFF0C", "},"))]
        for fn in ["grain.json", "corn.json"]:
            with open(os.path.join(ROOT, fn), encoding="utf-8") as f:
                ext_qs += [it.get("问题", list(it.values())[0]) for it in json.load(f)]
    except Exception as e:
        print("外部数据载入失败（跳过交叉去重）:", e)
    ext_ngrams = [ngrams(q) for q in ext_qs]
    # 3-gram 倒排索引加速交叉近重复检查
    inv = {}
    for i, gset in enumerate(ext_ngrams):
        for g in gset:
            inv.setdefault(g, []).append(i)

    seen_exact = set()
    used_ids = set()
    results = []
    stats_dup_exact = stats_dup_near = stats_dup_ext = 0

    def is_dup(q):
        nonlocal stats_dup_exact, stats_dup_near, stats_dup_ext
        nq = norm(q)
        h = hashlib.md5(nq.encode()).hexdigest()
        if h in seen_exact:
            stats_dup_exact += 1
            return True
        g = ngrams(q)
        # 块内近重复：与已有同块问题比对
        cands = [i for gl in inv.values() for i in []]  # placeholder
        # 交叉外部近重复
        from collections import Counter
        cnt = Counter()
        for gm in g:
            for i in inv.get(gm, []):
                cnt[i] += 1
        if cnt:
            for i, c in cnt.most_common(6):
                inter = len(g & ext_ngrams[i])
                union = len(g | ext_ngrams[i])
                if union and inter / union > 0.6:
                    stats_dup_ext += 1
                    return True
        seen_exact.add(h)
        return False

    def add(q, a, qt, block, fact_key=None):
        if is_dup(q):
            return False
        crop = block["crop"]; stage = block["stage"]
        cc, sc = CROP_CODE.get(crop, "X"), STAGE_CODE.get(stage, "XXX")
        seq = len(results) + 1
        qa = {
            "id": f"QA-{cc}-{sc}-{seq:05d}",
            "instruction": q,
            "input": "",
            "output": a,
            "metadata": {
                "作物": crop, "环节": stage, "主题": block["topic"],
                "地区": block.get("region", ""), "生育期": block.get("growth", ""),
                "条件": block.get("condition", ""), "问题类型": qt,
                "知识块": block["id"], "事实键": fact_key or "",
                "依据": block["source"],
            },
        }
        results.append(qa)
        return True

    # 块综合级 + 组合级
    for b in blocks:
        for q, a, qt, fk in gen_block_qas(b, rng):
            add(q, a, qt, b, fk)
        for q, a, qt, fk in combo_qas(b, rng):
            add(q, a, qt, b, fk)

    # 事实级逐级扩展
    TARGET = 13000
    for level, types in enumerate(LEVEL_TYPES):
        if len(results) >= TARGET:
            break
        for b in blocks:
            if len(results) >= TARGET:
                break
            for (key, value) in b["facts"]:
                for qi, qt in enumerate(types):
                    if qt == "compare" and b.get("region", "") in ("通用", ""):
                        continue
                    if qt in ("howmuch", "reverse") and not re.search(r"\d", value):
                        continue
                    q = fact_question(b, key, value, qi + level, qt, rng)
                    a = fact_answer(b, key, value, qt, rng)
                    add(q, a, qt, b, key)

    print(f"候选生成 {len(results)} 条（精确重复 {stats_dup_exact}，外部近重复剔除 {stats_dup_ext}）")

    # 全局近重复清理（倒排索引加速，阈值 0.72，保留先出现者）
    from collections import defaultdict, Counter
    allgs = [ngrams(qa["instruction"]) for qa in results]
    inv2 = defaultdict(list)
    for i, g in enumerate(allgs):
        for x in g:
            inv2[x].append(i)
    drop = set()
    for i in range(len(results)):
        if i in drop:
            continue
        cand = Counter()
        for x in allgs[i]:
            for j in inv2[x]:
                if j > i:
                    cand[j] += 1
        for j, c in cand.most_common(10):
            if j in drop:
                continue
            union = len(allgs[i]) + len(allgs[j]) - c
            if union and c / union > 0.72:
                drop.add(j); stats_dup_near += 1
    results = [qa for i, qa in enumerate(results) if i not in drop]
    print(f"全局近重复剔除 {stats_dup_near}，现有 {len(results)} 条")

    # 打乱并重编 ID
    rng.shuffle(results)
    for i, qa in enumerate(results, 1):
        cc = CROP_CODE.get(qa["metadata"]["作物"], "X")
        sc = STAGE_CODE.get(qa["metadata"]["环节"], "XXX")
        qa["id"] = f"QA-{cc}-{sc}-{i:05d}"

    # 拆分 train / val（按作物×环节分层、按比例精确凑 500）
    from collections import defaultdict as dd
    groups = dd(list)
    for qa in results:
        groups[(qa["metadata"]["作物"], qa["metadata"]["环节"])].append(qa)
    val, train = [], []
    total = len(results)
    for k, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        take = min(len(g), round(len(g) / total * 500) or 1)
        val += g[:take]; train += g[take:]
    while len(val) > 500:
        train.append(val.pop())
    while len(val) < 500 and train:
        val.append(train.pop())
    print(f"train {len(train)} 条 / val {len(val)} 条")

    os.makedirs(os.path.join(KG, "qa_pairs"), exist_ok=True)
    with open(os.path.join(KG, "qa_pairs", "train_10000.json"), "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=1)
    with open(os.path.join(KG, "qa_pairs", "val_500.json"), "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=1)
    print("已写出 train / val")

if __name__ == "__main__":
    main()
