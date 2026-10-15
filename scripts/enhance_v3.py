# -*- coding: utf-8 -*-
"""v3 增强：针对微调实测暴露的问题全面改进
  A 一题多问 combo 大扩充（v2 仅 64 条 → 模型漏答后半问）：双问拼接式，
    答案用"①②"分点逐一回应每个子问题，训练模型答全所有子问
  B 病虫"识别+用药"组合问（最常见的一题两问场景）
  C 跨环节全流程问答（种到收的关键时间与用量串联）
  D 答案结尾去模板化（v2 中 Top6 相同结尾占 33% → 重构分布）
输出：qa_pairs/train_v3.json、val_v3.json
"""
import json, os, random, re, sys, hashlib
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qa_generator import load_blocks, ngrams, norm, _obj_phrase, CROP_CODE, STAGE_CODE
from templates import get_key_questions, CLOSERS

KG = r"D:\桌面\桌面总\wew\agri_dataset"
rng = random.Random(20261002)
CODE = dict(STAGE_CODE); CODE["知识边界"] = "BND"; CODE["全流程"] = "FLOW"

def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

train = jload(os.path.join(KG, "qa_pairs", "train_v2.json"))
val = jload(os.path.join(KG, "qa_pairs", "val_v2.json"))
blocks = load_blocks()

# ---------------- 去重门 ----------------
seen = set(); allg = []; inv = defaultdict(list); rejected = Counter()

def gate(q, tag):
    nq = norm(q); h = hashlib.md5(nq.encode()).hexdigest()
    if h in seen:
        rejected[tag + "#exact"] += 1; return False
    g = ngrams(q); cnt = Counter()
    for x in g:
        for i in inv[x]:
            cnt[i] += 1
    for i, c in cnt.most_common(8):
        u = len(g) + len(allg[i]) - c
        if u and c / u > 0.72:
            rejected[tag + "#near"] += 1; return False
    seen.add(h); allg.append(g)
    for x in g:
        inv[x].append(len(allg) - 1)
    return True

for q in train + val:
    gate(q["instruction"], "old")

new_items = []

def md(b, qt, fk):
    return {"作物": b["crop"], "环节": b["stage"], "主题": b["topic"],
            "地区": b.get("region", ""), "生育期": b.get("growth", ""),
            "条件": b.get("condition", ""), "问题类型": qt,
            "知识块": b["id"], "事实键": fk or "", "依据": b["source"]}

def md_flow(crop, blist, qt):
    return {"作物": crop, "环节": "全流程", "主题": f"{crop}从种到收全流程",
            "地区": "", "生育期": "", "条件": "", "问题类型": qt,
            "知识块": "+".join(b["id"] for b in blist), "事实键": "",
            "依据": "；".join(dict.fromkeys(b["source"] for b in blist))}

def add(q, a, qt, meta):
    if not gate(q, qt):
        return False
    meta = dict(meta); meta["问题类型"] = qt
    cc = CODE.get(meta["作物"], "X"); sc = CODE.get(meta["环节"], "XXX")
    new_items.append({"id": f"QA3-{cc}-{sc}-{len(new_items)+1:05d}",
                      "instruction": q, "input": "", "output": a,
                      "metadata": meta})
    return True

def rctx_of(b):
    r = b.get("region", "")
    return r if r and r not in ("通用", "区域对比", "分区域") else ""

def subj_light(b):
    """轻主语：病虫草用'作物+病名'，其余用'地区+作物'（不带主题词，问题更自然）"""
    o = _obj_phrase(b)
    return o or f"{rctx_of(b)}{b['crop']}"

TIME_H = ("期", "时", "天", "时间", "阶段")
AMT_H = ("量", "多少", "公斤", "毫升", "厘米", "株", "万", "立方米", "克", "升", "倍液", "深度", "行距", "密度", "含水量", "水分")
STD_H = ("标准", "指标", "判断", "特征", "识别", "症状")
DRUG_H = ("药剂", "用药", "首选", "配方", "对水量")

def bucket_of(k):
    if any(h in k for h in DRUG_H): return "drug"
    if any(h in k for h in STD_H): return "std"
    if any(h in k for h in TIME_H): return "time"
    if any(h in k for h in AMT_H): return "amt"
    return None

# ---------------- A 一题多问（双问拼接） ----------------
def section_multi():
    n0 = len(new_items)
    for b in blocks:
        if b["stage"] in ("病害防治", "虫害防治", "草害防治"):
            continue  # 病虫走 B 节的识别+用药组合
        numeric = [(k, v) for k, v in b["facts"] if re.search(r"\d", v)]
        if len(numeric) < 2:
            continue
        timed = [(k, v) for k, v in numeric if any(h in k for h in TIME_H)]
        rest = [f for f in numeric if f[0] not in {t[0] for t in timed}]
        pairs = []
        if timed:
            for o in rest[:3]:                     # 时间×数值 优先
                pairs.append((timed[0], o))
        for i in range(min(2, max(0, len(rest) - 1))):   # 数值×数值
            pairs.append((rest[i], rest[i + 1]))
        seenp, pruned = set(), []
        for p in pairs:
            if p[0][0] != p[1][0] and (p[0][0], p[1][0]) not in seenp:
                seenp.add((p[0][0], p[1][0])); pruned.append(p)
        pairs = pruned[:4]
        subj = subj_light(b)
        paired_keys = {p[0][0] for p in pairs} | {p[1][0] for p in pairs}
        others = [v for k, v in b["facts"] if k not in paired_keys]
        for (k1, v1), (k2, v2) in pairs:
            q1 = get_key_questions(k1, b["stage"], b["crop"])[rng.randrange(3)]
            q2 = get_key_questions(k2, b["stage"], b["crop"])[rng.randrange(3)]
            q = rng.choice([f"{subj}{q1}？{q2}？",
                            f"请教下，{subj}{q1}？{q2}？",
                            f"{subj}{q1}？另外{q2}？"])
            lead = "分两点说：" if rng.random() < 0.3 else ""
            ans = f"{lead}①{k1}：{v1}。②{k2}：{v2}。"
            if others:
                ans += f"配套提醒：{rng.choice(others)}。"
            add(q, ans, "combo", md(b, "combo", f"{k1}+{k2}"))
    return len(new_items) - n0

# ---------------- B 病虫识别+用药组合问 ----------------
def section_disease_combo():
    n0 = len(new_items)
    for b in blocks:
        if b["stage"] not in ("病害防治", "虫害防治", "草害防治"):
            continue
        facts = dict(b["facts"])
        idk = next((k for k in facts if any(w in k for w in ("识别", "症状", "特征", "危害"))), None)
        druk = next((k for k in facts if any(h in k for h in DRUG_H)), None)
        stdk = next((k for k in facts if "指标" in k or "适期" in k), None)
        if not (idk and druk):
            continue
        obj = _obj_phrase(b) or f"{rctx_of(b)}{b['crop']}{b['topic']}"
        for q in (f"{obj}怎么识别？发病后用什么药、亩用量多少？",
                  f"{obj}有什么症状？打什么药、怎么打？"):
            ans = f"①识别：{facts[idk]}。②药剂与用量：{facts[druk]}。"
            if stdk:
                ans += f"③{stdk}：{facts[stdk]}。"
            add(q, ans, "combo", md(b, "combo", f"{idk}+{druk}"))
    return len(new_items) - n0

# ---------------- C 跨环节全流程 ----------------
def section_pipeline():
    n0 = len(new_items)
    STAGE_PREF = [("播种", "播种"), ("育秧移栽", "育秧移栽"), ("施肥", "施肥"),
                  ("灌溉排水", "灌溉"), ("病害防治", "病害防治"), ("收获", "收获")]
    for crop in ["小麦", "水稻", "玉米"]:
        picked = []
        for st, label in STAGE_PREF:
            cand = [b for b in blocks if b["crop"] == crop and b["stage"] == st]
            if not cand:
                continue
            b = cand[0]
            sig = next(((k, v) for k, v in b["facts"]
                        if any(h in k for h in TIME_H + AMT_H)), b["facts"][0])
            picked.append((label, sig, b))
        if len(picked) < 4:
            continue
        blist = [p[2] for p in picked]
        for q in (f"种{crop}从种到收，各环节关键的时间和用量帮我捋一遍",
                  f"{crop}一季的重点农事和具体参数有哪些？按顺序说",
                  f"新手第一次种{crop}，从播种到收获的关键节点怎么安排？",
                  f"{crop}全年栽培管理日历怎么定？把播期、施肥、灌溉、收获的时间说全",
                  f"给我一份{crop}从种到管的参数清单，要具体数字"):
            parts = [f"{chr(0x2460+i)}{label}（{k}）：{v}"
                     for i, (label, (k, v), _) in enumerate(picked)]
            ans = "\n".join(parts) + "\n以上为通用推荐方案，各地农时有差异，以当地农技部门当年指导为准。"
            add(q, ans, "pipeline", md_flow(crop, blist, "pipeline"))
    return len(new_items) - n0

# ---------------- D 结尾去模板化（对 v2 既有条目做安全手术） ----------------
NEW_CLOSERS = ["参数为推荐区间，年际间有波动，要结合当年气候。",
               "大面积应用前，建议先小面积试一下再推开。",
               "不同地块条件有差异，拿不准时先问下当地农技站。",
               ""]

def diversify_closers(items):
    hit = changed = 0
    for it in items:
        o = it["output"]
        for c in CLOSERS:
            if c and o.endswith(c):
                hit += 1
                r = rng.random()
                if r < 0.30:                       # 直接删掉结尾句
                    it["output"] = o[:-len(c)].rstrip()
                    changed += 1
                elif r < 0.65:                     # 换成新结尾
                    it["output"] = o[:-len(c)] + rng.choice(NEW_CLOSERS)
                    changed += 1
                break
    return hit, changed

def main():
    a = section_multi()
    b = section_disease_combo()
    c = section_pipeline()
    print(f"A一题多问 {a}｜B识别+用药 {b}｜C全流程 {c}｜合计新增 {len(new_items)}")
    print("去重剔除:", dict(rejected) or "无")

    hit, changed = diversify_closers(train)
    print(f"D结尾去模板化: 命中旧结尾 {hit} 条，改写/删除 {changed} 条")

    # 每类抽 5% 进验证集
    grp = defaultdict(list)
    for it in new_items:
        grp[it["metadata"]["问题类型"]].append(it)
    val_add = []
    for k, lst in grp.items():
        take = max(1, round(len(lst) * 0.05))
        val_add += lst[:take]
    rest = [it for it in new_items if it not in val_add]

    train_v3 = train + rest
    val_v3 = val + val_add
    with open(os.path.join(KG, "qa_pairs", "train_v3.json"), "w", encoding="utf-8") as f:
        json.dump(train_v3, f, ensure_ascii=False, indent=1)
    with open(os.path.join(KG, "qa_pairs", "val_v3.json"), "w", encoding="utf-8") as f:
        json.dump(val_v3, f, ensure_ascii=False, indent=1)
    print(f"train_v3: {len(train_v3)} 条（v2 {len(train)} + 新增 {len(rest)}）")
    print(f"val_v3:   {len(val_v3)} 条（v2 {len(val)} + 新增 {len(val_add)}）")

    # 结尾分布复核
    tail = Counter(q["output"][-12:] for q in train_v3)
    print("改后答案结尾 Top3:", [(t, n) for t, n in tail.most_common(3)])
    combo_total = sum(1 for q in train_v3 + val_v3 if q["metadata"]["问题类型"] == "combo")
    print(f"combo 类型总数（含 v2）: {combo_total}")

if __name__ == "__main__":
    main()
