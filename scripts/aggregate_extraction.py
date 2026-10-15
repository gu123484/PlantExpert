# -*- coding: utf-8 -*-
"""抽取结果 JSONL → 图谱知识块模块（scripts/data/extract_*.py）。
============================================================
用法（本地）：python -X utf8 scripts/aggregate_extraction.py
输入：extract/ 下的 *_full.jsonl（extract_knowledge.py 在 AutoDL 上的产物，先下载回本地）
输出：scripts/data/extract_{作物}_01.py，与手工模块同格式，build_triples.py 自动装载

合并规则（防脏数据，宁缺毋滥）：
1. 单作物语料（rich/corn/wheat）作物按来源锁定，不信模型判断；
   跨作物语料（natesc 官网全文）按每条记录自报的作物分桶，非三大作物整条丢弃；
2. 环节必须在白名单内，否则归"田间管理"；水稻"播种"且主题含育秧/移栽/插秧 → 育秧移栽；
3. 地区只接受单产区，多产区/含糊 → ""（通用）；
4. 主题清洗：去数字标点空格、去作物前缀；病虫草害环节去"防治"后缀（对齐现有裸名实体如"稻瘟病"）；
   空/超长/等于环节名 → "{环节}知识"；
5. 事实键 ≤15 字、值 4—60 字；同实体内 (键, 值去空白) 去重；与现有 triples_all.json
   同头实体的已有尾实体值去重（重建后的 triples_all.json 已含抽取模块时以基准文件为准）；
6. 每实体块事实上限 10 条（按语料顺序保留），超限丢弃。
"""
import glob, importlib.util, json, os, re, sys
from collections import OrderedDict

KG = r"D:\桌面\桌面总\wew\agri_dataset"

# 语料 → (锁定作物, 模块名, 来源标注)。作物=None 表示跨作物语料（natesc 官网全文），
# 此时按每条记录自己报的作物分桶，报不在三大作物内 → 整条丢弃（宁缺毋滥）
SOURCES = [
    (os.path.join(KG, "extract", "rich_full.jsonl"),  "水稻", "extract_rice_01",  "《水稻优质高产栽培技术问答》问答语料"),
    (os.path.join(KG, "extract", "corn_full.jsonl"),  "玉米", "extract_corn_01",  "玉米栽培技术问答语料"),
    (os.path.join(KG, "extract", "wheat_full.jsonl"), "小麦", "extract_wheat_01", "小麦栽培技术问答语料"),
    (os.path.join(KG, "extract", "natesc_full.jsonl"), None,  "extract_natesc_01", "全国农技中心农事指导（官网全文）"),
]

STAGES = {"育种选种", "育秧移栽", "播种", "施肥", "灌溉排水", "病害防治", "虫害防治",
          "草害防治", "田间管理", "逆境应对", "收获", "储藏运输"}
PEST_STAGES = {"病害防治", "虫害防治", "草害防治"}
REGIONS = ("东北", "黄淮海", "长江中下游", "华南", "西南", "西北")
MAX_FACTS = 10

BAD_KEYS = {"作物", "环节", "地区", "主题", "跳过", "问题", "答案", "事实", "备注"}
TOPIC_STRIP = re.compile(r"[\s0-9，。、；：！？“”‘’（）()【】\[\]\-—~·./%]")
# 问句式主题（'稻蓟马危害水稻后会有什么症状'）截到疑问词前；叙述句主题
# （'稻蓟马危害后影响分蘖'）超 8 字且含叙述动词时截到动词前——否则会建出
# 问句实体名，检索锚定头把整句话印进答案
_QWORD = re.compile(r"有什么|怎么样|如何|怎么|为什么|哪些|多少|什么时候|啥|呢|吗")
_NARR = re.compile(r"危害|为害|影响|症状|发生|受害")
# 尾巴词迭代剥净：'稻瘟病防治'→'稻瘟病'、'药害的典型症状'→'药害'；作物名后缀
# 只在剥后仍≥4字时才剥（'夏玉米'要保住）
_TAIL = re.compile(r"(防治技术|防治措施|识别与防治|防除技术|防治|防除|识别|症状|危害|为害|影响|发生|受害|的|典型|主要|后)$")
_CROP_SUF = re.compile(r"(水稻|玉米|小麦)$")


def norm_tail(v):
    """尾实体去重键：去所有空白（全半角空格、换行）"""
    return re.sub(r"\s+", "", str(v))


def norm_topic(topic, crop, stage):
    """主题清洗成规范名词短语：剥问句尾巴/叙述尾巴/防治后缀，对齐现有裸名实体"""
    t = TOPIC_STRIP.sub("", str(topic or ""))
    t = _QWORD.split(t)[0]
    if len(t) > 8 and _NARR.search(t):
        t = _NARR.split(t)[0]
    while True:
        t2 = _TAIL.sub("", t)
        if t2 == t:
            t3 = _CROP_SUF.sub("", t)
            t2 = t3 if len(t3) >= 4 else t
        if t2 == t:
            break
        t = t2
    t = t.replace(crop, "") if t.startswith(crop) and stage not in PEST_STAGES else t
    t = t.strip("之与和及")
    if not t or t == crop or len(t) < 2 or len(t) > 14 or t in STAGES:
        return f"{stage}知识"
    return t


def clean_region(region):
    r = str(region or "").strip()
    return r if r in REGIONS else ""


def clean_stage(stage, topic, crop):
    st = str(stage or "").strip()
    if st not in STAGES:
        st = "田间管理"
    if crop == "水稻" and st == "播种" and re.search(r"育秧|移栽|插秧|抛秧", topic):
        st = "育秧移栽"
    return st


def clean_fact(k, v):
    try:
        k, v = str(k).strip(), str(v).strip()
    except Exception:
        return None
    if not k or len(k) > 15 or k in BAD_KEYS:
        return None
    if not (4 <= len(v) <= 60):
        return None
    if v.startswith(("如下", "见上", "同上")):
        return None
    if _BAD_VAL.search(v):        # 试验设计用语混进事实（'0.25m保护行''试验中为4月11日'）
        return None               # ——生产问答用不上，还把断言数值顶掉
    return k, v

# 试验/科研设计用语：值里出现即丢弃（生产问答不需要，还会挤掉正规药剂剂量）
_BAD_VAL = re.compile(r"保护行|试验中为|试验期间|试验处理|试验设计|试验结果|试验数据|"
                      r"田间试验|大田试验|小区试验|重复[234]次|随机区组")

_REGION_PREFIX = ("长江中下游", "黄淮海南片", "南方稻区", "北方稻区", "黄淮海",
                  "东北", "华南", "西南", "西北", "南方", "北方")

def _core(head):
    """头实体剥产区前缀（含'/'分隔）后的核心名：'长江中下游/黄淮海南片小麦赤霉病'
    →'小麦赤霉病'，用于抽取实体向旧精编实体对齐"""
    t = head
    changed = True
    while changed:
        changed = False
        t = t.lstrip("/－—-")
        for r in _REGION_PREFIX:
            if t.startswith(r):
                t = t[len(r):]
                changed = True
    return t

def _fold_map(old_heads):
    """旧头实体核心名 → 旧头实体名。抽取实体核心名包含某旧核心名（'水稻二化螟防治指标'
    ⊃'水稻二化螟'、'小麦赤霉病药剂'⊃'小麦赤霉病'）→ 并入旧实体，防止近义新实体
    在检索里把精编实体挤下第一名（'赤霉病药剂'顶掉'赤霉病'、问A答B的开头）"""
    m = {}
    for h in old_heads:
        c = _core(h)
        if len(c) >= 5 and c not in m:
            m[c] = h
    return m

def fold_head(head, folds):
    """折进旧精编实体：返回（规范头名, 是否折入）。只做旧⊇新方向（旧名优先）"""
    c = _core(head)
    for oc, oh in folds.items():
        if c == oc or oc in c:
            return oh, True
    return head, False


def fold_filter(facts, head):
    """折入块的事实必须真讲头实体对象：跨对象语料块（讲二化螟的文档捎带福寿螺、
    麦田指标）折进精编实体后，别家对象的事实会在药方预填/上下文里顶出来——
    '水稻二化螟'答出杀菌剂（苯甲·丙环唑治螟虫是错药）就是这么来的"""
    core = _core(head)
    for c in ("小麦", "水稻", "玉米"):
        if core.startswith(c):
            core = core[len(c):]
            break
    parts = [p for p in re.split(r"与|或|、|/", core) if len(p) >= 2]
    if not parts:
        return facts
    return [kv for kv in facts if any(p in kv[0] or p in kv[1] for p in parts)]


def load_existing_tails():
    """现有图谱每头实体已有的尾实体值集合（抽取事实与之去重）"""
    path = os.path.join(KG, "extract", "baseline_tails.json")
    if os.path.exists(path):                     # 抽取模块已入图后重跑聚合：用第一次存的基准
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    with open(os.path.join(KG, "triples", "triples_all.json"), encoding="utf-8") as f:
        triples = json.load(f)
    tails = {}
    for t in triples:
        tails.setdefault(t["head"], set()).add(norm_tail(t["tail"]))
    tails = {h: sorted(v) for h, v in tails.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tails, f, ensure_ascii=False)
    return tails


def head_name(crop, stage, region, topic):
    """与 build_triples.head_entity 同构：地区前缀 + 病虫草害裸名"""
    r = region
    if stage in PEST_STAGES:
        base = topic if topic.startswith(crop) else crop + topic
        return f"{r}{base}"
    return f"{r}{crop}{topic}"


def main():
    existing = load_existing_tails()
    existing = {h: set(v) for h, v in existing.items()}
    folds = _fold_map(existing)         # 旧精编实体核心名 → 全名（抽取实体对齐并入）
    report = [f"[对齐] 旧精编核心名 {len(folds)} 个（抽取实体核心名包含即并入）"]
    for src, crop0, mod_name, src_label in SOURCES:
        if not os.path.exists(src):
            report.append(f"[跳过] {os.path.basename(src)} 不存在（还没下载/没抽完）")
            continue
        # ---- 读 JSONL，按 (作物,环节,地区,主题) 归并，保持语料顺序 ----
        merged = OrderedDict()   # (crop, stage, region, topic) -> {"facts": OrderedDict, "n_q": int}
        n_line = n_skip = n_bad = n_fact = n_dup_old = n_dup_new = n_offcrop = n_fold = 0
        with open(src, encoding="utf-8") as f:
            for ln in f:
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                n_line += 1
                ex = rec.get("extract")
                if not ex or ex.get("跳过") or not ex.get("事实"):
                    n_skip += 1
                    continue
                if crop0 is None:                       # 跨作物语料：信记录自己报的作物
                    crop = str(ex.get("作物") or "").strip()
                    if crop not in ("小麦", "水稻", "玉米"):
                        n_offcrop += 1
                        continue
                else:
                    crop = crop0
                topic0 = str(ex.get("主题") or "")
                stage = clean_stage(ex.get("环节"), topic0, crop)
                topic = norm_topic(topic0, crop, stage)
                region = clean_region(ex.get("地区"))
                key = (crop, stage, region, topic)
                m = merged.setdefault(key, {"facts": OrderedDict(), "n_q": 0})
                m["n_q"] += 1
                head, folded = fold_head(head_name(crop, stage, region, topic), folds)
                m["head"] = head
                m["folded"] = m.get("folded") or folded
                for kv in ex["事实"]:
                    if not (isinstance(kv, (list, tuple)) and len(kv) >= 2):
                        continue
                    got = clean_fact(kv[0], kv[1])
                    if not got:
                        n_bad += 1
                        continue
                    k, v = got
                    if norm_tail(v) in existing.get(head, ()):   # 图里已有同值事实
                        n_dup_old += 1
                        continue
                    dk = (k, norm_tail(v))
                    if dk in m["facts"]:                          # 本次抽取内部重复
                        n_dup_new += 1
                        continue
                    if len(m["facts"]) >= MAX_FACTS:              # 块满即止（语料顺序=优先级）
                        continue
                    m["facts"][dk] = (k, v)
                    n_fact += 1
        # ---- 写模块文件 ----
        id_pfx = "EX" if crop0 else "NA"              # natesc 跨作物块用 NA 前缀防与 EX 撞号
        blocks, seq = [], 0
        for (crop, stage, region, topic), m in merged.items():
            facts = list(m["facts"].values())
            if m.get("folded") and m.get("head"):   # 折入块剔除别家对象的事实
                facts = fold_filter(facts, m["head"])
            if len(facts) < 2:            # 单事实块检索价值低，丢弃
                n_bad += len(facts)
                continue
            hc = next((c for c in ("小麦", "水稻", "玉米") if c in (m.get("head") or "")), None)
            if hc and hc != crop:
                crop = hc    # 单作物语料里混着别家作物的知识（玉米书讲小麦赤霉病）：
                             # 主题自证作物，按头名作物归属。否则合并实体挂多条'属于作物'
                             # 边，crop 过滤会把正确实体当跨作物剔除（问小麦白粉病反把
                             # '小麦白粉病'实体剔掉，检索第一名落到期播日历上）
            seq += 1
            b = dict(
                id=f"{crop[0]}-{id_pfx}-{seq:03d}", crop=crop, stage=stage, topic=topic,
                region=region or "通用", condition="", growth="",
                facts=facts,
                source=f"{src_label}（微调模型结构化抽取，事实照抄答案原文）")
            if m.get("head") and m["head"] != head_name(crop, stage, region, topic):
                b["head"] = m["head"]     # 并入旧精编实体：build_triples 用覆盖名做头
                n_fold += 1
            blocks.append(b)
        if not blocks:
            report.append(f"[空] {os.path.basename(src)}：无可合并知识块")
            continue
        out = os.path.join(KG, "scripts", "data", f"{mod_name}.py")
        with open(out, "w", encoding="utf-8") as f:
            f.write(f'# -*- coding: utf-8 -*-\n"""{src_label} 自动抽取的知识块（aggregate_extraction.py 生成）。\n'
                    f'{"作物按记录自报分桶（仅小麦/水稻/玉米）" if crop0 is None else f"作物按语料锁定为{crop0}"}；'
                    f'事实值照抄答案原文，与手工模块同格式。"""\n\nBLOCKS = [\n')
            for b in blocks:
                fs = ",\n     ".join(json.dumps(x, ensure_ascii=False) for x in b["facts"])
                f.write(f'dict(id="{b["id"]}", crop="{b["crop"]}", stage="{b["stage"]}", '
                        f'topic="{b["topic"]}", region="{b["region"]}",\n'
                        f'     ' + (f'head="{b["head"]}", ' if b.get("head") else "")
                        + 'condition="", growth="",\n'
                        f'     facts=[{fs}],\n'
                        f'     source="{b["source"]}"),\n\n')
            f.write("]\n")
        report.append(
            f"[OK] {os.path.basename(src)}：{n_line} 行，跳过 {n_skip}"
            + (f"、非三大作物 {n_offcrop}" if n_offcrop else "") +
            f"，新增事实 {n_fact}（旧图去重 {n_dup_old}、内部去重 {n_dup_new}、丢弃 {n_bad}）"
            f"→ {len(blocks)} 块（{mod_name}.py），其中 {n_fold} 块并入旧精编实体，头实体如："
            + "、".join(head_name(*k)[:24] for k in list(merged)[:5]))
    print("\n".join(report))


if __name__ == "__main__":
    main()
