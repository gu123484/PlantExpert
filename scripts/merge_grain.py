# -*- coding: utf-8 -*-
"""把 grain 原始语料（8806条）规范化并入 train_v2，产出 train_with_grain.json
grain 数据为 {问题, 答案} 结构，无 metadata —— 作物/环节按关键词识别，
依据统一标注为原始语料来源说明。
"""
import json, os, re
from collections import Counter

ROOT = r"D:\桌面\桌面总\wew"
KG = os.path.join(ROOT, "agri_dataset")

def load_pair_file(path, fix_fw=True):
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    if fix_fw:
        txt = txt.replace("}\uFF0C", "},")
    return json.loads(txt)

items = []
for fn in ["grain/train_data.json", "grain/val_data.json"]:
    items += load_pair_file(os.path.join(ROOT, fn))
for fn in ["grain.json", "corn.json"]:
    with open(os.path.join(ROOT, fn), encoding="utf-8") as f:
        items += json.load(f)
print(f"grain 原始语料 {len(items)} 条")

STAGE_KW = [
    ("育秧移栽", ["育秧", "浸种", "催芽", "移栽", "插秧", "秧田", "抛秧", "旱育"]),
    ("播种", ["播种", "播期", "播量", "条播", "穴播", "免耕", "整地", "造墒", "镇压"]),
    ("育种选种", ["品种", "选种", "种子", "育种", "包衣", "拌种", "自交系", "杂交种", "纯度", "发芽率"]),
    ("施肥", ["施肥", "底肥", "基肥", "追肥", "尿素", "氮磷钾", "测土", "配方肥", "有机肥", "叶面肥"]),
    ("灌溉排水", ["灌水", "浇水", "灌溉", "排水", "晒田", "墒情", "滴灌", "喷灌", "涝"]),
    ("病害防治", ["病", "锈病", "赤霉病", "稻瘟", "纹枯", "白粉", "大斑病", "小斑病", "霉病", "菌核"]),
    ("虫害防治", ["虫", "蚜虫", "螟虫", "飞虱", "粘虫", "蝗", "螨", "蓟马"]),
    ("草害防治", ["草", "除草", "杂草", "封闭药", "苗后除草"]),
    ("收获", ["收获", "收割", "收割机", "适期", "脱粒", "晾晒", "烘干", "含水量", "产量"]),
    ("储藏运输", ["储藏", "贮藏", "仓储", "保管", "熏蒸", "运输", "入库"]),
    ("田间管理", ["管理", "中耕", "培土", "蹲苗", "化控", "叶龄", "控旺", "镇压", "划锄"]),
]

def guess_crop(text):
    cnt = {c: text.count(c) for c in ("小麦", "水稻", "玉米")}
    best = max(cnt, key=cnt.get)
    return best if cnt[best] > 0 else "通用"

def guess_stage(text):
    for st, kws in STAGE_KW:
        if any(k in text for k in kws):
            return st
    return "田间管理"

normalized = []
for i, it in enumerate(items, 1):
    q = it.get("instruction") or it.get("问题") or ""
    a = it.get("output") or it.get("答案") or it.get("回答") or ""
    if not q or not a:
        continue
    text = q + a
    normalized.append({
        "id": f"QG-G-G-{i:05d}",
        "instruction": q, "input": "", "output": a,
        "metadata": {"作物": guess_crop(text), "环节": guess_stage(text),
                     "主题": "grain原始语料", "地区": "", "生育期": "", "条件": "",
                     "问题类型": "external", "知识块": "GRAIN-EXT",
                     "依据": "grain外部农业问答语料（采集自公开资料，未逐条标注出处）"}})
print(f"规范化 {len(normalized)} 条")
print("作物分布:", dict(Counter(x['metadata']['作物'] for x in normalized)))
print("环节Top8:", dict(Counter(x['metadata']['环节'] for x in normalized).most_common(8)))

with open(os.path.join(KG, "qa_pairs", "train_v2.json"), encoding="utf-8") as f:
    train_v2 = json.load(f)
combined = train_v2 + normalized
print(f"train_with_grain: {len(train_v2)} (v2) + {len(normalized)} (grain) = {len(combined)} 条")
with open(os.path.join(KG, "qa_pairs", "train_with_grain.json"), "w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=1)
print("已写出 qa_pairs/train_with_grain.json")
