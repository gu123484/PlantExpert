# -*- coding: utf-8 -*-
"""深度体检：模板痕迹、覆盖空洞、依据多样性、数值合理性、答案开头结尾重复度"""
import json, os, re, sys
from collections import Counter, defaultdict

KG = r"D:\桌面\桌面总\wew\agri_dataset"

def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

train = load(os.path.join(KG, "qa_pairs", "train_10000.json"))
val = load(os.path.join(KG, "qa_pairs", "val_500.json"))
qa = train + val
print(f"共 {len(qa)} 条")

# 1 覆盖空洞（作物×环节=0）
mat = defaultdict(int)
for q in qa:
    mat[(q["metadata"]["作物"], q["metadata"]["环节"])] += 1
STAGES = ["育种选种", "播种", "育秧移栽", "施肥", "灌溉排水", "病害防治", "虫害防治", "草害防治",
          "田间管理", "时期判断", "逆境应对", "收获", "储藏运输", "区域综合"]
print("\n[1] 覆盖为 0 的 作物×环节：")
for c in ["小麦", "水稻", "玉米"]:
    zero = [s for s in STAGES if mat[(c, s)] == 0]
    print(f"   {c}: {zero if zero else '无（水稻无播种属设计内，旱直播并入育秧移栽）' if c=='水稻' else zero or '无'}")

# 2 问题类型分布
qt = Counter(q["metadata"]["问题类型"] for q in qa)
print("\n[2] 问题类型分布:", dict(qt.most_common()))

# 3 模板痕迹：答案结尾相同比例（CLOSERS 类短语）、连接词密度
tail = Counter(q["output"][-12:] for q in qa)
print("\n[3] 答案结尾 Top6（结尾雷同度）：")
for t, n in tail.most_common(6):
    print(f"   {n:5d} 条  …{t}")
conn = sum(1 for q in qa if "另外，" in q["output"] or "还要注意" in q["output"])
print(f"   含'另外/还要注意'连接词: {conn}/{len(qa)} = {conn/len(qa)*100:.0f}%")

# 4 依据多样性
src = Counter(q["metadata"]["依据"] for q in qa)
print(f"\n[4] 依据去重后 {len(src)} 种，Top5：")
for s, n in src.most_common(5):
    print(f"   {n:5d} 条  {s[:50]}")

# 5 知识块集中度
blk = Counter(q["metadata"]["知识块"] for q in qa)
print(f"\n[5] 知识块 {len(blk)} 个，平均 {len(qa)/len(blk):.0f} 条/块，Top5：")
for b, n in blk.most_common(5):
    print(f"   {n:5d} 条  {b}")

# 6 长度分布
ol = sorted(len(q["output"]) for q in qa)
print(f"\n[6] 答案长度分位: P5={ol[len(ol)//20]} P50={ol[len(ol)//2]} P95={ol[len(ol)*19//20]} max={ol[-1]}")

# 7 数值合理性抽查（可疑值扫描）
sus = []
for q in qa:
    o = q["output"]
    for mnum, unit in [(r"(\d+(?:\.\d+)?)\s*公斤", "公斤"), (r"(\d+(?:\.\d+)?)\s*立方米", "立方米"),
                       (r"(\d+(?:\.\d+)?)\s*℃", "℃")]:
        for mm in re.finditer(mnum, o):
            v = float(mm.group(1))
            if unit == "公斤" and v > 300 and "产量" not in o[:mm.start()] and "产量" not in o[mm.end():]:
                sus.append((q["id"], f"{v}公斤", o[max(0,mm.start()-20):mm.end()+10]))
            if unit == "立方米" and v > 200:
                sus.append((q["id"], f"{v}立方米", o[max(0,mm.start()-20):mm.end()+10]))
            if unit == "℃" and (v > 45 or v < -25):
                sus.append((q["id"], f"{v}℃", o[max(0,mm.start()-20):mm.end()+10]))
print(f"\n[7] 可疑数值（非产量类>300公斤 / >200立方米 / >45℃）：{len(sus)} 处")
for sid, v, ctx in sus[:10]:
    print(f"   {sid} [{v}] …{ctx}…")

# 8 空字段统计
empty_reg = sum(1 for q in qa if not q["metadata"].get("地区"))
empty_gr = sum(1 for q in qa if not q["metadata"].get("生育期"))
print(f"\n[8] 地区为空: {empty_reg} 条（通用条目，正常）｜生育期为空: {empty_gr} 条")
