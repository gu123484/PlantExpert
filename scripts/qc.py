# -*- coding: utf-8 -*-
"""全量质检：重复、覆盖、依据、数值具体性、长度、JSON 合法性"""
import json, os, re, sys, random
from collections import defaultdict

KG = r"D:\桌面\桌面总\wew\agri_dataset"
ROOT = r"D:\桌面\桌面总\wew"

def norm(s):
    return re.sub(r"[，。？?！!、\s""''（）()【】—\-]", "", s)

def ngrams(s, n=3):
    s = norm(s)
    return set(s[i:i+n] for i in range(max(len(s)-n+1, 1)))

def main():
    import sys
    tfn = sys.argv[1] if len(sys.argv) > 1 else "train_v2.json"
    vfn = sys.argv[2] if len(sys.argv) > 2 else "val_v2.json"
    with open(os.path.join(KG, "qa_pairs", tfn), encoding="utf-8") as f:
        train = json.load(f)
    with open(os.path.join(KG, "qa_pairs", vfn), encoding="utf-8") as f:
        val = json.load(f)
    allqa = train + val
    print(f"1) 数量: train {len(train)} / val {len(val)} / 合计 {len(allqa)}  {'✓≥10000' if len(train) >= 10000 else '✗'}")

    # 2) ID 唯一
    ids = [q["id"] for q in allqa]
    dupids = len(ids) - len(set(ids))
    print(f"2) ID唯一性: {'✓' if dupids == 0 else f'✗ {dupids} 个重复'}")

    # 3) 问题精确重复
    qs = [norm(q["instruction"]) for q in allqa]
    dupq = len(qs) - len(set(qs))
    print(f"3) 问题精确重复: {'✓ 无' if dupq == 0 else f'✗ {dupq} 条'}")

    # 4) 整条重复（问+答）
    pairs = [norm(q["instruction"]) + "|" + norm(q["output"]) for q in allqa]
    duppair = len(pairs) - len(set(pairs))
    print(f"4) 整条重复: {'✓ 无' if duppair == 0 else f'✗ {duppair} 条'}")

    # 5) 近重复（全量，倒排索引加速，阈值0.72）
    gs = [ngrams(q) for q in qs]
    inv = defaultdict(list)
    for i, g in enumerate(gs):
        for x in g:
            inv[x].append(i)
    near = 0
    from collections import Counter
    for i in range(len(gs)):
        cand = Counter()
        for x in gs[i]:
            for j in inv[x]:
                if j > i:
                    cand[j] += 1
        for j, c in cand.most_common(8):
            inter = c
            union = len(gs[i]) + len(gs[j]) - inter
            if union and inter / union > 0.72:
                near += 1
    print(f"5) 问题近重复(>0.72): {'✓ 无' if near == 0 else f'✗ {near} 对'}")

    # 6) 与 grain 现有数据交叉重复
    ext = []
    for fn in ["grain/train_data.json", "grain/val_data.json"]:
        with open(os.path.join(ROOT, fn), encoding="utf-8") as f:
            ext += [list(it.values())[0] for it in json.loads(f.read().replace("}\uFF0C", "},"))]
    for fn in ["grain.json", "corn.json"]:
        with open(os.path.join(ROOT, fn), encoding="utf-8") as f:
            ext += [it.get("问题", list(it.values())[0]) for it in json.load(f)]
    extn = set(norm(x) for x in ext)
    cross_exact = sum(1 for q in qs if q in extn)
    print(f"6) 与grain精确重复: {'✓ 无' if cross_exact == 0 else f'✗ {cross_exact} 条'}")

    # 7) 依据完整性
    no_src = [q["id"] for q in allqa if not q["metadata"].get("依据")]
    print(f"7) 依据标注: {'✓ 100%' if not no_src else f'✗ 缺失 {len(no_src)}: {no_src[:5]}'}")

    # 8) 覆盖矩阵
    mat = defaultdict(int)
    for q in allqa:
        mat[(q["metadata"]["作物"], q["metadata"]["环节"])] += 1
    print("8) 覆盖矩阵（作物×环节）:")
    crops = ["小麦", "水稻", "玉米", "通用"]
    stages = ["育种选种", "播种", "育秧移栽", "施肥", "灌溉排水", "病害防治", "虫害防治", "草害防治",
              "田间管理", "时期判断", "逆境应对", "收获", "储藏运输", "区域综合", "诊断急救", "知识边界"]
    hdr = "环节\\作物".ljust(10) + "".join(c.ljust(8) for c in crops)
    print("   " + hdr)
    for st in stages:
        row = st.ljust(13)
        for c in crops:
            row += str(mat.get((c, st), 0)).ljust(8)
        print("   " + row)

    # 9) 数值具体性
    pat = re.compile(r"\d+(?:\.\d+)?\s*(?:公斤|千克|kg|斤|亩|毫升|升|克|g|%|天|日|℃|厘米|米|立方米|株|万|头|次)")
    withnum = sum(1 for q in allqa if pat.search(q["output"]))
    print(f"9) 答案含具体数值: {withnum}/{len(allqa)} = {withnum/len(allqa)*100:.1f}%")

    # 10) 长度
    ol = [len(q["output"]) for q in allqa]
    il = [len(q["instruction"]) for q in allqa]
    print(f"10) 长度: 问题均值 {sum(il)/len(il):.0f} 字（{min(il)}—{max(il)}），答案均值 {sum(ol)/len(ol):.0f} 字（{min(ol)}—{max(ol)}）")
    short = [q["id"] for q in allqa if len(q["output"]) < 30]
    print(f"    过短答案(<30字): {len(short)} 条", short[:5])

    # 11) 格式完备
    bad = [q["id"] for q in allqa if set(q.keys()) != {"id", "instruction", "input", "output", "metadata"}]
    print(f"11) 字段完备: {'✓' if not bad else f'✗ {len(bad)}'}")

    # 12) 随机抽样供人工复核
    random.seed(99)
    print("\n===== 随机抽样 8 条（人工复核）=====")
    for q in random.sample(allqa, 8):
        m = q["metadata"]
        print(f"\n[{q['id']}][{m['作物']}/{m['环节']}/{m['问题类型']}][地区:{m['地区'] or '通用'}]")
        print(f"问: {q['instruction']}")
        print(f"答: {q['output'][:160]}{'...' if len(q['output'])>160 else ''}")
        print(f"据: {m['依据'][:60]}")

if __name__ == "__main__":
    main()
