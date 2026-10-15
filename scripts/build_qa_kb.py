# -*- coding: utf-8 -*-
"""从问答集中选取代表性问答，编入知识库'应用问答库'（Markdown），并统计知识库总字数"""
import json, os, sys
from collections import defaultdict

KG = r"D:\桌面\桌面总\wew\agri_dataset"
MAX_PER_FILE = 60

def main():
    with open(os.path.join(KG, "qa_pairs", "train_10000.json"), encoding="utf-8") as f:
        data = json.load(f)
    groups = defaultdict(list)
    for qa in data:
        m = qa["metadata"]
        if m["问题类型"] in ("comprehensive", "combo", "how", "why", "list", "scenario"):
            groups[(m["作物"], m["环节"])].append(qa)

    outdir = os.path.join(KG, "knowledge_base", "应用问答库")
    os.makedirs(outdir, exist_ok=True)
    total = 0
    files = 0
    for (crop, stage), items in sorted(groups.items()):
        if not items:
            continue
        items = sorted(items, key=lambda x: x["id"])
        fn = os.path.join(outdir, f"{crop}_{stage}.md")
        lines = [f"# {crop}·{stage} 应用问答\n\n> 精选 {min(len(items), MAX_PER_FILE)} 条代表性问答（全部 {len(items)} 条见 qa_pairs/）。\n\n"]
        for qa in items[:MAX_PER_FILE]:
            lines.append(f"## {qa['instruction']}\n\n")
            lines.append(f"{qa['output']}\n\n")
            lines.append(f"*依据：{qa['metadata']['依据']}｜地区：{qa['metadata'].get('地区') or '—'}｜知识块 {qa['metadata']['知识块']}*\n\n")
        text = "".join(lines)
        with open(fn, "w", encoding="utf-8") as f:
            f.write(text)
        total += len(text)
        files += 1
    print(f"应用问答库：{files} 个文件，约 {total} 字符")

    # 统计知识库总字数（所有 md）
    grand = 0
    for root, _, fs in os.walk(os.path.join(KG, "knowledge_base")):
        for fn in fs:
            if fn.endswith(".md"):
                with open(os.path.join(root, fn), encoding="utf-8") as f:
                    grand += len(f.read())
    print(f"知识库全部 Markdown 合计约 {grand} 字符（要求 ≥100000）")
    if grand >= 100000:
        print("✓ 满足 10 万字要求")
    else:
        print(f"✗ 距 10 万字还差 {100000 - grand} 字符")

if __name__ == "__main__":
    main()
