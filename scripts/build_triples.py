# -*- coding: utf-8 -*-
"""构建三元组（含属性标注）+ 渲染知识库 Markdown"""
import glob, importlib.util, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from templates import REL_MAP

KG = r"D:\桌面\桌面总\wew\agri_dataset"

def load_blocks():
    blocks = []
    files = sorted(glob.glob(os.path.join(KG, "scripts", "data", "*.py")),
                   # 抽取模块排最后：同一头实体（合并后）事实按模块序取——手工精编事实
                   # 领先抽取事实，药剂预填/上下文前几条拿到的先是核对过的内容
                   key=lambda f: (os.path.basename(f).startswith("extract_"),
                                  os.path.basename(f)))
    for f in files:
        name = "m_" + os.path.basename(f).replace(".py", "").replace("-", "_")
        spec = importlib.util.spec_from_file_location(name, f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        blocks += mod.BLOCKS
    return blocks

def head_entity(block):
    """构造头实体：地区+作物+主题（病虫主题=作物+病名）。
    聚合模块可带 head= 覆盖名（抽取实体并入旧精编实体时写入）"""
    if block.get("head"):
        return block["head"]
    region = block.get("region", "")
    crop, topic, stage = block["crop"], block["topic"], block["stage"]
    r = region if region and region not in ("通用", "区域对比", "分区域") else ""
    if stage in ("病害防治", "虫害防治", "草害防治"):
        base = topic if topic.startswith(crop) else crop + topic
        return f"{r}{base}"
    if stage == "诊断急救":
        return topic
    if stage == "区域综合":
        return f"{r}{crop}{topic}"
    return f"{r}{crop}{topic}"

def main():
    blocks = load_blocks()

    # ---------- 三元组 ----------
    triples = []
    seq = 0
    for b in blocks:
        h = head_entity(b)
        for key, value in b["facts"]:
            seq += 1
            rel = REL_MAP.get(key, key)
            triples.append({
                "id": f"T-{seq:05d}",
                "head": h,
                "relation": rel,
                "tail": value,
                "attributes": {
                    "作物": b["crop"], "环节": b["stage"], "主题": b["topic"],
                    "地区": b.get("region", ""), "生育期": b.get("growth", ""),
                    "条件": b.get("condition", ""),
                    "知识块": b["id"], "依据": b["source"],
                },
            })
    # 块级上下文三元组：作物-包含环节-主题
    for b in blocks:
        seq += 1
        triples.append({
            "id": f"T-{seq:05d}",
            "head": f"{b['crop']}{b['stage']}",
            "relation": "包含技术主题",
            "tail": b["topic"],
            "attributes": {"作物": b["crop"], "环节": b["stage"], "地区": b.get("region", ""),
                           "知识块": b["id"], "依据": b["source"]},
        })

    os.makedirs(os.path.join(KG, "triples"), exist_ok=True)
    with open(os.path.join(KG, "triples", "triples_all.json"), "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False, indent=1)
    # CSV（UTF-8 BOL 便于 Excel）
    with open(os.path.join(KG, "triples", "triples_all.csv"), "w", encoding="utf-8-sig") as f:
        f.write("id,头实体,关系,尾实体,作物,环节,主题,地区,生育期,条件,知识块,依据\n")
        for t in triples:
            a = t["attributes"]
            row = [t["id"], t["head"], t["relation"], t["tail"], a["作物"], a["环节"],
                   a.get("主题", ""), a.get("地区", ""), a.get("生育期", ""), a.get("条件", ""),
                   a.get("知识块", ""), a.get("依据", "")]
            f.write(",".join('"' + str(x).replace('"', '""') + '"' for x in row) + "\n")
    print(f"三元组 {len(triples)} 条已写出（json + csv）")

    # ---------- 知识库 Markdown ----------
    kb_dir = os.path.join(KG, "knowledge_base")
    os.makedirs(kb_dir, exist_ok=True)
    crop_files = {
        "小麦": ["01_育种选种.md", "02_播种.md", "03_施肥.md", "04_灌溉排水.md", "05_病害防治.md",
                "06_虫害草害防治.md", "07_田间管理与时期判断.md", "08_收获与储运.md", "09_区域特点.md"],
        "水稻": ["01_育种选种.md", "02_育秧移栽.md", "03_施肥.md", "04_灌溉排水.md", "05_病害防治.md",
                "06_虫害草害防治.md", "07_田间管理与时期判断.md", "08_收获与储运.md", "09_区域特点.md"],
        "玉米": ["01_育种选种.md", "02_播种.md", "03_施肥.md", "04_灌溉排水.md", "05_病害防治.md",
                "06_虫害草害防治.md", "07_田间管理与时期判断.md", "08_收获与储运.md", "09_区域特点.md"],
        "通用": ["10_跨作物综合.md"],
    }
    STAGE2FILE = {
        "小麦": {"育种选种": 0, "播种": 1, "施肥": 2, "灌溉排水": 3, "病害防治": 4,
                 "虫害防治": 5, "草害防治": 5, "田间管理": 6, "时期判断": 6, "逆境应对": 6,
                 "收获": 7, "储藏运输": 7, "区域综合": 8},
        "水稻": {"育种选种": 0, "育秧移栽": 1, "施肥": 2, "灌溉排水": 3, "病害防治": 4,
                 "虫害防治": 5, "草害防治": 5, "田间管理": 6, "时期判断": 6, "逆境应对": 6,
                 "收获": 7, "储藏运输": 7, "区域综合": 8},
        "玉米": {"育种选种": 0, "播种": 1, "施肥": 2, "灌溉排水": 3, "病害防治": 4,
                 "虫害防治": 5, "草害防治": 5, "田间管理": 6, "时期判断": 6, "逆境应对": 6,
                 "收获": 7, "储藏运输": 7, "区域综合": 8},
    }

    from collections import defaultdict
    buckets = defaultdict(list)  # (crop, fileidx) -> blocks
    for b in blocks:
        crop = b["crop"]
        if crop in STAGE2FILE:
            idx = STAGE2FILE[crop].get(b["stage"], 6)
            buckets[(crop, idx)].append(b)
        else:
            buckets[("通用", 0)].append(b)

    headers = {
        ("小麦", 0): "# 小麦·育种与选种\n\n", ("小麦", 1): "# 小麦·整地播种\n\n",
        ("小麦", 2): "# 小麦·施肥\n\n", ("小麦", 3): "# 小麦·灌溉与排水\n\n",
        ("小麦", 4): "# 小麦·病害防治\n\n", ("小麦", 5): "# 小麦·虫害与草害防治\n\n",
        ("小麦", 6): "# 小麦·田间管理、时期判断与逆境应对\n\n",
        ("小麦", 7): "# 小麦·收获与储藏运输\n\n", ("小麦", 8): "# 小麦·区域生产特点\n\n",
        ("水稻", 0): "# 水稻·育种与选种\n\n", ("水稻", 1): "# 水稻·育秧与移栽\n\n",
        ("水稻", 2): "# 水稻·施肥\n\n", ("水稻", 3): "# 水稻·灌溉与排水\n\n",
        ("水稻", 4): "# 水稻·病害防治\n\n", ("水稻", 5): "# 水稻·虫害与草害防治\n\n",
        ("水稻", 6): "# 水稻·田间管理、时期判断与逆境应对\n\n",
        ("水稻", 7): "# 水稻·收获与储藏运输\n\n", ("水稻", 8): "# 水稻·区域生产特点\n\n",
        ("玉米", 0): "# 玉米·育种与选种\n\n", ("玉米", 1): "# 玉米·整地播种\n\n",
        ("玉米", 2): "# 玉米·施肥\n\n", ("玉米", 3): "# 玉米·灌溉与排水\n\n",
        ("玉米", 4): "# 玉米·病害防治\n\n", ("玉米", 5): "# 玉米·虫害与草害防治\n\n",
        ("玉米", 6): "# 玉米·田间管理、时期判断与逆境应对\n\n",
        ("玉米", 7): "# 玉米·收获与储藏运输\n\n", ("玉米", 8): "# 玉米·区域生产特点\n\n",
        ("通用", 0): "# 跨作物综合技术\n\n",
    }

    total_chars = 0
    for (crop, idx), blist in sorted(buckets.items()):
        fn = crop_files[crop][idx]
        path = os.path.join(kb_dir, crop, fn)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lines = [headers[(crop, idx)]]
        lines.append(f"> 本文件 {len(blist)} 个知识单元，每单元附来源依据。\n\n")
        for b in blist:
            meta = f"地区：{b.get('region','')}｜生育期：{b.get('growth','')}｜适用条件：{b.get('condition','') or '—'}"
            lines.append(f"## {b['topic']}（{b['id']}）\n\n")
            lines.append(f"*{meta}*\n\n")
            for k, v in b["facts"]:
                lines.append(f"- **{k}**：{v}\n")
            lines.append(f"\n> 依据：{b['source']}\n\n")
        text = "".join(lines)
        total_chars += len(text)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # 知识库总览
    with open(os.path.join(kb_dir, "00_知识库总览.md"), "w", encoding="utf-8") as f:
        f.write(f"""# 粮油作物种植技术知识库总览

## 规模
- 知识单元：{len(blocks)} 个
- 量化事实：{sum(len(b['facts']) for b in blocks)} 条
- 正文总量：约 {total_chars} 字符
- 覆盖作物：小麦、水稻、玉米
- 覆盖环节：育种选种、（育秧移栽）、播种、施肥、灌溉排水、病虫害防治、草害防治、田间管理、时期判断、逆境应对、收获、储藏运输
- 覆盖区域：东北、黄淮海、长江中下游、西北、华南、西南 + 跨区域对比

## 依据体系
每个知识单元末尾标注"依据"，来源包括：
1. GB/NY/T 国家与行业标准（GB 4404.1、GB 1350/1351/1353、GB 2761、GB 13078、GB/T 8321、GB/T 29890、NY/T 496 等）
2. 《水稻优质高产栽培技术问答》（费槐林主编，科学普及出版社，9787110040195）
3. 全国农业技术推广服务中心历年施肥/植保/栽培技术指导意见
4. 《小麦栽培学》《水稻栽培学》《玉米栽培学》及各产区栽培技术规程
5. 国家产业技术体系（小麦/水稻/玉米）技术方案

> 注意：数值为通用推荐范围，实际应用以当地农技部门当年指导为准。
""")
    print(f"知识库已渲染：{len(blocks)} 块，正文约 {total_chars} 字符")

if __name__ == "__main__":
    main()
