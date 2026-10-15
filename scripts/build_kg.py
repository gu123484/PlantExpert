# -*- coding: utf-8 -*-
"""把三元组导入 Neo4j（Aura 云实例），构建知识图谱
节点：Entity(技术实体) / Value(参数值) / Crop(作物) / Stage(环节) / Region(地区)
关系：动态类型（适宜时间/播种量/使用药剂…）+ 属于作物/属于环节/适用地区
用法：python agri_dataset/scripts/build_kg.py
"""
import json, os, re, sys
from collections import defaultdict
from neo4j import GraphDatabase

KG = r"D:\桌面\桌面总\wew\agri_dataset"
CONF = os.path.join(KG, "Neo4j-aaf3820f-Created-2026-10-07.txt")

def load_conf():
    cfg = {}
    with open(CONF, encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg

def rel_type(rel):
    """关系名 → 合法的关系类型（去空格/斜杠等）"""
    t = re.sub(r"[^\w\u4e00-\u9fff]", "_", rel)
    return t or "关联"

# ================= 图结构增强（从既有事实推导，不新增知识） =================
DRUG_RE = re.compile(r"(\d+(?:\.\d+)?%[\w一-鿿·]{0,12}?"
                     r"(?:乳油|可湿性粉剂|悬浮剂|水分散粒剂|水剂|颗粒剂|悬乳剂|"
                     r"湿拌种剂|微乳剂|可溶液剂|油悬浮剂|种衣剂))")
DOSE_RE = re.compile(r"(\d+(?:\.\d+)?(?:—\d+(?:\.\d+)?)?\s*(?:克|毫升|升)/亩"
                     r"|\d+(?:—\d+)?倍液|种子(?:量|重量)?\d?\.?\d+%[—\d%]*)")
DRUG_REL_KEYS = ("药剂", "用药", "防治", "拌种", "包衣", "浸种", "施药",
                 "配方", "封闭", "丢心", "喷雾", "除草")

STAGE_SEQ = {
    "小麦": ["播种", "出苗", "分蘖", "越冬", "返青", "起身", "拔节",
             "孕穗", "抽穗", "扬花", "灌浆", "成熟"],
    "水稻": ["播种", "秧田", "移栽返青", "分蘖", "晒田", "拔节",
             "孕穗", "抽穗", "扬花", "灌浆", "成熟"],
    "玉米": ["播种", "出苗", "三叶", "拔节", "小喇叭口", "大喇叭口",
             "抽雄", "吐丝", "灌浆", "成熟"],
}

STD_RE = re.compile(r"(GB/?T?\s*\d+(?:\.\d+)?(?:—\d+)?|NY/?T\s*\d+)")

CONFUSION = [
    ("大斑病", "小斑病", "斑形：梭形大斑灰褐 vs 椭圆小斑黄褐紫边；两病药剂基本通用"),
    ("白粉病", "条锈病", "粉层：白色粉状霉层 vs 鲜黄色粉堆沿叶脉排成条"),
    ("丝黑穗", "瘤黑粉", "部位：穗内黑粉顶部留丝 vs 任意部位肿瘤外膜白亮"),
    ("二化螟", "条纹叶枯", "枯心：整株枯死茎内有虫 vs 假枯心老叶仍绿（灰飞虱传毒）"),
    ("白叶枯", "细菌性条斑", "病斑：叶缘波状有蜜黄菌脓 vs 叶面短条对光半透明"),
    ("稻瘟病", "纹枯病", "位置：叶面菱形/纺锤斑 vs 叶鞘云纹斑后期生菌核"),
    ("麦蚜", "麦蜘蛛", "部位：穗部群集吸汁排蜜露 vs 叶面红褐色小螨叶斑"),
]

def enrich(s, triples):
    """四类结构增强：药剂网络 / 生育期时间线 / 标准溯源 / 易混淆关系"""
    # A 药剂网络：从药剂类事实中抽取 (药品, 用量) → 双向关联
    drug_rows = []
    for t in triples:
        if not any(k in t["relation"] for k in DRUG_REL_KEYS):
            continue
        for m in DRUG_RE.finditer(t["tail"]):
            d = DOSE_RE.search(t["tail"], m.end())
            dose = d.group(1) if d and d.start() - m.end() < 30 else ""
            drug_rows.append({"h": t["head"], "drug": m.group(1), "dose": dose})
    if drug_rows:
        s.run("""UNWIND $rows AS row
                 MERGE (d:药剂 {name: row.drug})
                 MERGE (h:Entity {name: row.h})
                 MERGE (h)-[r:防治用药]->(d) SET r.用量 = row.dose
                 MERGE (d)-[:可防治]->(h)""", rows=drug_rows)
    print(f"药剂网络: {len(drug_rows)} 条用药关系")

    # B 生育期时间线：每作物按农艺顺序建链，实体挂到发生时期
    for crop, toks in STAGE_SEQ.items():
        names = [f"{crop}{t}期" for t in toks]
        s.run("""UNWIND $rows AS row MERGE (p:生育期 {name: row.name})
                 SET p.作物 = row.crop, p.顺序 = row.ord""", rows=[
                     {"name": n, "crop": crop, "ord": i} for i, n in enumerate(names)])
        s.run("""UNWIND $pairs AS p
                 MATCH (a:生育期 {name: p[0]}), (b:生育期 {name: p[1]})
                 MERGE (a)-[:下一阶段]->(b)""", pairs=list(zip(names, names[1:])))
        s.run("""MATCH (c:Crop {name: $crop}), (p:生育期 {作物: $crop})
                 MERGE (c)-[:包含生育期]->(p)""", crop=crop)
        occ = []
        for t in triples:
            g = t["attributes"].get("生育期", "")
            if t["attributes"].get("作物") != crop or not g:
                continue
            for tok in toks:
                if tok in g:
                    occ.append({"h": t["head"], "p": f"{crop}{tok}期"})
        if occ:
            s.run("""UNWIND $occ AS o MATCH (h:Entity {name: o.h}), (p:生育期 {name: o.p})
                     MERGE (h)-[:发生时期]->(p)""", occ=occ)
    n_stg = s.run("MATCH (:Crop)-[:包含生育期]->(p) RETURN count(p) AS c").single()["c"]
    print(f"生育期时间线: {n_stg} 个节点已建链")

    # C 标准溯源：从依据字段抽 GB/NY/T 编号
    std_rows = []
    for t in triples:
        for m in STD_RE.finditer(t["attributes"].get("依据", "")):
            std_rows.append({"h": t["head"], "std": re.sub(r"\s+", "", m.group(1))})
    if std_rows:
        s.run("""UNWIND $rows AS row MERGE (st:标准 {编号: row.std})
                 MERGE (h:Entity {name: row.h}) MERGE (h)-[:依据标准]->(st)""",
              rows=std_rows)
    print(f"标准溯源: {len(std_rows)} 条（去重 {len({r['std'] for r in std_rows})} 个标准）")

    # D 易混淆关系（人工核实的鉴别对）
    names = [r["h"] for r in s.run("MATCH (h:Entity) RETURN h.name AS h")]
    n_conf = 0
    for fa, fb, note in CONFUSION:
        ea = next((n for n in names if fa in n), None)
        eb = next((n for n in names if fb in n and n != ea), None)
        if ea and eb:
            s.run("""MATCH (a:Entity {name: $ea}), (b:Entity {name: $eb})
                     MERGE (a)-[:易混淆 {要点: $note}]->(b)
                     MERGE (b)-[:易混淆 {要点: $note}]->(a)""",
                  ea=ea, eb=eb, note=note)
            n_conf += 1
    print(f"易混淆关系: {n_conf} 对（计划 {len(CONFUSION)} 对）")

def main():
    cfg = load_conf()
    uri = cfg["NEO4J_URI"]
    user = cfg["NEO4J_USERNAME"]
    pwd = cfg.get("NEO4J_PASSWORD") or cfg.get("NEOO4J_PASSWORD")  # 容错配置文件拼写
    db = cfg.get("NEO4J_DATABASE") or "neo4j"
    print(f"连接 {uri} …")

    with open(os.path.join(KG, "triples", "triples_all.json"), encoding="utf-8") as f:
        triples = json.load(f)
    print(f"三元组 {len(triples)} 条")

    import time
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    for attempt in range(5):                      # Aura 刚建实例可能要等几分钟
        try:
            driver.verify_connectivity()
            break
        except Exception as e:
            if attempt == 4:
                raise
            print(f"  连接失败（{e.__class__.__name__}），30 秒后重试 {attempt + 1}/4 …")
            time.sleep(30)

    try:
        with driver.session(database=db) as s:
            s.run("RETURN 1").consume()
    except Exception:
        db = "neo4j"                               # Aura 常见默认库名回退
        print(f"  指定库不可用，回退到默认库 neo4j")

    with driver.session(database=db) as s:
        # 清空旧图（保证可重复导入）
        s.run("MATCH (n) DETACH DELETE n").consume()
        # 唯一性约束（兼作索引）
        for label in ("Entity", "Value", "Crop", "Stage", "Region", "Topic", "药剂", "生育期"):
            s.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (x:{label}) REQUIRE x.name IS UNIQUE").consume()
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (x:标准) REQUIRE x.编号 IS UNIQUE").consume()
        print("已清空旧图并建立约束")

        # 维度节点
        crops = sorted({t["attributes"]["作物"] for t in triples})
        stages = sorted({t["attributes"]["环节"] for t in triples})
        regions = sorted({t["attributes"].get("地区", "") for t in triples
                          if t["attributes"].get("地区") and t["attributes"]["地区"] not in ("通用", "区域对比", "分区域")})
        s.run("UNWIND $names AS n MERGE (x:Crop {name:n})", names=crops)
        s.run("UNWIND $names AS n MERGE (x:Stage {name:n})", names=stages)
        s.run("UNWIND $names AS n MERGE (x:Region {name:n})", names=regions)
        print(f"维度节点: 作物{len(crops)} 环节{len(stages)} 地区{len(regions)}")

        # 按关系类型分组导入（动态关系类型需按类型分批）
        by_rel = defaultdict(list)
        for t in triples:
            a = t["attributes"]
            by_rel[rel_type(t["relation"])].append(
                {"h": t["head"], "r": t["relation"], "t": t["tail"], "seq": t["id"],
                 "crop": a.get("作物", ""), "stage": a.get("环节", ""),
                 "region": a.get("地区", ""), "growth": a.get("生育期", ""),
                 "cond": a.get("条件", ""), "src": a.get("依据", ""), "blk": a.get("知识块", "")})
        for rt, rows in by_rel.items():
            s.run(f"""
                UNWIND $rows AS row
                MERGE (h:Entity {{name: row.h}})
                MERGE (t:Value {{name: row.t}})
                CREATE (h)-[:`{rt}` {{关系: row.r, 生育期: row.growth, 条件: row.cond,
                                      依据: row.src, 知识块: row.blk, 序号: row.seq,
                                      作物: row.crop, 环节: row.stage}}]->(t)
                WITH h, row
                MATCH (c:Crop {{name: row.crop}}) MERGE (h)-[:属于作物]->(c)
                WITH h, row
                MATCH (st:Stage {{name: row.stage}}) MERGE (h)-[:属于环节]->(st)
                WITH h, row
                MATCH (rg:Region {{name: row.region}}) MERGE (h)-[:适用地区]->(rg)
            """, rows=rows)
        print(f"已导入 {len(by_rel)} 种关系类型、{len(triples)} 条三元组")

        # 图结构增强（药剂网络/生育期时间线/标准溯源/易混淆）
        enrich(s, triples)

        # 校验
        n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        r = s.run("MATCH ()-[x]->() RETURN count(x) AS c").single()["c"]
        print(f"\n校验: 节点 {n} 个，关系 {r} 条")
        print("\n节点分布:")
        for rec in s.run("MATCH (n) RETURN labels(n)[0] AS lb, count(*) AS c ORDER BY c DESC"):
            print(f"   {rec['lb']}: {rec['c']}")
        print("\n关系类型 Top10:")
        for rec in s.run("MATCH ()-[x]->() RETURN type(x) AS t, count(*) AS c ORDER BY c DESC LIMIT 10"):
            print(f"   {rec['t']}: {rec['c']}")
        print("\n作物×实体数:")
        for rec in s.run("MATCH (:Entity)-[:属于作物]->(c:Crop) RETURN c.name AS n, count(*) AS c ORDER BY c DESC"):
            print(f"   {rec['n']}: {rec['c']}")
        print("\n药剂网络 Top8（按防治对象数）:")
        for rec in s.run("""MATCH (d:药剂)<-[:防治用药]-(:Entity)
                            RETURN d.name AS n, count(*) AS c ORDER BY c DESC LIMIT 8"""):
            print(f"   {rec['n']}: {rec['c']}")
        print("\n标准溯源（编号×引用实体数）:")
        for rec in s.run("""MATCH (:Entity)-[:依据标准]->(st:标准)
                            RETURN st.编号 AS n, count(*) AS c ORDER BY c DESC LIMIT 8"""):
            print(f"   {rec['n']}: {rec['c']}")
    driver.close()
    print("\n知识图谱构建完成 ✔  打开 https://console.neo4j.io → Instance01 → Query，粘下面任一条：")
    print("  MATCH p=(h:Entity)-[r]->(t) WHERE h.name CONTAINS '条锈病' RETURN p LIMIT 50")
    print("  MATCH p=(h:Entity)-[r]->(t) WHERE h.name CONTAINS '浸种' RETURN p LIMIT 50")
    print("  MATCH p=(c:Crop)<--(h:Entity)-[r]->(t) WHERE c.name='小麦' RETURN p LIMIT 100")

if __name__ == "__main__":
    main()
