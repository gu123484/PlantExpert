# -*- coding: utf-8 -*-
"""把 Aura 上的知识图谱导出成独立 HTML（pyvis 力导向图）
用途：浏览器直接打开演示，或 <iframe src="kg_graph.html"></iframe> 嵌入自己的网站
用法：python agri_dataset/scripts/export_graph_html.py
输出：agri_dataset/kg_graph.html
"""
import os
from neo4j import GraphDatabase
from pyvis.network import Network

KG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = os.path.join(KG, "Neo4j-37aabc62-Created-2026-10-01.txt")

def load_conf():
    cfg = {}
    with open(CONF, encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg

# 各类节点配色/字号（标签: (颜色, 大小)）
STYLE = {
    "Crop":   ("#e15759", 34),   # 作物：红
    "Stage":  ("#b07aa1", 24),   # 环节：紫
    "Region": ("#76b7b2", 18),   # 地区：青
    "Entity": ("#4e79a7", 15),   # 技术实体：蓝
    "Value":  ("#94b8d1", 7),    # 参数值：浅蓝小点
    "药剂":    ("#59a14f", 14),   # 绿
    "生育期":  ("#f28e2b", 13),   # 橙
    "标准":    ("#9c755f", 12),   # 棕
}

def main():
    cfg = load_conf()
    driver = GraphDatabase.driver(
        cfg["NEO4J_URI"], auth=(cfg["NEO4J_USERNAME"],
                                cfg.get("NEO4J_PASSWORD") or cfg.get("NEOO4J_PASSWORD")))
    net = Network(height="900px", width="100%", directed=True,
                  bgcolor="#ffffff", font_color="#333333",
                  select_menu=True, filter_menu=True,
                  heading="小麦/水稻/玉米 栽培知识图谱（1458 节点 / 2066 关系）",
                  cdn_resources="remote")
    with driver.session(database=cfg.get("NEO4J_DATABASE") or "neo4j") as s:
        nodes = s.run("""MATCH (n) RETURN elementId(n) AS id, labels(n)[0] AS lb,
                         coalesce(n.name, n.编号) AS name""").data()
        edges = s.run("""MATCH (a)-[r]->(b)
                         RETURN elementId(a) AS s, elementId(b) AS t, type(r) AS rel""").data()
    driver.close()

    for n in nodes:
        color, size = STYLE.get(n["lb"], ("#999999", 10))
        net.add_node(n["id"], label=n["name"], title=f"[{n['lb']}] {n['name']}",
                     color=color, size=size)
    for e in edges:
        net.add_edge(e["s"], e["t"], title=e["rel"], arrows="to",
                     color={"color": "#c9c9c9"})
    net.repulsion(node_distance=110, spring_length=120)   # 力导向布局参数

    out = os.path.join(KG, "kg_graph.html")
    try:
        net.write_html(out, open=False)          # pyvis ≥0.4
    except TypeError:
        net.write_html(out)                      # 旧版没有 open 参数
    except AttributeError:
        net.save_html(out)                       # 更旧版
    # 注：必须 directed=True —— 无向模式会把药剂网络/易混淆的双向边当重复丢弃(86 条)
    # pyvis 在 Windows 上按系统默认编码(GBK)写文件，中文标题会乱码 → 重存为 UTF-8
    try:
        open(out, encoding="utf-8").read()
    except UnicodeDecodeError:
        text = open(out, encoding="gbk").read()
        open(out, "w", encoding="utf-8", newline="").write(text)
    print(f"节点 {len(nodes)} / 边 {len(edges)} → {out}")
    print("用法：浏览器直接打开；或嵌入网页 <iframe src='kg_graph.html' "
          "style='width:100%;height:900px'></iframe>")

if __name__ == "__main__":
    main()
