# -*- coding: utf-8 -*-
"""GraphRAG 问答：微调模型 + Neo4j 知识图谱检索
================================================
流程：用户提问 → 从 Neo4j 图谱检索相关实体/事实 → 拼进提示词 → 微调模型生成回答

用法（AutoDL 上）:
  pip install neo4j -i https://pypi.tuna.tsinghua.edu.cn/simple
  python rag.py            # 交互问答（加载微调模型）
  python rag.py --test     # 只测检索效果，不加载模型（无 GPU 本机也能跑）
"""
import os, re, sys
from types import SimpleNamespace

# ================= 配置（只改这里） =================
NEO4J_URI = "neo4j+s://aaf3820f.databases.neo4j.io"
NEO4J_USER = "aaf3820f"
# 密码从环境变量 NEO4J_PASSWORD 读取；也可在本目录放 neo4j_password.txt（已被 .gitignore 排除，不会上传）
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
if not NEO4J_PASSWORD:
    _pw_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neo4j_password.txt")
    if os.path.exists(_pw_file):
        with open(_pw_file, encoding="utf-8") as _f:
            NEO4J_PASSWORD = _f.read().strip()
NEO4J_DATABASE = "aaf3820f"

BASE    = "/root/autodl-tmp/crop_cultivation"
MODEL   = os.path.join(BASE, "Qwen3_4b")
ADAPTER = os.path.join(BASE, "output", "qwen3_4b_agri_lora", "adapter")
MERGED  = os.path.join(BASE, "output", "qwen3_4b_agri_merged")
TOP_ENTITIES = 4      # 检索实体数（太多会稀释、诱发串味）
TOP_FACTS    = 3      # 每实体取事实数

# ================= 知识图谱检索 =================
CROPS = ("小麦", "水稻", "玉米")
# 口语症状词 → 图谱实体名线索（轻量实体链接，帮口语问法命中图谱）
SYM_HINTS = {
    "黄斑|条斑|大斑|叶斑": ("大斑病", "小斑病", "褐斑病", "锈病"),   # 不含泛称'叶斑'——
                                   # 抽取扩图后有'玉米叶斑病'泛称实体，症状题会把它顶到
                                   # 第一、预写'很可能是叶斑病'，把大斑病/小斑病挤掉
    "锈|黄粉|脓包":        ("锈病",),
    "白粉|白毛":           ("白粉病",),
    "黑粉|黑包":           ("丝黑穗", "瘤黑粉"),
    "白穗|穗发白":         ("全蚀", "茎基腐", "纹枯", "穗颈"),
    "青枯|整株枯":         ("茎腐",),
    "徒长|细高":           ("恶苗",),
    "卷叶|卷尖|葱管":      ("纵卷叶螟", "蓟马", "条纹叶枯"),
    "枯心|钻心|一拔就断|拔就断": ("二化螟", "玉米螟", "条纹叶枯"),
    "蚜虫|蜜虫":           ("蚜虫",),
    "蜘蛛|螨|叶片变红":    ("红蜘蛛", "麦蜘蛛", "叶螨"),
    "矮|长不高":           ("粗缩", "丛矮", "黄矮"),
    "烂根|根黑|根腐":      ("全蚀", "根腐", "茎基腐"),
    "倒伏|倒了":           ("倒伏",),
    "发霉|霉变":           ("穗腐", "赤霉", "稻曲"),
}
# 口语词 → 图谱术语扩展（命中任一关键词就把对应术语并入问题再做匹配）
Q_EXPAND = {
    "存": "储藏 贮藏 晾晒", "收完": "收获 储藏", "收了": "收获", "收回": "收获",
    "不会霉": "储藏 安全含水量 霉变", "防霉": "储藏 安全含水量 霉变",
    "发霉": "储藏 安全含水量 霉变", "生霉": "储藏 安全含水量 霉变",
    "打药": "药剂 施药", "打啥药": "药剂", "用药": "药剂", "浸种": "浸种 消毒",
    "浇水": "灌溉 灌水 需水", "浇地": "灌溉 灌水", "浇水吗": "灌溉 灌水",
    "拌种": "药剂 种子处理 拌种 包衣", "包衣": "药剂 种子处理 拌种 包衣",
    "多少株": "密度 密植 播量 保苗", "多大密度": "密度 密植 保苗", "一亩种多少": "密度 播量 保苗",
    "追肥": "施肥 追肥", "上肥": "施肥",
    "产量": "目标产量", "品种": "选种", "除草": "草害 除草",
    # 农时日历问法：问具体日期/月份 → 播期收获期日历类实体（图谱有各产区具体日期）
    # （"时候种/时候播"不加：播种块本身排第一，加了反而把日历实体顶到前面挤掉播量事实）
    "几号": "日期 播期 收获期", "几月": "月份 播期 收获期",
    "时候收": "收获期 日历", "上旬": "日期 播期 收获期", "中旬": "日期 播期 收获期",
}
def bigrams(s):
    return {s[i:i+2] for i in range(len(s) - 1)} if len(s) > 1 else {s}

def dup_fact(a, b, th=0.45):
    """两条事实值是否近重复（bigram 重叠 ≥ 阈值）——不同实体常各有一条同义'返青镇压'。
    预填池里用更严的 0.35：'干旱实体的灌水量'与'需水实体的关键灌水'重叠 ~0.42，
    0.45 挡不住，两条水事实会把穗肥挤出预填前四"""
    ka, kb = bigrams(a), bigrams(b)
    ov = len(ka & kb)
    return bool(ov) and ov / min(len(ka), len(kb)) >= th

# 生育期词（问某期怎么管/怎么追肥时，事实里真提到该期的实体要排上来——
# "大喇叭口期"常写在事实值里（"大喇叭口期重施穗肥"），不看值就会漏掉穗肥实体）
STAGE_TOKENS = ("大喇叭口", "小喇叭口", "喇叭口", "拔节", "抽雄", "吐丝", "孕穗", "抽穗",
                "齐穗", "扬花", "灌浆", "分蘖", "越冬", "冬前", "返青", "起身", "三叶", "出苗",
                "乳熟", "蜡熟", "苗期", "穗期", "花期", "成熟期")

# 枚举题（"小麦有什么病害？""玉米有哪些虫害"）：词面与任何病名都重叠不上、语义也难过
# 0.60 门槛 → 整题掉到纯模型。这类题正确做法=按作物+防治环节直接列图谱实体清单（零幻觉）
ENUM_PEST = re.compile(r"(病虫害|病害|虫害|草害)")
ENUM_ASK = re.compile(r"什么|哪些|啥|几种|都有|常见|主要|列表|一览|种类")
ENUM_STAGES = {"病虫害": ("病害防治", "虫害防治"), "病害": ("病害防治",),
               "虫害": ("虫害防治",), "草害": ("草害防治",)}

# 枚举清单题的名字过滤：扩图后防治环节下混着操作块/主题块/跨阶段实体
# （'一喷多防''春季病虫草害的防治药剂''小麦蚜虫'都挂在病害防治环节），
# 清单必须只留真病名/真虫名——病名以'病'结尾、虫名含虫部字
_ENUM_JUNK = re.compile(
    r"防治|药剂|一喷|统防|防控|监测|预警|应急|综合|区别|记录|模式|问题|原则|效果|传播|识别|利用|"
    r"判断|毒素|保鲜|青贮|熏蒸|带状|轮作|抗倒|耐盐|缺锌|缺苗|密植|茎秆|播期|施肥|追肥|"
    r"幼虫|成虫|习性|形态|产卵|寄主|治理|化学|生物|蚜茧蜂|灯诱|作业|栽培|品种|试验|哑")
_PEST_CH = re.compile(r"螟|虱|蚜|蝇|螨|蝗|蛾|虫|蓟马|蜘蛛|地老虎")

def _enum_ok(word, core, crop):
    """枚举清单只收真病名/真虫名：伞块（'小麦病害''返青期病虫害'）、操作块（'一喷多防'）、
    指标/药剂/天敌块、拼贴名（'叶部病害大斑病小斑病'）、别家作物对象（麦/玉米语料里
    抽出的'稻瘟病'块）全部剔除"""
    if core in ("病害", "虫害", "草害", "病虫害") or "病虫害" in core or "病害" in core:
        return False
    if any(c in core for c in CROPS if c != crop):
        return False
    if crop != "水稻" and core.startswith("稻"):
        return False           # 跨作物知识自证对象（稻瘟/稻曲是水稻病害，麦书里讲它不算小麦病害）
    if _ENUM_JUNK.search(core):
        return False
    is_dis = core.endswith("病")
    is_pst = bool(_PEST_CH.search(core)) and "虫害" not in core
    if word == "病害":
        return is_dis
    if word == "虫害":
        return is_pst
    if word == "草害":
        return True            # 草名无公共字根（节节麦/雀麦/稗草），只剔伞块与操作块
    return is_dis or is_pst    # 问'病虫害'：两类都要

_ENUM_UMB = re.compile(r"与|或|其他|等")   # 伞名（'茎基腐病与根腐病'）排在具体病名之后

def load_graph():
    """一次性拉全图，本地建索引。实体维度（作物/环节）取首条事实自带的维度：
    精编模块先入库、序号最小——合并实体的首条事实必属精编块，环节不会被后并进
    来的抽取块翻牌（'水稻稻瘟病'曾被抽取块挂到'收获'环节，枚举清单直接漏掉它）"""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    ents = {}
    with driver.session(database=NEO4J_DATABASE) as s:
        for r in s.run("""
            MATCH (h:Entity)-[r]->(t:Value)
            RETURN h.name AS h, coalesce(r.关系, type(r)) AS rel, t.name AS t,
                   r.生育期 AS g, r.条件 AS c, r.依据 AS src,
                   r.作物 AS crop, r.环节 AS stage
            ORDER BY r.序号"""):      # 按建图原始顺序：事实注入顺序确定，重建图谱不再洗牌
            e = ents.setdefault(r["h"], {"crop": None, "stage": None, "facts": []})
            if e["crop"] is None and (r["crop"] or r["stage"]):
                e["crop"], e["stage"] = r["crop"], r["stage"]
            e["facts"].append((r["rel"], r["t"], r["g"] or "", r["c"] or "", r["src"] or ""))
    driver.close()
    return ents

REGIONS = ("东北", "黄淮海", "长江中下游", "华南", "西南", "西北")

def _curated(e):
    """精编实体（事实来自手工核对模块，非语料抽取）。抽取扩图后近义新实体
    （裸名'白粉病'、'玉米小斑病防治指标'、'小麦春季病虫草害的防治药剂'）会在
    词法/语义上压过精编实体并抢走检索第一名——精编事实逐条核对过，同等相关时
    必须优先；抽取实体只在精编没覆盖的题目里顶上来"""
    return any(f[4] and "抽取" not in f[4] for f in e["facts"])

def _lexical(ents, q, noun=""):
    """词法打分：实体名/事实键 bigram 重叠 + 术语扩展 + 症状线索 + 生育期加成 + 作物过滤
    + 名词点名加成（'空秆''秃尖'没写成实体名、写在别家实体的'空秆原因'键里 → 该实体加分）"""
    extra = " ".join(t for k, t in Q_EXPAND.items() if k in q)
    gq = bigrams(q + " " + extra)
    stages = [t for t in STAGE_TOKENS if t in q]
    syns = _op_syns(q)          # 问操作（追肥/浇水/拌种…）：操作词本体+同义术语（'追肥'→'施肥'）
    crop_in_q = next((c for c in CROPS if c in q), None)
    region_in_q = next((r for r in REGIONS if r in q), None)   # 问产区：名字带该产区的实体优先
    hints = [tgt for alts, tgts in SYM_HINTS.items()            # （'黄淮海小麦播期播量'要压过
             if any(a in q for a in alts.split("|")) for tgt in tgts]  # 泛称'播种关键参数'——
    scored = []                                                 # 泛称实体混着春播日期）
    for name, e in ents.items():
        sc = len(gq & bigrams(name))
        if e["facts"]:
            sc += 0.5 * len(gq & bigrams(" ".join(f[0] for f in e["facts"])))
        if stages and any(t in name or any(t in f[0] or t in f[1] for f in e["facts"])
                          for t in stages):
            sc += 2      # 问生育期：事实里真提该期的实体加分（"大喇叭口期重施穗肥"写在值里，
        if hints and any(t in name for t in hints):   # 只看键会漏掉穗肥/玉米螟这类实体）
            sc += 1.5 if any(w in name for w in ("与", "其他", "等")) else 3
                         # 泛称伞实体（'褐斑病与其他叶斑病''根腐病与茎基腐'）也含病名线索，
                         # 但症状题要落到具体病上——伞名减半加成，'大斑病'压过'褐斑病与
                         # 其他叶斑病'，否则预写头会说'很可能是褐斑病与其他叶斑病'
        if noun and len(noun) >= 2 and any(noun in f[0] for f in e["facts"]):
            sc += 2      # 问的名词真写在事实键里（'空秆原因'这类专门键；值里顺嘴一提不算——
                         # 粳稻生产特点的值里提'防烂秧'不等于它管烂秧）
        if syns and (any(s in name for s in syns)
                     or any(any(s in f[0] for f in e["facts"]) for s in syns)):
            sc += 2.5    # 问操作（'大喇叭口期怎么追肥'）：名字或键里真管这操作的实体优先——
                         # 否则生育期加成会让'大喇叭口期干旱指标'这类顺带提该期的键占第一
        if region_in_q and region_in_q in name:
            sc += 2.5    # 问产区（'黄淮海…播种'）：带该产区名的实体（产区专属播期/品种）
                         # 压过不带产区的泛称实体——泛称的事实常是别的产区参数
        if _curated(e):
            sc += 2      # 精编先验：同等相关时手工核对实体优先于抽取实体
        if crop_in_q:
            sc += 1.5 if e["crop"] == crop_in_q else -2
        if sc > 0:
            scored.append((sc, name, e))
    scored.sort(key=lambda x: -x[0])
    return scored

# ================= 语义向量（可选：bge-small-zh，混合检索用） =================
EMB_PATHS = [os.path.join(BASE, "bge_small"),
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "bge_small")]
DENSE_MIN = 0.60        # 语义相似度门槛：低于它的实体不注入（越界问题→无命中→拒答）
LEX_MIN   = 4.0         # 词法直通门槛：词面命中很硬时不必依赖向量

ENC, ENT_MAT, ENT_NAMES = None, None, []

def init_dense(ents):
    """有 bge 模型就预编码全部实体（208 个，秒级）；失败自动退回纯词法"""
    global ENC, ENT_MAT, ENT_NAMES
    path = next((p for p in EMB_PATHS if os.path.exists(p)), None)
    if not path:
        print("[检索] 未找到向量模型 → 使用纯词法检索（同义表兜底）")
        return False
    try:
        import torch as T
        from transformers import AutoModel, AutoTokenizer
        dev = "cuda" if T.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModel.from_pretrained(path).to(dev).eval()
        def encode(texts, batch=16):
            out = []
            for i in range(0, len(texts), batch):
                enc = tok(texts[i:i + batch], padding=True, truncation=True,
                          max_length=512, return_tensors="pt").to(dev)
                with T.no_grad():
                    h = mdl(**enc).last_hidden_state[:, 0]      # [CLS]
                out.append(T.nn.functional.normalize(h, dim=1).cpu())
            return T.cat(out)
        ENC = SimpleNamespace(encode=encode)
        ENT_NAMES = sorted(ents)
        texts = []
        for n in ENT_NAMES:
            e = ents[n]
            body = "；".join(f"{r}：{t}" for r, t, *_ in e["facts"])
            texts.append(f"{n}（{e['crop'] or ''}{e['stage'] or ''}）{body}"[:500])
        ENT_MAT = encode(texts)
        print(f"[检索] 混合检索就绪：bge-small-zh 已编码 {len(ENT_NAMES)} 个实体（{dev}）")
        return True
    except Exception as ex:
        print(f"[检索] 向量模型加载失败（{ex}）→ 退回纯词法检索")
        ENC = None
        return False

OUT_CROPS = ("棉花", "花生", "大豆", "油菜", "土豆", "马铃薯", "甘薯", "红薯", "烟草", "茶叶",
             "果树", "苹果", "柑橘", "蔬菜", "番茄", "西红柿", "洋柿子", "圣女果", "黄瓜",
             "辣椒", "西瓜", "甜瓜", "哈密瓜", "草莓", "葡萄", "樱桃", "猕猴桃", "桃", "梨",
             "香蕉", "芒果", "荔枝", "龙眼", "甘蔗", "甜菜", "高粱", "谷子", "食用菌", "蘑菇",
             "中药材", "花卉", "大蒜", "生姜", "韭菜", "芹菜", "养鸡", "养猪", "养牛", "养羊")

def retrieve(ents, q, topk=TOP_ENTITIES):
    """混合检索：领域门 → 词法+语义排序 → RRF 融合 → 相关性门槛过滤"""
    in_q = next((c for c in CROPS if c in q), None)
    out_q = next((c for c in OUT_CROPS if c in q), None)
    if _MKT_RE.search(q):
        return [], -2.0          # 市场行情/补贴政策题：固定拒答（-2 专用文案），不给模型发挥机会
    if out_q and not in_q:
        return [], -1.0          # 主问范围外作物：不检索，让模型走拒答
    # 枚举题直通道：按作物+防治环节选实体，名字过滤只留真病名/真虫名，精编优先、
    # 短名去重（产区变体只留一个）。伪高分通过门槛过滤，rag_prompt 识别同款正则走清单头
    if in_q and ENUM_PEST.search(q) and ENUM_ASK.search(q):
        word = "病虫害" if "病虫害" in q else ENUM_PEST.search(q).group(1)
        scored = []
        for n, e in ents.items():
            if e["crop"] != in_q or e["stage"] not in ENUM_STAGES[word]:
                continue
            s = short_name(n)
            core = s[2:] if s.startswith(in_q) else s
            scored.append((0 if _curated(e) else 1, s, n, core))
        scored.sort(key=lambda x: (x[0], 1 if _ENUM_UMB.search(x[3]) else 0, x[1]))
                              # 精编优先 → 具体病名优先于伞名 → 名字序（'大斑病'排在
                              # '南方锈病与小斑病'这类伞实体前面，常见清单才像常见清单）
        cands, seen = [], set()
        for cu, s, n, core in scored:
            if s in seen or not _enum_ok(word, core, in_q):
                continue
            seen.add(s)
            cands.append(n)
            if len(cands) >= 10:
                break
        if cands:
            return [(1.0, n, ents[n], 9.0, 0.9) for n in cands], 0.9
    if _BROAD_RE.search(q):               # 宽泛管理题：多给实体，供中性主题头下综合作答
        topk = max(topk, 6)
    lex = _lexical(ents, q, noun=_residue_noun(q, in_q))[:15]   # 名词点名加成（空秆→'空秆原因'键）
    dense, sims_top = [], 0.0
    if ENC is not None:
        sims = (ENC.encode([q])[0] @ ENT_MAT.T).tolist()
        order = sorted(range(len(ENT_NAMES)), key=lambda i: -sims[i])
        sims_top = sims[order[0]]
        dense = [(sims[i], ENT_NAMES[i], ents[ENT_NAMES[i]]) for i in order[:15]]
    pool = {}
    for lst in (lex, dense):
        for r, (s, name, e) in enumerate(lst):
            p = pool.setdefault(name, {"rrf": 0.0, "e": e, "lex": 0.0, "sim": 0.0})
            p["rrf"] += 1.0 / (60 + r + 1)
            p["lex"] = max(p["lex"], s if (s, name, e) in lex else 0.0)
            p["sim"] = max(p["sim"], s if (s, name, e) in dense else 0.0)
    hits = [(v["rrf"], n, v["e"], v["lex"], v["sim"]) for n, v in pool.items()
            if v["sim"] >= DENSE_MIN or v["lex"] >= LEX_MIN]
    hits.sort(key=lambda x: -x[0])
    # 精编先验（融合层）：语义通道把近义抽取实体顶到第一时拉回——加成 0.02 只在
    # RRF 分差 <2 个名次内起效，全新话题（精编没覆盖）不受影响
    hits.sort(key=lambda h: -(h[0] + (0.02 if _curated(h[2]) else 0)))
    # 词面强命中（lex≥6）被语义融合压到后面、且比第一名词面分低档差 >1.0 → 提到第一：
    # 问"拌种"被"种子质量标准"语义带偏、把"种子药剂处理"挤出 topk 的修正
    strong = max((h for h in hits if h[3] >= 6.0), key=lambda h: h[3], default=None)
    if strong and strong is not hits[0] and hits[0][3] < strong[3] - 1.0:
        hits.remove(strong)
        hits.insert(0, strong)
    noun = _residue_noun(q, in_q)            # 名词点名（'空秆'）：事实键含该名词的实体直接提前
    if noun and len(noun) >= 2 and hits:     # （'空秆原因'键在化控防倒与授粉实体里，只认键不认值）
        for i, h in enumerate(hits):
            if any(noun in f[0] for f in h[2]["facts"]):
                if i:
                    hits.insert(0, hits.pop(i))
                break
    psyns = {s for s in _op_syns(q) if s != "管理"}   # 问操作：排第一的实体不管这操作、后面
    if psyns and not _BROAD_RE.search(q) and hits:    # 的实体管 → 换上来（RRF 只看名次，语义通道
        def _does_op(h):                              # 可能把'水肥一体化'顶到追肥题第一）。宽泛
            return (any(s in h[1] for s in psyns)     # 管理题豁免：中性头不锚实体，别锚到苗期实体
                    or any(any(s in f[0] for f in h[2]["facts"]) for s in psyns))
        if not _does_op(hits[0]):
            for i, h in enumerate(hits):
                if i and _does_op(h):
                    hits.insert(0, hits.pop(i))
                    break
    if in_q:                                   # 问题点名作物时剔除其他作物的实体（防跨作物串味）
        same = [h for h in hits if h[2]["crop"] in (in_q, None, "")]
        hits = same or hits[:2]
    if in_q and _DATE_ASK.search(q) and not any(r in q for r in REGIONS) \
       and not STAGE_HINT.search(q):
        sow_ask, har_ask = bool(re.search(r"播|种|育", q)), "收" in q
        for i, h in enumerate(hits):          # 不带产区的农时问法（'玉米什么时候播种'）：
            nm = h[1]                         # 日历/月历实体提到第一——产区专属播种实体
            if ("日历" in nm or "月历" in nm) and \
               ((sow_ask and re.search(r"播|种|育", nm)) or (har_ask and "收" in nm)):
                if i:                         # （东北春玉米播种）会抢走第一名，锚定头就只答
                    hits.insert(0, hits.pop(i))   # 一个产区；问播只提播期日历、问收只提收获日历
                break
    return hits[:topk], sims_top

HEAD_SYM = "从描述看，很可能是"    # 症状描述题预写的结论开头（模型续写病名，防答非所问）

def _names_entity(q, hits):
    """问题里是否点名了某命中实体的病/虫名（点名→不是纯症状描述，注入全量实体）"""
    for _, n, *_ in hits:
        s = short_name(n)
        body = s[2:] if s[:2] in CROPS else s
        if len(body) >= 2 and body in q:
            return True
    return False

_Q_STRIP = re.compile(r"[怎么如何咋怎样什么啥多少几时候呢吗办呀哦啊？?，。、！!\s]"
                      r"|怎么回事|咋回事|回事|防治|防|治|用药")
_OP_WORDS = ("浇水", "浇地", "灌溉", "灌水", "施肥", "追肥", "上肥", "除草", "打药", "拌种",
             "包衣", "浸种", "管理", "种好", "收获", "储藏", "晾晒", "控倒", "化控", "育种",
             "育秧", "插秧", "移栽", "烤田", "晒田")

def _op_syns(q):
    """问题里的操作词及其同义术语（'追肥'→{'追肥','施肥'}）：词法加成与实体提升共用"""
    ops = {w for w in _OP_WORDS if w in q}
    syns = set(ops)
    for w in ops:
        syns.update(Q_EXPAND.get(w, "").split())
    return syns

def _residue_noun(q, crop):
    """问题剥掉问法词/作物/生育期词/通用操作词后剩下的具体名词（'烂秧''蓟马''秃尖'），
    没有名词=泛操作问法（浇水/施肥/大喇叭口期怎么管），有名词=点名了具体对象。
    生育期词必须剥：'大喇叭口期怎么追肥'的名词若含'大喇叭口期'，会给干旱实体的
    '大喇叭口期干旱指标'键乱加分，把追肥实体从第一名挤掉"""
    r = _Q_STRIP.sub("", q)
    for w in _OP_WORDS + CROPS + STAGE_TOKENS + (crop or "",) + ("期",):
        r = r.replace(w, "")
    return r

def _object_match(q, hits):
    """问题问的对象图谱里到底有没有：点名实体 / 症状线索落到首实体 / 词面硬重叠（lex≥4）/
    语义强匹配（sim≥0.70）且①纯操作问法（浇水/施肥）或②问的具体名词真出现在首实体事实原文里
    （'秃尖'在'化控防倒与授粉'的键里）。皆不满足 = 图谱没有该对象（玉米蓟马、水稻烂秧：
    泡田整地语义相似 0.75 但事实里没提'烂秧'）→ 不硬灌（防'问A答B'），走模型自身知识作答。"""
    if _names_entity(q, hits):
        return True
    if _DATE_ASK.search(q) and ("日历" in hits[0][1] or "月历" in hits[0][1]):
        return True      # 农时问法命中日历实体：日历就是答这个的（键是产区名，词面/名词规则
                         # 全对不上，不豁免会被误判'图谱没有'踢去纯模型记忆作答）
    hints = [tgt for alts, tgts in SYM_HINTS.items()
             if any(a in q for a in alts.split("|")) for tgt in tgts]
    if hints and any(t in hits[0][1] for t in hints):
        return True
    if hits[0][3] >= 4.0:
        return True
    noun = _residue_noun(q, hits[0][2]["crop"])
    if noun and any(noun in f[0] for f in hits[0][2]["facts"]):
        return True                          # 问的名词真写在首实体的事实键里（'秃尖'→'秃尖原因'）
    if hits[0][4] >= 0.70:
        return not noun                      # 泛操作问法（'什么时候浇水'）语义即对象；
    return False                             # 点名名词但事实没提=图谱没有（烂秧→泡田 0.75 也不放）

def short_name(name):
    """实体名去掉产区前缀：'东北/黄淮海玉米大斑病' → '玉米大斑病'"""
    for c in CROPS:
        i = name.find(c)
        if i >= 0:
            return name[i:]
    return name

DRUG_REL = re.compile(r"药|防|治|施|喷|指标")
DRUG0_RE = re.compile(r"药|拌种|包衣|浸种|除草|喷雾|封闭")   # 键里真有药剂类内容才预写"防治用药："头
_BROAD_RE = re.compile(r"(怎么|如何|咋|怎样).{0,6}(管理|管|种好)|田间管理")  # 宽泛管理题：不锚单一实体
                                                          # （"预防措施"只有"防"不算——倒伏实体没药，
                                                          #  预写药头会逼模型编"无高效防治药剂"）
_DATE_ASK = re.compile(r"几号|几月|什么时候|啥时候|何时|日期|月份|上旬|中旬|下旬")   # 问日期/农时
_DATE_KEY = re.compile(r"播期|播时间|适期|收获期|日期|定植期")   # 日期类键：问农时的事实排序用
                                                              # （'播期播量'实体的日期键常排在
                                                              #  '播种量/播种深度'后面被配额切掉）
_DESC_RE = re.compile(r"判断|序列|标志|特征|识别|症状|类型|原因|天数|时间")
                                   # 描述性键（'大喇叭口判断''生育期序列'）：管理题里不是操作要点，
                                   # 但'序列'的值把各期名都列一遍、bigram 重叠反而最高 → 必须垫底

_TOPIC = (("水", re.compile(r"水|灌|溉|旱|涝")),
          ("肥", re.compile(r"肥|氮|磷|钾|尿素|微肥")),
          ("倒", re.compile(r"倒|扶|扎|化控|控倒")),
          ("病虫", re.compile(r"病|虫|药|螟|虱|蛾|锈|霉|腐|斑")),
          ("草", re.compile(r"草")),
          ("温", re.compile(r"温|热|冻|寒|冷")))

def _topic(k):
    """事实键的主题分类（水/肥/倒/病虫/草/温）——宽泛管理题预填用，同主题限 2 条，
    '干旱指标+灌水量+需水临界期'三条全是水，会把穗肥挤出预填"""
    for name, rex in _TOPIC:
        if rex.search(k):
            return name
    return "其他"

def _stage_hit(text, st0):
    """文本里对某生育期的提及是否算数：'大喇叭口期前可用矮壮素控高'、'拔节前追施尿素'
    是上一期的操作，只在字面上带了期名——这类'期前'提及不算该期事实（过滤与预填池共用）"""
    for m in re.finditer(re.escape(st0), text):
        tail = text[m.end():m.end() + 2]
        if not tail.startswith("前") and not tail.startswith("期前"):
            return True
    return False
STAGE_HINT = re.compile(           # 问题点名生育期 → 与该生育期相关的事实排最前
    r"喇叭口|拔节|抽雄|抽穗|孕穗|齐穗|扬花|灌浆|分蘖|越冬|返青|起身|三叶|出苗|乳熟|蜡熟|成熟|苗期|穗期|花期")
_DIAG_RE = re.compile(r"怎么回事|咋回事|什么病|啥病|什么问题|这是什么|是不是.{0,6}病")
                                   # 问诊题特征：问"是啥病/咋回事"才配"从描述看很可能是X"头；
                                   # "怎么拌种"是操作题，直接锚实体+药头，不说"可能是"

def _wants_drug(q):
    """问诊/防治类问题：问药、问治、问'是什么病/什么问题'、拌种包衣浸种、或有症状描述 → 药剂类事实优先注入"""
    if re.search(r"药|治|防|咋办|怎么办|回事|用什么|什么病|啥病|什么问题|拌种|包衣|种子处理", q):
        return True
    return any(a in q for alts in SYM_HINTS for a in alts.split("|"))

def _rank_facts(facts, q, crop=None):
    """事实排序（build_context 注入与 rag_prompt 预填共用，保证两边顺序一致）。
    名词点名的键（'空秆怎么回事'→'空秆原因'）最前；药剂题再分三级：真药剂内容键
    （药/拌种/包衣/浸种）→ 泛防治键（'综合治理'类空话）→ 其余。
    贪心解码答完第一单元就停——第一条必须是真药方/具体操作，不能是宣传语。"""
    noun = _residue_noun(q, crop)
    if _wants_drug(q):
        return sorted(facts, key=lambda f: 0 if noun and len(noun) >= 2 and noun in f[0] else
                             1 if DRUG0_RE.search(f[0]) else
                             2 if DRUG_REL.search(f[0]) else 3)
    return list(facts)

def _stage_sorted(facts, st0):
    """生育期事实排序（build_context 注入与 rag_prompt 预填共用同一顺序）：
    键含该期的动作键 → 值提该期的动作键 → 一般动作键 → 键含该期的描述键 →
    值提该期的描述键（'生育期序列'各期名都列、重叠反而高，必须垫底）→ 其余"""
    return sorted(facts, key=lambda f: (0 if st0 in f[0] and not _DESC_RE.search(f[0]) else
                                        1 if not _DESC_RE.search(f[0]) and st0 in f[1] else
                                        2 if not _DESC_RE.search(f[0]) else
                                        3 if st0 in f[0] else
                                        4 if st0 in f[1] else 5))

def _q_rel_sort(facts, q):
    """通用事实排序（问题无生育期/药剂/农时线索时）：按问题（含术语扩展）与事实键的
    bigram 重叠排——问'怎么存才不会霉'时'储藏安全含水量''霉变防控'排前、'运输要求'
    垫底。稳定排序，同分保持建图顺序。受控采样下模型自由枚举可能早停漏掉核心事实
    （仓储题漏'安全含水量≤14%'），注入与预填都改用这个顺序就不再赌模型挑哪条"""
    extra = " ".join(t for k, t in Q_EXPAND.items() if k in q)
    gq = bigrams(q + " " + extra)
    return sorted(facts, key=lambda f: -len(gq & bigrams(f[0])))

def build_context(hits, q=""):
    """检索结果拼成参考资料：每行自带病名（防模型认不出通用键属于谁），头条多给事实。
    问诊/防治类问题把'防治用药/防治指标'类事实排前——否则只注入识别类事实，模型无药可答"""
    want_drug = _wants_drug(q)
    stage_t = next((t for t in STAGE_TOKENS if t in q), "")
    st0 = stage_t[:-1] if stage_t.endswith("期") else stage_t
    noun = _residue_noun(q, hits[0][2]["crop"]) if hits else ""
    noun = noun if len(noun) >= 2 else ""   # '空秆怎么回事'→'空秆原因'键最先注入
    lines, seen_fact = [], set()       # 跨实体去重：同一事实只保留排序更高实体下的那份
    for rank, (rrf, name, e, lex, sim) in enumerate(hits):
        short = short_name(name)
        facts = list(e["facts"])
        if want_drug:                      # 稳定四级排序（同 _rank_facts）：名词点名键最前（'空秆
            facts.sort(key=lambda f: 0 if noun and noun in f[0] else      # 原因'别被 quota 切掉），
                             1 if DRUG0_RE.search(f[0]) else   # 再真药剂键；泛防治空话别占第一——
                             2 if DRUG_REL.search(f[0]) else 3)  # 贪心解码答完第一条就停
        elif st0:                          # 问具体生育期 → 各实体内该期动作事实优先（穗肥先于
            if _BROAD_RE.search(q):        # 苗肥）；宽泛管理题更狠：只留该期相关事实，描述键
                facts = [f for f in facts  # （判断/序列）与下期内容（'拔节前追施尿素'——
                         if _stage_hit(f[0], st0) or _stage_hit(f[1], st0)]  # 期前提及不算）全剔除
            facts = _stage_sorted(facts, st0)
        elif _DATE_ASK.search(q):         # 问农时：各实体内日期键（播期/适期/收获期）排最前——
            facts.sort(key=lambda f: 0 if _DATE_KEY.search(f[0]) else 1)  # 否则会被'播种深度'这类
                                                                           # 排在实体前面的键挤掉配额
        else:                             # 无期无药无农时：按问题相关度排（仓储题'储藏安全
            facts = _q_rel_sort(facts, q)  # 含水量'排到'运输要求'前），注入与预填头同序
        took, quota = 0, (len(facts) if rank == 0 and ("日历" in name or "月历" in name) else
                          TOP_FACTS + 3 if len(hits) == 1 else          # 单实体（症状诊断）：放宽到 6 条
                          TOP_FACTS + 1 if rank == 0 else TOP_FACTS - 1)
        if rank and st0 and _BROAD_RE.search(q):
            quota = 1                      # 宽泛+生育期：副实体只留最相关 1 条，穗肥能进上下文
        block = [f"◆ {short}" + (f"（{e['crop']}·{e['stage']}）" if e["crop"] else "")]
        for rel, tail, g, c, src in facts:
            if took >= quota:
                break
            took += 1                     # 每扫一条占一个名额（含跳过的重复——保持旧切片口径，
            key = (rel, tail)             #   不然多实体上下文变长会扰动贪心解码）
            if key in seen_fact:          # 别的实体已注入过（如多病共用"合理密植增钾"）→ 跳过
                continue
            seen_fact.add(key)
            cond = "；".join(x for x in (g, c) if x)
            block.append(f"  · {short}｜{rel}：{tail}" + (f"（{cond}）" if cond else ""))
        if len(block) > 1 or rank == 0:    # 生育期过滤后一条不剩的副实体：整块丢弃——
            lines.extend(block)            # 留个光杆◆头只会诱导模型扯无关实体
    return "\n".join(lines)

# 越界问题的固定拒答（领域门拦截后直接返回，不让基座知识有发挥机会）
REFUSAL = ("这个问题超出了我的知识范围——我主要掌握小麦、水稻、玉米三大作物全生育期的栽培技术"
           "（品种选择、播期播量、施肥灌溉、病虫草害防治、收获储藏等）。"
           "这方面的问题建议咨询当地农技部门，或拨打12316三农服务热线。")
# 市场/政策类问题的固定拒答（ retrieve 返回 -2 触发）：没有实时行情数据，不猜
REFUSAL_MKT = ("粮价行情、补贴政策这类市场和政策问题不属于种植技术范围，我也没有实时行情数据，"
               "没法给您靠谱的答案。建议关注当地粮食和物资储备局、农业农村部门发布的信息，"
               "或拨打12316三农服务热线咨询。")
_MKT_RE = re.compile(r"价格|行情|粮价|收购价|市场行情|卖粮|补贴|政策")

# ================= 模型生成 =================
SYS = ("你是农业种植技术专家，只精通小麦、水稻、玉米三大作物的栽培技术。回答要求："
       "①先给结论（如'从描述看很可能是大斑病'或直接答案），再分点给出具体参数和操作要点，不要照抄罗列参考资料原文；"
       "②参考资料中有相关内容时，数值必须采用资料中的，并注意区分适用条件不同的数值"
       "（如机械收获适宜含水量与储藏安全含水量是两回事，储藏标准以 GB 1351小麦/GB 1353玉米 为准）；"
       "资料没覆盖到的部分，用你自己掌握的可靠农业技术知识正常补充作答，不要一句'资料未提及'就结束；"
       "只有药剂用量等具体数值资料里没有时不要凭空编造，可讲通用原则；"
       "③资料为空且问题超出小麦、水稻、玉米范围（其他作物、养殖、政策、价格等）时，"
       "如实说明这超出你的知识范围，建议咨询当地农技部门或12316热线，不要编造具体参数。")
SYS_PLAIN = ("你是农业种植技术专家，熟悉小麦、水稻、玉米的栽培技术。"
             "回答第一句先给结论、直接回答问题，再分点给出具体参数和操作要点；"
             "不要用'具体如下''根据资料'之类开头；分点不超过6条，每条一个主题，语句要通顺连贯；"
             "拿不准的具体数值不要凭空编造，可给通用原则并建议咨询当地农技部门。"
             "政策、补贴、市场行情、价格预测等非技术问题如实说明这不属于种植技术范围，"
             "建议咨询当地农业农村部门；其他作物（大豆、棉花、蔬菜等）只讲通用原则并说明你的专长是三大粮作物。")
IM_S, IM_E = "<|im_start|>", "<|im_end|>"

def hist_text(hist):
    """多轮对话记录块：没有就空串（首轮单问完全不受影响）"""
    if not hist:
        return ""
    lines = [f"{'用户' if m.get('role') == 'user' else '助手'}：{m.get('content', '')}"
             for m in hist if m.get("content")]
    return ("\n【对话记录（最新在最后；'它/这个/那'等指代以对话为准，只回答当前问题）】\n"
            + "\n".join(lines)) if lines else ""

def rag_text(ctx, q, head="", hist=None):
    """head：预写在 assistant 开头的文字（症状诊断题='结论句+防治用药：'，模型只能续写内容）"""
    return (f"{IM_S}system\n{SYS}{IM_E}\n{IM_S}user\n"
            f"【参考资料（检索自知识图谱，◆后是实体名，事实行'实体｜关系：内容'）】\n{ctx}\n\n"
            f"【要求】①第一句必须是结论：症状描述题说'从描述看很可能是X病'，其他题先用一句话直接回答问题的第一问；"
            f"禁止用'具体如下''根据参考资料'之类开头。"
            f"②围绕问题作答：问时机先答时机、问药先答药，以第一个◆实体为主；"
            f"别的实体的参数只能用于它自己，不得安到主实体上，无关的不写。"
            f"③问题里问了几个点就必须答几个点（如既问播期又问播量，两问都要给出数值），"
            f"分点最多6条，每条一个主题，同一要点不重复。"
            f"④数值一律取自资料并注明适用条件；资料里有药剂名称和用量的必须写出。"
            f"⑤参考资料只覆盖了问题的一部分时，未覆盖的部分用你自身掌握的可靠农业技术知识补充作答，"
            f"不要以'资料未提及'结束；只有拿不准的具体数值（药剂用量、时期等）不凭空编造，可给通用原则。"
            f"{hist_text(hist)}\n\n【问题】{q}{IM_E}\n{IM_S}assistant\n{head}")

def rag_prompt(hits, q, cls_q=None, hist=None):
    """检索→提示词。纯症状描述题（问诊但没点名病/虫名）只注入第 1 实体，且预写
    '结论句+防治用药：'开头——4B 模型会挑错病、漏药剂，检索第一名+预写能两头锁死。
    cls_q：分类/排序用的问题文本（多轮时=上一问+当前问，追问才检索得准）；q 只用于展示。
    返回 (提示词, 是否带资料, 预写开头文本)"""
    c = cls_q or q
    # 对象一致性闸门：图谱没有问题所问的对象 → 不灌图谱、不预写结论，走模型自身知识
    # （否则"玉米蓟马怎么防治"会被硬灌成大斑病——问A答B，前后说的不是一个问题）
    if hits and not _object_match(c, hits):
        return chat_text(q, hist), False, ""
    # 枚举清单题（"小麦有什么病害？"）：retrieve 已按作物+防治环节选出实体，
    # 这里预写实体名清单（照抄图谱，零幻觉），模型顺着展开各对象要点
    if ENUM_PEST.search(c) and ENUM_ASK.search(c) and hits:
        word = "病虫害" if "病虫害" in c else ENUM_PEST.search(c).group(1)
        crop_w = next((cp for cp in CROPS if cp in c), "") or (hits[0][2]["crop"] or "")
        head = f"{crop_w}常见{word}主要有：{'、'.join(short_name(n) for _, n, *_ in hits)}。"
        return rag_text(build_context(hits, c), q, head, hist), True, head
    # 追问（"换成水稻纹枯病呢？"）句面不含"药/治"——去掉实体名和语气词后近乎空句的追问，
    # 继承最近两问的问药意图；"那玉米怎么追肥"这种换了话题的追问不继承
    users = [m["content"] for m in (hist or []) if m.get("role") == "user"]
    residue = q
    for _, n, *_ in hits:
        residue = residue.replace(short_name(n), "")
    for crop in CROPS:
        residue = residue.replace(crop, "")
    residue = re.sub(r"[呢那换它这什么呢？?，,。、！!]", "", residue)
    followup = bool(users) and len(residue) <= 2
    want_drug = _wants_drug(c) or (followup and any(_wants_drug(u) for u in users[-2:]))
    diag = _DIAG_RE.search(c) or any(a in c for alts in SYM_HINTS for a in alts.split("|"))
    sym = (want_drug and diag and not _names_entity(c, hits)
           and hits[0][2]["stage"] in ("病害防治", "虫害防治", "草害防治"))
    # 症状描述+问诊断 → 预写"很可能是X"——但首实体得真是病/虫/草害（"空秆怎么回事"命中
    # 田间管理实体，说"很可能是化控防倒与授粉"不像诊断结论 → 走实体锚定头+名词键优先）
    cal = bool(hits) and bool(_DATE_ASK.search(q)) \
          and ("日历" in hits[0][1] or "月历" in hits[0][1]) \
          and not any(r in q for r in REGIONS)
    ctx = build_context(hits[:1] if (sym or cal) else hits, c)
          # 无产区农时问法只注入日历实体：多实体上下文会让模型答完日历后把产区
          # 播种实体的播量/播深/行距也拖进答案（问'什么时候'却续答操作参数，离题），
          # 还会连上下文里的（生育期；条件）注记一起复述成（播种期；麦茬直播）尾巴
    if not ctx:
        return chat_text(q, hist), False, ""
    drug0 = any(DRUG0_RE.search(f[0]) for f in hits[0][2]["facts"])
    if sym:                                  # 症状描述题：预写结论句，有药剂事实再预写完整药方
        head = f"{HEAD_SYM}{short_name(hits[0][1])}。"
        _DRUG_VAL = re.compile(r"药剂|用药|乳油|悬浮剂|可湿性粉|水分散粒|颗粒剂|水剂|倍液")
        drug_f = next((f for f in _rank_facts(hits[0][2]["facts"], c, hits[0][2]["crop"])
                       if DRUG0_RE.search(f[0])), None) \
            or next((f for f in hits[0][2]["facts"] if _DRUG_VAL.search(f[1])), None)
                   # 键不带'药'字但值里真是药方（'配套要点：药剂同大斑病（苯醚甲环唑…）'）
        if drug_f:                           # 药方全值预写、不留裸'防治用药：'——受控采样下
            if "药" in drug_f[0]:            # 裸键名会被 EOS 击中，答案就停在结论句（开头即停）。
                head += f"{drug_f[0]}：{drug_f[1]}"     # 键名本身含'药'（防治用药/使用药剂）直接用
            else:
                head += f"防治用药：{drug_f[0]}：{drug_f[1]}"
    elif want_drug and drug0:                # 点名病害/操作问'咋治/打多少/怎么拌种'：药剂就是答案，
        head = f"{short_name(hits[0][1])}。防治用药："   # 预写药头 + 排序后前三条完整药方（数值照抄
        drugs, dk = [], set()                # 图谱，杜绝模型抄串邻居药方的量）。同名键只预写
        for f in _rank_facts(hits[0][2]["facts"], c,   # 一条——合并实体里'施药方法'常有两份近义
                             hits[0][2]["crop"]):      # 事实，连写两个同名键读着像复读。全部预写
            if DRUG0_RE.search(f[0]) and f[0] not in dk:   # 完整值、不留悬空键名——受控采样下
                dk.add(f[0])                # 悬空键可能被 EOS 击中，答案会以光杆"键名："收尾
                drugs.append(f)
                if len(drugs) >= 3:
                    break
        if drugs:
            head += "；".join(f"{d[0]}：{d[1]}" for d in drugs)
        if _DATE_ASK.search(q):                     # 双问'什么时候打药+打多少'：药方预写后时机
            tm = next((f for f in hits[0][2]["facts"]   # 也要预写——时机键（防治适期/施药时间）
                       if re.search(r"时机|适期|时间|时期|窗口", f[0])), None)  # 不含'药'字，
            if tm:                                  # 进不了 DRUG0 药方；受控采样下模型自由
                head += f"；{tm[0]}：{tm[1]}"       # 枚举常早停在药剂段，第二问就漏答了
    elif _BROAD_RE.search(c) and not _names_entity(c, hits):
        st = next((t for t in STAGE_TOKENS if t in c), "")    # 宽泛管理题（"大喇叭口期怎么管理"）：
        if st and not st.endswith("期"):                       # 不锚单一实体（否则被排第一的
            st += "期"                                         # 干旱/倒伏实体钉死），中性主题头
        crop_w = next((cp for cp in CROPS if cp in c), "") or (hits[0][2]["crop"] or "")
        head = f"{crop_w}{st}管理要点："
        if st:                                                 # 再预写跨实体的该期要点前两条
            st0 = st[:-1] if st.endswith("期") else st         # （穗肥在追肥实体、关键水在灌溉实体），
            pool, taken = [], set()                            # 把模型按进枚举中间——贪心答完一句
            def _pool_ok(f):                                   # "是需水临界期"这种结论腔就收工的兜底
                if f[0] in (st, st0):        # 键就是生育期名（'返青期：日均温回升…'）＝定义不是要点
                    return False
                kg = bigrams(f[0])
                for p in pool:               # 键面近重复（'大喇叭口后倒伏'与'…倒伏处理'）只取一条，
                    pg = bigrams(p[0])       # 腾出名额给别的主题（关键期灌水量）——要点覆盖面优先
                    if kg and pg and len(kg & pg) / min(len(kg), len(pg)) >= 0.6:
                        return False
                return True
            for _, _, e, *_ in hits:
                for f in e["facts"]:
                    if (_stage_hit(f[0], st0) or _stage_hit(f[1], st0)) \
                       and not _DESC_RE.search(f[0]) \
                       and (f[0], f[1]) not in taken \
                       and _pool_ok(f) \
                       and not any(dup_fact(f[1], p[1], 0.35) or f[0] == p[0] for p in pool):
                        taken.add((f[0], f[1]))                # 跨实体同义事实（两条'返青镇压'、
                        pool.append(f)                         # 两条'大喇叭口灌水'）只取一条。
                                                               # 描述键（序列/判断/标志）值里把各期名
                                                               # 列一遍、_stage_hit 全过，但管理题里
                                                               # 从不是操作要点——直接不进池
            pool.sort(key=lambda f: (0 if st0 in f[0] and not _DESC_RE.search(f[0]) else
                                     1 if not _DESC_RE.search(f[0]) else 2))  # 键里点名该期的
            sel, cnt = [], {}                                 # 专门动作最前，描述键垫底；同主题限 2
            for f in pool:                                    # 条——第 4 要点留给肥/倒等别的主题
                t = _topic(f[0])                              # （穗肥在追肥实体、跨实体组合成答案）
                if cnt.get(t, 0) >= 2:
                    continue
                cnt[t] = cnt.get(t, 0) + 1
                sel.append(f)
                if len(sel) >= 4:
                    break
            if len(sel) >= 2:                                 # （指标/倒伏/灌水/穗肥）。选中要点全部
                head += "；".join(f"{f[0]}：{f[1]}" for f in sel)  # 预写完整值、不留悬空键名——受控采样
                                                               # 下悬空键可能被 EOS 击中，答案会以光杆
                                                               # "键名："收尾；全值预填后模型停了也已是
                                                               # 完整答案，续写只是加分
    else:
        head = f"{short_name(hits[0][1])}："   # 锚定主实体——正好是微调训练的答案格式，
        st0 = next((t for t in STAGE_TOKENS if t in c), "")
        st0 = st0[:-1] if st0.endswith("期") else st0
        if st0:                                # 问某期的具体操作（'大喇叭口期怎么追肥'）：
            top = [f for f in _stage_sorted(hits[0][2]["facts"], st0)    # 该期动作事实预填前两条
                   if _stage_hit(f[0], st0) or _stage_hit(f[1], st0)]    # 全值——不预填时模型会裸吐
        elif want_drug:                        # '拔节期（7—8展开叶）…'这种不带键名的值
            top = _rank_facts(hits[0][2]["facts"], c, hits[0][2]["crop"])
        elif _DATE_ASK.search(c):              # 问农时：日期键（播期/适期）最前，与 build_context
            top = [f for f in sorted(hits[0][2]["facts"],   # 注入同序——问'什么时候'预填的第一条
                          key=lambda f: 0 if _DATE_KEY.search(f[0]) else 1)  # 必须是日期，不能是
                     if f[0] != "判断基准"] or list(hits[0][2]["facts"])  # '播种量'（词面重叠更高）
        else:                                  # 无期无药：问题相关度排序的事实（'怎么存才不会
            top = _q_rel_sort(hits[0][2]["facts"], c)    # 霉'→安全含水量+霉变防控最前）
        if len(top) >= 2:                      # 预填前两条完整值：受控采样下锚定头+自由枚举
            noun = _residue_noun(c, hits[0][2]["crop"])   # 可能早停，漏掉核心事实（仓储题漏
            if len(noun) >= 2 and any(noun in f[0] for f in top):      # '安全含水量≤14%'）。
                head += f"{top[0][0]}：{top[0][1]}"        # 但问题点名了具体对象（'空秆怎么回事'）
            else:                                # 时该对象的键就是全部答案，只预填这一条——
                head += f"{top[0][0]}：{top[0][1]}；{top[1][0]}：{top[1][1]}"  # 多塞邻居键（化控防倒）反而离题
        elif top:
            head += f"{top[0][0]}：{top[0][1]}"
    # 日历/月历类实体或带播期键的播种实体 + 问题真在问时间：日期单元预写完整值——
    # "什么时候播+播多少"双问预填即两问都答全；锚定头已写过的单元不再重复。
    # 不带产区的日历问法（'玉米什么时候播种'）各产区日期全写：只写前两条会漏产区，
    # 答案就变成只答抢到第一的那个产区；判断基准压轴（地温/积温总原则收束）。
    # 键候选跳过"判断基准"（Cypher 返回顺序不保证，不能信 facts[0]）
    has_sow_date = any(re.search(r"播期|播时间|适期", f[0]) for f in hits[0][2]["facts"])
    if ("日历" in head or "月历" in head or has_sow_date) and not want_drug \
       and _DATE_ASK.search(q):
        def _unit(f):
            return f"{f[0]}：{f[1]}"
        in_head = sum(_unit(f) in head for f in hits[0][2]["facts"])
        cands = [f for f in hits[0][2]["facts"]
                 if f[0] != "判断基准" and _unit(f) not in head] \
                or list(hits[0][2]["facts"])
        if cands and (cal or in_head < 2):
            add = cands[:5 if cal else 2 - in_head]
            if cal:
                jb = next((f for f in hits[0][2]["facts"] if f[0] == "判断基准"), None)
                if jb:
                    add.append(jb)
            head += ("" if head.endswith("：") else "；") + "；".join(_unit(f) for f in add)
    if cal and not head.endswith("。"):
        head += "。"      # 日历头收尾句号：答案已完整，给模型停机信号、别再续写下文
    return rag_text(ctx, q, head, hist), True, head  # 模型会顺着枚举该实体的事实（双问自然答全）

def chat_text(q, hist=None):
    """域内但检索无命中 → 用微调模型的原生格式（v4 行为直接激活）"""
    return f"{IM_S}system\n{SYS_PLAIN}{IM_E}\n{IM_S}user\n{hist_text(hist)}\n【问题】{q}{IM_E}\n{IM_S}assistant\n"

def main():
    test_only = "--test" in sys.argv
    print("连接知识图谱 …")
    ents = load_graph()
    print(f"图谱加载: {len(ents)} 个实体、{sum(len(e['facts']) for e in ents.values())} 条事实")
    init_dense(ents)

    samples = ["俺家玉米叶子上一条一条的黄斑，咋回事？",
               "玉米大斑病怎么防治？用药量多少？",
               "水稻浸种消毒用什么药？",
               "黄淮海小麦什么时候播种？亩基本苗多少？",
               "水稻稻飞虱咋治？",
               "玉米收完怎么存才不会霉？",
               "小麦赤霉病什么时候打药？打多少？",
               "种棉花怎么选品种？"]
    if test_only:
        for q in samples:
            hits, top_sim = retrieve(ents, q)
            print("\n" + "=" * 60 + f"\n问: {q}")
            if hits:
                print("命中实体: " + "、".join(
                    f"{n}(rrf {s:.3f}｜词法 {lx:g}｜语义 {sm:.2f})"
                    for s, n, _, lx, sm in hits))
            else:
                why = ("问到范围外作物（棉花/花生等），领域门拦截"
                       if top_sim < 0 else
                       (f"最高语义相似度 {top_sim:.2f} < 门槛 {DENSE_MIN}" if ENC else "词法未命中"))
                print(f"无命中（{why}）→ 不注入资料，交给模型自身处理（拒答/追问）")
        return

    import torch                       # 懒加载：server.py 复用本文件检索逻辑时不必装 torch
    assert torch.cuda.is_available(), "请在 GPU 实例上运行"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    if os.path.exists(MERGED):
        print("[加载] 合并后的完整模型")
        try:
            model = AutoModelForCausalLM.from_pretrained(MERGED, **kw)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(MERGED, dtype=torch.bfloat16,
                                                         device_map="cuda", trust_remote_code=True)
    elif os.path.exists(ADAPTER):
        print("[加载] 基座模型 + LoRA 适配器")
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                         device_map="cuda", trust_remote_code=True)
        model = PeftModel.from_pretrained(model, ADAPTER)
    else:
        print("[警告] 未找到微调产物，用原版基座回答！")
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                         device_map="cuda", trust_remote_code=True)
    model.eval()
    eos = tok.convert_tokens_to_ids(IM_E)

    print("\nGraphRAG 问答已启动（输入 q 退出）\n" + "-" * 50)
    while True:
        try:
            q = input("问: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("q", "quit", "exit", "退出"):
            break
        hits, top_sim = retrieve(ents, q)
        if top_sim < 0:                      # 领域门拦截 → 固定拒答，不给基座知识发挥机会
            print("答: " + (REFUSAL_MKT if top_sim < -1.5 else REFUSAL) + "\n")
            continue
        ctx = build_context(hits)
        prompt = rag_text(ctx, q) if ctx else chat_text(q)
        print(f"[检索] {'、'.join(n for _, n, *_ in hits) or '无命中（域内问题，纯模型作答）'}")
        ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=500, do_sample=False,
                               pad_token_id=tok.pad_token_id or eos, eos_token_id=eos)
        print(f"答: " + tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip() + "\n")

if __name__ == "__main__":
    main()
