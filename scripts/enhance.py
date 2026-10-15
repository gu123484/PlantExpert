# -*- coding: utf-8 -*-
"""数据集增强器 v2：补齐体检发现的短板
  A 玉米逆境应对 QA（新知识块 C-STR-001~003）
  B 时间+用量组合型（combo 型原仅 23 条，重点补充）
  C 口语化问法改写（增强真实用户问法泛化，答案重新组装）
  D 知识边界拒答/引导（防幻觉）
  E 农药安全急救
  F 多轮对话（sharegpt 格式，独立文件 multiturn_300.json）
去重：与现有 10947 条 + 新增互相，精确 + 3gram 近重复(0.72)
输出：qa_pairs/train_v2.json、val_v2.json、multiturn_300.json
"""
import json, os, random, re, sys, hashlib
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qa_generator import (load_blocks, ngrams, norm, fact_question, fact_answer,
                          compose_answer, gen_block_qas, CROP_CODE, STAGE_CODE,
                          _obj_phrase, _clean_subject)

KG = r"D:\桌面\桌面总\wew\agri_dataset"
rng = random.Random(20261001)
CODE = dict(STAGE_CODE); CODE["知识边界"] = "BND"

def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

train = jload(os.path.join(KG, "qa_pairs", "train_10000.json"))
val = jload(os.path.join(KG, "qa_pairs", "val_500.json"))
blocks = load_blocks()
BMAP = {b["id"]: b for b in blocks}

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

def add(q, a, qt, meta):
    if not gate(q, qt):
        return False
    meta = dict(meta); meta["问题类型"] = qt
    cc = CODE.get(meta["作物"], "X"); sc = CODE.get(meta["环节"], "XXX")
    new_items.append({"id": f"QA2-{cc}-{sc}-{len(new_items)+1:05d}",
                      "instruction": q, "input": "", "output": a,
                      "metadata": meta})
    return True

# ---------------- A 玉米逆境应对 ----------------
def section_corn_stress():
    n0 = len(new_items)
    for b in [x for x in blocks if x.get("id", "").startswith("C-STR")]:
        for q, a, qt, fk in gen_block_qas(b, rng):
            add(q, a, qt, md(b, qt, fk))
        for (k, v) in b["facts"]:
            for qi, qt in enumerate(["direct", "howmuch", "why", "scenario",
                                     "criterion", "detail"]):
                if qt == "howmuch" and not re.search(r"\d", v):
                    continue
                q = fact_question(b, k, v, qi + 3, qt, rng)
                a = fact_answer(b, k, v, qt, rng)
                add(q, a, qt, md(b, qt, k))
    return len(new_items) - n0

# ---------------- B 组合型补充 ----------------
TIME_HINT = ("期", "时", "天", "时间", "阶段")
AMT_HINT = ("量", "多少", "公斤", "毫升", "厘米", "株", "万", "立方米", "克", "升", "倍液")

def section_combo():
    n0 = len(new_items)
    for b in blocks:
        tf = [(k, v) for k, v in b["facts"] if any(h in k for h in TIME_HINT)]
        af = [(k, v) for k, v in b["facts"] if any(h in k for h in AMT_HINT)]
        if not tf or not af:
            continue
        rctx = b.get("region", "")
        rctx = rctx if rctx and rctx not in ("通用", "区域对比", "分区域") else ""
        obj = _obj_phrase(b) or _clean_subject(rctx, b["crop"], b["topic"])
        pairs = [(tf[0], af[len(af) // 2]), (tf[len(tf) // 2], af[-1])]
        for t, a in pairs:
            if t[0] == a[0]:
                continue
            others = [v for k, v in b["facts"] if k not in (t[0], a[0])]
            rng.shuffle(others)
            ans = f"{t[0]}：{t[1]}。{a[0]}：{a[1]}。"
            for o in others[:2]:
                ans += f"配套要点：{o}。"
            for tpl in (f"{obj}这一季从时间到用量给捋一下：{t[0]}和{a[0]}分别怎么定？",
                        f"种{b['crop']}的话，{t[0]}与{a[0]}的搭配安排是怎样的？"):
                add(tpl, ans, "combo", md(b, "combo", t[0]))
    return len(new_items) - n0

# ---------------- C 口语化改写 ----------------
SYN = [("什么时候", "啥时候"), ("怎么", "咋"), ("如何", "咋"),
       ("请给具体数字", "给个准数"), ("请展开说明具体细节", "说细点"), ("方面", "这块")]
FRONT = ["", "请问哈，", "想问一下，", "俺们村里问：", "合作社的师傅问：",
         "新手第一次种地，", "帮家里老人问：", "请教一下，"]
BACK = ["", "在线等", "谢谢啦", "说详细点哈", "给个准话", "急用"]

def section_colloquial(target=600):
    pool = [q for q in train
            if q["metadata"].get("事实键") and q["metadata"]["知识块"] in BMAP
            and q["metadata"]["问题类型"] in ("direct", "howmuch", "criterion", "detail")]
    rng.shuffle(pool)
    kept = 0
    for it in pool:
        if kept >= target:
            break
        m = it["metadata"]
        q2 = it["instruction"]
        for a, b in SYN:
            q2 = q2.replace(a, b)
        q2 = rng.choice(FRONT) + q2 + rng.choice(BACK)
        blk = BMAP[m["知识块"]]
        ans = compose_answer(blk, focus_keys=[m["事实键"]], qtype="direct", rng=rng)
        meta = dict(m); meta["问题类型"] = "colloquial"
        if add(q2, ans, "colloquial", meta):
            kept += 1
    return kept

# ---------------- D 知识边界拒答 ----------------
OUT_CROPS = ["大豆", "棉花", "花生", "油菜", "马铃薯", "甘薯", "高粱", "谷子", "荞麦",
             "绿豆", "烟草", "茶叶", "苹果", "柑橘", "葡萄", "大棚蔬菜", "番茄", "黄瓜",
             "辣椒", "食用菌", "中药材", "苜蓿"]
LIVE = ["生猪", "蛋鸡", "肉牛", "山羊", "鱼塘养鱼", "南美白对虾", "蜜蜂"]
OTHER = ["最近的天气", "今年的粮食补贴政策", "现在的小麦价格行情", "农机购置补贴",
         "农村宅基地审批", "无人机飞行驾照", "哪个牌子的化肥好", "种子商标纠纷", "征地补偿标准"]

REF_ANS = [
    "抱歉，这个问题超出了我目前掌握的范围。我专注的是小麦、水稻、玉米三种粮食作物的种植技术，涵盖选种、播种、施肥、灌溉、病虫害防治、收获储运等环节。关于{sub}的问题，建议咨询当地农技推广站，或拨打12316三农服务热线。您有小麦、水稻、玉米方面的技术问题，我可以给出具体的参数和操作方案。",
    "您好，{sub}不在我当前的知识覆盖内。我的知识库围绕小麦、水稻、玉米的种植技术构建（从育种选种到收获储运全流程）。这类问题更可靠的做法是：1）咨询当地农业农村局或农技推广站；2）参考该领域的官方技术手册。换成小麦、水稻、玉米的问题，我可以直接给您具体数字。",
    "这个问题我目前答不好——{sub}属于我知识范围之外的内容。我能提供的是小麦、水稻、玉米三作物全生育期的量化技术方案（播期、播量、施肥、灌溉、用药等）。为避免给您不准确的建议，{sub}的问题请以当地农技部门的意见为准。",
]
PER_QA = [
    ("我家这块地今年种小麦还是玉米好？",
     "这个决定需要结合您地块的具体条件，建议从这几方面判断：1）前茬作物与地力（高产田宜小麦—玉米两熟，瘦地可优先养地作物）；2）灌溉条件（玉米关键期需水多，无水浇条件需慎重）；3）当地积温与种植习惯；4）上年的病虫害发生情况。最稳妥的做法是带着地块信息（乡镇、前茬、地力、灌溉条件）咨询当地农技站，做针对性安排。如果您告诉我大概的区域和条件，我也可以给出小麦或玉米各自的关键技术参数供您比较。"),
    ("帮我看看我地里到底得的是什么病？",
     "仅凭口头描述还不能下结论。麻烦补充这些信息：1）作物和生育期（如小麦拔节期、玉米大喇叭口期）；2）症状部位（叶片/茎秆/穗部/根部）和形态（斑点、锈粉、霉层、腐烂、萎蔫）；3）发生范围（零星还是成片，从下部叶还是上部叶开始）；4）最近天气和管理（降雨、施肥、打过什么药）。有照片更好。补充后我可以按症状对照给出可能的病害和对应防治方案。"),
    ("我们村的具体施肥方案帮我定一下",
     "施肥方案要依据测土结果来定才准确。建议：1）先取土送检（县级农技中心一般提供测土配方施肥服务），测有机质、碱解氮、有效磷、速效钾；2）拿到结果后按目标产量折算施肥量。在此之前，可参考我知识库中的分区通用方案（如黄淮海小麦亩施纯氮14—16公斤、五氧化二磷6—8公斤、氧化钾4—6公斤），但具体到村到田，以测土配方结果和当地农技部门指导为准。"),
    ("我这块地的土壤检测报告怎么看、怎么用？",
     "看土壤检测报告重点抓四项：1）有机质（低于10克/公斤说明地力偏低，需增施有机肥）；2）碱解氮/有效磷/速效钾（对应氮磷钾施肥量，可按目标产量法折算，一般每100公斤小麦籽粒约需纯氮3公斤左右，再减去土壤供给量）；3）pH值（小于5.5需施用石灰调理，大于8.5注意锌铁硼有效性）；4）微量元素（缺锌地块玉米对锌敏感，可基施硫酸锌1—2公斤/亩）。拿不准时带着报告去县农技中心，他们可以直接给出施肥配方。"),
]

def section_refusal():
    n0 = len(new_items)
    meta0 = {"作物": "通用", "环节": "知识边界", "主题": "拒答与引导", "地区": "",
             "生育期": "", "条件": "", "知识块": "BOUNDARY-001", "事实键": "",
             "依据": "知识库范围声明：仅覆盖小麦/水稻/玉米种植技术；引导渠道参考12316全国三农服务热线"}
    ai = 0
    for c in OUT_CROPS:
        for qt in (f"{c}怎么种才能高产？", f"{c}的病虫害怎么防治？",
                   f"我想种{c}，选什么品种、一亩用多少种子？"):
            add(qt, REF_ANS[ai % 3].format(sub=c), "refusal", dict(meta0)); ai += 1
    for c in LIVE:
        for qt in (f"养{c}要注意什么？", f"{c}的饲料怎么配？"):
            add(qt, REF_ANS[ai % 3].format(sub=c), "refusal", dict(meta0)); ai += 1
    for o in OTHER:
        add(f"{o}，您了解吗？", REF_ANSS(ai, o), "refusal", dict(meta0)); ai += 1
    for qt, ans in PER_QA:
        add(qt, ans, "refusal", dict(meta0))
    return len(new_items) - n0

def REF_ANSS(i, sub):
    return REF_ANS[i % 3].format(sub=sub)

# ---------------- E 农药安全急救 ----------------
SAFETY = [
    ("打农药时不小心喝进嘴里怎么办？",
     "这是急症，请立即处理：1）马上拨打120，并带上该农药的标签或说明书就医；2）不要自行催吐（强碱强腐蚀性农药催吐会造成二次灼伤，标签注明可催吐的除外）；3）清醒者可饮少量温水；4）记录误服时间、农药名称和大概量告诉医生。常见有机磷中毒表现为头晕、恶心、多汗、瞳孔缩小，救治需用阿托品等特效解毒药，必须由医院处置。"),
    ("农药溅到皮肤上怎么处理？",
     "立即脱去污染衣物，用大量流动清水冲洗接触部位15分钟以上，眼角、指缝等部位都要冲到；冲洗后如出现红肿、刺痛或水疱，携农药标签就医。注意：敌百虫遇碱毒性增强，不可用肥皂水冲洗；多数有机磷农药可用肥皂水清洗，拿不准时先用大量清水。"),
    ("农药进了眼睛怎么急救？",
     "立即用流动清水或生理盐水持续冲洗眼睛至少15分钟，冲洗时拉开眼睑、转动眼球，让结膜穹隆充分冲洗；冲洗后无论是否缓解都应携农药标签到眼科检查，角膜化学性损伤早期可能症状轻但后果重。"),
    ("打药时间长了头晕恶心怎么办？",
     "这是农药（尤其有机磷类）中毒的典型前兆。立即：1）离开施药现场到空气新鲜处，脱去防护衣物；2）用肥皂和清水清洗暴露皮肤；3）症状不缓解或加重（呕吐、视物模糊、肌肉颤抖）立即就医并告知医生农药名称；4）当天不再接触农药。预防上，连续施药不超过4小时，中午高温时段不打药。"),
    ("家里剩下的农药怎么存放？",
     "1）原包装密封保存，不可转入饮料瓶、食品容器（误服事故多因此发生）；2）放在儿童和畜禽接触不到的阴凉、避光、通风处，与粮食、饲料、种子分开；3）保留原标签，破损的写明名称和含量；4）粉剂和水分开存放防潮结块。依据农药标签管理规定和农药管理条例的贮存要求执行。"),
    ("用完的农药瓶、包装袋怎么处理？",
     "农药包装废弃物不可乱丢或焚烧。依据《农药包装废弃物回收处理管理办法》：1）清洗3次（清洗液倒入喷雾器再用）后压扁；2）交到村里的农药包装废弃物回收点或农药店的回收桶；3）各地回收有补贴政策可咨询。随意丢弃会造成田面源污染，也污染灌溉水。"),
    ("中午高温的时候能打药吗？",
     "不建议。原因：1）中午气温高，药液蒸发快，人体吸入和经皮中毒风险成倍增加；2）高温下作物气孔张开，药液易产生药害；3）部分农药（如部分杀虫剂）高温下分解失效或毒性增强。最佳施药时段是上午9—11时露水干后、下午16时以后，风速3级以下。按GB/T 8321农药合理使用准则，还应做好防护。"),
    ("刚打完药的粮食蔬菜多久能收？",
     "要看农药的安全间隔期——最后一次施药到采收的最少天数，印在农药标签上（如某吡虫啉制剂在小麦上14天、某三环唑在水稻上21天）。收获前严禁施药；不知道间隔期时查标签或问农技部门，切不可估摸着收。超间隔期采收的残留一般可控制在限量内，这是GB 2763食品中农药最大残留限量的基础。"),
    ("小孩误食了拌过药的种子怎么办？",
     "立即拨打120，同时：1）保留种子包装/药剂标签带给医生；2）不要盲目催吐，听从急救人员指挥（昏迷、抽搐时催吐有窒息风险）；3）记下误食大概时间和数量。拌种剂（如含克百威、甲拌磷的高毒成分）对儿童极危险，事后务必把拌药种子存放在儿童拿不到的地方，播剩的不可再作粮食或饲料。"),
    ("怎么判断买的农药是不是假药？",
     "1）查标签：农药登记证号（可在'中国农药信息网'查真伪）、生产许可证号、执行标准号三证齐全；2）看外观：乳油应透明无沉淀，粉剂不结块，悬浮剂摇匀后均匀；3）扫码：新包装多有追溯二维码；4）渠道：到有经营许可证的门店购买并索要票据。怀疑假劣药可向当地农业综合执法大队举报。"),
]

def section_safety():
    n0 = len(new_items)
    meta0 = {"作物": "通用", "环节": "诊断急救", "主题": "农药安全急救", "地区": "",
             "生育期": "", "条件": "", "知识块": "SAFETY-001", "事实键": "",
             "依据": "GB/T 8321农药合理使用准则；农药标签管理规定；GB 2763食品中农药最大残留限量"}
    for qt, ans in SAFETY:
        add(qt, ans, "safety", dict(meta0))
    return len(new_items) - n0

# ---------------- F 多轮对话 ----------------
def build_multiturn(n=300):
    out = []
    for i, b in enumerate(rng.sample(blocks, min(n, len(blocks)))):
        if len(b["facts"]) < 3:
            continue
        rctx = b.get("region", "")
        rctx = rctx if rctx and rctx not in ("通用", "区域对比", "分区域") else ""
        obj = _obj_phrase(b) or f"{rctx}{b['crop']}{b['topic']}"
        f0, f1, f2 = b["facts"][0], b["facts"][1], b["facts"][2]
        k0 = f0[0]
        q1 = (f"{obj}，{k0}是多少？" if any(h in k0 for h in ("量", "数", "率", "指标", "含量", "浓度"))
              else f"{obj}，{k0}怎么把握？")
        conv = [
            {"from": "human", "value": q1},
            {"from": "gpt", "value": f"{f0[0]}：{f0[1]}。"},
            {"from": "human", "value": f"那{f1[0]}呢？"},
            {"from": "gpt", "value": f"{f1[0]}：{f1[1]}。"},
            {"from": "human", "value": "干的时候还有啥要特别注意的？"},
            {"from": "gpt", "value": f"{f2[0]}：{f2[1]}。以上参数为推荐范围，具体请结合地力与当年气候，以当地农技部门指导为准。"},
        ]
        out.append({"id": f"MT-{i:04d}", "conversations": conv,
                    "metadata": {"作物": b["crop"], "环节": b["stage"], "主题": b["topic"],
                                 "地区": b.get("region", ""), "生育期": b.get("growth", ""),
                                 "条件": b.get("condition", ""), "问题类型": "multiturn",
                                 "知识块": b["id"], "事实键": "", "依据": b["source"]}})
    return out

def main():
    a = section_corn_stress()
    b2 = section_combo()
    c = section_colloquial()
    d = section_refusal()
    e = section_safety()
    print(f"A玉米逆境 {a}｜B组合型 {b2}｜C口语化 {c}｜D拒答 {d}｜E安全 {e}｜合计新增 {len(new_items)}")
    print("去重剔除:", dict(rejected) or "无")

    # 每类抽 5% 进验证集
    grp = defaultdict(list)
    for it in new_items:
        grp[it["metadata"]["问题类型"]].append(it)
    val_add = []
    for k, lst in grp.items():
        take = max(1, round(len(lst) * 0.05))
        val_add += lst[:take]
    rest = [it for it in new_items if it not in val_add]

    train_v2 = train + rest
    val_v2 = val + val_add
    with open(os.path.join(KG, "qa_pairs", "train_v2.json"), "w", encoding="utf-8") as f:
        json.dump(train_v2, f, ensure_ascii=False, indent=1)
    with open(os.path.join(KG, "qa_pairs", "val_v2.json"), "w", encoding="utf-8") as f:
        json.dump(val_v2, f, ensure_ascii=False, indent=1)

    mt = build_multiturn()
    with open(os.path.join(KG, "qa_pairs", "multiturn_300.json"), "w", encoding="utf-8") as f:
        json.dump(mt, f, ensure_ascii=False, indent=1)

    print(f"train_v2: {len(train_v2)} 条（原 {len(train)} + 新增 {len(rest)}）")
    print(f"val_v2:   {len(val_v2)} 条（原 {len(val)} + 新增 {len(val_add)}）")
    print(f"multiturn: {len(mt)} 组（sharegpt 格式，可选混入）")
    # 新增类型统计
    print("新增问题类型:", dict(Counter(it["metadata"]["问题类型"] for it in new_items)))

if __name__ == "__main__":
    main()
