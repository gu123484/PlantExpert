# 农业知识库与微调数据集（小麦/水稻/玉米）

为 LoRA 微调 Qwen2.5 农业智能问答模型构建：**知识库资料 → 三元组（知识图谱底料）→ 微调问答对**。
质检结论见 `质检报告.md`。GraphRAG 问答（微调模型 + Neo4j 图谱检索）：`python rag.py`（AutoDL），`python rag.py --test` 仅测检索（无 GPU 本机可跑）。

## 目录结构（实际产物）
```
agri_dataset/
├── README.md                 # 本说明
├── 质检报告.md               # 全量质检结果
├── samples/                  # 最初送审的10条样例 + 三元组格式预览
├── knowledge_base/           # 知识库（合计约 75 万字符 ≥ 10 万要求）
│   ├── 00_知识库总览.md
│   ├── 小麦/                 # 9 个环节文件（育种选种→区域特点）
│   ├── 水稻/                 # 9 个环节文件
│   ├── 玉米/                 # 9 个环节文件
│   ├── 综述/                 # 4 篇长文：三作物栽培通论 + 跨作物总论
│   └── 应用问答库/           # 41 个文件，按作物×环节精选问答（含依据）
├── triples/
│   ├── triples_all.json      # 1110 条三元组（含属性：地区/生育期/条件/依据）
│   └── triples_all.csv       # CSV（UTF-8 BOM，可直接 LOAD CSV 进 Neo4j）
├── qa_pairs/
│   ├── train_v4.json         # ★推荐训练用：11670 条（v3 + 看症辨病106，治症状幻觉）
│   ├── val_v4.json           # 570 条验证
│   ├── train_v3.json         # 11571 条（v2 + 一题多问335 + 全流程9，治漏答）
│   ├── val_v3.json           # 563 条验证
│   ├── train_v2.json         # 11245 条（v1 + 玉米逆境/口语化/拒答/安全）
│   ├── val_v2.json           # 545 条
│   ├── multiturn_300.json    # 167 组多轮对话（sharegpt 格式，可选混入）
│   ├── train_with_grain.json # 20051 条（v2 + grain 外部语料合并版，可选）
│   ├── train_10000.json      # v1 原版 10447 条
│   └── val_500.json          # v1 原版 500 条
├── run_finetune.py           # LoRA 一键微调（AutoDL 上跑）
├── chat.py                   # 纯模型交互测试（不带检索）
├── rag.py                    # GraphRAG 问答：微调模型 + Neo4j 图谱检索
├── server.py                 # ★网站后端：对话 + 图谱可视化 API（FastAPI）
├── static/                   # ★网站前端：index.html + echarts.min.js（本地自带图表库）
├── selftest.py               # ★部署自检（9 项体检，start.sh 会自动调用）
├── graph_snapshot.json       # ★图谱本地快照（Aura 连不上时自动兜底）
├── deploy_gpu/               # ★一键部署包·GPU 版（传 AutoDL，bash start.sh）
├── deploy_cpu/               # ★一键部署包·CPU 版（先转 GGUF，bash start.sh）
├── 部署说明.md               # ★模型上服务器 + 本地访问（GPU/CPU 两条路径）
├── kg_graph.html             # 全量图谱静态导出（pyvis，可独立嵌入网页）
└── scripts/
    ├── data/                 # 19 个结构化知识数据文件（167 块 / 943 事实，含玉米逆境）
    ├── templates.py          # 问法模板库
    ├── qa_generator.py       # 问答生成器（去重+目标驱动）
    ├── enhance.py            # v2 增强：玉米逆境/组合型/口语化/拒答/安全/多轮
    ├── enhance_v3.py         # v3 增强：一题多问/识别+用药/全流程/结尾去模板
    ├── enhance_v4.py         # v4 增强：看症辨病/易混淆鉴别/模糊症状追问
    ├── merge_grain.py        # grain 外部语料规范化合并
    ├── build_triples.py      # 三元组 + 知识库渲染
    ├── build_kg.py           # 三元组导入 Neo4j Aura（读根目录 Neo4j-*.txt 凭据）
    ├── build_qa_kb.py        # 应用问答库渲染 + 字数统计
    ├── qc.py                 # 全量质检（可重复运行，默认查 v2）
    └── qc_deep.py            # 深度体检（模板痕迹/数值合理性/依据多样性）
```

## 问答对格式
Alpaca 格式（LLaMA-Factory 可直接使用），训练时剔除 `id` 与 `metadata`：
```json
{
  "id": "QA-W-SOW-00001",
  "instruction": "黄淮海小麦播期播量方面，什么时候播种？",
  "input": "",
  "output": "……10月5日—15日……亩基本苗18万—22万……",
  "metadata": {"作物": "小麦", "环节": "播种", "主题": "播期播量", "地区": "黄淮海",
               "生育期": "播种期", "条件": "", "问题类型": "direct",
               "知识块": "W-SOW-001", "依据": "全国农技中心秋播技术指导意见；黄淮海麦区小麦栽培技术规程"}
}
```

## 三元组格式
```json
{"head": "华北小麦拔节水", "relation": "灌溉量", "tail": "40—50立方米/亩",
 "attributes": {"作物": "小麦", "地区": "华北", "生育期": "拔节初期", "依据": "…"}}
```

## 知识图谱（Neo4j Aura，已导入 2026-10-01）
`build_kg.py` 一键重建（清空重导，可重复执行），规模：**1110 条三元组 + 结构增强 → 1458 节点 / 2066 关系**。
- 基础层：`Entity`(208) / `Value`(1104)，动态关系类型（使用药剂/防治指标/播种量…），边带 生育期/条件/依据 属性，维度 `Crop`(4)/`Stage`(15)/`Region`(31)
- **结构增强**（从既有事实推导，不新增知识）：
  - `药剂`(51)：`防治用药`(带用量属性)↔`可防治` 双向边——"三唑酮能治什么"一查便知；
  - `生育期`(33)：三作物按农艺顺序 `下一阶段` 建链，实体 `发生时期` 挂载（97 条）；
  - `标准`(12)：GB/NY/T 编号节点，`依据标准` 溯源（40 条）；
  - `易混淆`(5 对)：大斑↔小斑、白粉↔条锈、稻瘟↔纹枯、二化螟↔条纹叶枯、麦蚜↔麦蜘蛛，边上带鉴别要点
- 可视化（Neo4j 官方）：console.neo4j.io → Instance02 → Query，例如
  `MATCH p=(h:Entity)-[r]->(t) WHERE h.name CONTAINS '条锈病' RETURN p LIMIT 50`
  `MATCH p=(d:药剂)<-[:防治用药]-(h)-[:属于作物]->(c) WHERE d.name CONTAINS '戊唑醇' RETURN p`（某药治哪些病）
  `MATCH p=(c:Crop)-[:包含生育期]->(n)-[:下一阶段*..11]->(m)<-[:发生时期]-(h) WHERE c.name='玉米' AND m.name='玉米大喇叭口期' RETURN p`（某生育期的农事）
- **自建网站展示**：`python agri_dataset/scripts/export_graph_html.py` → 生成 `kg_graph.html`（全量 1458 节点/2066 边的力导向图，按节点类型着色，带类型筛选与搜索定位；数据内嵌、不含数据库密码，可安全公开）。嵌入网页：`<iframe src="kg_graph.html" style="width:100%;height:900px"></iframe>`

## GraphRAG 问答（rag.py）
微调模型 + 图谱检索：提问 → 从 Neo4j 拉取的 208 实体/1110 事实中**混合检索** → 拼成【参考资料】注入提示词 → 微调模型生成。

检索三道工序：
1. **领域门**：主问范围外作物（棉花/花生/养殖等）直接不检索 → 模型走拒答，防止硬塞资料；
2. **双通道**：词法（实体名/事实键 bigram + 口语症状线索表 + 术语同义扩展）与语义（bge-small-zh 向量，把实体全文预编码成 208×512 矩阵）各自排序；
3. **RRF 融合 + 相关性门槛**：倒数排名融合两路结果，语义相似度 ≥0.60 或词法 ≥4 分才注入；无向量模型时自动退回纯词法模式。

```
pip install neo4j && python rag.py          # AutoDL 上交互问答（自动加载微调模型）
python rag.py --test                        # 只看检索命中，不加载模型（本机可跑）
```
向量模型：`modelscope download --model BAAI/bge-small-zh-v1.5 --local_dir agri_dataset/bge_small`（95MB，放 rag.py 同级即可）。
实测：口语症状"叶子上一条一条的黄斑"→ 大斑病/褐斑病/锈病三候选 + 语义通道额外召回"玉米苗期异常诊断"；"玉米收完怎么存才不会霉"→ 仓储管理（含安全含水量标准）；"种棉花怎么选品种"→ 领域门拦截不注入。检索层与 v4 诊断训练互补：图谱保数值可溯源，模型保口语理解。

## 网站部署（server.py，本地浏览器访问）
`python server.py [--port 8000]` 启动后浏览器打开即用——左侧**对话问答**（GraphRAG 流式输出、
检索命中实体 chips、超范围问题拒答卡片），右侧**知识图谱**（总览/按作物过滤/实体邻域搜索、
节点详情卡含事实与依据、可点击邻居继续下钻）。模型后端自动二选一：GPU 加载合并模型 /
CPU 加载 GGUF 量化版；图表库随 `static/` 本地提供，无外网 CDN 依赖。Neo4j 只在启动时
连一次（全量图谱进内存），断网可用本地快照 `graph_snapshot.json` 兜底（`python server.py --dump` 生成）。
详细上传清单与两条部署路径（AutoDL GPU / 无 GPU CPU 服务器）见 **`部署说明.md`**。

## 复现方式
```
python agri_dataset/scripts/qa_generator.py     # 生成 v1 问答对（含去重）
python agri_dataset/scripts/enhance.py          # 生成 v2 增强版（train_v2/val_v2/multiturn）
python agri_dataset/scripts/merge_grain.py      # 可选：合并 grain 语料 → train_with_grain
python agri_dataset/scripts/build_triples.py    # 生成三元组 + 知识库框架
python agri_dataset/scripts/build_kg.py         # 三元组导入 Neo4j Aura 建知识图谱
python agri_dataset/scripts/build_qa_kb.py      # 渲染应用问答库 + 字数统计
python agri_dataset/scripts/qc.py               # 全量质检（默认查 v2）
python agri_dataset/scripts/qc_deep.py          # 深度体检
```

## v2 相对 v1 的增强（深度体检后补齐）
1. **玉米逆境应对**：新增 3 个知识块（倒伏/干旱高温/涝渍雹灾，21 条量化事实）→ 101 条 QA，补齐唯一真实覆盖空洞；
2. **口语化改写 600 条**：方言/口语问法（"啥时候""咋弄""俺们村里问"）缓解模板问法与真实用户问法的差距，答案重新组装非复制；
3. **拒答与引导 91 条**：超出范围（其他作物/养殖/政策/价格）→ 说明边界并引导 12316/当地农技站，个人化问题 → 追问所需信息，防幻觉；
4. **农药安全急救 10 条**：误服/溅肤/入眼/高温施药/安全间隔期/包装回收等；
5. **组合型（时间+用量）补充 41 条**；
6. **多轮对话 167 组**（sharegpt 格式，LLaMA-Factory 可直接注册使用）；
7. 去重门与 v1 相同（精确 + 3gram 近重复 0.72），全量复检零重复、100% 依据、91.9% 含数值。

## v3 相对 v2 的增强（首轮微调实测反馈驱动）
微调实测暴露"一题两问漏答后半问"（combo 型仅 64 条太薄），v3 针对性改进：
1. **一题多问 335 条**（combo 总数 64→399）："问1？问2？"双问拼接式，答案以"①②"逐点回应每个子问——直接训练模型答全所有子问（如"什么时候播种？亩基本苗多少？"）；
2. **病虫"识别+用药"组合 32 条**：最常见的一题两问场景（怎么认？打什么药？）；
3. **跨环节全流程问答 9 条**：从种到收的时间用量串联（pipeline 型）；
4. **答案结尾去模板化**：v2 中 Top6 相同结尾占 33% → 改写/删除 2516 条，结尾分布均化（Top3 为 340/339/326）；
5. 复检全过：train 11571 / val 563、零重复、100% 依据。复现：`python agri_dataset/scripts/enhance_v3.py`。

## v4 相对 v3 的增强（二轮微调实测反馈驱动）
二轮实测暴露"口语症状描述→幻觉编病名"（问玉米条状黄斑，编出'女峰病'），根因是只有正向"病名→症状"、缺反向"症状→病名"：
1. **看症辨病 87 条**：口语症状（"俺家玉米叶子上一条一条的黄斑"）→"很可能是X病" + 判别要点 + 防治指标 + 药剂用量，41 个病虫对象全覆盖、症状措辞与真实用户描述对齐；
2. **易混淆鉴别 7 条**：大斑vs小斑、白粉vs条锈、二化螟枯心vs条纹叶枯假枯心、丝黑穗vs瘤黑粉等，练"对比判断"；
3. **模糊症状追问 7 条**：信息不足（"玉米叶子发黄是啥原因？"）→ 列可能原因+各自判别动作+反问补充信息，**不下单一结论**，直接对治幻觉；
4. 复检全过：train 11670 / val 570、零重复、100% 依据、数值率 92.0%。复现：`python agri_dataset/scripts/enhance_v4.py`。

## 依据与免责
知识依据 GB/NY/T 标准、费槐林《水稻优质高产栽培技术问答》、全国农技中心技术意见、
三作物栽培学教材及产区规程编写；数值为通用推荐范围，实际应用以当地农技部门指导为准，
生产决策前建议经当地专家复核。
