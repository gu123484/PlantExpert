# -*- coding: utf-8 -*-
"""回答质量回归评测：16 问 × 关键词断言。
改 rag.py / server.py / 提示词之后必跑——改 A 坏 B 一眼现形。
用法：python eval.py [URL]     （URL 缺省 = 线上 AutoDL 实例）
断言：must 每项都要命中；"A|B" 表示任一命中即过。
"""
import io, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ask_web import ask                      # 复用同一个 SSE 客户端

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":   # Windows GBK 控制台
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

URL = sys.argv[1] if len(sys.argv) > 1 else ""
if URL:
    import ask_web; ask_web.URL = URL

# (问题, 必须命中项)——断言词取自图谱事实原文（模型从注入资料照抄，稳定可断言）
CASES = [
    # --- 症状诊断：结论句 + 药剂 ---
    ("玉米叶子一条一条的黄斑，咋回事？",          ["大斑病", "多菌灵|百菌清|嘧菌酯"]),
    ("小麦叶片上有一层白粉一样的东西，是啥病？咋治？", ["白粉病", "三唑酮|腈菌唑|戊唑醇"]),
    ("水稻叶子发黄慢慢枯死，心叶一拔就断，是什么问题？", ["条纹叶枯|二化螟"]),
    # --- 点名问药：药剂名 + 用量单位 ---
    ("小麦赤霉病什么时候打药？打多少？",           ["扬花", "戊唑醇|氰烯菌酯", "毫升/亩"]),
    ("水稻稻飞虱咋治？",                          ["吡蚜酮|烯啶虫胺|呋虫胺", "克/亩|毫升/亩"]),
    ("玉米大斑病怎么防治？用药量多少？",            ["多菌灵|百菌清|嘧菌酯", "倍液|毫升/亩"]),
    ("小麦条锈病打什么药？",                      ["三唑酮|戊唑醇|烯唑醇"]),
    ("水稻纹枯病咋治？",                          ["井冈霉素|噻呋酰胺|苯甲·丙环唑"]),
    ("水稻二化螟怎么防治？",                      ["氯虫苯甲酰胺|阿维|甲氧虫酰肼"]),
    ("水稻浸种消毒用什么药？",                     ["咪鲜胺|强氯精"]),
    # --- 参数双问 / 管理类 ---
    ("黄淮海小麦什么时候播种？亩基本苗多少？",       ["10月"]),
    ("玉米大喇叭口期怎么追肥？",                   ["尿素", "公斤"]),
    ("玉米倒伏了怎么办？",                         ["扶|捆扎|恢复"]),
    ("玉米收完怎么存才不会霉？",                    ["14", "含水量|储藏"]),
    # --- 越界拒答 ---
    ("种棉花怎么选品种？",                         ["超出.*知识范围|知识范围"]),
    ("养猪怎么防疫？",                             ["超出.*知识范围|知识范围"]),
]

def check(text, must):
    for pat in must:
        if not re.search(pat, text):
            return pat
    return None

def main():
    print(f"回答质量回归评测：{len(CASES)} 问\n" + "=" * 62)
    npass, failed = 0, []
    for i, (q, must) in enumerate(CASES, 1):
        t0 = time.time()
        try:
            meta, text = ask(q)
        except Exception as ex:
            meta, text = {}, f"[请求异常：{ex}]"
        dt = time.time() - t0
        bad = check(text, must)
        if bad is None:
            npass += 1
            print(f"✅ {i:2d}. {q[:22]}…（{dt:.0f}s｜{meta.get('mode','-')}）")
        else:
            failed.append((q, bad, text))
            print(f"❌ {i:2d}. {q[:22]}…（{dt:.0f}s）未命中断言：{bad}")
    print("=" * 62)
    print(f"结果：{npass}/{len(CASES)} 通过")
    for q, bad, text in failed:
        print(f"\n--- 失败：{q}\n    断言：{bad}\n    回答：{text[:300]}")
    sys.exit(0 if not failed else 1)

if __name__ == "__main__":
    main()
