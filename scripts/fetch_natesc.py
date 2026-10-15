# -*- coding: utf-8 -*-
"""全国农技中心（natesc.agri.cn）农事指导全文抓取 → 问答语料。
================================================================
用法（本地，服务器被 WAF 拦）：python -X utf8 scripts/fetch_natesc.py
输入：extract/web/natesc_guides.json（GetNewsListGroupTopCache 列表筛选出的指导文档）
输出：extract/web/natesc_qa.json —— [{"问题","答案"}]，每答案 ≤550 字（抽取提示词截 600）

正文接口（页面 JS 渲染，但 API 直出 JSON）：
  列表：GET /Home/GetNewsListGroupTopCache         （730 篇元数据，含 NewsId/标题/分类）
  正文：POST /news/GetNewsEntity，表单只带 NewsId  （GET/带中文 NewsTitle 都拿不到正文）
"""
import html as _html
import json, os, re, time, urllib.request

KG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
GUIDES = os.path.join(KG, "extract", "web", "natesc_guides.json")
OUT = os.path.join(KG, "extract", "web", "natesc_qa.json")
CHUNK = 550          # 答案上限（extract_knowledge.py 的提示词截 600）


def get(url, data=None, retry=3):
    for i in range(retry):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA,
                              "X-Requested-With": "XMLHttpRequest"},
                data=data.encode() if data else None)
            with urllib.request.urlopen(req, timeout=25) as r:
                return r.read().decode("utf-8", errors="ignore")
        except Exception as e:
            if i == retry - 1:
                print(f"  [失败] {url} → {e}")
                return ""
            time.sleep(3)


def strip_html(html):
    """正文 HTML（API 里是转义过的）→ 纯文本段落列表（保序去空）"""
    html = _html.unescape(html)               # &lt;p&gt; → <p>
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", "", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</p>", "\n", html)
    txt = re.sub(r"<[^>]+>", "", html)
    txt = (txt.replace("&nbsp;", " ").replace("&ldquo;", "“").replace("&rdquo;", "”")
           .replace("&mdash;", "—").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
           .replace("&#39;", "'").replace("&quot;", '"'))
    lines = []
    for ln in txt.splitlines():
        ln = re.sub(r"\s+", " ", ln).strip()
        if len(ln) >= 8:                     # 丢弃残行（页眉导航等短碎片）
            lines.append(ln)
    return lines


def clean_title(t):
    """'全国农技中心关于印发《X技术指导意见》的通知' → 'X技术指导意见'"""
    m = re.search(r"《(.+?)》", t)
    if m:
        return m.group(1)
    t = re.sub(r"^全国农技中心关于印发?", "", t)
    t = re.sub(r"的通知$", "", t)
    return t or t


def main():
    guides = json.load(open(GUIDES, encoding="utf-8"))
    pairs, n_art = [], 0
    for g in guides:
        raw = get("https://natesc.agri.cn/news/GetNewsEntity",
                  data=f"NewsId={g['id']}")
        if not raw:
            continue
        try:
            art = json.loads(raw)
        except Exception:
            print(f"  [坏JSON] {g['title']}")
            continue
        content = art.get("NewsContent") or ""
        paras = strip_html(content)
        if len(paras) < 3:
            print(f"  [无正文] {g['title']}")
            continue
        n_art += 1
        title = clean_title(g["title"])
        body = [p for p in paras if not re.match(r"^[（(]?发布时间|来源[:：]|浏览[:：]", p)]
        # 分块：按段落累积 ≤550 字；数字标题行（一、二、 / 1. 2. / （一））起头优先开新块
        chunks, cur = [], []
        cur_len = 0
        for p in body:
            heading = bool(re.match(r"^[一二三四五六七八九十]{1,3}[、．.]|^\d{1,2}[、．.^]|^[（(][一二三四五六七八九十\d]{1,3}[)）]", p))
            if cur and (cur_len + len(p) > CHUNK or (heading and cur_len > 300)):
                chunks.append("".join(cur))
                cur, cur_len = [], 0
            if len(p) > CHUNK:                # 超长段硬切
                for i in range(0, len(p), CHUNK):
                    chunks.append(p[i:i + CHUNK])
                continue
            cur.append(p)
            cur_len += len(p)
        if cur:
            chunks.append("".join(cur))
        for i, c in enumerate(chunks, 1):
            q = title if len(chunks) == 1 else f"{title}（第{i}部分）"
            pairs.append({"问题": q, "答案": c, "_src": title, "_url":
                          f"https://natesc.agri.cn/News/des?id={g['id']}"})
        print(f"[OK] {title}：{len(body)} 段 → {len(chunks)} 块")
        time.sleep(1.5)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=1)
    print(f"\n共 {n_art} 篇 → {len(pairs)} 个问答块 → {OUT}")


if __name__ == "__main__":
    main()
