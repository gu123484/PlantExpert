# -*- coding: utf-8 -*-
"""向部署好的网站提问（走公网，SSE 流式收全量）
用法：python ask_web.py "问题" [URL]     （URL 默认线上 AutoDL 实例）
"""
import json, sys, urllib.request

URL = sys.argv[2] if len(sys.argv) > 2 else \
    "https://u824853-9181-bfa494ba.bjb2.seetacloud.com:8443"

def ask(q, timeout=300):
    req = urllib.request.Request(URL + "/ask",
        data=json.dumps({"question": q}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    meta, text = {}, ""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r.read().decode("utf-8").split("\n"):
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                if ev["type"] == "meta":   meta = ev
                elif ev["type"] == "delta": text += ev.get("t", "")
    return meta, text

if __name__ == "__main__":
    meta, text = ask(sys.argv[1])
    print(f"[模式 {meta.get('mode')}] 命中：{'、'.join(e['name'] for e in meta.get('entities', []))}")
    print(text)
