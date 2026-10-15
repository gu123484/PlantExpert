#!/bin/bash
cd /root/autodl-tmp/crop_cultivation
while ! grep -q ALL_EXTRACTION_DONE extract/full.log 2>/dev/null; do sleep 60; done
P=/root/miniconda3/envs/rag/bin/python
$P extract_knowledge.py --src qa/natesc_qa.json --out extract/natesc_full.jsonl >> extract/full.log 2>&1
echo WEB_EXTRACTION_DONE >> extract/full.log
