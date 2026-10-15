# -*- coding: utf-8 -*-
"""微调模型交互问答：python chat.py
在终端里输入问题回车即问，输入 q 退出。
优先用合并后的完整模型，没有就自动加载 基座+LoRA适配器。
"""
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE   = "/root/autodl-tmp/crop_cultivation"
MODEL  = os.path.join(BASE, "Qwen3_4b")
ADAPTER = os.path.join(BASE, "output", "qwen3_4b_agri_lora", "adapter")
MERGED = os.path.join(BASE, "output", "qwen3_4b_agri_merged")

SYS = "你是农业种植技术专家，熟悉小麦、水稻、玉米的栽培技术，回答要给出具体参数和操作要点。"
IM_S, IM_E = "<|im_start|>", "<|im_end|>"

def chat_text(q):
    return (f"{IM_S}system\n{SYS}{IM_E}\n{IM_S}user\n{q}{IM_E}\n{IM_S}assistant\n")

def main():
    assert torch.cuda.is_available(), "请在 GPU 实例上运行"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kw = dict(torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
    except TypeError:
        kw["dtype"] = kw.pop("torch_dtype")
        model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)

    if os.path.exists(MERGED):
        del model
        print("[加载] 合并后的完整模型")
        try:
            model = AutoModelForCausalLM.from_pretrained(MERGED, **kw)
        except TypeError:
            kw2 = dict(dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(MERGED, **kw2)
    elif os.path.exists(ADAPTER):
        print("[加载] 基座模型 + LoRA 适配器")
        model = PeftModel.from_pretrained(model, ADAPTER)
    else:
        print("[警告] 未找到微调产物，将用原版基座模型回答！")
    model.eval()

    eos = tok.convert_tokens_to_ids(IM_E)
    print("\n可以开始提问了（输入 q 退出）\n" + "-" * 50)
    while True:
        try:
            q = input("问: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("q", "quit", "exit", "退出"):
            break
        ids = tok(chat_text(q), add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=400, do_sample=False,
                               pad_token_id=tok.pad_token_id or eos, eos_token_id=eos)
        ans = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"答: {ans}\n")

if __name__ == "__main__":
    main()
