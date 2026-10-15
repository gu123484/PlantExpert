# -*- coding: utf-8 -*-
"""
Qwen3-4B 农业问答 LoRA 一键微调脚本（兼容 transformers 4.x / 5.x）
=====================================================
用法（AutoDL 上共 2 条命令）:
  1) pip install -U transformers peft accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
  2) python run_finetune.py

说明：不使用 apply_chat_template（transformers 5.x 行为有变），改为手动拼接
      Qwen 对话格式 + 经典 tokenizer 调用，任何版本结果一致。
"""
import csv, json, os, torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments, set_seed)
from peft import LoraConfig, get_peft_model

# ================= 路径（只改这里） =================
BASE   = "/root/autodl-tmp/crop_cultivation"
MODEL  = os.path.join(BASE, "Qwen3_4b")
QADIR  = os.path.join(BASE, "agri_dataset", "qa_pairs")
OUT    = os.path.join(BASE, "output", "qwen3_4b_agri_lora")
MERGED = os.path.join(BASE, "output", "qwen3_4b_agri_merged")
MAX_LEN = 1024

# ================= 超参 =================
EPOCHS, LR = 3.0, 5e-5
BS, GA = 4, 4   # RTX5090(32G)/A100 用 (4,4)；24G 卡改 (2,8)；16G 卡改 (1,16)

SYS = "你是农业种植技术专家，熟悉小麦、水稻、玉米的栽培技术，回答要给出具体参数和操作要点。"
IM_S, IM_E = "<|im_start|>", "<|im_end|>"

def chat_text(q, a=None):
    """Qwen 对话格式：system + user [+ assistant]"""
    t = (f"{IM_S}system\n{SYS}{IM_E}\n"
         f"{IM_S}user\n{q}{IM_E}\n"
         f"{IM_S}assistant\n")
    if a is not None:
        t = t + a + IM_E
    return t

def encode(text, tok):
    return tok(text, add_special_tokens=False)["input_ids"]


def pick_file(*names):
    """按优先级挑第一个'存在且有内容'的数据文件，过程全部打印"""
    for n in names:
        p = os.path.join(QADIR, n)
        if not os.path.exists(p):
            print(f"[数据] {n}: 不存在")
            continue
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, list) and len(d) > 0:
                print(f"[数据] {n}: {len(d)} 条  ✓ 使用")
                return p
            print(f"[数据] {n}: 文件存在但内容为空 → 上传不完整！")
        except Exception as e:
            print(f"[数据] {n}: 解析失败 {e}")
    return None


def build_examples(path, tok):
    """问答 → 带 loss 掩码样本（问题部分不算 loss）"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for it in data:
        q = it["instruction"] + (("\n" + it["input"]) if it.get("input") else "")
        a = it["output"]
        p_ids = encode(chat_text(q), tok)
        f_ids = encode(chat_text(q, a), tok)[:MAX_LEN]
        if len(f_ids) <= len(p_ids):      # 答案被截没，跳过并计数
            continue
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        out.append({"input_ids": f_ids, "labels": labels,
                    "attention_mask": [1] * len(f_ids)})
    return out


class ListDataset(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def quick_test(model, tok, questions):
    print("\n===== 微调效果自测 =====")
    model.eval()
    try: model.config.use_cache = True
    except Exception: pass
    eos = tok.convert_tokens_to_ids(IM_E)
    for q in questions:
        ids = tok(chat_text(q), add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=300, do_sample=False,
                               pad_token_id=tok.pad_token_id or eos, eos_token_id=eos)
        print(f"\n问: {q}\n答: " + tok.decode(o[0][ids.shape[1]:],
                                             skip_special_tokens=True).strip())


def main():
    assert torch.cuda.is_available(), "未检测到 GPU，请在 GPU 实例上运行"
    set_seed(42)
    bf16 = torch.cuda.is_bf16_supported()
    print(f"GPU: {torch.cuda.get_device_name(0)} | bf16={bf16}")

    TRAIN = pick_file("train_v4.json", "train_v3.json", "train_v2.json", "train_10000.json")
    VALP  = pick_file("val_v4.json", "val_v3.json", "val_v2.json", "val_500.json")
    assert TRAIN, "没有可用的训练数据！请检查 agri_dataset/qa_pairs/ 上传是否完整"

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("处理数据 ...")
    train_ds = ListDataset(build_examples(TRAIN, tok))
    val_ds   = ListDataset(build_examples(VALP, tok)) if VALP else None
    print(f"训练 {len(train_ds)} 条 / 验证 {len(val_ds) if val_ds else 0} 条")
    assert len(train_ds) > 0, "训练样本为 0：请把本条输出截图反馈"
    # 打印一条样本核对格式
    s = train_ds[0]
    print("样本格式核对:", tok.decode(s["input_ids"])[:150].replace("\n", "⏎"), "…")

    kw = dict(torch_dtype=torch.bfloat16 if bf16 else torch.float16,
              device_map="cuda", trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
    except TypeError:                      # 新版 transformers 参数改名 dtype
        kw["dtype"] = kw.pop("torch_dtype")
        model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
    model.config.use_cache = False
    model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules="all-linear"))
    model.print_trainable_parameters()

    common = dict(
        output_dir=OUT,
        per_device_train_batch_size=BS,
        gradient_accumulation_steps=GA,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=bf16, fp16=not bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20,
        save_strategy="epoch", save_total_limit=2,
        eval_steps=200, per_device_eval_batch_size=4,
        dataloader_num_workers=2,
        report_to=[], seed=42)
    if val_ds:
        common["eval_strategy"] = "steps"
    try:
        args = TrainingArguments(**common)
    except TypeError:                      # 旧版 transformers 参数名
        common["evaluation_strategy"] = common.pop("eval_strategy")
        args = TrainingArguments(**common)

    trainer = Trainer(model=model, args=args,
                      train_dataset=train_ds,
                      eval_dataset=val_ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True,
                                                           label_pad_token_id=-100))
    trainer.train()

    # 保存 LoRA 适配器 + 训练日志
    model.save_pretrained(os.path.join(OUT, "adapter"))
    tok.save_pretrained(os.path.join(OUT, "adapter"))
    with open(os.path.join(OUT, "training_log.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["step", "loss", "eval_loss"])
        for h in trainer.state.log_history:
            w.writerow([h.get("step", ""), h.get("loss", ""), h.get("eval_loss", "")])
    print(f"\nLoRA 适配器已保存: {OUT}/adapter")

    quick_test(model, tok, [
        "黄淮海小麦什么时候播种？亩基本苗多少？",
        "水稻育秧时怎么浸种消毒？",
        "玉米大斑病怎么防治？用药量多少？"])

    # 合并导出完整模型（可直接部署；失败也不影响已存的 LoRA）
    try:
        print("\n合并 LoRA 并导出完整模型 ...")
        merged = model.merge_and_unload()
        merged.config.use_cache = True
        merged.save_pretrained(MERGED)
        tok.save_pretrained(MERGED)
        print(f"完整模型已导出: {MERGED}")
    except Exception as e:
        print("合并导出跳过:", e)

    print("\n全部完成 ✔  产物位置:")
    print(f"  LoRA 适配器: {OUT}/adapter")
    print(f"  完整模型:    {MERGED}")
    print(f"  训练日志:    {OUT}/training_log.csv")


if __name__ == "__main__":
    main()
