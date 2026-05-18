from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_dora=False,
)

prompt_style = """Below is an instruction that describes a task. \
Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

def fix_output(x):
    if isinstance(x["output"], list):
        return {"output": x["output"][0]}
    return x

def formatting_prompts_func(examples):
    texts = [
        prompt_style.format(i, o) + tokenizer.eos_token
        for i, o in zip(examples["instruction"], examples["output"])
    ]
    return {"text": texts}

# ── データセット ──────────────────────────────────────────

# Magicoder（コーディング：10000件・MIT）
ds_magicoder = (
    load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
    .shuffle(seed=42)
    .select(range(10000))
    .rename_columns({"instruction": "instruction", "response": "output"})
    .map(fix_output)
)

# MetaMathQA（数学・文章題：10000件・MIT）
ds_meta = (
    load_dataset("meta-math/MetaMathQA", split="train")
    .shuffle(seed=42)
    .select(range(10000))
    .rename_columns({"query": "instruction", "response": "output"})
    .map(fix_output)
)

# NuminaMath-CoT（高度数学＋推論：10000件・Apache 2.0）
ds_numina = (
    load_dataset("AI-MO/NuminaMath-CoT", split="train")
    .shuffle(seed=42)
    .select(range(10000))
    .rename_columns({"problem": "instruction", "solution": "output"})
    .map(fix_output)
)

# 結合 → プロンプト整形
dataset = (
    concatenate_datasets([ds_magicoder, ds_meta, ds_numina])
    .map(formatting_prompts_func, batched=True)
)

# ── 学習設定 ──────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        max_steps=4500,
        learning_rate=5e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        output_dir="outputs",
        seed=3407,
    ),
)

trainer.train()

# ── 保存 ──────────────────────────────────────────────────
model.save_pretrained("qwen2.5-1.5b-lora-all")
tokenizer.save_pretrained("qwen2.5-1.5b-lora-all")

model.save_pretrained_merged(
    "qwen2.5-1.5b-specialized-v2",
    tokenizer,
    save_method="merged_16bit",
)