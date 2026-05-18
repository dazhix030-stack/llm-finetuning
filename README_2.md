# LLM Fine-tuning with LoRA

Unsloth + LoRA を用いた Qwen2.5-1.5B のファインチューニング実験。コーディング・数学・推論の3種データセットを混合学習。

## 概要

RTX 4050 (6GB VRAM) のローカル環境で、Qwen2.5-1.5B-Instruct を LoRA + 4bit量子化でファインチューニング。

## データセット（計30,000件）

| データセット | 件数 | 内容 | ライセンス |
|-------------|------|------|----------|
| Magicoder-Evol-Instruct-110K | 10,000 | コーディング | MIT |
| MetaMathQA | 10,000 | 数学・文章題 | MIT |
| NuminaMath-CoT | 10,000 | 高度数学 + CoT推論 | Apache 2.0 |

## 学習設定

```python
# LoRA設定
r = 16
lora_alpha = 16
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# 学習設定
learning_rate = 5e-5
max_steps = 4500
per_device_train_batch_size = 1
gradient_accumulation_steps = 8  # 実効バッチサイズ = 8
optim = "adamw_8bit"
```

## 動作環境

- Python 3.12
- PyTorch 2.x
- [Unsloth](https://github.com/unslothai/unsloth)（高速LoRAファインチューニング）
- bitsandbytes（4bit量子化）
- GPU: RTX 4050 (6GB VRAM)

## 実行方法

```bash
pip install unsloth trl transformers datasets bitsandbytes
python train.py
```

学習済みモデルは `qwen2.5-1.5b-lora-all`（LoRAアダプタのみ）と  
`qwen2.5-1.5b-specialized-v2`（マージ済み16bit）に保存される。

## 関連リポジトリ

- [llm-neuron-analysis](../llm-neuron-analysis) — ニューロン単位の内部解釈実験
- [llm-layer-pruning](../llm-layer-pruning) — 残差補正による層削除軽量化
