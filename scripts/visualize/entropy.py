import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, required=True)
args = parser.parse_args()

# ==============================================================================
# --- 0. 配置区域 ---
# ==============================================================================
# --- 模型与数据路径 ---
MODEL_ID = "Qwen/Qwen3-8B"
DATASET_PATH = "dataset/timetabling.jsonl"
OUTPUT_DIR = "logs/entropy-analysis"  # 所有分析结果将保存到此文件夹

# --- 分析参数 ---
LAYER_TO_ANALYZE = args.layer
MAX_NEW_TOKENS = 16384  # 建议减小以加快分析速度，16384会非常慢且消耗大量内存
TASKS_TO_ANALYZE = 20  # 分析数据集中的前N个任务，设为None则分析全部

# ==============================================================================
# --- 1. 初始化与加载 ---
# ==============================================================================

# 创建输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("正在加载模型和分词器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
print("模型加载完成。")


# ==============================================================================
# --- 2. 核心分析与绘图函数 ---
# ==============================================================================


def calculate_attention_entropy(attentions):
    epsilon = 1e-9
    entropy = -torch.sum(attentions * torch.log2(attentions + epsilon), dim=-1)
    return entropy


def analyze_attention_for_prompt(prompt: str):
    """
    对给定的prompt进行生成，并返回多种分析所需的数据。
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        output_attentions=True,
        return_dict_in_generate=True,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    attentions_per_step = outputs.attentions
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[0, input_len:]
    all_token_ids = outputs.sequences[0]

    mean_entropies = []
    per_head_entropies_trace = []  # 记录每一步所有头的熵
    attention_matrices_for_heatmap = {}  # 存储特定步骤的注意力矩阵

    for step_idx, step_attentions in enumerate(attentions_per_step):
        layer_attention = step_attentions[LAYER_TO_ANALYZE]
        attention_for_last_token = layer_attention[0, :, -1, :]

        # 归一化每个头的注意力分布
        normalized_attention_per_head = attention_for_last_token / attention_for_last_token.sum(dim=-1, keepdim=True)

        # 计算每个头的熵
        per_head_entropy = calculate_attention_entropy(normalized_attention_per_head)
        per_head_entropies_trace.append(per_head_entropy.cpu().float().numpy())

        # 计算平均熵
        avg_head_attention = normalized_attention_per_head.mean(dim=0)
        mean_entropy = calculate_attention_entropy(avg_head_attention)
        mean_entropies.append(mean_entropy.item())

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    all_tokens = [tokenizer.decode(token_id) for token_id in all_token_ids]

    return mean_entropies, per_head_entropies_trace, attention_matrices_for_heatmap, generated_text, all_tokens


def plot_entropy_evolution(df, dir, filename_prefix):
    """绘制熵演化图"""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(6, 4))

    # 处理数据
    min_len = df["mean_entropies"].apply(len).min()
    entropy_data = []
    for index, row in df.iterrows():
        for step, entropy_val in enumerate(row["mean_entropies"][:min_len]):
            entropy_data.append({"step": step, "entropy": entropy_val, "prompt_type": row["prompt_type"]})
    entropy_df = pd.DataFrame(entropy_data)

    # 绘图
    ax = sns.lineplot(data=entropy_df, x="step", y="entropy", hue="prompt_type", errorbar="sd")

    title = f"Attention Entropy Evolution (Layer {LAYER_TO_ANALYZE})"

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Generated Token Step", fontsize=12)
    ax.set_ylabel("Average Attention Entropy (bits)", fontsize=12)
    plt.legend(title="Prompt Type")
    plt.grid(True)
    filepath = os.path.join(dir, f"{filename_prefix}.svg")
    plt.savefig(filepath)
    print(f"熵演化图已保存到: {filepath}")
    plt.close()


# ==============================================================================
# --- 3. 主分析流程 ---
# ==============================================================================
print("开始分析数据集...")
try:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f]
except FileNotFoundError:
    print(f"错误: 数据集文件 '{DATASET_PATH}' 未找到。")
    tasks = []

if TASKS_TO_ANALYZE is not None:
    tasks = tasks[:TASKS_TO_ANALYZE]

results = []
heatmap_data_store = []

for i, item in enumerate(tasks):
    task_text = item.get("question", item.get("task", ""))  # 兼容不同key
    print(f"\n--- 正在处理任务 {i + 1}/{len(tasks)}: {task_text[:50]}... ---")

    # 定义两种prompt (根据您的格式)
    messages_short = [{"role": "user", "content": task_text + "\nThink step by step before answering." + " /no_think"}]
    prompt_short = tokenizer.apply_chat_template(
        messages_short, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    messages_long_cot = [{"role": "user", "content": task_text + " /think"}]
    prompt_long_cot = tokenizer.apply_chat_template(
        messages_long_cot, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )

    prompts_to_run = {"Long-CoT": prompt_long_cot, "Short-CoT": prompt_short}

    for p_type, p_text in prompts_to_run.items():
        mean_entropies, per_head_entropies, heatmaps, text, tokens = analyze_attention_for_prompt(p_text)
        results.append(
            {
                "task": task_text,
                "prompt_type": p_type,
                "mean_entropies": mean_entropies,
                "per_head_entropies": per_head_entropies,
                "generated_text": text,
            }
        )
        print(f"  [{p_type}] 生成文本: {text[:100]}...")

# ==============================================================================
# --- 4. 结果聚合与最终可视化 ---
# ==============================================================================
if results:
    df = pd.DataFrame(results)

    plot_entropy_evolution(df, OUTPUT_DIR, f"layer{LAYER_TO_ANALYZE}-entropy")
else:
    print("没有成功处理任何任务，无法生成图表。")
