import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, required=True)
args = parser.parse_args()

# ==============================================================================
# --- 0. 配置区域 ---
# ==============================================================================
# --- 模型与数据路径 ---
MODEL_ID = "/var/s3fs/guanjiannan/models/Qwen3-8B" 
DATASET_PATH = "../dataset/timetabling.jsonl"
OUTPUT_DIR = "logs/entropy_analysis" # 所有分析结果将保存到此文件夹

# --- 分析参数 ---
LAYER_TO_ANALYZE = args.layer
MAX_NEW_TOKENS = 16384 # 建议减小以加快分析速度，16384会非常慢且消耗大量内存
TASKS_TO_ANALYZE = 20   # 分析数据集中的前N个任务，设为None则分析全部

# --- 可视化控制 ---
# 放大图：聚焦于生成过程的前N个Token
ZOOM_IN_X_LIMIT = 2048

# 注意力头分析：是否生成每个头熵的分布图
ANALYZE_HEADS = True

# 热图可视化：是否为特定任务生成注意力热图
VISUALIZE_HEATMAP = True
HEATMAP_TASK_INDEX = 0  # 选择第几个任务来生成热图 (0 for the first task)
HEATMAP_STEP_INDICES = [50, 100, 200, 500, 1000, 2000] # 选择在第几个生成步骤生成热图
HEATMAP_CONTEXT_WINDOW = 256 # 热图只显示最近的N个token，防止图像过大

# ==============================================================================
# --- 1. 初始化与加载 ---
# ==============================================================================

# 创建输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("正在加载模型和分词器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
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
        pad_token_id=tokenizer.eos_token_id
    )
    
    attentions_per_step = outputs.attentions
    input_len = inputs['input_ids'].shape[1]
    generated_ids = outputs.sequences[0, input_len:]
    all_token_ids = outputs.sequences[0]

    mean_entropies = []
    per_head_entropies_trace = [] # 记录每一步所有头的熵
    attention_matrices_for_heatmap = {} # 存储特定步骤的注意力矩阵

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

        # 如果当前步骤需要保存热图，则保存
        if VISUALIZE_HEATMAP and step_idx in HEATMAP_STEP_INDICES:
            # .clone().detach() 是个好习惯
            attention_matrices_for_heatmap[step_idx] = layer_attention.clone().detach()

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    all_tokens = [tokenizer.decode(token_id) for token_id in all_token_ids]
    
    return mean_entropies, per_head_entropies_trace, attention_matrices_for_heatmap, generated_text, all_tokens


def plot_entropy_evolution(df, dir, filename_prefix, x_limit=None):
    """绘制熵演化图（完整版和放大版）"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))
    
    # 处理数据
    min_len = df['mean_entropies'].apply(len).min()
    entropy_data = []
    for index, row in df.iterrows():
        for step, entropy_val in enumerate(row['mean_entropies'][:min_len]):
            entropy_data.append({"step": step, "entropy": entropy_val, "prompt_type": row['prompt_type']})
    entropy_df = pd.DataFrame(entropy_data)
    
    # 绘图
    ax = sns.lineplot(data=entropy_df, x="step", y="entropy", hue="prompt_type", errorbar="sd")
    
    title = f'Attention Entropy Evolution (Layer {LAYER_TO_ANALYZE})'
    if x_limit:
        ax.set_xlim(0, x_limit)
        title += f' - First {x_limit} Steps'
        
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Generated Token Step', fontsize=12)
    ax.set_ylabel('Average Attention Entropy (bits)', fontsize=12)
    plt.legend(title='Prompt Type')
    plt.grid(True)
    filepath = os.path.join(dir, f"{filename_prefix}.png")
    plt.savefig(filepath)
    print(f"熵演化图已保存到: {filepath}")
    plt.close()


def plot_head_entropy_distribution(df, dir, filename):
    """绘制每个注意力头熵的分布箱形图"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))
    
    head_data = []
    for index, row in df.iterrows():
        # 将所有步骤的所有头的熵展平
        all_head_entropies = np.array(row['per_head_entropies']).flatten()
        for entropy_val in all_head_entropies:
            head_data.append({"entropy": entropy_val, "prompt_type": row['prompt_type']})
    head_df = pd.DataFrame(head_data)

    ax = sns.boxplot(data=head_df, x="prompt_type", y="entropy")
    ax.set_title(f'Distribution of Per-Head Entropies (Layer {LAYER_TO_ANALYZE})', fontsize=16)
    ax.set_xlabel('Prompt Type', fontsize=12)
    ax.set_ylabel('Attention Entropy (bits)', fontsize=12)
    
    filepath = os.path.join(dir, f"{filename}.png")
    plt.savefig(filepath)
    print(f"注意力头熵分布图已保存到: {filepath}")
    plt.close()


def plot_attention_heatmap(matrix, tokens, task_text, step_idx, prompt_type, dir, filename):
    """为单一步骤绘制注意力热图"""
    # 我们只关心最后一个token（query）对所有历史（key）的注意力
    attention_to_plot = matrix[0, :, -1, :].mean(dim=0).cpu().float().numpy() # (key_len,)
    
    # <--- 这里是关键的修复逻辑 --- >
    # 1. 首先，获取当前注意力向量的实际长度
    actual_seq_len = attention_to_plot.shape[0]
    
    # 2. 然后，根据这个实际长度，从完整的tokens列表中截取对应的部分
    #    这确保了 token 列表和 attention 向量的长度在这一步是完全匹配的
    tokens_for_this_step = tokens[:actual_seq_len]
    
    # 3. 接下来，对已经匹配好的 tokens 和 attention_to_plot 应用“上下文窗口”截断
    if len(tokens_for_this_step) > HEATMAP_CONTEXT_WINDOW:
        tokens_to_plot = tokens_for_this_step[-HEATMAP_CONTEXT_WINDOW:]
        attention_to_plot = attention_to_plot[-HEATMAP_CONTEXT_WINDOW:]
    else:
        tokens_to_plot = tokens_for_this_step
    # <--- 修复逻辑结束 --->

    # 现在 tokens_to_plot 和 attention_to_plot 的长度保证一致
    df = pd.DataFrame(attention_to_plot.reshape(1, -1), columns=tokens_to_plot, index=['attention_score'])
    
    plt.figure(figsize=(20, 5))
    ax = sns.heatmap(df, cmap="viridis", cbar_kws={"label": "Attention Score"})
    ax.set_title(f"Attention Heatmap for '{prompt_type}' at Step {step_idx}\nTask: {task_text[:80]}...", fontsize=14, wrap=True)
    ax.set_xlabel("Attended Tokens (Context)")
    ax.set_ylabel("Generated Token (Query)")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks([])
    plt.tight_layout()
    
    filepath = os.path.join(dir, f"{filename}.png")
    plt.savefig(filepath)
    print(f"注意力热图已保存到: {filepath}")
    plt.close()


# ==============================================================================
# --- 3. 主分析流程 ---
# ==============================================================================
print("开始分析数据集...")
try:
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f]
except FileNotFoundError:
    print(f"错误: 数据集文件 '{DATASET_PATH}' 未找到。")
    tasks = []

if TASKS_TO_ANALYZE is not None:
    tasks = tasks[:TASKS_TO_ANALYZE]

results = []
heatmap_data_store = []

for i, item in enumerate(tasks):
    task_text = item.get('question', item.get('task', '')) # 兼容不同key
    print(f"\n--- 正在处理任务 {i+1}/{len(tasks)}: {task_text[:50]}... ---")
    
    # 定义两种prompt (根据您的格式)
    messages_short = [{"role": "user", "content": task_text + "\nThink step by step before answering." + " /no_think"}]
    prompt_short = tokenizer.apply_chat_template(messages_short, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    
    messages_long_cot = [{"role": "user", "content": task_text + " /think"}]
    prompt_long_cot = tokenizer.apply_chat_template(messages_long_cot, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    
    prompts_to_run = {"Short-CoT": prompt_short, "Long-CoT": prompt_long_cot}

    for p_type, p_text in prompts_to_run.items():
        # try:
        mean_entropies, per_head_entropies, heatmaps, text, tokens = analyze_attention_for_prompt(p_text)
        results.append({
            "task": task_text,
            "prompt_type": p_type,
            "mean_entropies": mean_entropies,
            "per_head_entropies": per_head_entropies,
            "generated_text": text
        })
        print(f"  [{p_type}] 生成文本: {text[:100]}...")

        # 如果这是我们想要可视化热图的任务，就保存相关数据
        if VISUALIZE_HEATMAP and i == HEATMAP_TASK_INDEX:
            for step_idx, matrix in heatmaps.items():
                heatmap_data_store.append({
                    "matrix": matrix, "tokens": tokens, "task_text": task_text,
                    "step_idx": step_idx, "prompt_type": p_type
                })
        # except Exception as e:
            # print(f"  [{p_type}] 分析失败: {e}")

# ==============================================================================
# --- 4. 结果聚合与最终可视化 ---
# ==============================================================================
if results:
    df = pd.DataFrame(results)

    # 绘制熵演化图（完整版）
    plot_entropy_evolution(df, OUTPUT_DIR, f"layer{LAYER_TO_ANALYZE}_entropy_evolution_full")
    
    # 绘制熵演化图（放大版）
    plot_entropy_evolution(df, OUTPUT_DIR, f"layer{LAYER_TO_ANALYZE}_entropy_evolution_zoomed", x_limit=ZOOM_IN_X_LIMIT)

    # 绘制注意力头熵分布图
    if ANALYZE_HEADS:
        plot_head_entropy_distribution(df, OUTPUT_DIR, f"layer{LAYER_TO_ANALYZE}_head_entropy_distribution")

    # 绘制选定任务的注意力热图
    if VISUALIZE_HEATMAP and heatmap_data_store:
        print("\n正在生成注意力热图...")
        for data in heatmap_data_store:
            filename = f"layer{LAYER_TO_ANALYZE}_heatmap_task{HEATMAP_TASK_INDEX}_step{data['step_idx']}_{data['prompt_type']}"
            plot_attention_heatmap(
                data['matrix'], data['tokens'], data['task_text'], 
                data['step_idx'], data['prompt_type'], OUTPUT_DIR, filename
            )
else:
    print("没有成功处理任何任务，无法生成图表。")

print("\n--- 分析全部完成 ---")