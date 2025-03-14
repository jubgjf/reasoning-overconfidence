#!/usr/bin/bash

#SBATCH -J sglang-server
#SBATCH -t 8:00:00
#SBATCH -o logs/sglang-server.log
#SBATCH -p compute
#SBATCH -c 32
#SBATCH --mem 128GB
#SBATCH --gres gpu:nvidia_rtx_a6000:8

. .venv/bin/activate

python -m sglang.launch_server \
    --model-path /home/share/models/Qwen2.5-7B-Instruct \
    --served-model-name qwen2.5-7b \
    --tp 1 \
    --dp 8 \
    --host 0.0.0.0 \
    --port 33333 \
