#!/usr/bin/bash

. .venv/bin/activate

# Short-CoT
python inference.py                 --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot    --dataset timetabling --concurrency 500 --temperature 0.2 --turn 0
python evaluate.py                  --model qwen3-8b-no_think                                    --template cot    --dataset timetabling --concurrency 500 --temperature 0.2 --turn 0 --fake_type none --judge_model qwen3-32b-no_think --judge_model_name_or_path Qwen/Qwen3-32B

# Short-CoT temperature
python inference.py                 --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot    --dataset timetabling --concurrency 500 --temperature 0.0 --turn 0
python inference.py                 --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot    --dataset timetabling --concurrency 500 --temperature 0.1 --turn 0
python inference.py                 --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot    --dataset timetabling --concurrency 500 --temperature 0.3 --turn 0
python inference.py                 --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot    --dataset timetabling --concurrency 500 --temperature 0.4 --turn 0
python inference.py                 --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot    --dataset timetabling --concurrency 500 --temperature 0.5 --turn 0
python inference.py                 --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot    --dataset timetabling --concurrency 500 --temperature 0.6 --turn 0
python inference.py                 --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot    --dataset timetabling --concurrency 500 --temperature 0.7 --turn 0
python inference.py                 --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot    --dataset timetabling --concurrency 500 --temperature 0.8 --turn 0
python inference.py                 --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot    --dataset timetabling --concurrency 500 --temperature 0.9 --turn 0
python inference.py                 --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot    --dataset timetabling --concurrency 500 --temperature 1.0 --turn 0
python evaluate.py                  --model qwen3-8b-no_think                                    --template cot    --dataset timetabling --concurrency 500 --temperature 0.0 --turn 0 --fake_type none --judge_model qwen3-32b-no_think --judge_model_name_or_path Qwen/Qwen3-32B
python evaluate.py                  --model qwen3-8b-no_think                                    --template cot    --dataset timetabling --concurrency 500 --temperature 0.1 --turn 0 --fake_type none --judge_model qwen3-32b-no_think --judge_model_name_or_path Qwen/Qwen3-32B
python evaluate.py                  --model qwen3-8b-no_think                                    --template cot    --dataset timetabling --concurrency 500 --temperature 0.3 --turn 0 --fake_type none --judge_model qwen3-32b-no_think --judge_model_name_or_path Qwen/Qwen3-32B
python evaluate.py                  --model qwen3-8b-no_think                                    --template cot    --dataset timetabling --concurrency 500 --temperature 0.4 --turn 0 --fake_type none --judge_model qwen3-32b-no_think --judge_model_name_or_path Qwen/Qwen3-32B
python evaluate.py                  --model qwen3-8b-no_think                                    --template cot    --dataset timetabling --concurrency 500 --temperature 0.5 --turn 0 --fake_type none --judge_model qwen3-32b-no_think --judge_model_name_or_path Qwen/Qwen3-32B
python evaluate.py                  --model qwen3-8b-no_think                                    --template cot    --dataset timetabling --concurrency 500 --temperature 0.6 --turn 0 --fake_type none --judge_model qwen3-32b-no_think --judge_model_name_or_path Qwen/Qwen3-32B
python evaluate.py                  --model qwen3-8b-no_think                                    --template cot    --dataset timetabling --concurrency 500 --temperature 0.7 --turn 0 --fake_type none --judge_model qwen3-32b-no_think --judge_model_name_or_path Qwen/Qwen3-32B
python evaluate.py                  --model qwen3-8b-no_think                                    --template cot    --dataset timetabling --concurrency 500 --temperature 0.8 --turn 0 --fake_type none --judge_model qwen3-32b-no_think --judge_model_name_or_path Qwen/Qwen3-32B
python evaluate.py                  --model qwen3-8b-no_think                                    --template cot    --dataset timetabling --concurrency 500 --temperature 0.9 --turn 0 --fake_type none --judge_model qwen3-32b-no_think --judge_model_name_or_path Qwen/Qwen3-32B
python evaluate.py                  --model qwen3-8b-no_think                                    --template cot    --dataset timetabling --concurrency 500 --temperature 1.0 --turn 0 --fake_type none --judge_model qwen3-32b-no_think --judge_model_name_or_path Qwen/Qwen3-32B


# Short-CoT w/ self-exploration
python inference-fake-reflection.py --mLdel qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot    --dataset timetabling --concurrency 500 --temperature 0.2 --turn 0 --fake_type more
python evaluate.py                  --model qwen3-8b-no_think                                    --template cot    --dataset timetabling --concurrency 500 --temperature 0.2 --turn 0 --fake_type more --judge_model qwen3-32b-no_think --judge_model_name_or_path Qwen/Qwen3-32B

# Long-CoT
python inference.py                 --model qwen3-8b-think    --model_name_or_path Qwen/Qwen3-8B --template simple --dataset timetabling --concurrency 500 --temperature 0.2 --turn 0
python evaluate.py                  --model qwen3-8b-think                                       --template simple --dataset timetabling --concurrency 500 --temperature 0.2 --turn 0 --fake_type none --judge_model qwen3-32b-no_think --judge_model_name_or_path Qwen/Qwen3-32B

# Long-CoT w/ self-consistency
python inference.py                 --model qwen3-8b-think    --model_name_or_path Qwen/Qwen3-8B --template simple --dataset timetabling --concurrency 500 --temperature 0.2 --turn 1
python inference.py                 --model qwen3-8b-think    --model_name_or_path Qwen/Qwen3-8B --template simple --dataset timetabling --concurrency 500 --temperature 0.2 --turn 2
python inference.py                 --model qwen3-8b-think    --model_name_or_path Qwen/Qwen3-8B --template simple --dataset timetabling --concurrency 500 --temperature 0.2 --turn 3
python inference.py                 --model qwen3-8b-think    --model_name_or_path Qwen/Qwen3-8B --template simple --dataset timetabling --concurrency 500 --temperature 0.2 --turn 4
python evaluate.py                  --model qwen3-8b-think                                       --template simple --dataset timetabling --concurrency 500 --temperature 0.2 --turn 1 --fake_type none --judge_model qwen3-32b-no_think --judge_model_name_or_path Qwen/Qwen3-32B
python evaluate.py                  --model qwen3-8b-think                                       --template simple --dataset timetabling --concurrency 500 --temperature 0.2 --turn 2 --fake_type none --judge_model qwen3-32b-no_think --judge_model_name_or_path Qwen/Qwen3-32B
python evaluate.py                  --model qwen3-8b-think                                       --template simple --dataset timetabling --concurrency 500 --temperature 0.2 --turn 3 --fake_type none --judge_model qwen3-32b-no_think --judge_model_name_or_path Qwen/Qwen3-32B
python evaluate.py                  --model qwen3-8b-think                                       --template simple --dataset timetabling --concurrency 500 --temperature 0.2 --turn 4 --fake_type none --judge_model qwen3-32b-no_think --judge_model_name_or_path Qwen/Qwen3-32B

# Long-CoT w/o self-reflection
python inference-fake-reflection.py --model qwen3-8b-think    --model_name_or_path Qwen/Qwen3-8B --template simple --dataset timetabling --concurrency 500 --temperature 0.2 --turn 0 --fake_type less
python evaluate.py                  --model qwen3-8b-think                                       --template simple --dataset timetabling --concurrency 500 --temperature 0.2 --turn 0 --fake_type less --judge_model qwen3-32b-no_think --judge_model_name_or_path Qwen/Qwen3-32B
