# Reasoning Overconfidence

Code for **Beware of Reasoning Overconfidence: Pitfalls in the Reasoning Process for Multi-solution Tasks**

## Environment Setup

```shell
$ pip install -r requirements.txt
$ . .venv/bin/activate
```

## Run Experiments

### Dataset

`scripts/gen-timetabling.py` and `scripts/gen-subsetsum.py` are used to generate dataset, which will be saved to `dataset/`.
```shell
$ export PYTHONPATH=$(pwd)
$ python scripts/gen-timetabling.py
```

The dataset used in our paper has already been placed in `dataset/`, and you can use it directly.

### Main Experiments

We provide two program entry points: `inference.py` and `inference-fake-reflection.py`.

`inference.py` is used for conducting basic model inference experiments.
Its parameters are as follows:

```python
model: ModelName = ModelName.QWEN3_8B_THINK      # Model name, see `model.py` for options
model_name_or_path: str = "Qwen/Qwen3-8B"        # Model name or path in HuggingFace format
dataset: DatasetName = DatasetName.TimeTabling   # Dataset name, see `dataset.py` for options
template: Template = "simple"                    # Prompt template name. Use "cot" for Short-CoT models and "simple" for Long-CoT models.
temperature: float = 0.2                         # Inference temperature
max_completion_tokens: int = 20480               # Maximum number of tokens to generate during inference
force_update: bool = False                       # Disables checkpoint continuation, forces update of the experiment result log
concurrency: int = 100                           # Request concurrency
turn: int = 0                                    # Experiment round
```

`inference-fake-reflection.py` continues the experiments (Reflection and Exploration) based on the model inference results from `inference.py`.
Most of its parameters are the same as `inference.py`, with the only addition being the `--fake_type` parameter.

```python
model: ModelName = ModelName.QWEN3_8B_THINK
model_name_or_path: str = "Qwen/Qwen3-8B"
dataset: DatasetName = DatasetName.TimeTabling
template: Template = "simple"
temperature: float = 0.2
fake_type: FakeType = FakeType.less  # `FakeType.less` is for the Reflection experiment, and `FakeType.more` is for the Exploration experiment
force_update: bool = False
concurrency: int = 5
turn: int = 0
```

- Run Short-CoT Baseline
```shell
$ python inference.py --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot --dataset timetabling --concurrency 500 --temperature 0.2 --turn 0
```

- Run Short-CoT Temperature
```shell
$ python inference.py --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot --dataset timetabling --concurrency 500 --temperature 0.0 --turn 0
$ python inference.py --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot --dataset timetabling --concurrency 500 --temperature 0.1 --turn 0
$ python inference.py --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot --dataset timetabling --concurrency 500 --temperature 0.3 --turn 0
$ python inference.py --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot --dataset timetabling --concurrency 500 --temperature 0.4 --turn 0
$ python inference.py --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot --dataset timetabling --concurrency 500 --temperature 0.5 --turn 0
$ python inference.py --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot --dataset timetabling --concurrency 500 --temperature 0.6 --turn 0
$ python inference.py --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot --dataset timetabling --concurrency 500 --temperature 0.7 --turn 0
$ python inference.py --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot --dataset timetabling --concurrency 500 --temperature 0.8 --turn 0
$ python inference.py --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot --dataset timetabling --concurrency 500 --temperature 0.9 --turn 0
$ python inference.py --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot --dataset timetabling --concurrency 500 --temperature 1.0 --turn 0
```

- Run Short-CoT w/ Exploration
```shell
$ python inference-fake-reflection.py --model qwen3-8b-no_think --model_name_or_path Qwen/Qwen3-8B --template cot    --dataset timetabling --concurrency 500 --temperature 0.2 --turn 0 --fake_type more
```

- Run Long-CoT Baseline
```shell
$ python inference.py --model qwen3-8b-think --model_name_or_path Qwen/Qwen3-8B --template simple --dataset timetabling --concurrency 500 --temperature 0.2 --turn 0
```

- Run Long-CoT w/ Self-Consistency
```shell
$ python inference.py --model qwen3-8b-think --model_name_or_path Qwen/Qwen3-8B --template simple --dataset timetabling --concurrency 500 --temperature 0.2 --turn 1
$ python inference.py --model qwen3-8b-think --model_name_or_path Qwen/Qwen3-8B --template simple --dataset timetabling --concurrency 500 --temperature 0.2 --turn 2
$ python inference.py --model qwen3-8b-think --model_name_or_path Qwen/Qwen3-8B --template simple --dataset timetabling --concurrency 500 --temperature 0.2 --turn 3
$ python inference.py --model qwen3-8b-think --model_name_or_path Qwen/Qwen3-8B --template simple --dataset timetabling --concurrency 500 --temperature 0.2 --turn 4
```

- Run Long-CoT w/o Reflection
```shell
$ python inference-fake-reflection.py --model qwen3-8b-think --model_name_or_path Qwen/Qwen3-8B --template simple --dataset timetabling --concurrency 500 --temperature 0.2 --turn 0 --fake_type less
```

All experiment logs will be saved to `logs/`
For convenience, we provide a script `scripts/inference.example.sh` to run experiments above.

### Visualize Results

We provide multiple result visualization scripts, located in `scripts/visualize/*.py`.

For example, to plot a Performance vs. Recall 3D graph:
```shell
$ export PYTHONPATH=$(pwd)
$ python scripts/visualize/performance.py
```

The only special case is plotting attention entropy, which needs to be launched with `accelerate`.
The only parameter for `entropy.py` is `--layer`, which indicates the layer id to analyze.
```shell
$ export PYTHONPATH=$(pwd)
$ accelerate launch scripts/visualize/entropy.py --layer 0
```

We also provide a script `scripts/entropy.example.sh` for this.

## Reproducibility

We have provided all the experimental logs from this paper in [HuggingFace](https://huggingface.co/datasets/jubgjf/reasoning-overconfidence-logs-reproduce).

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="jubgjf/reasoning-overconfidence-logs-reproduce",
    repo_type="dataset",
    local_dir="logs-reproduce/",
)
```

You can directly use the visualization code to reproduce the results for all figures and tables in the paper.
