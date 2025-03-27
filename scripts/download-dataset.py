import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path

from datasets import load_dataset

from confidence.data import ARCData, GAOKAOData, GSM8KData, LogiQAData
from confidence.utils import gsm8k_postprocess

if __name__ == "__main__":
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    save_to = Path("./dataset/gsm8k.jsonl")
    if not save_to.exists():
        save_to.parent.mkdir(exist_ok=True)
    with open(save_to, "w") as f:
        for i, data in enumerate(dataset):
            answer_num = gsm8k_postprocess(data["answer"])
            if answer_num.is_err():
                print(answer_num.err_value)
                continue
            data = GSM8KData(
                question_id=i,
                question=data["question"],
                answer=data["answer"],
                answer_num=float(gsm8k_postprocess(data["answer"]).ok_value[0]),
            )
            f.write(data.model_dump_json() + "\n")

    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    save_to = Path("./dataset/arc.jsonl")
    if not save_to.exists():
        save_to.parent.mkdir(exist_ok=True)
    with open(save_to, "w") as f:
        for data in dataset:
            data = ARCData(
                question_id=data["id"],
                question=data["question"],
                choices={k: v for k, v in zip(data["choices"]["label"], data["choices"]["text"])},
                answer_key=data["answerKey"],
            )
            f.write(data.model_dump_json() + "\n")

    dataset = load_dataset("logikon/logikon-bench", name="logiqa", split="test")
    save_to = Path("./dataset/logiqa.jsonl")
    if not save_to.exists():
        save_to.parent.mkdir(exist_ok=True)
    with open(save_to, "w") as f:
        for i, data in enumerate(dataset):
            data = LogiQAData(
                question_id=i,
                passage=data["passage"],
                question=data["question"],
                choices={k: v for k, v in zip(["A", "B", "C", "D"], data["options"])},
                answer_key=["A", "B", "C", "D"][data["answer"]],
            )
            f.write(data.model_dump_json() + "\n")

    dataset = load_dataset("hails/agieval-gaokao-physics", split="test")
    save_to = Path("./dataset/gaokao_physics.jsonl")
    if not save_to.exists():
        save_to.parent.mkdir(exist_ok=True)
    with open(save_to, "w") as f:
        for i, data in enumerate(dataset):
            data = GAOKAOData(
                question_id=i,
                question_and_choices=data["query"],
                answer_keys=" ".join([["A", "B", "C", "D"][answer_index] for answer_index in data["gold"]]),
            )
            f.write(data.model_dump_json() + "\n")
