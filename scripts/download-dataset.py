import os

from confidence.utils import gsm8k_postprocess

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path

from datasets import load_dataset
from confidence.data import GSM8KData, ARCData

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
                id=i,
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
                id=data["id"],
                question=data["question"],
                choices={k: v for k, v in zip(data["choices"]["label"], data["choices"]["text"])},
                answer_key=data["answerKey"],
            )
            f.write(data.model_dump_json() + "\n")
