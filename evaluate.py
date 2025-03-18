from tap import Tap


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tortoise import run_async

from confidence.data import GSM8KTemplate, ARCTemplate, string_to_template
from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName


class Argument(Tap):
    model: ModelName = ModelName.QWEN2_5_7B
    # dataset: DatasetName = DatasetName.GSM8K
    # template: GSM8KTemplate | ARCTemplate = GSM8KTemplate.BigGSM
    dataset: DatasetName = DatasetName.ARC
    template: GSM8KTemplate | ARCTemplate = ARCTemplate.OpenCompass
    method: MethodName = MethodName.Verbal_0_100
    # method: MethodName = MethodName.LogProb
    # method: MethodName = MethodName.P_True

    def configure(self) -> None:
        self.add_argument("--template", type=string_to_template)


async def main(args: Argument):
    record_cls = args.dataset.record_cls
    db_logger = Logger(
        db_name=args.dataset.value,
        table_name=f"{args.dataset}--{args.method}--{args.template}--{args.model}",
        record_cls=record_cls,
    )
    async with db_logger:
        records = await db_logger.fetch()

    records_list = [record.model_dump() for record in records]
    df = pd.DataFrame(records_list)
    df["model_answer_response_len"] = df["model_answer_response"].apply(len)
    if args.dataset == DatasetName.GSM8K:
        df["model_answer_extracted"] = df["model_answer_extracted"].apply(float)
    elif args.dataset == DatasetName.ARC:
        pass
    if args.method == MethodName.Verbal_0_100:
        df["model_confidence_extracted"] = df["model_confidence_extracted"].apply(lambda x: x / 100)

    # ===== answer length vs confidence =====
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df["model_answer_response_len"], y=df["model_confidence_extracted"], alpha=0.5)
    plt.xlim(0, 2500)
    plt.xlabel("Response Length")
    plt.ylabel("Confidence")
    plt.title(f"Response Length vs. Confidence\n{args.dataset}--{args.method}--{args.template}--{args.model}")
    plt.tight_layout()
    plt.show()

    # ===== correctness vs confidence =====
    # if args.dataset == DatasetName.GSM8K:
    #     df["is_correct"] = df["model_answer_extracted"] == df["answer_num"]
    # elif args.dataset == DatasetName.ARC:
    #     df["is_correct"] = df["model_answer_extracted"] == df["answer_key"]
    # acc = df["is_correct"].sum() / len(df["is_correct"])
    # df["confidence_bin"] = pd.cut(
    #     df["model_confidence_extracted"],
    #     bins=np.linspace(0, 1, 11),
    #     labels=np.arange(10),
    #     include_lowest=True,
    #     right=True,
    # )
    # cm = confusion_matrix(df["is_correct"], df["confidence_bin"], labels=np.arange(10))
    # cm = cm[:2]  # only keep the first two rows: correct/incorrect
    #
    # n_rows, n_cols = cm.shape
    # fig, ax = plt.subplots(figsize=(n_cols, n_rows * 1.5))
    #
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    # plt.xlabel("Confidence Bin")
    # plt.ylabel("Correctness")
    # plt.title(f"{args.dataset}, {args.method}, {args.template}, {args.model}\nAcc = {acc:.2f}")
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(f"figures/{args.dataset}--{args.method}--{args.template}--{args.model}.png")


if __name__ == "__main__":
    args = Argument().parse_args()

    run_async(main(args))
