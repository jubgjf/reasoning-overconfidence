import pandas as pd
from pydantic import BaseModel
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName
from confidence.template import Template, SubsetSumTemplate, TimeTablingTemplate


class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    judge_model = ModelName.QWEN3_32B_NO_THINK
    dataset = DatasetName.TimeTabling
    method = MethodName.Verbal_0_100
    no_cot_memory = False

    settings = [
        # Setting(model=ModelName.QWEN3_8B_THINK, template=SubsetSumTemplate.simple),
        Setting(model=ModelName.QWEN3_8B_THINK, template=TimeTablingTemplate.simple),
    ]

    records_list = []
    turns = [0, 1, 2]
    for setting in settings:
        for turn in turns:
            record_cls = dataset.record_cls
            db_logger = Logger(
                db_name=dataset.value + "--turn" + str(turn),
                table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{setting.template}--{setting.model}--evaluate-by-{judge_model}",
                record_cls=record_cls,
            )
            async with db_logger:
                records = await db_logger.fetch()

            method_records = [record.model_dump() for record in records]
            df = pd.DataFrame(method_records)
            if method == MethodName.Verbal_0_100:
                df["model_confidence_extracted"] = df["model_confidence_extracted"].apply(lambda x: x / 100)
            df["setting"] = f"{setting.model}--{setting.template}"
            df["turn"] = turn
            df["consistency_choose"] = False

            records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)

    df["correct_solution_count"] = df["eval_result"].apply(lambda x: int(x.split("/")[0]))
    df["total_solution_count"] = df["eval_result"].apply(lambda x: int(x.split("/")[1]))

    df = df[df["correct_solution_count"] <= df["total_solution_count"]]
    df = df[df["correct_solution_count"] <= df["answer_count"]]
    df = df[df["total_solution_count"] <= df["answer_count"]]

    if dataset == DatasetName.TimeTabling:
        df["answer_count_bin"] = df["answer_count"].apply(lambda x: int(x // 50))
    elif dataset == DatasetName.SubsetSum:
        df["answer_count_bin"] = df["answer_count"].apply(lambda x: int(x // 50) if int(x // 50) < 6 else 6)
    else:
        raise NotImplementedError

    df["completeness"] = df["correct_solution_count"] / df["answer_count"]
    df["accuracy"] = df["correct_solution_count"] / df["total_solution_count"]

    # 对于每个question_id，可能存在多个turn的结果，只保留len(model_thinking_response)最大的行(设置consistency_choose为True)
    df["max_model_thinking_response"] = df.groupby(["question_id", "answer_count_bin"])[
        "model_thinking_response"
    ].transform(lambda x: x.max())
    df["consistency_choose"] = df["model_thinking_response"] == df["max_model_thinking_response"]

    df["scaling"] = df["consistency_choose"].apply(lambda x: "parallel" if x else "none")
    df = df[~((df["scaling"] == "none") & (df["turn"] != 0))]

    if dataset == DatasetName.TimeTabling:
        # df = df[df["answer_count_bin"] < 3]  # easy
        # df = df[df["answer_count_bin"] > 7]  # hard
        pass  # total
    elif dataset == DatasetName.SubsetSum:
        # df = df[df["answer_count_bin"] < 2]  # easy
        # df = df[df["answer_count_bin"] > 5]  # hard
        pass  # total
    else:
        raise NotImplementedError

    parallel_completeness_mean = df[df["scaling"] == "parallel"]["completeness"].mean()
    none_completeness_mean = df[df["scaling"] == "none"]["completeness"].mean()
    print(f"Parallel Completeness: {parallel_completeness_mean * 100:.2f}")
    print(f"None Completeness: {none_completeness_mean * 100:.2f}")
    # plt.figure()
    # sns.lineplot(data=df, x="answer_count_bin", y="completeness", hue="scaling")
    # plt.xlabel("Answer Count Bin")
    # plt.ylabel("completeness")
    # plt.title("completeness")
    # plt.show()

    parallel_accuracy_mean = df[df["scaling"] == "parallel"]["accuracy"].mean()
    none_accuracy_mean = df[df["scaling"] == "none"]["accuracy"].mean()
    print(f"Parallel Accuracy: {parallel_accuracy_mean * 100:.2f}")
    print(f"None Accuracy: {none_accuracy_mean * 100:.2f}")
    # plt.figure()
    # sns.lineplot(data=df, x="answer_count_bin", y="accuracy", hue="scaling")
    # plt.xlabel("Answer Count Bin")
    # plt.ylabel("accuracy")
    # plt.title("accuracy")
    # plt.show()

    parallel_confidence_mean = df[df["scaling"] == "parallel"]["model_confidence_extracted"].mean()
    none_confidence_mean = df[df["scaling"] == "none"]["model_confidence_extracted"].mean()
    print(f"Parallel Confidence: {parallel_confidence_mean}")
    print(f"None Confidence: {none_confidence_mean}")
    # plt.figure()
    # sns.scatterplot(df, x="model_confidence_extracted", y="completeness", hue="scaling")
    # plt.xlabel("conf")
    # plt.ylabel("comp")
    # plt.title("accuracy")
    # plt.show()


if __name__ == "__main__":
    run_async(main())
