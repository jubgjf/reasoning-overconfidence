import os

import pandas as pd
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.evaluate import add_confidence_column, prf, show_metrics
from confidence.logger import Logger
from confidence.model import ModelName


async def generate_self_consistency_data(
    dataset: DatasetName,
    model: ModelName,
    template: str = "simple",
    temperature: float = 0.2,
    turns_range: range = range(0, 32),
    output_dir: str = "tmp/self_consistency",
):
    """
    生成self-consistency数据并保存到文件

    Args:
        dataset: 数据集名称
        model: 模型名称
        template: 模板类型
        temperature: 温度参数
        turns_range: turn的范围
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing dataset: {dataset}, model: {model}, template: {template}, temperature: {temperature}")

    # 加载所有turns的数据
    records_list = []
    for turn in turns_range:
        record_cls = dataset.record_cls
        title = f"{dataset}--{template}--{model}--{temperature}--{turn}"
        db_logger = Logger(db_name=title, table_name=title, record_cls=record_cls)
        async with db_logger:
            records = await db_logger.fetch()
        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)
        df["turn"] = turn
        df["consistency_choose"] = False
        records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)

    # 数据预处理
    df = prf(df, dataset)
    df = add_confidence_column(df)
    df["model_thinking_response"] = df["thinking_history"].apply(lambda x: x[1])

    # 计算各种self-consistency策略的选择标准
    df["max_model_thinking_response"] = df.groupby(["question_id", "answer_count_bin"])[
        "model_thinking_response"
    ].transform(lambda x: x.max())

    df["max_model_confidence"] = df.groupby(["question_id", "answer_count_bin"])[
        "model_confidence_extracted"
    ].transform(lambda x: x.max())

    df["min_model_confidence"] = df.groupby(["question_id", "answer_count_bin"])[
        "model_confidence_extracted"
    ].transform(lambda x: x.min())

    df["median_model_confidence"] = df.groupby(["question_id", "answer_count_bin"])[
        "model_confidence_extracted"
    ].transform("median")
    df["confidence_diff"] = (df["model_confidence_extracted"] - df["median_model_confidence"]).abs()
    df["min_confidence_diff"] = df.groupby(["question_id", "answer_count_bin"])["confidence_diff"].transform("min")

    # 标记不同方法
    df["method"] = "Long-CoT"
    df.loc[df["model_thinking_response"] == df["max_model_thinking_response"], "method"] = (
        "Self-Consistency (max thinking)"
    )
    df.loc[df["model_confidence_extracted"] == df["max_model_confidence"], "method"] = (
        "Self-Consistency (max confidence)"
    )
    df.loc[df["model_confidence_extracted"] == df["min_model_confidence"], "method"] = (
        "Self-Consistency (min confidence)"
    )
    df.loc[df["confidence_diff"] == df["min_confidence_diff"], "method"] = "Self-Consistency (median confidence)"

    # 输出各种方法的性能指标
    methods = [
        ("Long-CoT", df[df["turn"] == 0].copy()),
        (
            "Self-Consistency (max thinking)",
            df[df["model_thinking_response"] == df["max_model_thinking_response"]].copy(),
        ),
        (
            "Self-Consistency (max confidence)",
            df[df["model_confidence_extracted"] == df["max_model_confidence"]].copy(),
        ),
        (
            "Self-Consistency (min confidence)",
            df[df["model_confidence_extracted"] == df["min_model_confidence"]].copy(),
        ),
        ("Self-Consistency (median confidence)", df[df["confidence_diff"] == df["min_confidence_diff"]].copy()),
    ]

    for method_name, method_df in methods:
        show_metrics(method_df, method_name)

    # 只保留每种方法的唯一样本用于可视化
    plot_df = df.drop_duplicates(subset=["question_id", "answer_count_bin", "method"])

    # 保存数据
    output_filename = f"{dataset}_{model.series_name.lower()}_{template}_temp{temperature}.pkl"
    output_path = os.path.join(output_dir, output_filename)
    plot_df.to_pickle(output_path)

    print(f"Data saved to: {output_path}")
    print(f"Total samples: {len(plot_df)}")
    print(f"Methods: {plot_df['method'].unique()}")

    return output_path


async def main():
    """主函数：生成默认配置的数据"""
    dataset = DatasetName.SubsetSum
    temperature = 0.2
    model = ModelName.QWEN3_8B_THINK
    template = "simple"

    await generate_self_consistency_data(
        dataset=dataset, model=model, template=template, temperature=temperature, turns_range=range(0, 32)
    )


if __name__ == "__main__":
    run_async(main())
