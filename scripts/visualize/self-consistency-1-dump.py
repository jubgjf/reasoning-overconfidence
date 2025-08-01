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

    # 计算majority voting的结果
    def calculate_majority_voting_solutions(group):
        """计算majority voting的解决方案并集和confidence"""
        from confidence.evaluate import extract_all_solutions

        # 提取所有turns的解决方案，并记录每个解决方案的confidence
        solution_to_confidences = {}  # 解决方案 -> confidence列表的映射
        all_confidences = []

        for _, row in group.iterrows():
            # 从chat_history中提取解决方案（而不是model_output）
            chat_history = row["chat_history"]
            confidence = row["model_confidence_extracted"]
            all_confidences.append(confidence)

            if len(chat_history) >= 2:
                model_output = chat_history[1]["content"]
                solutions = extract_all_solutions(model_output, dataset)

                # 为每个解决方案记录其对应的confidence
                for solution in solutions:
                    if solution not in solution_to_confidences:
                        solution_to_confidences[solution] = []
                    solution_to_confidences[solution].append(confidence)

        all_solutions = set(solution_to_confidences.keys())

        # 计算majority voting的confidence（基于解决方案频率的加权方案）
        if solution_to_confidences:
            # 方案1: 基于解决方案出现频率的加权confidence
            total_weighted_confidence = 0.0
            total_weight = 0

            for solution, confidences in solution_to_confidences.items():
                # 该解决方案的频率作为权重
                frequency = len(confidences)
                # 该解决方案的平均confidence
                avg_confidence = sum(confidences) / len(confidences)

                total_weighted_confidence += frequency * avg_confidence
                total_weight += frequency

            majority_confidence = total_weighted_confidence / total_weight if total_weight > 0 else 0.0
        else:
            # 如果没有找到任何解决方案，使用所有confidence的平均值
            majority_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

        # 重构model_output以包含所有解决方案
        if all_solutions:
            if dataset == DatasetName.TimeTabling:
                # TimeTabling格式需要表格形式
                solution_strs = []
                for i, solution in enumerate(sorted(all_solutions), 1):
                    # 解析标准化的解决方案字符串回到字典格式
                    solution_dict = {}
                    parts = solution.split(";")
                    for part in parts:
                        course_info = part.split(":")
                        if len(course_info) == 2:
                            course = course_info[0]
                            time_room_teacher = course_info[1].split(",")
                            if len(time_room_teacher) == 3:
                                solution_dict[course] = {
                                    "time": time_room_teacher[0],
                                    "room": time_room_teacher[1],
                                    "teacher": time_room_teacher[2],
                                }

                    # 生成表格格式
                    if solution_dict:
                        table_lines = []
                        table_lines.append("| Course | Time | Room | Teacher |")
                        table_lines.append("|--------|------|------|---------|")
                        for course in sorted(solution_dict.keys()):
                            assignment = solution_dict[course]
                            table_lines.append(
                                f"| {course} | {assignment['time']} | {assignment['room']} | {assignment['teacher']} |"
                            )

                        solution_str = f"Solution {i}:\n" + "\n".join(table_lines)
                        solution_strs.append(solution_str)

                majority_output = (
                    "\n\n".join(solution_strs) + f"\n\nTotal {len(all_solutions)} feasible solutions shown above."
                )
            else:
                # SubsetSum格式
                solution_strs = []
                for i, solution in enumerate(sorted(all_solutions), 1):
                    solution_strs.append(f"Solution {i}: {solution}")

                majority_output = (
                    "\n".join(solution_strs) + f"\n\nTotal {len(all_solutions)} feasible solutions shown above."
                )
        else:
            majority_output = "No solutions found."

        return pd.Series(
            {
                "majority_solutions": all_solutions,
                "majority_confidence": majority_confidence,
                "majority_output": majority_output,
                "majority_solution_count": len(all_solutions),
            }
        )

    # 为每个question_id和answer_count_bin组合计算majority voting结果
    majority_results = (
        df.groupby(["question_id", "answer_count_bin"]).apply(calculate_majority_voting_solutions).reset_index()
    )

    # 将majority voting结果合并回原DataFrame
    df = df.merge(majority_results, on=["question_id", "answer_count_bin"], how="left")

    # 标记不同方法
    df["method"] = "Long-CoT"
    # df.loc[df["model_thinking_response"] == df["max_model_thinking_response"], "method"] = (
    #     "Self-Consistency (max thinking)"
    # )
    # df.loc[df["model_confidence_extracted"] == df["max_model_confidence"], "method"] = (
    #     "Self-Consistency (max confidence)"
    # )
    # df.loc[df["model_confidence_extracted"] == df["min_model_confidence"], "method"] = (
    #     "Self-Consistency (min confidence)"
    # )
    df.loc[df["confidence_diff"] == df["min_confidence_diff"], "method"] = "Self-Consistency (median confidence)"

    # 为majority voting创建特殊的数据行
    majority_voting_rows = []
    for _, group in df.groupby(["question_id", "answer_count_bin"]):
        # 取第一行作为模板
        template_row = group.iloc[0].copy()

        # 更新关键字段
        template_row["method"] = "Self-Consistency (majority voting)"
        majority_output = group.iloc[0]["majority_output"]
        template_row["model_confidence_extracted"] = group.iloc[0]["majority_confidence"]
        template_row["turn"] = -1  # 特殊标记表示这是majority voting结果

        # 创建新的chat_history，将majority voting结果放入其中
        original_chat_history = template_row["chat_history"]
        new_chat_history = original_chat_history.copy()
        if len(new_chat_history) >= 2:
            new_chat_history[1]["content"] = majority_output
        template_row["chat_history"] = new_chat_history

        # 重新计算precision和recall
        from confidence.evaluate import compute_precision_recall

        pr_results = compute_precision_recall(
            majority_output, template_row["answers"]["0"], template_row["answer_count"], dataset
        )
        template_row["precision"] = pr_results["precision"]
        template_row["recall"] = pr_results["recall"]
        template_row["correct_solution_count"] = pr_results["correct_solution_count"]
        template_row["total_solution_count"] = pr_results["total_solution_count"]

        majority_voting_rows.append(template_row)

    # 将majority voting行添加到DataFrame
    if majority_voting_rows:
        majority_df = pd.DataFrame(majority_voting_rows)
        df = pd.concat([df, majority_df], ignore_index=True)

    # 输出各种方法的性能指标
    methods = [
        ("Long-CoT", df[df["turn"] == 0].copy()),
        # (
        #     "Self-Consistency (max thinking)",
        #     df[df["model_thinking_response"] == df["max_model_thinking_response"]].copy(),
        # ),
        # (
        #     "Self-Consistency (max confidence)",
        #     df[df["model_confidence_extracted"] == df["max_model_confidence"]].copy(),
        # ),
        # (
        #     "Self-Consistency (min confidence)",
        #     df[df["model_confidence_extracted"] == df["min_model_confidence"]].copy(),
        # ),
        ("Self-Consistency (median confidence)", df[df["confidence_diff"] == df["min_confidence_diff"]].copy()),
        ("Self-Consistency (majority voting)", df[df["method"] == "Self-Consistency (majority voting)"].copy()),
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
    dataset = DatasetName.TimeTabling
    temperature = 0.2
    model = ModelName.QWEN3_8B_THINK
    template = "simple"

    await generate_self_consistency_data(
        dataset=dataset, model=model, template=template, temperature=temperature, turns_range=range(0, 32)
    )


if __name__ == "__main__":
    run_async(main())
