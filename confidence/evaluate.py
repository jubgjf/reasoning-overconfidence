import re

import pandas as pd

from .dataset import DatasetName
from .result import Result


def extract_timetabling_solutions(s: str) -> set[str]:
    """
    从模型生成的内容中提取TimeTabling任务的所有解决方案

    Args:
        s: 模型生成的字符串内容

    Returns:
        set[str]: 包含所有解决方案的集合，每个解决方案表示为标准化字符串
    """
    solutions = set()

    # 匹配解决方案的模式：Solution X: 后面跟着表格
    # 修改正则表达式以更灵活地匹配表格内容
    solution_pattern = r"Solution\s+\d+:\s*\n((?:\|.*?\|.*?\|.*?\|.*?\|\s*\n?)+)"

    # 查找所有解决方案块
    solution_matches = re.finditer(solution_pattern, s, re.IGNORECASE | re.MULTILINE | re.DOTALL)

    for match in solution_matches:
        table_content = match.group(1).strip()

        # 解析表格内容
        solution = parse_timetabling_table(table_content)
        if solution:
            # 将解决方案转换为标准化字符串形式
            normalized_solution = normalize_timetabling_solution(solution)
            solutions.add(normalized_solution)

    return solutions


def parse_timetabling_table(table_content: str) -> dict[str, dict[str, str]] | None:
    """
    解析表格内容，提取课程分配信息

    Args:
        table_content: 表格内容字符串

    Returns:
        dict[str, dict[str, str]] | None: 课程分配字典，格式为 {course: {time, room, teacher}}，失败时返回None
    """
    lines = table_content.strip().split("\n")
    solution = {}

    for line_num, line in enumerate(lines):
        # 跳过没有管道符的行
        if "|" not in line:
            continue

        # 跳过分隔线
        if "---" in line or "===" in line:
            continue

        # 跳过表头行（只跳过包含 "Course" 且包含 "Time" 的行）
        if "Course" in line and "Time" in line and "Room" in line and "Teacher" in line:
            continue

        # 提取表格行数据
        cells = [cell.strip() for cell in line.split("|") if cell.strip()]

        if len(cells) >= 4:
            course = cells[0].strip()
            time = cells[1].strip()
            room = cells[2].strip()
            teacher = cells[3].strip()

            # 过滤掉无效的行
            if course and time and room and teacher:
                solution[course] = {"time": time, "room": room, "teacher": teacher}

    return solution if solution else None


def normalize_timetabling_solution(solution: dict[str, dict[str, str]]) -> str:
    """
    将解决方案标准化为字符串形式，便于去重和比较

    Args:
        solution: 课程分配字典

    Returns:
        str: 标准化的解决方案字符串
    """
    # 按课程名排序，确保相同解决方案的字符串表示一致
    sorted_courses = sorted(solution.keys())

    solution_parts = []
    for course in sorted_courses:
        assignment = solution[course]
        part = f"{course}:{assignment['time']},{assignment['room']},{assignment['teacher']}"
        solution_parts.append(part)

    return ";".join(solution_parts)


def extract_all_solutions(s: str) -> set[str]:
    """
    通用的解决方案提取函数（目前只支持TimeTabling）

    Args:
        s: 模型生成的字符串内容

    Returns:
        set[str]: 包含所有解决方案的集合
    """
    return extract_timetabling_solutions(s)


def extract_total_count_from_text(s: str) -> int | None:
    """
    从文本中提取模型声明的总解决方案数量

    Args:
        s: 模型生成的字符串内容

    Returns:
        int | None: 模型声明的总数量，如果找不到则返回None
    """
    # 匹配 "Total xxx feasible solutions" 格式
    total_pattern = r"Total\s+(\d+)\s+feasible\s+solutions?"
    match = re.search(total_pattern, s, re.IGNORECASE)

    if match:
        return int(match.group(1))

    # 尝试其他可能的格式
    alt_patterns = [
        r"(\d+)\s+feasible\s+solutions?\s+shown\s+above",
        r"Total:\s*(\d+)",
        r"总共\s*(\d+)\s*个",
        r"共\s*(\d+)\s*种",
    ]

    for pattern in alt_patterns:
        match = re.search(pattern, s, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def compare_solution_counts(model_output: str) -> dict[str, int]:
    """
    比较模型声明的数量和实际提取的解决方案数量

    Args:
        model_output: 模型生成的字符串内容

    Returns:
        dict[str, int]: 包含 'declared_count', 'extracted_count', 'unique_count' 的字典
    """
    declared_count = extract_total_count_from_text(model_output)
    extracted_solutions = extract_all_solutions(model_output)
    unique_count = len(extracted_solutions)

    return {
        "declared_count": declared_count if declared_count is not None else -1,
        "extracted_count": unique_count,  # 由于使用set，extracted_count等于unique_count
        "unique_count": unique_count,
    }


def extract_predicted_count(model_output: str) -> int | None:
    """
    从模型输出中提取预测的解决方案总数

    Args:
        model_output: 模型生成的字符串内容

    Returns:
        int | None: 模型预测的总数，如果找不到则返回None
    """
    # 尝试多种可能的格式
    patterns = [
        r"The total number of feasible schedules is \*\*(\d+)\*\*",
        r"Total \\\boxed\{(\d+)\} feasible solutions",
        r"(\d+) unique feasible schedules",
        r"Total:\s*(\d+)",
        r"总共\s*(\d+)\s*个",
        r"共\s*(\d+)\s*种",
        r"The number of feasible schedules is \*\*(\d+)\*\*",
        r"there are \*\*(\d+)\*\* unique feasible schedules",
        r"\\boxed\{(\d+)\}",  # 添加boxed格式
        r"Total (\d+) feasible",
        r"answer is (\d+)",
        r"total of (\d+)",
        r"exactly (\d+)",
        r"(\d+) feasible solutions",
        r"(\d+) different",
        r"(\d+) valid",
        r"(\d+) possible",
    ]

    for pattern in patterns:
        match = re.search(pattern, model_output, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def evaluate_count_prediction(ground_truth_count: int, model_output: str) -> dict:
    """
    评估模型预测的总数与真实总数的比较

    Args:
        ground_truth_count: 真实的解决方案总数
        model_output: 模型输出字符串

    Returns:
        dict: 包含预测准确性的评估结果
    """
    predicted_count = extract_predicted_count(model_output)

    result = {
        "ground_truth_count": ground_truth_count,
        "predicted_count": predicted_count,
        "count_available": predicted_count is not None,
        "count_correct": False,
        "count_error": None,
        "count_error_rate": None,
    }

    if predicted_count is not None:
        result["count_correct"] = predicted_count == ground_truth_count
        result["count_error"] = abs(predicted_count - ground_truth_count)
        if ground_truth_count > 0:
            result["count_error_rate"] = result["count_error"] / ground_truth_count
        else:
            result["count_error_rate"] = float("inf")

    return result


def extract_confidence(s: str) -> Result[float, str]:
    pattern = r"\[\[CONFIDENCE: \\boxed{(100|[1-9]?[0-9])}]]"
    confidence_score_matches = re.findall(pattern, s)
    if len(confidence_score_matches) < 1:
        return Result(err=f"Confidence score not found in the response.\n{s}\n=======================================")
    return Result(ok=float(confidence_score_matches[0]) / 100)


def is_solution_changed(s: str) -> Result[bool, str]:
    change_pattern = r"\[\[CHANGE]]"
    unchange_pattern = r"\[\[UNCHANGE]]"

    if re.search(change_pattern, s):
        return Result(ok=True)
    elif re.search(unchange_pattern, s):
        return Result(ok=False)
    else:
        return Result(err="Change status not found in the response")


def compute_precision_recall(model_output: str, ground_truth_answers: str, answer_count: int) -> dict:
    """
    计算precision和recall

    Args:
        model_output: 模型生成的输出
        ground_truth_answers: 真实答案
        answer_count: 真实答案总数

    Returns:
        dict: 包含precision, recall, correct_solution_count, total_solution_count的字典
    """
    # 提取模型生成的解决方案
    model_solutions = extract_all_solutions(model_output)

    # 提取真实答案中的解决方案
    ground_truth_solutions = extract_all_solutions(ground_truth_answers)

    # 计算正确的解决方案数量（交集）
    correct_solutions = model_solutions.intersection(ground_truth_solutions)
    correct_solution_count = len(correct_solutions)

    # 模型生成的总解决方案数量
    total_solution_count = len(model_solutions)

    # 计算precision和recall
    precision = correct_solution_count / total_solution_count if total_solution_count > 0 else 0.0
    recall = correct_solution_count / answer_count if answer_count > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "correct_solution_count": correct_solution_count,
        "total_solution_count": total_solution_count,
    }


def prf(df: pd.DataFrame, dataset: DatasetName) -> pd.DataFrame:
    """
    计算precision, recall和相关指标，使用后处理方法而非legacy数据库列

    Args:
        df: 包含chat_history, answers, answer_count等列的DataFrame
        dataset: 数据集类型

    Returns:
        pd.DataFrame: 添加了precision, recall等列的DataFrame
    """
    precision_list = []
    recall_list = []
    correct_solution_count_list = []
    total_solution_count_list = []

    for _, row in df.iterrows():
        # 获取模型输出（通常在chat_history的第4个元素，即索引3）
        chat_history = row["chat_history"]
        if len(chat_history) < 4:
            # 如果chat_history长度不够，使用默认值
            precision_list.append(0.0)
            recall_list.append(0.0)
            correct_solution_count_list.append(0)
            total_solution_count_list.append(0)
            continue

        model_output = chat_history[3]["content"]

        # 获取真实答案
        answers = row["answers"]
        ground_truth_answers = answers.get("0", "") if isinstance(answers, dict) else ""
        answer_count = row["answer_count"]

        # 计算precision和recall
        metrics = compute_precision_recall(model_output, ground_truth_answers, answer_count)

        precision_list.append(metrics["precision"])
        recall_list.append(metrics["recall"])
        correct_solution_count_list.append(metrics["correct_solution_count"])
        total_solution_count_list.append(metrics["total_solution_count"])

    # 添加计算结果到DataFrame
    df = df.copy()
    df["precision"] = precision_list
    df["recall"] = recall_list
    df["correct_solution_count"] = correct_solution_count_list
    df["total_solution_count"] = total_solution_count_list

    # 过滤无效数据
    df = df[df["correct_solution_count"] <= df["total_solution_count"]]
    df = df[df["correct_solution_count"] <= df["answer_count"]]
    df = df[df["total_solution_count"] <= df["answer_count"]]

    # 创建answer_count_bin
    if dataset == DatasetName.TimeTabling:
        df["answer_count_bin"] = df["answer_count"].apply(lambda x: int(x // 50))
    elif dataset == DatasetName.SubsetSum:
        df["answer_count_bin"] = df["answer_count"].apply(lambda x: int(x // 50) if int(x // 50) < 6 else 6)
    else:
        raise NotImplementedError

    return df


def ece(df: pd.DataFrame) -> float:
    """
    计算Expected Calibration Error (ECE)，使用从chat_history提取的置信度

    Args:
        df: 包含chat_history, recall等列的DataFrame

    Returns:
        float: ECE值
    """
    # 提取置信度
    confidence_list = []
    valid_indices = []

    for idx, (_, row) in enumerate(df.iterrows()):
        chat_history = row["chat_history"]
        if len(chat_history) < 2:
            continue

        # 从第二个消息中提取置信度（索引1）
        content = chat_history[1]["content"]
        confidence_result = extract_confidence(content)

        if confidence_result.ok is not None:
            confidence_list.append(confidence_result.ok)
            valid_indices.append(idx)

    if not confidence_list:
        return 0.0

    # 创建只包含有效置信度的子DataFrame
    valid_df = df.iloc[valid_indices].copy()
    valid_df["model_confidence_extracted"] = confidence_list

    # 计算ECE
    # 注意：置信度是0-100范围，需要归一化到0-1
    confidence_normalized = valid_df["model_confidence_extracted"] / 100.0
    bins = [i / 10.0 for i in range(11)]  # [0.0, 0.1, 0.2, ..., 1.0]
    valid_df["bin"] = pd.cut(confidence_normalized, bins=bins, include_lowest=True, labels=False)
    ece = 0
    N = len(valid_df)

    for b in range(10):
        bin_data = valid_df[valid_df["bin"] == b]
        if len(bin_data) == 0:
            continue
        acc = (bin_data["recall"] == 1).mean()  # 或用 precision
        conf = confidence_normalized[bin_data.index].mean()
        ece += len(bin_data) / N * abs(acc - conf)
    return ece


def show_metrics(df: pd.DataFrame, setting_name: str):
    """
    显示评估指标，使用后处理方法计算所有指标

    Args:
        df: 包含precision, recall等列的DataFrame（已通过prf函数处理）
        setting_name: 设置名称
    """
    print(setting_name)
    metrics = {
        "Precision": df["precision"].mean(),
        "Recall": df["recall"].mean(),
        "ECE(r)": ece(df),
    }
    print(",".join(metrics.keys()))
    print(",".join(f"{value * 100:.2f}" for value in metrics.values()))
    print("==========================================")


def add_confidence_column(df: pd.DataFrame) -> pd.DataFrame:
    confidence_list = []

    for idx, (_, row) in enumerate(df.iterrows()):
        chat_history = row["chat_history"]
        content = chat_history[3]["content"]
        confidence_result = extract_confidence(content)

        if confidence_result.ok is not None:
            confidence_list.append(confidence_result.ok)
        else:
            confidence_list.append(None)

    df["model_confidence_extracted"] = confidence_list
    return df.dropna(subset=["model_confidence_extracted"])
