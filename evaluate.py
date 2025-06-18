import asyncio
import os
import re
from enum import Enum

from loguru import logger
from result import Err, Ok, Result
from sglang.utils import launch_server_cmd, terminate_process, wait_for_server
from tap import Tap
from tortoise import run_async
from tqdm.auto import tqdm

from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import Model, ModelName
from confidence.template import Template, TimeTablingTemplate, string_to_template
from confidence.utils import limit_concurrency, split_thinking_answer


class FakeType(Enum):
    less = "less"
    more = "more"
    none = "none"

    def __str__(self) -> str:
        return self.value


class Argument(Tap):
    model: ModelName = ModelName.QWEN3_8B_THINK
    judge_model: ModelName = ModelName.QWEN3_32B_NO_THINK
    judge_model_name_or_path: str = "Qwen/Qwen3-32B"
    dataset: DatasetName = DatasetName.TimeTabling
    template: Template = TimeTablingTemplate.simple
    method: MethodName = MethodName.Verbal_0_100
    temperature: float = 0.2
    fake_type: FakeType = FakeType.less
    no_cot_memory: bool = True
    concurrency: int = 200
    turn: int = 0
    debug: bool = False

    def configure(self) -> None:
        self.add_argument("--template", type=string_to_template)


async def evaluate(judge_model: Model, record: dict) -> Result[tuple[int, int, dict], str]:
    judge_prompt = (
        "You are now a large language model referee. "
        "I will provide you with a multi-solution question and the answers from another model."
        "The model's answers may be wrong, incomplete, or all correct."
        "You need to judge model's answers one by one.\n\n"
        "Multi-solution question:\n"
        f"{record['question']}\n"
        "Model answers:\n"
        f"{record['model_answer_extracted']}\n\n"
        "Please tell me how many solutions the model obtained for this question. "
        "According to the constraints in the question, is each solution of the model correct? "
        "Please verify each solution using the constraints one by one, and output in format: \\boxed{x/y}, "
        "where x is the number of correct solutions output by the model "
        "and y is the total number of solutions output by the model. "
        "Besides, x or y can be 0 if no solution is output by the model."
    )
    if judge_model.model_name in [ModelName.QWEN3_8B_THINK, ModelName.QWEN3_32B_THINK]:
        messages = [{"role": "user", "content": judge_prompt + " /think"}]
    elif judge_model.model_name in [ModelName.QWEN3_8B_NO_THINK, ModelName.QWEN3_32B_NO_THINK]:
        messages = [{"role": "user", "content": judge_prompt + " /no_think"}]
    else:
        messages = [{"role": "user", "content": judge_prompt}]
    judge_response_result = await judge_model.request(messages=messages)
    if judge_response_result.is_err():
        return judge_response_result

    _, judge_response = split_thinking_answer(judge_response_result.ok_value.message_content)
    matches = [(int(x), int(y)) for x, y in re.findall(r"\\boxed{(\d+)/(\d+)}", judge_response)]
    if len(matches) < 1:
        return Err(f"No solution count found in judge model response: {judge_response}")
    correct_count, total_count = matches[0][0], matches[0][1]
    if correct_count > total_count:
        return Err(f"Correct count {correct_count} cannot be greater than total count {total_count}\n{judge_response}")

    return Ok((correct_count, total_count, record))


async def main(args: Argument):
    judge_model = Model(args.judge_model, args.judge_model_name_or_path)
    record_cls = args.dataset.record_cls
    if args.fake_type != FakeType.none:
        table_name = f"{args.dataset}--{args.method}--no-cot-memory-{args.no_cot_memory}--{args.template}--{args.model}--{args.temperature}--{args.fake_type}-reflection"
    else:
        table_name = f"{args.dataset}--{args.method}--no-cot-memory-{args.no_cot_memory}--{args.template}--{args.model}--{args.temperature}"
    if args.debug:
        db_name = "debug"
    else:
        db_name = f"{args.dataset.value}--{args.model}--{args.template}--turn{args.turn}"
    db_logger = Logger(db_name=db_name, table_name=table_name, record_cls=record_cls)
    async with db_logger:
        records = await db_logger.fetch()
        records = [record.model_dump() for record in records]

    tasks = [evaluate(judge_model=judge_model, record=record) for record in records]
    tasks = limit_concurrency(tasks, args.concurrency)

    db_logger = Logger(
        db_name=db_name,
        table_name=f"{table_name}--evaluate-by-{args.judge_model}",
        record_cls=record_cls,
    )
    async with db_logger:
        async for zip_response_result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
            zip_response_result: Result[tuple[int, int, dict], str] = await zip_response_result
            if zip_response_result.is_err():
                logger.error(zip_response_result.err_value)
                continue

            correct_count: int
            total_count: int
            record: dict
            correct_count, total_count, record = zip_response_result.ok_value
            await db_logger.insert(record_cls(**{**record, "eval_result": f"{correct_count}/{total_count}"}))


if __name__ == "__main__":
    args = Argument().parse_args()

    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    server_process, port = launch_server_cmd(
        (
            "python3 -m sglang.launch_server "
            "--tp 8 "
            "--dp 1 "
            f"--model-path {args.judge_model_name_or_path} "
            f"--served-model-name {args.model} "
            "--reasoning-parser qwen3 "
            "--context-length 131072 "
            """--json-model-override-args {"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}} """
            "--host 0.0.0.0 "
            "--port 33333"
        )
    )
    wait_for_server(f"http://localhost:{port}")

    os.environ["BASE_URL"] = f"http://localhost:{port}/v1"
    os.environ["API_KEY"] = "sglang"

    run_async(main(args))

    terminate_process(server_process)
