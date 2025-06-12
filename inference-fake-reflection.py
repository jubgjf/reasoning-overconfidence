import asyncio
import os
from enum import Enum

from loguru import logger
from sglang.utils import launch_server_cmd, terminate_process, wait_for_server
from tap import Tap
from tortoise import run_async
from tqdm.auto import tqdm

from confidence.data import Data
from confidence.dataset import DatasetName
from confidence.extractor import extract_answer_and_confidence
from confidence.logger import Logger, list_history_to_dict
from confidence.method import Method, MethodName, Response
from confidence.model import Model, ModelName
from confidence.template import Template, TimeTablingTemplate, string_to_template
from confidence.utils import flatten, last_git_hash, limit_concurrency, split_thinking_answer


class FakeType(Enum):
    less = "less"
    more = "more"

    def __str__(self) -> str:
        return self.value


class Argument(Tap):
    model: ModelName = ModelName.QWEN3_8B_THINK
    model_name_or_path: str = "Qwen/Qwen3-8B"
    dataset: DatasetName = DatasetName.TimeTabling
    template: Template = TimeTablingTemplate.simple
    method: MethodName = MethodName.Verbal_0_100
    temperature: float = 0.2
    fake_type: FakeType = FakeType.less
    no_cot_memory: bool = False
    force_update: bool = False
    concurrency: int = 5
    turn: int | None = None
    debug: bool = False

    def configure(self) -> None:
        self.add_argument("--template", type=string_to_template)


async def main(args: Argument):
    record_cls = args.dataset.record_cls

    # ===== full-reflection chat history =====
    if args.debug:
        db_name = "debug"
    elif args.turn is None:
        db_name = args.dataset.value
    else:
        db_name = f"{args.dataset.value}--turn{args.turn}"
    load_title = f"{args.dataset}--{args.method}--no-cot-memory-{args.no_cot_memory}--{args.template}--{args.model}--{args.temperature}"
    db_logger = Logger(db_name=db_name, table_name=load_title, record_cls=record_cls)
    async with db_logger:
        # ===== dataset =====
        dataset_cls = args.dataset.dataset_cls
        dataset = dataset_cls().load_processed_dataset(await db_logger.already_processed_question_ids())
        dataset = {data.question_id: data for data in dataset}

        # ===== history =====
        history = await db_logger.chat_history()
        logger.info(f"Loaded {len(history)} chat history")

        # {question_id: (data, history), ...}
        dataset_history_pair = {question_id: (dataset[question_id], turn) for question_id, turn in history.items()}

    save_title = f"{args.dataset}--{args.method}--no-cot-memory-{args.no_cot_memory}--{args.template}--{args.model}--{args.temperature}--{args.fake_type}-reflection"
    db_logger = Logger(db_name=db_name, table_name=save_title, record_cls=record_cls, force_update=args.force_update)
    async with db_logger:
        # ===== model =====
        model = Model(args.model, args.model_name_or_path)

        # ===== method =====
        method = Method(name=args.method)
        request_builder = (
            method.build_less_reflection_requests
            if args.fake_type == FakeType.less
            else method.build_more_reflection_requests
        )

        # ===== task =====
        tasks = [
            request_builder(
                model=model,
                data=data,
                template=args.template,
                temperature=args.temperature,
                history_thinking_content=split_thinking_answer(turn["assistant_0"])[0],
                max_tokens=16384,
                no_cot_memory=args.no_cot_memory,
            )
            for data, turn in dataset_history_pair.values()
        ]
        tasks = flatten(tasks)
        tasks = limit_concurrency(tasks, args.concurrency)

        async for zip_response_result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=save_title):
            zip_response_result = await zip_response_result
            if zip_response_result.is_err():
                logger.error(zip_response_result.err_value)
                continue

            data: Data
            response_result: Response
            data, response_result = zip_response_result.ok_value

            history, turn_0, turn_1 = (
                response_result.history,
                response_result.turn_0,
                response_result.turn_1,
            )
            assert history[-1]["role"] == "assistant"
            answer_and_confidence_result = extract_answer_and_confidence(
                method_name=args.method,
                dataset_name=args.dataset,
                question_turn=turn_0,
                confidence_turn=turn_1,
            )
            if answer_and_confidence_result.is_err():
                logger.error(answer_and_confidence_result.err_value)
                continue

            model_answer_extracted, confidence_extracted = answer_and_confidence_result.ok_value
            record = record_cls(
                **data.model_dump(),
                model_thinking_response=turn_0.thinking_content,
                model_answer_response=turn_0.answer_content,
                model_answer_extracted=model_answer_extracted,
                model_confidence_response=turn_1.answer_content if turn_1 is not None else "",
                model_confidence_extracted=confidence_extracted,
                template=args.template.value,
                method=args.method.value,
                history=list_history_to_dict(history),
                model=args.model.value,
                ref=f"{data.question_id}--variant",
                eval_result="",
                git_hash=last_git_hash(),
            )
            await db_logger.insert(record)


if __name__ == "__main__":
    args = Argument().parse_args()

    server_process, port = launch_server_cmd(
        (
            "python3 -m sglang.launch_server "
            "--tp 8 "
            "--dp 1 "
            f"--model-path {args.model_name_or_path} "
            f"--served-model-name {args.model} "
            "--reasoning-parser qwen3 "
            "--host 0.0.0.0 "
            "--port 33333"
        )
    )
    wait_for_server(f"http://localhost:{port}")

    os.environ["BASE_URL"] = f"http://localhost:{port}/v1"
    os.environ["API_KEY"] = "sglang"

    run_async(main(args))

    terminate_process(server_process)
