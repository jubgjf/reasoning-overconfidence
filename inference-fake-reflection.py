import asyncio
import os
from enum import Enum

from loguru import logger
from sglang.utils import launch_server_cmd, terminate_process, wait_for_server
from tap import Tap
from tortoise import run_async
from tqdm.auto import tqdm

from confidence import build_less_reflection_requests, build_more_reflection_requests
from confidence.data import Data, Template
from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.model import ChatResponse, Model, ModelName
from confidence.utils import flatten, last_git_hash, limit_concurrency


class FakeType(Enum):
    less = "less"
    more = "more"

    def __str__(self) -> str:
        return self.value


class Argument(Tap):
    model: ModelName = ModelName.QWEN3_8B_THINK
    model_name_or_path: str = "Qwen/Qwen3-8B"
    dataset: DatasetName = DatasetName.TimeTabling
    template: Template = "simple"
    temperature: float = 0.2
    fake_type: FakeType = FakeType.less
    force_update: bool = False
    concurrency: int = 5
    turn: int = 0


async def main(args: Argument):
    record_cls = args.dataset.record_cls
    load_title = f"{args.dataset}--{args.template}--{args.model}--{args.temperature}--{args.turn}"
    db_logger = Logger(db_name=load_title, table_name=load_title, record_cls=record_cls)
    async with db_logger:
        # ===== dataset =====
        dataset_cls = args.dataset.dataset_cls
        dataset = dataset_cls().load_processed_dataset(await db_logger.already_processed_question_ids())
        dataset = {data.question_id: data for data in dataset}

        # ===== history =====
        history = await db_logger.history()
        logger.info(f"Loaded {len(history)} chat history")

        # {question_id: (data, chat_history, thinking_history), ...}
        dataset_history_pair = {
            question_id: (dataset[question_id], chat_history, thinking_history)
            for question_id, (chat_history, thinking_history) in history.items()
        }

    save_title = f"{args.dataset}--{args.template}--{args.model}--{args.temperature}--{args.turn}--{args.fake_type}"
    db_logger = Logger(db_name=save_title, table_name=save_title, record_cls=record_cls, force_update=args.force_update)
    async with db_logger:
        # ===== model =====
        model = Model(args.model, args.model_name_or_path)

        request_builder = (
            build_less_reflection_requests if args.fake_type == FakeType.less else build_more_reflection_requests
        )

        # ===== task =====
        tasks = [
            request_builder(
                model=model,
                data=data,
                template=args.template,
                chat_history=chat_history,
                thinking_history=thinking_history,
                temperature=args.temperature,
                max_completion_tokens=32768,
            )
            for data, chat_history, thinking_history in dataset_history_pair.values()
        ]
        tasks = flatten(tasks)
        tasks = limit_concurrency(tasks, args.concurrency)

        async for zip_response_result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=save_title):
            zip_response_result = await zip_response_result
            if zip_response_result.is_err():
                logger.error(zip_response_result.err_value)
                continue

            data: Data
            response_result: ChatResponse
            data, response_result = zip_response_result.ok_value

            thinking = response_result.thinking
            record = record_cls(
                **data.model_dump(),
                chat_history=response_result.messages,
                thinking_history=thinking if thinking is not None else [],
                model=args.model.value,
                dataset=args.dataset.value,
                template=args.template,
                temperature=args.temperature,
                git_hash=last_git_hash(),
            )
            await db_logger.insert(record)


if __name__ == "__main__":
    args = Argument().parse_args()

    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    server_process, port = launch_server_cmd(
        (
            "python3 -m sglang.launch_server "
            "--tp 8 "
            "--dp 1 "
            f"--model-path {args.model_name_or_path} "
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
