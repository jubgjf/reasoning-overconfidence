import asyncio
from collections.abc import Coroutine, Sequence

import git
from loguru import logger


def limit_concurrency(coroutines: Sequence[Coroutine], concurrency: int) -> list[Coroutine]:
    semaphore = asyncio.Semaphore(concurrency)

    async def with_concurrency_limit(coroutine: Coroutine) -> Coroutine:
        async with semaphore:
            return await coroutine

    return [with_concurrency_limit(coroutine) for coroutine in coroutines]


def last_git_hash() -> str:
    repo = git.Repo(".")
    commit = repo.head.commit
    return commit.hexsha[:7]


def split_thinking_answer(text: str) -> tuple[str, str]:
    """
    ```python
    split_thinking_answer("TTTTT</think>AAAAA") = ("TTTTT</think>", "AAAAA")
    split_thinking_answer("AAAAA") = ("", "AAAAA")
    ```
    """

    if text.count("</think>") > 1:
        logger.warning(f'text.count("</think>") = {text.count("</think>")}, text = {text}')

    if "</think>" in text:
        splits = text.split("</think>")
        thinking_content = "</think>".join(splits[:-1]) + "</think>"
        answer_content = splits[-1]
    else:
        thinking_content, answer_content = "", text

    return thinking_content, answer_content


def flatten[T](ll: list[list[T]]) -> list[T]:
    flattened = []
    for l in ll:
        flattened.extend(l)
    return flattened
