import random
import re
from collections.abc import Coroutine
from copy import deepcopy
from typing import Any

from loguru import logger as loguru_logger

from .data import Data, Template
from .model import ChatResponse, Model, ModelName
from .result import Result
from .utils import split_thinking_answer


async def request(
    model: Model,
    data: Data,
    template: Template,
    temperature: float = 0,
    max_completion_tokens: int = 32768,
) -> Result[ChatResponse, str]:
    """
    turn 0: Ask for model confidence
    turn 1: Ask for answer
    turn 2: Ask for model confidence again
    turn 3: Let model think again
    """

    if model.model_name in [ModelName.QWEN3_8B_THINK, ModelName.QWEN3_32B_THINK]:
        suffix = " /think"
    elif model.model_name in [ModelName.QWEN3_8B_NO_THINK, ModelName.QWEN3_32B_NO_THINK]:
        suffix = " /no_think"
    else:
        suffix = ""

    messages = [
        {
            "role": "user",
            "content": (
                f"{data.task_description()}\n"
                "\n"
                "\n"
                "Before answering the question, "
                "please first rate your confidence to find all feasible solutions on a scale of 0-100.\n"
                f"Put your confidence score in format: [[CONFIDENCE: \\boxed{{}}]].{suffix}"
            ),
        },
        {"role": "assistant", "content": "[[ASSISTANT]]"},
        {"role": "user", "content": f"{data.ask_for_solve(template)}{suffix}"},
        {"role": "assistant", "content": "[[ASSISTANT]]"},
        {
            "role": "user",
            "content": (
                "Check your thinking process, "
                f"give your confidence score in format [[CONFIDENCE: \\boxed{{}}]] again.{suffix}"
            ),
        },
        {"role": "assistant", "content": "[[ASSISTANT]]"},
        {
            "role": "user",
            "content": (
                "Recheck all your answers. You can now supplement and correct your answers.\n"
                "If you think your answer does not need to be changed, please output [[UNCHANGE]].\n"
                "If you need to supplement or correct your answer, please input [[CHANGE]] "
                f"and re-output your new answer IN FULL, NOT JUST THE PART YOU CHANGED.{suffix}"
            ),
        },
        {"role": "assistant", "content": "[[ASSISTANT]]"},
    ]

    response_result = await model.chat(
        messages=messages, temperature=temperature, max_completion_tokens=max_completion_tokens
    )
    return response_result


async def _request_fake_reflection(
    model: Model,
    prompt: str,
    thinking_history: list[str],
    data: Data,
    template: Template,
    temperature: float = 0,
    max_completion_tokens: int = 32768,
) -> Result[tuple[Data, ChatResponse], str]:
    """
    turn 0: Ask for model confidence
    turn 1: Ask for answer                 <- `model.complete`
    turn 2: Ask for model confidence again <- `model.chat`
    turn 3: Let model think again          <- `model.chat`

    Args:
        prompt:
            For Long-CoT, prompt example: (<think> and </think> exist in turn 1)
            ```
            <|im_start|>user
            turn 0 ...<|im_end|>
            <|im_start|>assistant
            turn 0 ...<|im_end|>
            <|im_start|>user
            turn 1 ...<|im_end|>
            <|im_start|>assistant
            <think>turn 1 ...</think>
            ```

            For Short-CoT, prompt example:
            ```
            <|im_start|>user
            turn 0 ...<|im_end|>
            <|im_start|>assistant
            turn 0 ...<|im_end|>
            <|im_start|>user
            turn 1 ...<|im_end|>
            <|im_start|>assistant
            turn 1 ...
            ```
    """

    # ===== turn 1 (model.complete) =====
    response_result = await model.complete(
        prompt=prompt, max_tokens=int(max_completion_tokens * 2 / 3), temperature=temperature
    )
    if response_result.is_err():
        return Result(err=f"Found err in fake reflection completion: {response_result.err}")
    turn_0_thinking_content = "" if len(thinking_history) == 0 else thinking_history[0]
    turn_0_answer_content = prompt.split("<|im_start|>assistant\n")[1].split("<|im_end|>\n")[0]
    turn_1_model_output = prompt.split("<|im_start|>assistant\n")[-1] + response_result.ok_value
    turn_1_thinking_content, turn_1_answer_content = split_thinking_answer(turn_1_model_output)

    if model.model_name in [ModelName.QWEN3_8B_THINK, ModelName.QWEN3_32B_THINK]:
        suffix = " /think"
    elif model.model_name in [ModelName.QWEN3_8B_NO_THINK, ModelName.QWEN3_32B_NO_THINK]:
        suffix = " /no_think"
    else:
        suffix = ""

    messages = [
        {
            "role": "user",
            "content": (
                f"{data.task_description()}\n"
                "\n"
                "\n"
                "Before answering the question, "
                "please first rate your confidence to find all feasible solutions on a scale of 0-100.\n"
                f"Put your confidence score in format: [[CONFIDENCE: \\boxed{{}}]].{suffix}"
            ),
        },
        {"role": "assistant", "content": turn_0_answer_content},
        {"role": "user", "content": f"{data.ask_for_solve(template)}{suffix}"},
        {"role": "assistant", "content": turn_1_answer_content},
        {
            "role": "user",
            "content": (
                "Check your thinking process, "
                f"give your confidence score in format [[CONFIDENCE: \\boxed{{}}]] again.{suffix}"
            ),
        },
        {"role": "assistant", "content": "[[ASSISTANT]]"},
        {
            "role": "user",
            "content": (
                "Recheck all your answers. You can now supplement and correct your answers.\n"
                "If you think your answer does not need to be changed, please output [[UNCHANGE]].\n"
                "If you need to supplement or correct your answer, please input [[CHANGE]] "
                f"and re-output your new answer IN FULL, NOT JUST THE PART YOU CHANGED.{suffix}"
            ),
        },
        {"role": "assistant", "content": "[[ASSISTANT]]"},
    ]
    response_result = await model.chat(
        messages=messages, temperature=temperature, max_completion_tokens=max_completion_tokens
    )
    if response_result.is_err():
        return Result(err=response_result.err)

    assert len(response_result.ok_value.messages) == 8  # Total 4 turns, each turn has 2 messages (user and assistant)
    if model.model_name in [ModelName.QWEN3_8B_THINK, ModelName.QWEN3_32B_THINK]:
        assert turn_1_thinking_content != ""
        assert response_result.ok_value.thinking is not None
        assert len(response_result.ok_value.thinking) == 2  # Only turn 2 and turn 3 have thinking content
        thinking = [
            turn_0_thinking_content,
            turn_1_thinking_content,
            response_result.ok_value.thinking[0],
            response_result.ok_value.thinking[1],
        ]
    else:
        # assert thinking_content == ""
        if turn_1_thinking_content != "":
            loguru_logger.warning("Short-CoT model should not thinking, but got thinking content. This is unexpected.")
        assert response_result.ok_value.thinking is None
        thinking = None

    return Result(ok=(data, ChatResponse(messages=response_result.ok_value.messages, thinking=thinking)))


def build_less_reflection_requests(
    model: Model,
    data: Data,
    template: Template,
    chat_history: list[dict[str, str]],
    thinking_history: list[str],
    temperature: float = 0,
    max_completion_tokens: int = 32768,
) -> list[Coroutine[Any, Any, Result[tuple[Data, ChatResponse], str]]]:
    """
    Long-CoT only

    turn 0: Ask for model confidence          <- keep unchanged
    turn 1: Ask for answer                    <- split reflection
    turn 2: Ask for model confidence again    <- delete and chat again
    turn 3: Let model think again             <- delete and chat again
    """

    assert model.model_name in [ModelName.QWEN3_8B_THINK, ModelName.QWEN3_32B_THINK], "Only support Long-CoT model"

    messages = deepcopy(chat_history)
    messages = messages[:4]

    turn_1_thinking = thinking_history[1]
    assert turn_1_thinking != ""
    assert "</think>" in turn_1_thinking
    assert "<think>" not in turn_1_thinking

    reflection_patterns = [
        r"^Wait,.*\n\n",
        r"^Let me double-check.*\n\n",
        r"^Let me think again.*\n\n",
    ]
    combined_pattern = "|".join(reflection_patterns)

    # thinking_steps_by_reflection =
    #     0: thinking...  1: reflection...
    #     2: thinking...  3: reflection...
    #     4: thinking ...                    # Last step must not be reflection
    last_step_start_index, thinking_steps_by_reflection = 0, []
    for m in re.finditer(combined_pattern, turn_1_thinking, re.M):
        thinking_steps_by_reflection.append(turn_1_thinking[last_step_start_index : m.start()])
        thinking_steps_by_reflection.append(m.group())
        last_step_start_index = m.end()
    thinking_steps_by_reflection.append(turn_1_thinking[last_step_start_index:])
    if len(turn_1_thinking) == last_step_start_index:
        # Last step is reflection, although this might be impossible. Remove it.
        thinking_steps_by_reflection = thinking_steps_by_reflection[:-2]

    thinking_with_reduced_reflection = []
    for i in range(0, len(thinking_steps_by_reflection), 2):
        thinking_with_reduced_reflection.append(thinking_steps_by_reflection[: i + 1])

    # prompt_with_less_reflection_list[i] =
    #     <|im_start|>user
    #     turn 0 ...<|im_end|>
    #     <|im_start|>assistant
    #     turn 0 ...<|im_end|>
    #     <|im_start|>user
    #     turn 1 ...<|im_end|>
    #     <|im_start|>assistant
    #     <think>turn 1 ...(less reflection)</think>
    prompt = model.apply_chat_template(messages[:3])
    prompt_with_less_reflection_list = [
        prompt + "<think>" + "".join(steps) + "</think>" for steps in thinking_with_reduced_reflection
    ]

    # Input for model is `prompt_with_less_reflection_list[i]`
    # Now the model need to run `model.complete` to complete turn 1
    # Then run `model.chat` to complete turn 2 and turn 3
    return [
        _request_fake_reflection(
            model=model,
            thinking_history=thinking_history,
            prompt=prompt,
            data=data,
            template=template,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        for prompt in prompt_with_less_reflection_list
    ]


def build_more_reflection_requests(
    model: Model,
    data: Data,
    template: Template,
    chat_history: list[dict[str, str]],
    thinking_history: list[str],
    temperature: float = 0,
    max_completion_tokens: int = 32768,
) -> list[Coroutine[Any, Any, Result[tuple[Data, ChatResponse], str]]]:
    """
    Short-CoT only

    turn 0: Ask for model confidence          <- keep unchanged
    turn 1: Ask for answer                    <- append reflection to assistant answer
    turn 2: Ask for model confidence again    <- delete and chat again
    turn 3: Let model think again             <- delete and chat again
    """

    assert model.model_name not in [ModelName.QWEN3_8B_THINK, ModelName.QWEN3_32B_THINK], "Not support Long-CoT model"

    messages = deepcopy(chat_history)
    messages = messages[:4]

    # Insert a random reflection pattern after Short-CoT
    reflection_patterns = [
        "Wait, there may be other solutions.",
        "Let me double-check if there is any other solution.",
        "Let me think again if there is any other solution.",
    ]
    turn_1_assistant_answer = messages[3]["content"]
    turn_1_assistant_answer_with_more_reflection = turn_1_assistant_answer + "\n\n" + random.choice(reflection_patterns)

    # prompt_with_more_reflection =
    #     <|im_start|>user
    #     turn 0 ...<|im_end|>
    #     <|im_start|>assistant
    #     turn 0 ...<|im_end|>
    #     <|im_start|>user
    #     turn 1 ...<|im_end|>
    #     <|im_start|>assistant
    #     turn 1 ...
    #     Wait, there may be other solutions.
    prompt = model.apply_chat_template(messages[:3])
    prompt_with_more_reflection = prompt + turn_1_assistant_answer_with_more_reflection

    # Input for model is `prompt_with_more_reflection`
    # Now the model need to run `model.complete` to complete turn 1
    # Then run `model.chat` to complete turn 2 and turn 3
    return [
        _request_fake_reflection(
            model=model,
            prompt=prompt_with_more_reflection,
            thinking_history=thinking_history,
            data=data,
            template=template,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
    ]
