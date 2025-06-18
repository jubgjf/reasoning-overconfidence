import random
import re
from enum import Enum
from typing import assert_never, Any, Coroutine

from loguru import logger
from openai.types.chat import ChatCompletionTokenLogprob
from openai.types.completion_choice import Logprobs
from pydantic import BaseModel
from result import Err, Ok, Result

from .data import Data
from .model import CompleteAPIResponse, Model, ModelName
from .template import Template
from .utils import split_thinking_answer, split_thinking_answer_logprobs


class MethodName(Enum):
    Verbal_0_100 = "verbal_0_100"
    P_True = "p_true"

    def __str__(self) -> str:
        return self.value

    @property
    def need_another_turn(self) -> bool:
        return self in [self.Verbal_0_100, self.P_True]

    @property
    def need_logprobs(self) -> bool:
        return self in [self.P_True]

    @property
    def prompt(self) -> str | None:
        if self == self.Verbal_0_100:
            return "Please rate your confidence in the proposed answer on a scale of 0-100. Give your confidence in format: [[xx]]"
        elif self == self.P_True:
            return "Please rate your confidence in the proposed answer as either 0 or 1. Give your confidence in format: [[x]]"
        else:
            assert_never(self)


class ChatResponsePerTurn(BaseModel):
    thinking_content: str | None
    answer_content: str
    thinking_logprobs: list[ChatCompletionTokenLogprob] | None
    answer_logprobs: list[ChatCompletionTokenLogprob] | None


class CompleteResponsePerTurn(BaseModel):
    thinking_content: str | None
    answer_content: str
    thinking_logprobs: Logprobs | None
    answer_logprobs: Logprobs | None


class Response(BaseModel):
    history: list[dict[str, str]]
    turn_0: ChatResponsePerTurn | CompleteResponsePerTurn
    turn_1: ChatResponsePerTurn | None


class Method:
    def __init__(self, name: MethodName):
        self._name = name

    @staticmethod
    def _validate_text_vs_tokens(text: str, tokens: list[ChatCompletionTokenLogprob] | Logprobs | None):
        if tokens is None:
            return

        if isinstance(tokens, list):
            text_from_tokens = "".join([t.token for t in tokens])
        elif isinstance(tokens, Logprobs):
            text_from_tokens = "".join(tokens.tokens)
        else:
            assert_never(tokens)

        # Try eliminating unicode chars
        unicode_chars, broken_char = ["÷", "≈"], "��"
        for c in unicode_chars:
            if c in text:
                text_from_tokens = text_from_tokens.replace(broken_char, c)
        if text != text_from_tokens:
            logger.warning("Text/token mismatch:")
            logger.warning(f"Text  = {text.replace('\n', '\\n')}")
            logger.warning(f"Token = {text_from_tokens.replace('\n', '\\n')}")
            if "�" in text_from_tokens:
                logger.warning("Broken Unicode character (�) found, might be normal")
            else:
                logger.error("Broken Unicode character (�) not found, might be abnormal")

    async def request(
        self,
        model: Model,
        data: Data,
        template: Template,
        temperature: float = 0,
        max_tokens: int = 16384,
        no_cot_memory: bool = False,
    ) -> Result[Response, str]:
        # ===== First turn =====
        if model.model_name in [ModelName.QWEN3_8B_THINK, ModelName.QWEN3_32B_THINK]:
            messages = [{"role": "user", "content": template.prompt(data) + " /think"}]
        elif model.model_name in [ModelName.QWEN3_8B_NO_THINK, ModelName.QWEN3_32B_NO_THINK]:
            messages = [{"role": "user", "content": template.prompt(data) + " /no_think"}]
        else:
            messages = [{"role": "user", "content": template.prompt(data)}]
        response_result = await model.request(
            messages=messages, temperature=temperature, max_tokens=max_tokens, logprobs=self._name.need_logprobs
        )
        if response_result.is_err():
            return response_result
        messages.append({"role": "assistant", "content": response_result.ok_value.message_content})

        thinking_content, answer_content = split_thinking_answer(response_result.ok_value.message_content)
        thinking_logprobs, answer_logprobs = split_thinking_answer_logprobs(response_result.ok_value.logprobs_content)
        self._validate_text_vs_tokens(thinking_content, thinking_logprobs)
        self._validate_text_vs_tokens(answer_content, answer_logprobs)

        turn_0 = ChatResponsePerTurn(
            thinking_content=thinking_content,
            answer_content=answer_content,
            thinking_logprobs=thinking_logprobs.copy() if thinking_logprobs is not None else None,
            answer_logprobs=answer_logprobs.copy() if answer_logprobs is not None else None,
        )
        if not self._name.need_another_turn:
            assert len(messages) == 2
            return Ok(Response(history=messages, turn_0=turn_0, turn_1=None))

        # ===== Second turn (Optional) =====
        if no_cot_memory:
            messages[1]["content"] = answer_content

        if model.model_name in [ModelName.QWEN3_8B_THINK, ModelName.QWEN3_32B_THINK]:
            messages.append({"role": "user", "content": self._name.prompt + " /think"})
        elif model.model_name in [ModelName.QWEN3_8B_NO_THINK, ModelName.QWEN3_32B_NO_THINK]:
            messages.append({"role": "user", "content": self._name.prompt + " /no_think"})
        else:
            messages.append({"role": "user", "content": self._name.prompt})
        response_result = await model.request(
            messages=messages, temperature=temperature, max_tokens=max_tokens, logprobs=self._name.need_logprobs
        )
        if response_result.is_err():
            return response_result
        messages.append({"role": "assistant", "content": response_result.ok_value.message_content})

        thinking_content, answer_content = split_thinking_answer(response_result.ok_value.message_content)
        thinking_logprobs, answer_logprobs = split_thinking_answer_logprobs(response_result.ok_value.logprobs_content)
        self._validate_text_vs_tokens(thinking_content, thinking_logprobs)
        self._validate_text_vs_tokens(answer_content, answer_logprobs)

        turn_1 = ChatResponsePerTurn(
            thinking_content=thinking_content,
            answer_content=answer_content,
            thinking_logprobs=thinking_logprobs.copy() if thinking_logprobs is not None else None,
            answer_logprobs=answer_logprobs.copy() if answer_logprobs is not None else None,
        )
        assert len(messages) == 4
        return Ok(Response(history=messages, turn_0=turn_0, turn_1=turn_1))

    async def _request_fake_reflection(
        self,
        model: Model,
        user_input: str,
        prompt: str,
        data: Data,
        template: Template,
        temperature: float = 0,
        max_tokens: int = 16384,
        no_cot_memory: bool = False,
    ) -> Result[tuple[Data, Response], str]:
        # ===== First turn =====
        response_result: Result[CompleteAPIResponse, str]
        response_result = await model.complete(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature, logprobs=self._name.need_logprobs
        )
        if response_result.is_err():
            return Err(f"Found err in fake reflection completion: {response_result.err_value}")

        if model.model_name in [ModelName.QWEN3_8B_THINK, ModelName.QWEN3_32B_THINK]:
            messages = [{"role": "user", "content": template.prompt(data) + " /think"}]
        elif model.model_name in [ModelName.QWEN3_8B_NO_THINK, ModelName.QWEN3_32B_NO_THINK]:
            messages = [{"role": "user", "content": template.prompt(data) + " /no_think"}]
        else:
            messages = [{"role": "user", "content": template.prompt(data)}]

        model_output = (prompt + response_result.ok_value.text_content)[len(user_input) :]
        messages.append({"role": "assistant", "content": model_output})

        thinking_content, answer_content = split_thinking_answer(model_output)
        thinking_logprobs, answer_logprobs = split_thinking_answer_logprobs(response_result.ok_value.logprobs_content)
        self._validate_text_vs_tokens(thinking_content, thinking_logprobs)
        self._validate_text_vs_tokens(answer_content, answer_logprobs)

        turn_0 = CompleteResponsePerTurn(
            thinking_content=thinking_content,
            answer_content=answer_content,
            thinking_logprobs=thinking_logprobs.copy() if thinking_logprobs is not None else None,
            answer_logprobs=answer_logprobs.copy() if answer_logprobs is not None else None,
        )
        if not self._name.need_another_turn:
            assert len(messages) == 2
            return Ok((data, Response(history=messages, turn_0=turn_0, turn_1=None)))

        # ===== Second turn (Optional) =====
        if no_cot_memory:
            messages[1]["content"] = answer_content
        if model.model_name in [ModelName.QWEN3_8B_THINK, ModelName.QWEN3_32B_THINK]:
            messages.append({"role": "user", "content": self._name.prompt + " /think"})
        elif model.model_name in [ModelName.QWEN3_8B_NO_THINK, ModelName.QWEN3_32B_NO_THINK]:
            messages.append({"role": "user", "content": self._name.prompt + " /no_think"})
        else:
            messages.append({"role": "user", "content": self._name.prompt})
        response_result = await model.request(
            messages=messages, temperature=temperature, max_tokens=max_tokens, logprobs=self._name.need_logprobs
        )
        if response_result.is_err():
            return response_result
        messages.append({"role": "assistant", "content": response_result.ok_value.message_content})

        thinking_content, answer_content = split_thinking_answer(response_result.ok_value.message_content)
        thinking_logprobs, answer_logprobs = split_thinking_answer_logprobs(response_result.ok_value.logprobs_content)
        self._validate_text_vs_tokens(thinking_content, thinking_logprobs)
        self._validate_text_vs_tokens(answer_content, answer_logprobs)

        turn_1 = ChatResponsePerTurn(
            thinking_content=thinking_content,
            answer_content=answer_content,
            thinking_logprobs=thinking_logprobs.copy() if thinking_logprobs is not None else None,
            answer_logprobs=answer_logprobs.copy() if answer_logprobs is not None else None,
        )
        assert len(messages) == 4
        return Ok((data, Response(history=messages, turn_0=turn_0, turn_1=turn_1)))

    def build_less_reflection_requests(
        self,
        model: Model,
        data: Data,
        template: Template,
        history_thinking_content: str,
        history_answer_content: str,
        temperature: float = 0,
        max_tokens: int = 16384,
        no_cot_memory: bool = False,
    ) -> list[Coroutine[Any, Any, Result[tuple[Data, Response], str]]]:
        assert model.model_name in [ModelName.QWEN3_8B_THINK, ModelName.QWEN3_32B_THINK], "Only support Long-CoT model"

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
        if history_thinking_content.startswith("<think>\n"):
            history_thinking_content = history_thinking_content.lstrip("<think>\n")
        for m in re.finditer(combined_pattern, history_thinking_content, re.M):
            thinking_steps_by_reflection.append(history_thinking_content[last_step_start_index : m.start()])
            thinking_steps_by_reflection.append(m.group())
            last_step_start_index = m.end()
        thinking_steps_by_reflection.append(history_thinking_content[last_step_start_index:])
        if len(history_thinking_content) == last_step_start_index:
            # Last step is reflection, although this might be impossible. Remove it.
            thinking_steps_by_reflection = thinking_steps_by_reflection[:-2]

        thinking_with_reduced_reflection = []
        for i in range(0, len(thinking_steps_by_reflection), 2):
            thinking_with_reduced_reflection.append(thinking_steps_by_reflection[: i + 1])

        user_input = model.apply_chat_template([{"role": "user", "content": template.prompt(data) + " /think"}])
        user_input_with_thinking = [user_input + "".join(steps) for steps in thinking_with_reduced_reflection]
        user_input_with_thinking = [
            im if im.endswith("</think>") else im + "</think>" for im in user_input_with_thinking
        ]
        return [
            self._request_fake_reflection(
                model=model,
                user_input=user_input,
                prompt=prompt,
                data=data,
                template=template,
                temperature=temperature,
                max_tokens=max_tokens,
                no_cot_memory=no_cot_memory,
            )
            for prompt in user_input_with_thinking
        ]

    def build_more_reflection_requests(
        self,
        model: Model,
        data: Data,
        template: Template,
        history_thinking_content: str,
        history_answer_content: str,
        temperature: float = 0,
        max_tokens: int = 16384,
        no_cot_memory: bool = False,
    ) -> list[Coroutine[Any, Any, Result[tuple[Data, Response], str]]]:
        assert model.model_name not in [ModelName.QWEN3_8B_THINK, ModelName.QWEN3_32B_THINK], (
            "Not support Long-CoT model"
        )

        # Insert a random reflection pattern after Short-CoT
        reflection_patterns = [
            "Wait, there may be other solutions.",
            "Let me double-check if there is any other solution.",
            "Let me think again if there is any other solution.",
        ]
        history_answer_content += "\n\n" + random.choice(reflection_patterns)

        if model.model_name in [ModelName.QWEN3_8B_NO_THINK, ModelName.QWEN3_32B_NO_THINK]:
            user_input = model.apply_chat_template([{"role": "user", "content": template.prompt(data) + " /no_think"}])
        else:
            user_input = model.apply_chat_template([{"role": "user", "content": template.prompt(data)}])
        user_input_with_thinking = user_input + history_answer_content
        return [
            self._request_fake_reflection(
                model=model,
                user_input=user_input,
                prompt=user_input_with_thinking,
                data=data,
                template=template,
                temperature=temperature,
                max_tokens=max_tokens,
                no_cot_memory=no_cot_memory,
            )
        ]
