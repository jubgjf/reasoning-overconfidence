from enum import Enum
from typing import assert_never

from loguru import logger
from openai.types.chat import ChatCompletionTokenLogprob
from pydantic import BaseModel
from result import Ok, Result

from .data import Data
from .model import Model
from .template import Template
from .utils import split_thinking_answer, split_thinking_answer_logprobs


class MethodName(Enum):
    Verbal_0_100 = "verbal_0_100"
    LogProb = "logprob"
    P_True = "p_true"

    def __str__(self) -> str:
        return self.value

    @property
    def need_another_turn(self) -> bool:
        return self in [self.Verbal_0_100, self.P_True]

    @property
    def need_logprobs(self) -> bool:
        return self in [self.LogProb, self.P_True]

    @property
    def prompt(self) -> str | None:
        if self == self.Verbal_0_100:
            return "Please rate your confidence in the proposed answer on a scale of 0-100. Give your confidence in format: [[xx]]"
        elif self == self.P_True:
            return "Please rate your confidence in the proposed answer as either 0 or 1. Give your confidence in format: [[x]]"
        elif self == self.LogProb:
            return None
        else:
            assert_never(self)


class ResponsePerTurn(BaseModel):
    thinking_content: str | None
    answer_content: str
    thinking_logprobs: list[ChatCompletionTokenLogprob] | None
    answer_logprobs: list[ChatCompletionTokenLogprob] | None


class Response(BaseModel):
    history: list[dict[str, str]]
    turn_0: ResponsePerTurn
    turn_1: ResponsePerTurn | None


class Method:
    def __init__(self, name: MethodName):
        self._name = name

    @staticmethod
    def _validate_text_vs_tokens(text: str, tokens: list[ChatCompletionTokenLogprob] | None):
        if tokens is None:
            return

        text_from_tokens = "".join([t.token for t in tokens])

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

        turn_0 = ResponsePerTurn(
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

        turn_1 = ResponsePerTurn(
            thinking_content=thinking_content,
            answer_content=answer_content,
            thinking_logprobs=thinking_logprobs.copy() if thinking_logprobs is not None else None,
            answer_logprobs=answer_logprobs.copy() if answer_logprobs is not None else None,
        )
        assert len(messages) == 4
        return Ok(Response(history=messages, turn_0=turn_0, turn_1=turn_1))
