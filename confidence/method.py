from enum import Enum
from typing import assert_never

from openai.types.chat import ChatCompletionTokenLogprob
from pydantic import BaseModel
from result import Ok, Result

from .data import Data
from .model import Model
from .template import Template


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
            return "Please rate your confidence in the proposed answer as either 0 or 1. Give your confidence in format: [[xx]]"
        elif self == self.LogProb:
            return None
        else:
            assert_never(self)


class Response(BaseModel):
    messages: list[dict[str, str]]
    logprobs: list[ChatCompletionTokenLogprob] | None


class Method:
    def __init__(self, name: MethodName):
        self._name = name

    async def request(
        self,
        model: Model,
        data: Data,
        template: Template,
        temperature: float = 0,
        max_tokens: int = 16384,
    ) -> Result[Response, str]:
        # ===== First turn =====
        messages = [{"role": "user", "content": template.prompt(data)}]
        response_result = await model.request(
            messages=messages, temperature=temperature, max_tokens=max_tokens, logprobs=self._name.need_logprobs
        )
        if response_result.is_err():
            return response_result
        messages.append({"role": "assistant", "content": response_result.ok_value.message_content})
        if not self._name.need_another_turn:
            assert len(messages) == 2
            return Ok(Response(messages=messages, logprobs=response_result.ok_value.logprobs_content))

        # ===== Second turn (Optional) =====
        messages.append({"role": "user", "content": self._name.prompt})
        response_result = await model.request(
            messages=messages, temperature=temperature, max_tokens=max_tokens, logprobs=self._name.need_logprobs
        )
        if response_result.is_err():
            return response_result
        messages.append({"role": "assistant", "content": response_result.ok_value.message_content})
        assert len(messages) == 4
        return Ok(Response(messages=messages, logprobs=response_result.ok_value.logprobs_content))
