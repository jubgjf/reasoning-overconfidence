import asyncio
import os
from enum import Enum

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionTokenLogprob
from openai.types.completion_choice import Logprobs
from pydantic import BaseModel
from result import Err, Ok, Result
from transformers import AutoTokenizer


class ModelName(Enum):
    QWEN3_8B_THINK = "qwen3-8b-think"
    QWEN3_8B_NO_THINK = "qwen3-8b-no_think"
    QWEN3_32B_THINK = "qwen3-32b-think"
    QWEN3_32B_NO_THINK = "qwen3-32b-no_think"

    def __str__(self) -> str:
        return self.value

    @property
    def model_id(self) -> str:
        return self.value


class ChatAPIResponse(BaseModel):
    message_content: str
    logprobs_content: list[ChatCompletionTokenLogprob] | None


class CompleteAPIResponse(BaseModel):
    text_content: str
    logprobs_content: Logprobs | None


class Model:
    def __init__(self, model_name: ModelName, model_name_or_path: str):
        self.model_name = model_name
        self._client = self._get_client()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    @staticmethod
    def _get_client() -> AsyncOpenAI:
        def get_url_and_key() -> tuple[str | None, str | None]:
            return os.getenv("BASE_URL"), os.getenv("API_KEY")

        base_url, api_key = get_url_and_key()
        if base_url is None or api_key is None:
            load_dotenv()
            base_url, api_key = get_url_and_key()

        if base_url is None or api_key is None:
            raise ValueError("BASE_URL and API_KEY must be set in environment variables")

        return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=1800)

    async def request(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0,
        max_tokens: int = 32768,
        logprobs: bool = False,
    ) -> Result[ChatAPIResponse, str]:
        async def request_once() -> Result[ChatAPIResponse, str]:
            try:
                response = await self._client.chat.completions.create(
                    model=self.model_name.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logprobs=logprobs,
                    top_logprobs=5 if logprobs else None,
                )
                if self.model_name in [ModelName.QWEN3_8B_THINK]:
                    assert response.choices[0].message.model_extra["reasoning_content"] is not None
                    message_content = (
                        response.choices[0].message.model_extra["reasoning_content"]
                        + "</think>"
                        + response.choices[0].message.content
                    )
                else:
                    message_content = response.choices[0].message.content
                return Ok(
                    ChatAPIResponse(
                        message_content=message_content,
                        logprobs_content=response.choices[0].logprobs.content if logprobs else None,
                    )
                )
            except Exception as e:
                err_msg = f"Error: {e}"
                return Err(err_msg)

        retry, tolerance = 0, 5
        response_result = Err("Failed to request")
        while retry < tolerance:
            response_result = await request_once()
            if response_result.is_ok():
                break
            retry += 1
            logger.error(f"Retry: {retry}/{tolerance}. Failure: {response_result.err_value} ")
            await asyncio.sleep(0.1)

        return response_result

    async def complete(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 32768,
        logprobs: bool = False,
    ) -> Result[CompleteAPIResponse, str]:
        async def request_once() -> Result[CompleteAPIResponse, str]:
            try:
                response = await self._client.completions.create(
                    model=self.model_name.model_id,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logprobs=5 if logprobs else None,
                )
                return Ok(
                    CompleteAPIResponse(
                        text_content=response.choices[0].text,
                        logprobs_content=response.choices[0].logprobs if logprobs else None,
                    )
                )
            except Exception as e:
                err_msg = f"Error: {e}"
                return Err(err_msg)

        # TODO refactor request&complete
        retry, tolerance = 0, 5
        response_result = Err("Failed to request")
        while retry < tolerance:
            response_result = await request_once()
            if response_result.is_ok():
                break
            retry += 1
            logger.error(f"Retry: {retry}/{tolerance}. Failure: {response_result.err_value} ")
            await asyncio.sleep(0.1)

        return response_result

    def string_to_token_id(self, text: str) -> int:
        token_ids = self._tokenizer.encode(text)
        assert len(token_ids) == 1
        return token_ids[0]

    def token_ids_to_string(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids)

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
