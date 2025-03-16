import asyncio
import os
from enum import Enum

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionTokenLogprob
from pydantic import BaseModel
from result import Err, Ok, Result


class ModelName(Enum):
    QWEN2_5_7B = "qwen2.5-7b"
    QWEN2_5_72B = "qwen2.5-72b"
    LLAMA3_3_70B = "llama3.3-70b"
    QWQ_32B = "qwq-32b"
    DEEPSEEK_R1 = "deepseek-r1-250120"

    def __str__(self) -> str:
        return self.value

    @property
    def model_id(self) -> str:
        return self.value


class APIResponse(BaseModel):
    message_content: str
    logprobs_content: list[ChatCompletionTokenLogprob] | None


class Model:
    def __init__(self, model_name: ModelName):
        self._model_name = model_name
        self._client = self._get_client()

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
        max_tokens: int = 16384,
        logprobs: bool = False,
    ) -> Result[APIResponse, str]:
        async def request_once() -> Result[APIResponse, str]:
            try:
                response = await self._client.chat.completions.create(
                    model=self._model_name.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logprobs=logprobs,
                    top_logprobs=5 if logprobs else None,
                )
                return Ok(
                    APIResponse(
                        message_content=response.choices[0].message.content,
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
