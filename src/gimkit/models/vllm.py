# Adapted from https://github.com/dottxt-ai/outlines/blob/main/outlines/models/vllm.py


from typing import Any, Literal, overload

from openai import AsyncOpenAI as AsyncOpenAIClient
from openai import OpenAI as OpenAIClient
from outlines.models.vllm import VLLM as OutlinesVLLM
from outlines.models.vllm import AsyncVLLM as OutlinesAsyncVLLM

from gimkit.contexts import Query, Result
from gimkit.models.base import _acall, _call
from gimkit.schemas import ContextInput


class VLLM(OutlinesVLLM):
    def __call__(
        self,
        model_input: ContextInput | Query,
        output_type: Literal["cfg", "json"] | None = "cfg",
        backend: str | None = None,
        use_gim_prompt: bool = False,
        **inference_kwargs: Any,
    ) -> Result | list[Result]:
        return _call(
            self,
            model_input,
            output_type,
            backend,
            use_gim_prompt,
            **inference_kwargs,
        )


class AsyncVLLM(OutlinesAsyncVLLM):
    async def __call__(
        self,
        model_input: ContextInput | Query,
        output_type: Literal["cfg", "json"] | None = "cfg",
        backend: str | None = None,
        use_gim_prompt: bool = False,
        **inference_kwargs: Any,
    ) -> Result | list[Result]:
        return await _acall(
            self,
            model_input,
            output_type,
            backend,
            use_gim_prompt,
            **inference_kwargs,
        )


@overload
def from_vllm(client: OpenAIClient, model_name: str | None = None) -> VLLM: ...


@overload
def from_vllm(client: AsyncOpenAIClient, model_name: str | None = None) -> AsyncVLLM: ...


def from_vllm(
    client: OpenAIClient | AsyncOpenAIClient,
    model_name: str | None = None,
) -> VLLM | AsyncVLLM:
    if isinstance(client, OpenAIClient):
        return VLLM(client, model_name)
    elif isinstance(client, AsyncOpenAIClient):
        return AsyncVLLM(client, model_name)
    else:
        raise ValueError(
            f"Unsupported client type: {type(client)}.\n"
            "Please provide an OpenAI or AsyncOpenAI instance."
        )
