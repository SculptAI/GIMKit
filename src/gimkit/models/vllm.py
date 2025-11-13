# Adapted from https://github.com/dottxt-ai/outlines/blob/main/outlines/models/vllm.py


from typing import Any, Literal, overload

from openai import AsyncOpenAI as AsyncOpenAIClient
from openai import OpenAI as OpenAIClient
from outlines.models.vllm import VLLM as OutlinesVLLM
from outlines.models.vllm import AsyncVLLM as OutlinesAsyncVLLM

from gimkit.contexts import Query, Result
from gimkit.models.base import _acall, _call
from gimkit.schemas import RESPONSE_SUFFIX, ContextInput


class VLLM(OutlinesVLLM):
    def __call__(
        self,
        model_input: ContextInput | Query,
        output_type: Literal["cfg", "json"] | None = "cfg",
        backend: str | None = None,
        use_gim_prompt: bool = False,
        **inference_kwargs: Any,
    ) -> Result | list[Result]:
        # TODO: Using `stop=RESPONSE_SUFFIX` is just a temporary workaround. The ending string
        # has already been defined in the lark grammar, and the intermediate regex matching is
        # non-greedy. However, for some reason, it still matches multiple ending strings. Solving
        # this issue involves five packages, including `outlines`, `vllm`, `guidance`,
        # `llguidance`, and `lark`. Due to its complexity, it will be addressed in future updates.
        # Same for vllm_offline.py
        return _call(
            self,
            model_input,
            output_type,
            backend,
            use_gim_prompt,
            stop=RESPONSE_SUFFIX,
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
            stop=RESPONSE_SUFFIX,
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
