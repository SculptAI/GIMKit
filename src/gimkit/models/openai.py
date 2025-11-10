# Adapted from https://github.com/dottxt-ai/outlines/blob/main/outlines/models/openai.py

from typing import Any, Literal, overload

from openai import AsyncAzureOpenAI as AsyncAzureOpenAIClient
from openai import AsyncOpenAI as AsyncOpenAIClient
from openai import AzureOpenAI as AzureOpenAIClient
from openai import OpenAI as OpenAIClient
from outlines.models.openai import AsyncOpenAI as OutlinesAsyncOpenAI
from outlines.models.openai import OpenAI as OutlinesOpenAI

from gimkit.contexts import Query, Result
from gimkit.models.base import _acall, _call
from gimkit.schemas import ContextInput


class OpenAI(OutlinesOpenAI):
    def __call__(
        self,
        model_input: ContextInput | Query,
        output_type: Literal["json"] | None = None,
        backend: str | None = None,
        use_gim_prompt: bool = False,
        **inference_kwargs: Any,
    ) -> Result | list[Result]:
        return _call(self, model_input, output_type, backend, use_gim_prompt, **inference_kwargs)


class AsyncOpenAI(OutlinesAsyncOpenAI):
    async def __call__(
        self,
        model_input: ContextInput | Query,
        output_type: Literal["json"] | None = None,
        backend: str | None = None,
        use_gim_prompt: bool = False,
        **inference_kwargs: Any,
    ) -> Result | list[Result]:
        return await _acall(
            self, model_input, output_type, backend, use_gim_prompt, **inference_kwargs
        )


@overload
def from_openai(
    client: OpenAIClient | AzureOpenAIClient, model_name: str | None = None
) -> OpenAI: ...


@overload
def from_openai(
    client: AsyncOpenAIClient | AsyncAzureOpenAIClient, model_name: str | None = None
) -> AsyncOpenAI: ...


def from_openai(
    client: OpenAIClient | AsyncOpenAIClient | AzureOpenAIClient | AsyncAzureOpenAIClient,
    model_name: str | None = None,
) -> OpenAI | AsyncOpenAI:
    import openai

    if isinstance(client, openai.OpenAI):
        return OpenAI(client, model_name)
    elif isinstance(client, openai.AsyncOpenAI):
        return AsyncOpenAI(client, model_name)
    else:
        raise ValueError(
            "Invalid client type. The client must be an instance of "
            "`openai.OpenAI` or `openai.AsyncOpenAI`."
        )
