# Adapted from https://github.com/dottxt-ai/outlines/blob/main/outlines/models/vllm_offline.py


from typing import TYPE_CHECKING, Any, Literal

from outlines.models.vllm_offline import VLLMOffline as OutlinesVLLMOffline

from gimkit.contexts import Query, Result
from gimkit.models.base import _call
from gimkit.schemas import ContextInput


if TYPE_CHECKING:
    from vllm import LLM


class VLLMOffline(OutlinesVLLMOffline):
    def __call__(
        self,
        model_input: ContextInput | Query,
        output_type: Literal["cfg", "json"] | None = "cfg",
        backend: str | None = None,
        use_gim_prompt: bool = False,
        **inference_kwargs: Any,
    ) -> Result | list[Result]:
        return _call(self, model_input, output_type, backend, use_gim_prompt, **inference_kwargs)


def from_vllm_offline(model: "LLM") -> VLLMOffline:
    return VLLMOffline(model)
