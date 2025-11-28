# Adapted from https://github.com/dottxt-ai/outlines/blob/main/outlines/models/vllm_offline.py


from typing import TYPE_CHECKING, Any, Literal

from outlines.models.vllm_offline import VLLMOffline as OutlinesVLLMOffline

from gimkit.contexts import Query, Result
from gimkit.models.base import _call
from gimkit.schemas import RESPONSE_SUFFIX, ContextInput


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
        inference_kwargs = self._ensure_response_suffix(inference_kwargs)
        return _call(self, model_input, output_type, backend, use_gim_prompt, **inference_kwargs)

    def _ensure_response_suffix(self, inference_kwargs: dict[str, Any]) -> dict[str, Any]:
        # Using `stop=RESPONSE_SUFFIX` is preferred for two reasons:
        # 1. The model might not be trained well enough to generate EOS tokens immediately after RESPONSE_SUFFIX.
        # 2. Even with CFG, inference engines like vLLM do not guarantee termination when the CFG is satisfied (See https://github.com/vllm-project/vllm/issues/29632).
        if "sampling_params" not in inference_kwargs:
            from vllm import SamplingParams

            inference_kwargs["sampling_params"] = SamplingParams(stop=[RESPONSE_SUFFIX])
        elif (
            isinstance(inference_kwargs["sampling_params"].stop, list)
            and RESPONSE_SUFFIX not in inference_kwargs["sampling_params"].stop
        ):
            inference_kwargs["sampling_params"].stop.append(RESPONSE_SUFFIX)
        return inference_kwargs

def from_vllm_offline(model: "LLM") -> VLLMOffline:
    return VLLMOffline(model)
