# Adapted from https://github.com/dottxt-ai/outlines/blob/main/outlines/models/llamacpp.py


from typing import TYPE_CHECKING, Any, Literal, cast

from outlines.generator import Generator
from outlines.models.llamacpp import LlamaCpp as OutlinesLlamaCpp

from gimkit.contexts import Query, Result
from gimkit.log import get_logger
from gimkit.models.utils import get_outlines_model_input, get_outlines_output_type, infill_responses
from gimkit.schemas import RESPONSE_SUFFIX, ContextInput


logger = get_logger(__name__)

if TYPE_CHECKING:
    from llama_cpp import Llama


class LlamaCpp(OutlinesLlamaCpp):
    def __call__(
        self,
        model_input: ContextInput | Query,
        output_type: Literal["cfg", "json"] | None = "cfg",
        backend: str | None = None,
        use_gim_prompt: bool = False,
        **inference_kwargs: Any,
    ) -> Result | list[Result]:
        # Using `stop=RESPONSE_SUFFIX` is preferred for two reasons:
        # 1. The model might not be trained well enough to generate EOS tokens immediately after RESPONSE_SUFFIX.
        # 2. Even with CFG, inference engines may not guarantee termination when the CFG is satisfied.
        inference_kwargs = self._ensure_response_suffix(inference_kwargs)

        outlines_model_input = get_outlines_model_input(model_input, output_type, use_gim_prompt)
        outlines_output_type = get_outlines_output_type(model_input, output_type)
        generator = Generator(self, outlines_output_type, backend)
        raw_responses = generator(outlines_model_input, **inference_kwargs)
        logger.debug(f"Raw responses of {self}: {raw_responses}")
        return infill_responses(
            model_input,
            cast("str | list[str]", raw_responses),
            json_responses=(output_type == "json"),
        )

    def _ensure_response_suffix(self, inference_kwargs: dict[str, Any]) -> dict[str, Any]:
        stop = inference_kwargs.get("stop")
        if stop is None:
            inference_kwargs["stop"] = [RESPONSE_SUFFIX]
        elif isinstance(stop, list) and RESPONSE_SUFFIX not in stop:
            inference_kwargs["stop"] = [*stop, RESPONSE_SUFFIX]
        elif isinstance(stop, str) and stop != RESPONSE_SUFFIX:
            inference_kwargs["stop"] = [stop, RESPONSE_SUFFIX]
        return inference_kwargs


def from_llamacpp(model: "Llama") -> LlamaCpp:
    return LlamaCpp(model)
