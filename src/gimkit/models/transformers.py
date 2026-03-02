# Adapted from https://github.com/dottxt-ai/outlines/blob/main/outlines/models/transformers.py


from typing import TYPE_CHECKING, Any, Literal

from outlines.generator import Generator
from outlines.models.transformers import Transformers as OutlinesTransformers

from gimkit.contexts import Query, Result
from gimkit.log import get_logger
from gimkit.models.utils import get_outlines_model_input, get_outlines_output_type, infill_responses
from gimkit.schemas import ContextInput


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


class Transformers(OutlinesTransformers):
    def __call__(
        self,
        model_input: ContextInput | Query,
        output_type: Literal["cfg", "json"] | None = "cfg",
        backend: str | None = None,
        use_gim_prompt: bool = False,
        **inference_kwargs: Any,
    ) -> Result | list[Result]:
        outlines_model_input = get_outlines_model_input(model_input, output_type, use_gim_prompt)
        outlines_output_type = get_outlines_output_type(model_input, output_type)
        generator = Generator(self, outlines_output_type, backend)
        raw_responses = generator(outlines_model_input, **inference_kwargs)
        logger.debug(f"Raw responses of {self}: {raw_responses}")
        return infill_responses(
            model_input,
            raw_responses,  # type: ignore[arg-type]
            json_responses=(output_type == "json"),
        )


def from_transformers(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
) -> Transformers:
    return Transformers(model, tokenizer)
