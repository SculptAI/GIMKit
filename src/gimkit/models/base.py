from typing import Any, Literal, cast

from outlines.generator import Generator
from outlines.models.base import AsyncModel, Model

from gimkit.contexts import Query, Result
from gimkit.log import get_logger
from gimkit.models.utils import infill_responses, transform_to_outlines
from gimkit.schemas import ContextInput


logger = get_logger(__name__)


def _call(
    self: Model,
    model_input: ContextInput | Query,
    output_type: Literal["cfg", "json"] | None = "cfg",
    backend: str | None = None,
    use_gim_prompt: bool = False,
    **inference_kwargs: Any,
) -> Result | list[Result]:
    outlines_model_input, outlines_output_type = transform_to_outlines(
        model_input, output_type, use_gim_prompt
    )
    generator = Generator(self, outlines_output_type, backend)
    raw_responses = generator(outlines_model_input, **inference_kwargs)
    logger.debug(f"Raw responses of {self}: {raw_responses}")
    return infill_responses(
        model_input, cast("str | list[str]", raw_responses), json_responses=(output_type == "json")
    )


async def _acall(
    self: AsyncModel,
    model_input: ContextInput | Query,
    output_type: Literal["cfg", "json"] | None = "cfg",
    backend: str | None = None,
    use_gim_prompt: bool = False,
    **inference_kwargs: Any,
) -> Result | list[Result]:
    outlines_model_input, outlines_output_type = transform_to_outlines(
        model_input, output_type, use_gim_prompt
    )
    generator = Generator(self, outlines_output_type, backend)
    raw_responses = await generator(outlines_model_input, **inference_kwargs)
    logger.debug(f"Raw responses of {self}: {raw_responses}")
    return infill_responses(
        model_input, cast("str | list[str]", raw_responses), json_responses=(output_type == "json")
    )
