from typing import Any, Literal, cast

from outlines.generator import Generator
from outlines.models.base import AsyncModel, Model

from gimkit.contexts import Query, Result
from gimkit.log import get_logger
from gimkit.models.types import ErrorMode, GenerationResult
from gimkit.models.utils import (
    get_outlines_model_input,
    get_outlines_output_type,
    parse_generation_responses,
    validate_error_mode,
)
from gimkit.schemas import ContextInput, TagField


logger = get_logger(__name__)


def _call(
    self: Model,
    model_input: ContextInput | Query,
    output_type: Literal["cfg", "json"] | None = "cfg",
    backend: str | None = None,
    use_gim_prompt: bool = False,
    visible_tag_fields: list[TagField] | None = None,
    *,
    error_mode: ErrorMode = "raise",
    **inference_kwargs: Any,
) -> Result | list[Result] | GenerationResult | list[GenerationResult]:
    validate_error_mode(error_mode)
    outlines_model_input = get_outlines_model_input(
        model_input, output_type, use_gim_prompt, visible_tag_fields
    )
    logger.debug(f"Outlines model input of {self}: {outlines_model_input}")
    outlines_output_type = get_outlines_output_type(model_input, output_type)
    generator = Generator(self, outlines_output_type, backend)
    raw_responses = generator(outlines_model_input, **inference_kwargs)
    logger.debug(f"Raw responses of {self}: {raw_responses}")
    return cast(
        "Result | list[Result] | GenerationResult | list[GenerationResult]",
        cast("Any", parse_generation_responses)(
            model_input,
            cast("str | list[str]", raw_responses),
            json_responses=(output_type == "json"),
            error_mode=error_mode,
        ),
    )


async def _acall(
    self: AsyncModel,
    model_input: ContextInput | Query,
    output_type: Literal["cfg", "json"] | None = "cfg",
    backend: str | None = None,
    use_gim_prompt: bool = False,
    visible_tag_fields: list[TagField] | None = None,
    *,
    error_mode: ErrorMode = "raise",
    **inference_kwargs: Any,
) -> Result | list[Result] | GenerationResult | list[GenerationResult]:
    validate_error_mode(error_mode)
    outlines_model_input = get_outlines_model_input(
        model_input, output_type, use_gim_prompt, visible_tag_fields
    )
    logger.debug(f"Outlines model input of {self}: {outlines_model_input}")
    outlines_output_type = get_outlines_output_type(model_input, output_type)
    generator = Generator(self, outlines_output_type, backend)
    raw_responses = await generator(outlines_model_input, **inference_kwargs)
    logger.debug(f"Raw responses of {self}: {raw_responses}")
    return cast(
        "Result | list[Result] | GenerationResult | list[GenerationResult]",
        cast("Any", parse_generation_responses)(
            model_input,
            cast("str | list[str]", raw_responses),
            json_responses=(output_type == "json"),
            error_mode=error_mode,
        ),
    )
