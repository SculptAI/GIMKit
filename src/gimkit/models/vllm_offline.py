# Adapted from https://github.com/dottxt-ai/outlines/blob/main/outlines/models/vllm_offline.py


from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast, overload

from outlines.generator import Generator
from outlines.inputs import Chat
from outlines.models.vllm_offline import VLLMOffline as OutlinesVLLMOffline
from outlines.types.dsl import CFG, JsonSchema

from gimkit.contexts import Query, Result
from gimkit.log import get_logger
from gimkit.models.types import ErrorMode, GenerationResult
from gimkit.models.utils import (
    get_outlines_model_input,
    get_outlines_model_inputs,
    get_outlines_output_type,
    parse_batch_generation_responses,
    parse_generation_responses,
)
from gimkit.schemas import RESPONSE_SUFFIX, ContextInput, TagField


logger = get_logger(__name__)

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams


OutlinesModelInput: TypeAlias = str | Chat
OutlinesOutputType: TypeAlias = CFG | JsonSchema | None
VLLMFormattedInput: TypeAlias = str | list[object]


class VLLMOffline(OutlinesVLLMOffline):
    @overload
    def __call__(
        self,
        model_input: ContextInput | Query,
        output_type: Literal["cfg", "json"] | None = "cfg",
        backend: str | None = None,
        use_gim_prompt: bool = False,
        visible_tag_fields: list[TagField] | None = None,
        *,
        error_mode: Literal["raise"] = "raise",
        **inference_kwargs: Any,
    ) -> Result | list[Result]: ...

    @overload
    def __call__(
        self,
        model_input: ContextInput | Query,
        output_type: Literal["cfg", "json"] | None = "cfg",
        backend: str | None = None,
        use_gim_prompt: bool = False,
        visible_tag_fields: list[TagField] | None = None,
        *,
        error_mode: Literal["collect"],
        **inference_kwargs: Any,
    ) -> GenerationResult | list[GenerationResult]: ...

    def __call__(
        self,
        model_input: ContextInput | Query,
        output_type: Literal["cfg", "json"] | None = "cfg",
        backend: str | None = None,
        use_gim_prompt: bool = False,
        visible_tag_fields: list[TagField] | None = None,
        *,
        error_mode: ErrorMode = "raise",
        **inference_kwargs: Any,
    ) -> Result | list[Result] | GenerationResult | list[GenerationResult]:
        inference_kwargs = self._ensure_response_suffix(inference_kwargs)

        outlines_model_input = get_outlines_model_input(
            model_input,
            output_type,
            use_gim_prompt,
            visible_tag_fields=visible_tag_fields,
        )
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

    @overload
    def batch(
        self,
        model_input: Sequence[ContextInput | Query],
        output_type: Literal["cfg", "json"] | None = "cfg",
        backend: str | None = None,
        use_gim_prompt: bool = False,
        visible_tag_fields: list[TagField] | None = None,
        *,
        error_mode: Literal["raise"] = "raise",
        **inference_kwargs: Any,
    ) -> list[list[Result]]: ...

    @overload
    def batch(
        self,
        model_input: Sequence[ContextInput | Query],
        output_type: Literal["cfg", "json"] | None = "cfg",
        backend: str | None = None,
        use_gim_prompt: bool = False,
        visible_tag_fields: list[TagField] | None = None,
        *,
        error_mode: Literal["collect"],
        **inference_kwargs: Any,
    ) -> list[list[GenerationResult]]: ...

    def batch(
        self,
        model_input: Sequence[ContextInput | Query],
        output_type: Literal["cfg", "json"] | None = "cfg",
        backend: str | None = None,
        use_gim_prompt: bool = False,
        visible_tag_fields: list[TagField] | None = None,
        *,
        error_mode: ErrorMode = "raise",
        **inference_kwargs: Any,
    ) -> list[list[Result]] | list[list[GenerationResult]]:  # type: ignore[override]
        inference_kwargs = self._ensure_response_suffix(inference_kwargs)

        outlines_model_inputs = get_outlines_model_inputs(
            model_input,
            output_type,
            use_gim_prompt,
            visible_tag_fields=visible_tag_fields,
        )
        outlines_output_types = [
            get_outlines_output_type(batch_item, output_type) for batch_item in model_input
        ]
        raw_responses = self._generate_batch_with_output_types(
            outlines_model_inputs,
            outlines_output_types,
            inference_kwargs,
        )
        logger.debug(f"Raw batch responses of {self}: {raw_responses}")
        return parse_batch_generation_responses(
            model_input,
            raw_responses,
            json_responses=(output_type == "json"),
            error_mode=error_mode,
        )

    def _generate_batch_with_output_types(
        self,
        model_inputs: list[OutlinesModelInput],
        output_types: list[OutlinesOutputType],
        inference_kwargs: dict[str, Any],
    ) -> list[list[str]]:
        generation_kwargs = dict(inference_kwargs)
        sampling_params = generation_kwargs.pop("sampling_params", None)
        sampling_params_list = self._build_batch_sampling_params(sampling_params, output_types)

        formatted_inputs = [
            cast("VLLMFormattedInput", self.type_adapter.format_input(item))
            for item in model_inputs
        ]
        if formatted_inputs and isinstance(formatted_inputs[0], list):
            chat_messages = cast("list[list[Any]]", formatted_inputs)
            results = self.model.chat(
                messages=chat_messages,
                sampling_params=sampling_params_list,
                **generation_kwargs,
            )
        else:
            prompts = cast("list[str]", formatted_inputs)
            results = self.model.generate(
                prompts=prompts,
                sampling_params=sampling_params_list,
                **generation_kwargs,
            )
        return [[sample.text for sample in batch.outputs] for batch in results]

    def _build_batch_sampling_params(
        self,
        sampling_params: "SamplingParams | list[SamplingParams] | None",
        output_types: list[OutlinesOutputType],
    ) -> list["SamplingParams"]:
        if isinstance(sampling_params, list):
            if len(sampling_params) != len(output_types):
                raise ValueError(
                    "sampling_params list must have the same length as model_input: "
                    f"{len(sampling_params)} sampling params for {len(output_types)} input(s)."
                )
            return [
                self._build_generation_args({"sampling_params": params}, output_type)
                for params, output_type in zip(sampling_params, output_types, strict=True)
            ]

        return [
            self._build_generation_args({"sampling_params": sampling_params}, output_type)
            for output_type in output_types
        ]

    def _ensure_response_suffix(self, inference_kwargs: dict[str, Any]) -> dict[str, Any]:
        # Using `stop=RESPONSE_SUFFIX` is preferred for two reasons:
        # 1. The model might not be trained well enough to generate EOS tokens immediately after RESPONSE_SUFFIX.
        # 2. Even with CFG, inference engines like vLLM do not guarantee termination when the CFG is satisfied (See https://github.com/vllm-project/vllm/issues/29632).

        def _ensure_sampling_params_response_suffix(sampling_params: "SamplingParams") -> None:
            if sampling_params.stop is None:
                sampling_params.stop = [RESPONSE_SUFFIX]
            elif isinstance(sampling_params.stop, str):
                if sampling_params.stop != RESPONSE_SUFFIX:
                    sampling_params.stop = [sampling_params.stop, RESPONSE_SUFFIX]
            elif RESPONSE_SUFFIX not in sampling_params.stop:
                sampling_params.stop.append(RESPONSE_SUFFIX)

        if "sampling_params" not in inference_kwargs:
            from vllm import SamplingParams

            inference_kwargs["sampling_params"] = SamplingParams(stop=[RESPONSE_SUFFIX])
        elif isinstance(inference_kwargs["sampling_params"], list):  # For batch inference
            for sampling_params in inference_kwargs["sampling_params"]:
                _ensure_sampling_params_response_suffix(sampling_params)
        else:
            _ensure_sampling_params_response_suffix(inference_kwargs["sampling_params"])
        return inference_kwargs


def from_vllm_offline(model: "LLM") -> VLLMOffline:
    return VLLMOffline(model)
