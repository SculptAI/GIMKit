# Adapted from https://github.com/dottxt-ai/outlines/blob/main/outlines/models/vllm_offline.py


from typing import TYPE_CHECKING, Any, Literal, cast

from outlines.generator import Generator
from outlines.models.vllm_offline import VLLMOffline as OutlinesVLLMOffline

from gimkit.contexts import Query, Result
from gimkit.log import get_logger
from gimkit.models.utils import (
    get_outlines_model_input,
    get_outlines_model_inputs,
    get_outlines_output_type,
    infill_batch_responses,
    infill_responses,
)
from gimkit.schemas import RESPONSE_SUFFIX, ContextInput, TagField


logger = get_logger(__name__)

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams


class VLLMOffline(OutlinesVLLMOffline):
    def __call__(
        self,
        model_input: ContextInput | Query,
        output_type: Literal["cfg", "json"] | None = "cfg",
        backend: str | None = None,
        use_gim_prompt: bool = False,
        visible_tag_fields: list[TagField] | None = None,
        **inference_kwargs: Any,
    ) -> Result | list[Result]:
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
        return infill_responses(
            model_input,
            cast("str | list[str]", raw_responses),
            json_responses=(output_type == "json"),
        )

    def batch(
        self,
        model_input: list[Any],
        output_type: Any | None = "cfg",
        backend: str | None = None,
        use_gim_prompt: bool = False,
        visible_tag_fields: list[TagField] | None = None,
        **inference_kwargs: Any,
    ) -> list[Any]:
        inference_kwargs = self._ensure_response_suffix(inference_kwargs)
        model_inputs = cast("list[ContextInput | Query]", model_input)
        gim_output_type = cast("Literal['cfg', 'json'] | None", output_type)

        outlines_model_inputs = get_outlines_model_inputs(
            model_inputs,
            gim_output_type,
            use_gim_prompt,
            visible_tag_fields=visible_tag_fields,
        )
        outlines_output_types = [
            get_outlines_output_type(model_input, gim_output_type) for model_input in model_inputs
        ]
        raw_responses = self._generate_batch_with_output_types(
            outlines_model_inputs,
            outlines_output_types,
            inference_kwargs,
        )
        logger.debug(f"Raw batch responses of {self}: {raw_responses}")
        return infill_batch_responses(
            model_inputs,
            cast("list[str] | list[list[str]]", raw_responses),
            json_responses=(gim_output_type == "json"),
        )

    def _generate_batch_with_output_types(
        self,
        model_inputs: list[Any],
        output_types: list[Any],
        inference_kwargs: dict[str, Any],
    ) -> list[list[str]]:
        generation_kwargs = dict(inference_kwargs)
        sampling_params = generation_kwargs.pop("sampling_params", None)
        sampling_params_list = self._build_batch_sampling_params(sampling_params, output_types)

        formatted_inputs = [self.type_adapter.format_input(item) for item in model_inputs]
        if formatted_inputs and isinstance(formatted_inputs[0], list):
            results = self.model.chat(
                messages=formatted_inputs,
                sampling_params=sampling_params_list,
                **generation_kwargs,
            )
        else:
            results = self.model.generate(
                prompts=formatted_inputs,
                sampling_params=sampling_params_list,
                **generation_kwargs,
            )
        return [[sample.text for sample in batch.outputs] for batch in results]

    def _build_batch_sampling_params(
        self,
        sampling_params: Any,
        output_types: list[Any],
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
        if "sampling_params" not in inference_kwargs:
            from vllm import SamplingParams

            inference_kwargs["sampling_params"] = SamplingParams(stop=[RESPONSE_SUFFIX])
        elif isinstance(inference_kwargs["sampling_params"], list):
            for sampling_params in inference_kwargs["sampling_params"]:
                self._ensure_sampling_params_response_suffix(sampling_params)
        else:
            self._ensure_sampling_params_response_suffix(inference_kwargs["sampling_params"])
        return inference_kwargs

    def _ensure_sampling_params_response_suffix(self, sampling_params: "SamplingParams") -> None:
        if sampling_params.stop is None:
            sampling_params.stop = [RESPONSE_SUFFIX]
        elif isinstance(sampling_params.stop, str):
            if sampling_params.stop != RESPONSE_SUFFIX:
                sampling_params.stop = [sampling_params.stop, RESPONSE_SUFFIX]
        elif RESPONSE_SUFFIX not in sampling_params.stop:
            sampling_params.stop.append(RESPONSE_SUFFIX)


def from_vllm_offline(model: "LLM") -> VLLMOffline:
    return VLLMOffline(model)
