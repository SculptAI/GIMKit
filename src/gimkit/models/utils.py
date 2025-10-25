from typing import Any, Literal

from outlines.generator import Generator
from outlines.inputs import Chat
from outlines.types.dsl import CFG, JsonSchema

from gimkit.contexts import Query, Result, infill
from gimkit.prompts import DEMO_CONVERSATION_MSGS, SYSTEM_PROMPT_MSG
from gimkit.schemas import (
    RESPONSE_PREFIX,
    RESPONSE_SUFFIX,
    TAG_END,
    TAG_OPEN_LEFT,
    TAG_OPEN_RIGHT,
    ContextInput,
)


def build_cfg(query: Query) -> CFG:
    """Build a Lark-based CFG output type based on the query object."""
    num_tags = len(query.tags)
    grammar_first_line = f'''start: "{RESPONSE_PREFIX}" {" ".join(f"tag{i}" for i in range(num_tags))} "{RESPONSE_SUFFIX}"'''
    grammar_next_lines = (
        "\n".join(
            # `/(?s:.)*?/` is a non-greedy match for any character including newlines
            f'tag{i}: "{TAG_OPEN_LEFT} id=\\"m_{i}\\"{TAG_OPEN_RIGHT}" /(?s:.)*?/ "{TAG_END}"'
            for i in range(num_tags)
        )
        if num_tags > 0
        else ""
    )
    return CFG(f"{grammar_first_line}\n{grammar_next_lines}")


def build_json_schema(query: Query) -> JsonSchema:  # pragma: no cover  # TODO
    raise NotImplementedError("JSON schema generation is not implemented yet.")


def get_outlines_output_type(
    output_type: Literal["cfg", "json"] | None, query: Query
) -> None | CFG | JsonSchema:
    if output_type is None:
        return None
    elif output_type == "cfg":
        return build_cfg(query)
    elif output_type == "json":  # pragma: no cover  # TODO
        return build_json_schema(query)
    else:
        raise ValueError(f"Invalid output type: {output_type}")


def transform_to_outlines(
    model_input: ContextInput | Query,
    output_type: Literal["cfg", "json"] | None,
    use_gim_prompt: bool,
) -> tuple[str | Chat, None | CFG | JsonSchema]:
    query_obj = Query(model_input) if not isinstance(model_input, Query) else model_input
    outlines_model_input: str | Chat = str(query_obj)
    if use_gim_prompt:
        outlines_model_input = Chat(
            [
                SYSTEM_PROMPT_MSG,
                *DEMO_CONVERSATION_MSGS,
                {"role": "user", "content": outlines_model_input},
            ]
        )
    outlines_output_type = get_outlines_output_type(output_type, query_obj)
    return outlines_model_input, outlines_output_type


def infill_responses(
    query: ContextInput | Query, responses: str | list[str] | Any
) -> Result | list[Result]:
    if isinstance(responses, str):
        return infill(query, responses)

    if not isinstance(responses, list):
        raise TypeError(f"Expected responses to be str or list of str, got {type(responses)}")

    if not all(isinstance(resp, str) for resp in responses):
        raise TypeError(f"All items in the response list must be strings, got: {responses}")

    if len(responses) == 0:
        raise ValueError("Response list is empty.")

    return [infill(query, resp) for resp in responses]


def _call(
    self,
    model_input: ContextInput | Query,
    output_type: Literal["cfg", "json"] | None = "cfg",
    backend: str | None = None,
    use_gim_prompt: bool = False,
    **inference_kwargs: Any,
) -> Result | list[Result]:
    outlines_model_input, outlines_output_type = transform_to_outlines(
        model_input, output_type, use_gim_prompt
    )
    raw_response = Generator(self, outlines_output_type, backend)(
        outlines_model_input, **inference_kwargs
    )
    return infill_responses(model_input, raw_response)


async def _acall(
    self,
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
    return infill_responses(model_input, raw_responses)
