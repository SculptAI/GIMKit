from typing import Any, Literal

from outlines.generator import Generator
from outlines.types.dsl import CFG, JsonSchema

from gimkit.contexts import Query, Result, infill
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
    output_type = CFG(f"{grammar_first_line}\n{grammar_next_lines}")
    return output_type


def build_json_schema(query: Query) -> JsonSchema:  # pragma: no cover  # TODO
    raise NotImplementedError("JSON schema generation is not implemented yet.")


def get_output_type(
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
):
    query_obj = Query(model_input) if not isinstance(model_input, Query) else model_input
    outlines_output_type = get_output_type(output_type, query_obj)
    outlines_model_input = str(query_obj)
    return outlines_output_type, outlines_model_input


def ensure_str(response: Any) -> str:
    if isinstance(response, str):
        return response
    if (
        isinstance(response, list)
        and len(response) > 0
        and all(isinstance(item, str) for item in response)
    ):
        return response[0]  # pragma: no cover  # TODO: Handle multiple responses
    else:
        raise TypeError("Response is not a string.")


def _call(
    self,
    model_input: ContextInput | Query,
    output_type: Literal["cfg", "json"] | None = "cfg",
    backend: str | None = None,
    **inference_kwargs: Any,
) -> Result:
    outlines_output_type, outlines_model_input = transform_to_outlines(model_input, output_type)
    raw_response = Generator(self, outlines_output_type, backend)(
        outlines_model_input, **inference_kwargs
    )
    str_response = ensure_str(raw_response)
    return infill(model_input, str_response)


async def _acall(
    self,
    model_input: ContextInput | Query,
    output_type: Literal["cfg", "json"] | None = "cfg",
    backend: str | None = None,
    **inference_kwargs: Any,
) -> Result:
    outlines_output_type, outlines_model_input = transform_to_outlines(model_input, output_type)
    generator = Generator(self, outlines_output_type, backend)
    raw_response = await generator(outlines_model_input, **inference_kwargs)
    str_response = ensure_str(raw_response)
    return infill(model_input, str_response)
