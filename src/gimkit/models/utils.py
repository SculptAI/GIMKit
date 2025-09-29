from typing import Any, Literal

from outlines.generator import Generator
from outlines.types import CFG, JsonSchema

from gimkit.contexts import Query, Response
from gimkit.schemas import MaskedTag


def build_query(model_input: str | MaskedTag | list[str | MaskedTag]) -> Query:
    return Query(model_input)


def build_cfg(query: Query) -> CFG:
    # TODO
    grammar_string = ""
    output_type = CFG(grammar_string)
    return output_type


def build_json_schema(query: Query) -> JsonSchema:
    raise NotImplementedError("JSON schema generation is not implemented yet.")


def get_output_type(
    output_type: Literal["cfg", "json"] | None, query: Query
) -> None | CFG | JsonSchema:
    if output_type is None:
        return None
    elif output_type == "cfg":
        return build_cfg(query)
    elif output_type == "json":
        return build_json_schema(query)
    else:
        raise ValueError(f"Invalid output type: {output_type}")


def ensure_str(response: Any) -> str:
    if isinstance(response, str):
        return response
    else:
        raise TypeError("Response is not a string.")


def _call(
    self,
    model_input: str | MaskedTag | list[str | MaskedTag],
    output_type: Literal["cfg", "json"] | None = "cfg",
    backend: str | None = None,
    **inference_kwargs: Any,
) -> Response:
    query_obj = Query(model_input)
    outlines_output_type = get_output_type(output_type, query_obj)
    outlines_model_input = str(query_obj)
    raw_response = Generator(self, outlines_output_type, backend)(
        outlines_model_input, **inference_kwargs
    )
    str_response = ensure_str(raw_response)
    return Response(str_response)


async def _acall(
    self,
    model_input: str | MaskedTag | list[str | MaskedTag],
    output_type: Literal["cfg", "json"] | None = "cfg",
    backend: str | None = None,
    **inference_kwargs: Any,
) -> Response:
    query_obj = Query(model_input)
    outlines_output_type = get_output_type(output_type, query_obj)
    outlines_model_input = str(query_obj)
    generator = Generator(self, outlines_output_type, backend)
    raw_response = await generator(outlines_model_input, **inference_kwargs)
    str_response = ensure_str(raw_response)
    return Response(str_response)
