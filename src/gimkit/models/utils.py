from typing import Any, Literal, overload

from outlines.generator import Generator
from outlines.inputs import Chat
from outlines.types.dsl import CFG, JsonSchema

from gimkit.contexts import Query, Response, Result, infill
from gimkit.prompts import DEMO_CONVERSATION_MSGS, SYSTEM_PROMPT_MSG
from gimkit.schemas import (
    RESPONSE_PREFIX,
    RESPONSE_SUFFIX,
    TAG_END,
    TAG_OPEN_LEFT,
    TAG_OPEN_RIGHT,
    ContextInput,
    MaskedTag,
)


def build_cfg(query: Query) -> CFG:
    """Build a Lark-based CFG output type based on the query object."""
    num_tags = len(query.tags)
    grammar_first_line = f'''start: "{RESPONSE_PREFIX}" {" ".join(f"tag{i}" for i in range(num_tags))} "{RESPONSE_SUFFIX}"'''

    grammar_rest_lines = []
    for i, tag in enumerate(query.tags):
        # `/(?s:.)*?/` is a non-greedy match for any character including newlines
        content_pattern = f"/{tag.regex}/" if tag.regex else "/(?s:.)*?/"
        grammar_rest_lines.append(
            f'tag{i}: "{TAG_OPEN_LEFT} id=\\"m_{i}\\"{TAG_OPEN_RIGHT}" {content_pattern} "{TAG_END}"'
        )

    grammar = grammar_first_line + "\n" + "\n".join(grammar_rest_lines)
    return CFG(grammar)


def build_json_schema(query: Query) -> JsonSchema:
    """Build a JSON schema output type based on the query object.

    The JSON schema represents the response structure where each masked tag
    becomes a field in the JSON object. The field name is "m_{id}" to match
    the tag id, and patterns are applied when regex is specified.
    """
    properties = {}
    required_fields = []

    for tag in query.tags:
        field_name = f"m_{tag.id}"
        field_schema = {"type": "string"}

        # Add regex pattern if specified
        if tag.regex is not None:
            field_schema["pattern"] = tag.regex

        # Add description if available
        if tag.desc is not None:
            field_schema["description"] = tag.desc

        properties[field_name] = field_schema
        required_fields.append(field_name)

    schema = {
        "type": "object",
        "properties": properties,
        "required": required_fields,
        "additionalProperties": False,
    }

    return JsonSchema(schema)


def get_outlines_output_type(
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


def json_responses_to_gim_response(json_response: str) -> str:
    """Convert a JSON response string to a GIM response string.

    Args:
        json_response: A JSON string representing the response.

    Returns:
        A properly formatted GIM response string.

    Raises:
        ValueError: If any key does not follow the "m_X" format where X is an integer.
    """
    import re

    import json_repair

    json_obj = json_repair.loads(json_response)
    assert isinstance(json_obj, dict), "Expected JSON response to be a dictionary."

    validated_items = []
    for field_name, content in json_obj.items():
        match_result = re.fullmatch(r"m_(\d+)", field_name)
        if not match_result:
            raise ValueError(
                f"Invalid field name in JSON response: {field_name}. Expected format 'm_X' where X is an integer."
            )
        tag_id = int(match_result.group(1))
        validated_items.append((tag_id, content))

    validated_items.sort(key=lambda x: x[0])
    return str(
        Response([MaskedTag(id=tag_id, content=content) for tag_id, content in validated_items])
    )


@overload
def infill_responses(
    query: ContextInput | Query, responses: str, json_responses: bool = False
) -> Result: ...


@overload
def infill_responses(
    query: ContextInput | Query, responses: list[str], json_responses: bool = False
) -> list[Result]: ...


def infill_responses(
    query: ContextInput | Query, responses: str | list[str], json_responses: bool = False
) -> Result | list[Result]:
    # Handle single string response
    if isinstance(responses, str):
        if json_responses:
            responses = json_responses_to_gim_response(responses)
        return infill(query, responses)

    # Handle list of responses
    if not isinstance(responses, list):
        raise TypeError(f"Expected responses to be str or list of str, got {type(responses)}")

    if len(responses) == 0:
        raise ValueError("Response list is empty.")

    if not all(isinstance(resp, str) for resp in responses):
        raise TypeError(f"All items in the response list must be strings, got: {responses}")

    return [infill_responses(query, resp) for resp in responses]


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
    raw_responses = Generator(self, outlines_output_type, backend)(
        outlines_model_input, **inference_kwargs
    )
    assert isinstance(raw_responses, str | list[str]), (
        "Expected raw responses to be a string or list."
    )
    return infill_responses(model_input, raw_responses, json_responses=(output_type == "json"))


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
    return infill_responses(model_input, raw_responses, json_responses=(output_type == "json"))
