from typing import Literal, overload

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


def get_grammar_spec(grammar: str) -> str:
    from llguidance import grammar_from

    # Borrowed from outlines source code at https://github.com/dottxt-ai/outlines/blob/87234d202924acce84ead694f8d06748608fd5f9/outlines/backends/llguidance.py#L296-L299
    # We try both lark and ebnf
    try:
        grammar_spec = grammar_from("grammar", grammar)
    except ValueError:  # pragma: no cover
        grammar_spec = grammar_from("lark", grammar)

    return grammar_spec


def validate_grammar_spec(grammar_spec: str) -> tuple[bool, list[str]]:
    from llguidance import LLMatcher

    is_error, msgs = LLMatcher.validate_grammar_with_warnings(grammar_spec)
    return is_error, msgs


def build_cfg(query: Query) -> CFG:
    """Build an LLGuidance context-free grammar (CFG) output type based on the query object."""
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

    is_error, msgs = validate_grammar_spec(get_grammar_spec(grammar))
    if is_error:
        raise ValueError(
            "Invalid CFG grammar constructed from the query object:\n"
            + "\n".join(msgs)
            + "\nWe recommend checking the syntax documentation at https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md"
        )
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
    """Transform the model input and output type to Outlines-compatible formats."""
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
    if not isinstance(json_obj, dict):
        raise ValueError(f"Expected JSON response to be a dictionary, got {type(json_obj)}")

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
    """Infill the provided query with content from the GIM responses or JSON responses."""
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

    return [infill_responses(query, resp, json_responses=json_responses) for resp in responses]
