"""Define DSL builders for various output types.

- `build_cfg` constructs a context-free grammar (CFG) using LLGuidance syntax
- `build_json_schema` constructs a JSON schema representing the response structure."""

from gimkit.contexts import Query
from gimkit.schemas import (
    RESPONSE_PREFIX,
    RESPONSE_SUFFIX,
    TAG_END,
    TAG_OPEN_LEFT,
    TAG_OPEN_RIGHT,
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


def build_cfg(query: Query) -> str:
    """Build an LLGuidance context-free grammar (CFG) string based on the query object.

    LLGuidance syntax reference: https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md
    """
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
    return grammar


def build_json_schema(query: Query) -> dict:
    """Build a JSON schema dictionary based on the query object.

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
            field_schema["pattern"] = f"^({tag.regex})$"

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

    return schema
