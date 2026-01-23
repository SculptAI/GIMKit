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

    Constructs a flattened grammar structure compatible with LLGuidance's suffix/capture logic.

    Ref:
    - https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md: Incomplete documentation of llguidance grammar syntax
    - https://github.com/guidance-ai/guidance/blob/main/guidance/_ast.py: LarkSerializer implementation
    - https://github.com/guidance-ai/llguidance: Source code

    Real-World Example:
    ```python
    query = '<|GIM_QUERY|>The capital of <|MASKED desc="single word" regex="中国|法国"|><|/MASKED|> is Beijing<|MASKED desc="punctuation mark" regex="\\."|><|/MASKED|><|/GIM_QUERY|>'
    print(repr(build_cfg(Query(query))))
    >>> '%llguidance {}\nstart: "<|GIM_RESPONSE|>" REGEX "<|MASKED id=\\"m_0\\"|>" m_0 REGEX "<|MASKED id=\\"m_1\\"|>" m_1 REGEX "<|/GIM_RESPONSE|>"\nREGEX: /\\s*/\nm_0[capture, suffix="<|/MASKED|>"]: T_0\nm_1[capture, suffix="<|/MASKED|>"]: T_1\nT_0: /中国|法国/\nT_1: /\\./\n'
    ```
    """
    num_tags = len(query.tags)

    # 1. Header declaration
    lines = ["%llguidance {}"]

    # 2. Build start rule
    # Target format: start: "PREFIX" REGEX "OPEN_TAG_0" m_0 REGEX "OPEN_TAG_1" m_1 ... REGEX "SUFFIX"
    start_parts = [f'"{RESPONSE_PREFIX}"']

    for i in range(num_tags):
        # Add whitespace rule reference
        start_parts.append("REGEX")

        # Add opening tag literal, e.g.: "<|MASKED id=\"m_0\"|>"
        # Note escaping: id=\"m_{i}\"
        open_tag_str = f'"{TAG_OPEN_LEFT} id=\\"m_{i}\\"{TAG_OPEN_RIGHT}"'
        start_parts.append(open_tag_str)

        # Add content rule reference (lowercase m_i)
        start_parts.append(f"m_{i}")

    # Add trailing whitespace and suffix
    start_parts.append("REGEX")
    start_parts.append(f'"{RESPONSE_SUFFIX}"')

    lines.append(f"start: {' '.join(start_parts)}")

    # 3. Define whitespace rule (named REGEX to match examples, usually can also be called WS)
    lines.append(r"REGEX: /\s*/")

    # 4. Collect unique patterns and create a mapping for terminal reuse
    # This optimization avoids creating duplicate terminal rules for tags with the same regex
    unique_pattern_terminals: dict[str, str] = {}
    terminal_definitions: list[str] = []

    for i, tag in enumerate(query.tags):
        # Note: When used with suffix, using greedy match /(?s:.*)/ instead of /(?s:.)*?/ is correct and legal.
        pattern = f"/{tag.regex}/" if tag.regex else "/(?s:.*)/"

        # Get or create a shared terminal for this pattern
        if pattern not in unique_pattern_terminals:
            # Create a new terminal name for this unique pattern
            terminal_name = f"T_{len(unique_pattern_terminals)}"
            unique_pattern_terminals[pattern] = terminal_name
            terminal_definitions.append(f"{terminal_name}: {pattern}")

        terminal_name = unique_pattern_terminals[pattern]

        # Rule m_i (logical layer):
        # - capture: tells the engine to capture this part.
        # - suffix: specifies the ending tag, the engine stops and consumes it when encountered.
        # Note: Here we reference the TAG_END constant (i.e., "<|/MASKED|>")
        lines.append(f'm_{i}[capture, suffix="{TAG_END}"]: {terminal_name}')

    # 5. Add all unique terminal definitions
    lines.extend(terminal_definitions)

    # 6. Assemble final string
    grammar = "\n".join(lines) + "\n"

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
