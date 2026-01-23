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

    Example:
    ```python
    print(build_cfg(query))
    %llguidance {}

    start: "<|GIM_RESPONSE|>" REGEX "<|MASKED id=\"m_0\"|>" m_0 REGEX "<|MASKED id=\"m_1\"|>" m_1 REGEX "<|MASKED id=\"m_2\"|>" m_2 REGEX "<|MASKED id=\"m_3\"|>" m_3 REGEX "<|MASKED id=\"m_4\"|>" m_4 REGEX "<|MASKED id=\"m_5\"|>" m_5 REGEX "<|MASKED id=\"m_6\"|>" m_6 REGEX "<|/GIM_RESPONSE|>"
    REGEX: /\s*/
    m_0[capture, suffix="<|/MASKED|>"]: M_0
    M_0: /CO₂|二氧化碳/
    m_1[capture, suffix="<|/MASKED|>"]: M_1
    M_1: /(?s:.*)/
    m_2[capture, suffix="<|/MASKED|>"]: M_2
    M_2: /(?s:.*)/
    m_3[capture, suffix="<|/MASKED|>"]: M_3
    M_3: /(?s:.*)/
    m_4[capture, suffix="<|/MASKED|>"]: M_4
    M_4: /(?s:.*)/
    m_5[capture, suffix="<|/MASKED|>"]: M_5
    M_5: /(?s:.*)/
    m_6[capture, suffix="<|/MASKED|>"]: M_6
    M_6: /(?s:.*)/
    ```
    """
    num_tags = len(query.tags)

    # 1. 头部声明
    lines = ["%llguidance {}"]

    # 2. 构建 start 规则
    # 目标格式: start: "PREFIX" REGEX "OPEN_TAG_0" m_0 REGEX "OPEN_TAG_1" m_1 ... REGEX "SUFFIX"
    start_parts = [f'"{RESPONSE_PREFIX}"']

    for i in range(num_tags):
        # 添加空白符规则引用
        start_parts.append("REGEX")

        # 添加开始标签的字面量，例如: "<|MASKED id=\"m_0\"|>"
        # 注意转义: id=\"m_{i}\"
        open_tag_str = f'"{TAG_OPEN_LEFT} id=\\"m_{i}\\"{TAG_OPEN_RIGHT}"'
        start_parts.append(open_tag_str)

        # 添加内容规则引用 (小写 m_i)
        start_parts.append(f"m_{i}")

    # 添加结尾的空白符和后缀
    start_parts.append("REGEX")
    start_parts.append(f'"{RESPONSE_SUFFIX}"')

    lines.append(f"start: {' '.join(start_parts)}")

    # 3. 定义空白符规则 (命名为 REGEX 以匹配你的合法示例，通常也可以叫 WS)
    lines.append(r"REGEX: /\s*/")

    # 4. 生成每个 tag 的具体规则
    for i, tag in enumerate(query.tags):
        # 注意：配合 suffix 使用时，使用贪婪匹配 /(?s:.*)/ 而不是 /(?s:.)*?/ 是正确且合法的。
        pattern = f"/{tag.regex}/" if tag.regex else "/(?s:.*)/"

        # 规则 m_i (逻辑层):
        # - capture: 告诉引擎捕获这个部分。
        # - suffix: 指定结束标签，引擎遇到它会停止并消费它。
        # 注意：这里引用 TAG_END 常量 (即 "<|/MASKED|>")
        lines.append(f'm_{i}[capture, suffix="{TAG_END}"]: M_{i}')

        # 规则 M_i (正则层):
        # 定义实际的匹配模式
        lines.append(f"M_{i}: {pattern}")

        # TODO: "/(?s:.*)/" 的 tags 可能有很多个, 可以将具有相同 pattern 的规则合并以优化效率

    # 5. 组合最终字符串
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
