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
    return CFG(f"{grammar_first_line}\n{grammar_next_lines}")


def build_json_schema(query: Query) -> JsonSchema:
    """Build a JSON schema output type based on the query object.
    
    This creates a JSON schema that enforces the GIM response format,
    useful for non-GIM LLMs with JSON mode support.
    """
    from outlines.types.dsl import JsonSchema
    
    # Build the schema for tags
    tag_properties = {}
    required_tags = []
    
    for i, tag in enumerate(query.tags):
        tag_id = f"m_{i}"
        tag_desc = tag.desc or "Generated content for this masked section"
        
        tag_properties[tag_id] = {
            "type": "string",
            "description": tag_desc,
        }
        required_tags.append(tag_id)
    
    # Create the overall schema
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "object",
                "properties": tag_properties,
                "required": required_tags,
                "additionalProperties": False,
            }
        },
        "required": ["tags"],
        "additionalProperties": False,
    }
    
    import json
    return JsonSchema(json.dumps(schema))


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
):
    query_obj = Query(model_input) if not isinstance(model_input, Query) else model_input
    outlines_model_input = str(query_obj)
    outlines_output_type = get_outlines_output_type(output_type, query_obj)
    return outlines_model_input, outlines_output_type


def infill_responses(
    query: ContextInput | Query, responses: str | list[str] | Any, output_type: Literal["cfg", "json"] | None = None
) -> Result | list[Result]:
    """Infill responses into query, handling both GIM format and JSON format responses.
    
    Args:
        query: The query with masked tags
        responses: The model response(s) - either GIM format string(s) or JSON format
        output_type: The output type used to generate the response ("cfg" or "json")
    
    Returns:
        Result or list of Results with infilled content
    """
    if isinstance(responses, str):
        # Handle JSON response format
        if output_type == "json":
            import json
            try:
                json_data = json.loads(responses)
                # Convert JSON format to GIM response format
                response_str = RESPONSE_PREFIX
                if "tags" in json_data:
                    for tag_id, content in json_data["tags"].items():
                        response_str += f'<|MASKED id="{tag_id}"|>{content}<|/MASKED|>'
                response_str += RESPONSE_SUFFIX
                return infill(query, response_str)
            except (json.JSONDecodeError, KeyError):
                # If JSON parsing fails, try to infill as-is
                return infill(query, responses)
        return infill(query, responses)

    if not isinstance(responses, list):
        raise TypeError(f"Expected responses to be str or list of str, got {type(responses)}")

    if not all(isinstance(resp, str) for resp in responses):
        raise TypeError(f"All items in the response list must be strings, got: {responses}")

    if len(responses) == 0:
        raise ValueError("Response list is empty.")

    return [infill_responses(query, resp, output_type) for resp in responses]


def _call(
    self,
    model_input: ContextInput | Query,
    output_type: Literal["cfg", "json"] | None = "cfg",
    backend: str | None = None,
    **inference_kwargs: Any,
) -> Result | list[Result]:
    outlines_model_input, outlines_output_type = transform_to_outlines(model_input, output_type)
    raw_response = Generator(self, outlines_output_type, backend)(
        outlines_model_input, **inference_kwargs
    )
    return infill_responses(model_input, raw_response, output_type)


async def _acall(
    self,
    model_input: ContextInput | Query,
    output_type: Literal["cfg", "json"] | None = "cfg",
    backend: str | None = None,
    **inference_kwargs: Any,
) -> Result | list[Result]:
    outlines_model_input, outlines_output_type = transform_to_outlines(model_input, output_type)
    generator = Generator(self, outlines_output_type, backend)
    raw_responses = await generator(outlines_model_input, **inference_kwargs)
    return infill_responses(model_input, raw_responses, output_type)
