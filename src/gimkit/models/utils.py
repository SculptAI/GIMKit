from typing import Literal, overload

from outlines.inputs import Chat
from outlines.types.dsl import CFG, JsonSchema

from gimkit.contexts import Query, Response, Result, infill
from gimkit.dsls import build_cfg, build_json_schema
from gimkit.prompts import (
    DEMO_CONVERSATION_MSGS,
    DEMO_CONVERSATION_MSGS_JSON,
    SYSTEM_PROMPT_MSG,
    SYSTEM_PROMPT_MSG_JSON,
)
from gimkit.schemas import ContextInput, MaskedTag


def get_outlines_model_input(
    model_input: ContextInput | Query,
    output_type: Literal["cfg", "json"] | None,
    use_gim_prompt: bool,
    force_chat_input: bool = False,
) -> str | Chat:
    """Transform the model input to an Outlines-compatible format."""
    query_obj = Query(model_input) if not isinstance(model_input, Query) else model_input
    outlines_model_input: str | Chat = str(query_obj)

    if use_gim_prompt:
        # Use JSON-specific prompts when output_type is "json"
        if output_type == "json":
            system_prompt = SYSTEM_PROMPT_MSG_JSON
            demo_msgs = DEMO_CONVERSATION_MSGS_JSON
        else:
            system_prompt = SYSTEM_PROMPT_MSG
            demo_msgs = DEMO_CONVERSATION_MSGS
        outlines_model_input = Chat(
            [
                system_prompt,
                *demo_msgs,
                {"role": "user", "content": outlines_model_input},
            ]
        )

    if force_chat_input and isinstance(outlines_model_input, str):
        outlines_model_input = Chat([{"role": "user", "content": outlines_model_input}])

    return outlines_model_input


def get_outlines_output_type(
    model_input: ContextInput | Query, output_type: Literal["cfg", "json"] | None
) -> None | CFG | JsonSchema:
    """Transform the output type to an Outlines-compatible format."""
    query_obj = Query(model_input) if not isinstance(model_input, Query) else model_input
    if output_type is None:
        return None
    elif output_type == "cfg":
        return CFG(build_cfg(query_obj))
    elif output_type == "json":
        return JsonSchema(build_json_schema(query_obj))
    else:
        raise ValueError(f"Invalid output type: {output_type}")


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

    from gimkit.log import get_logger

    logger = get_logger(__name__)

    result = json_repair.loads(json_response, logging=True)
    # When logging=True, json_repair.loads returns a tuple (json_obj, repair_log)
    if isinstance(result, tuple):
        json_obj, repair_log = result
        if repair_log:
            logger.warning(
                "JSON response required repair. Original: %s, Repair actions: %s",
                json_response,
                repair_log,
            )
    else:  # pragma: no cover
        # This shouldn't happen when logging=True, but handle gracefully
        json_obj = result  # type: ignore[assignment]
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
