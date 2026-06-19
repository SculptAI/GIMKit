from collections.abc import Sequence
from typing import Any, Literal, cast, overload

from outlines.inputs import Chat
from outlines.types.dsl import CFG, JsonSchema

from gimkit.contexts import Query, Response, Result, infill
from gimkit.dsls import build_cfg, build_json_schema
from gimkit.models.types import ErrorMode, GenerationResult
from gimkit.prompts import (
    DEMO_CONVERSATION_MSGS,
    DEMO_CONVERSATION_MSGS_JSON,
    SYSTEM_PROMPT_MSG,
    SYSTEM_PROMPT_MSG_JSON,
)
from gimkit.schemas import ContextInput, MaskedTag, TagField


def _ensure_query(model_input: ContextInput | Query) -> Query:
    return model_input if isinstance(model_input, Query) else Query(model_input)


def get_outlines_model_input(
    model_input: ContextInput | Query,
    output_type: Literal["cfg", "json"] | None,
    use_gim_prompt: bool,
    visible_tag_fields: list[TagField] | None = None,
) -> str | Chat:
    """Transform the model input to an Outlines-compatible format.

    Args:
        model_input: The query input, either a raw context or a Query object.
        output_type: The output type for the model ("cfg", "json", or None).
        use_gim_prompt: Whether to wrap the input with a GIM system prompt.
        visible_tag_fields: The tag fields to include in the serialized query string.
            Controls which attributes of each MaskedTag are visible to the model.
            If None, uses the Query default (["id", "desc", "content"]).
            Example: ["id", "name", "desc", "content", "regex"] to expose all fields.
    """
    query_obj = _ensure_query(model_input)
    outlines_model_input: str | Chat = (
        query_obj.to_string(fields=visible_tag_fields)
        if visible_tag_fields is not None
        else str(query_obj)
    )

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

    return outlines_model_input


def get_outlines_model_inputs(
    model_inputs: Sequence[ContextInput | Query],
    output_type: Literal["cfg", "json"] | None,
    use_gim_prompt: bool,
    visible_tag_fields: list[TagField] | None = None,
) -> list[str | Chat]:
    """Transform a batch of model inputs to Outlines-compatible formats."""
    if len(model_inputs) == 0:
        raise ValueError("Batch input list is empty.")
    return [
        get_outlines_model_input(
            model_input,
            output_type,
            use_gim_prompt,
            visible_tag_fields=visible_tag_fields,
        )
        for model_input in model_inputs
    ]


def get_outlines_output_type(
    model_input: ContextInput | Query, output_type: Literal["cfg", "json"] | None
) -> None | CFG | JsonSchema:
    """Transform the output type to an Outlines-compatible format."""
    query_obj = _ensure_query(model_input)
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


def validate_error_mode(error_mode: ErrorMode) -> None:
    if error_mode not in ("raise", "collect"):
        raise ValueError(f"Invalid error mode: {error_mode}. Expected 'raise' or 'collect'.")


@overload
def parse_generation_response(
    query: ContextInput | Query,
    raw_response: str,
    *,
    json_response: bool = False,
    error_mode: Literal["raise"] = "raise",
) -> Result: ...


@overload
def parse_generation_response(
    query: ContextInput | Query,
    raw_response: str,
    *,
    json_response: bool = False,
    error_mode: Literal["collect"],
) -> GenerationResult: ...


def parse_generation_response(
    query: ContextInput | Query,
    raw_response: str,
    *,
    json_response: bool = False,
    error_mode: ErrorMode = "raise",
) -> Result | GenerationResult:
    """Parse and infill one raw model generation.

    ``collect`` only isolates errors raised while parsing and infilling an
    already generated string. Model invocation and response-container errors
    remain whole-call failures.
    """
    validate_error_mode(error_mode)
    if not isinstance(raw_response, str):
        raise TypeError(f"Expected raw response to be str, got {type(raw_response)}")

    try:
        result = infill_responses(query, raw_response, json_responses=json_response)
    except Exception as exc:
        if error_mode == "raise":
            raise
        return GenerationResult(
            raw_response=raw_response,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    if error_mode == "raise":
        return result
    return GenerationResult(raw_response=raw_response, result=result)


@overload
def parse_generation_responses(
    query: ContextInput | Query,
    raw_responses: str | list[str],
    *,
    json_responses: bool = False,
    error_mode: Literal["raise"] = "raise",
) -> Result | list[Result]: ...


@overload
def parse_generation_responses(
    query: ContextInput | Query,
    raw_responses: str | list[str],
    *,
    json_responses: bool = False,
    error_mode: Literal["collect"],
) -> GenerationResult | list[GenerationResult]: ...


def parse_generation_responses(
    query: ContextInput | Query,
    raw_responses: str | list[str],
    *,
    json_responses: bool = False,
    error_mode: ErrorMode = "raise",
) -> Result | list[Result] | GenerationResult | list[GenerationResult]:
    """Parse one or more raw generations while preserving their container shape."""
    validate_error_mode(error_mode)
    if isinstance(raw_responses, str):
        return parse_generation_response(
            query,
            raw_responses,
            json_response=json_responses,
            error_mode=error_mode,
        )
    if not isinstance(raw_responses, list):
        raise TypeError(f"Expected responses to be str or list of str, got {type(raw_responses)}")
    if len(raw_responses) == 0:
        raise ValueError("Response list is empty.")
    if not all(isinstance(response, str) for response in raw_responses):
        raise TypeError(f"All items in the response list must be strings, got: {raw_responses}")

    parsed = [
        cast("Any", parse_generation_response)(
            query,
            raw_response,
            json_response=json_responses,
            error_mode=error_mode,
        )
        for raw_response in raw_responses
    ]
    return cast("list[Result] | list[GenerationResult]", parsed)


@overload
def parse_batch_generation_responses(
    queries: Sequence[ContextInput | Query],
    raw_responses: list[list[str]],
    *,
    json_responses: bool = False,
    error_mode: Literal["raise"] = "raise",
) -> list[list[Result]]: ...


@overload
def parse_batch_generation_responses(
    queries: Sequence[ContextInput | Query],
    raw_responses: list[list[str]],
    *,
    json_responses: bool = False,
    error_mode: Literal["collect"],
) -> list[list[GenerationResult]]: ...


def parse_batch_generation_responses(
    queries: Sequence[ContextInput | Query],
    raw_responses: list[list[str]],
    *,
    json_responses: bool = False,
    error_mode: ErrorMode = "raise",
) -> list[list[Result]] | list[list[GenerationResult]]:
    """Parse batch generations, preserving query and candidate dimensions."""
    validate_error_mode(error_mode)
    if len(queries) == 0:
        raise ValueError("Batch input list is empty.")
    if not isinstance(raw_responses, list):
        raise TypeError(f"Expected batch responses to be a list, got {type(raw_responses)}")
    if len(queries) != len(raw_responses):
        raise ValueError(
            "Mismatched number of batch inputs and responses: "
            f"{len(queries)} input(s), {len(raw_responses)} response group(s)."
        )
    if not all(isinstance(response_group, list) for response_group in raw_responses):
        invalid_group = next(
            response_group
            for response_group in raw_responses
            if not isinstance(response_group, list)
        )
        raise TypeError(
            f"Each batch response group must be a list of strings, got {type(invalid_group)}"
        )
    for response_group in raw_responses:
        if len(response_group) == 0:
            raise ValueError("Response list is empty.")
        if not all(isinstance(response, str) for response in response_group):
            raise TypeError(
                f"All items in the response list must be strings, got: {response_group}"
            )

    parsed = [
        cast(
            "list[Result] | list[GenerationResult]",
            cast("Any", parse_generation_responses)(
                query,
                response_group,
                json_responses=json_responses,
                error_mode=error_mode,
            ),
        )
        for query, response_group in zip(queries, raw_responses, strict=True)
    ]
    return cast("list[list[Result]] | list[list[GenerationResult]]", parsed)


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


@overload
def infill_batch_responses(
    queries: Sequence[ContextInput | Query], responses: list[str], json_responses: bool = False
) -> list[Result]: ...


@overload
def infill_batch_responses(
    queries: Sequence[ContextInput | Query],
    responses: list[list[str]],
    json_responses: bool = False,
) -> list[list[Result]]: ...


def infill_batch_responses(
    queries: Sequence[ContextInput | Query],
    responses: list[str] | list[list[str]],
    json_responses: bool = False,
) -> list[Result] | list[list[Result]]:
    """Infill each query in a batch with its corresponding response(s)."""
    if len(queries) == 0:
        raise ValueError("Batch input list is empty.")
    if not isinstance(responses, list):
        raise TypeError(f"Expected batch responses to be a list, got {type(responses)}")
    if len(queries) != len(responses):
        raise ValueError(
            "Mismatched number of batch inputs and responses: "
            f"{len(queries)} input(s), {len(responses)} response(s)."
        )

    if all(isinstance(response, str) for response in responses):
        return [
            infill_responses(query, cast("str", response), json_responses=json_responses)
            for query, response in zip(queries, responses, strict=True)
        ]

    if all(isinstance(response, list) for response in responses):
        return [
            infill_responses(query, cast("list[str]", response), json_responses=json_responses)
            for query, response in zip(queries, responses, strict=True)
        ]

    invalid_response = next(
        response for response in responses if not isinstance(response, (str, list))
    )
    raise TypeError(
        f"Each batch response must be a string or a list of strings, got {type(invalid_response)}"
    )
