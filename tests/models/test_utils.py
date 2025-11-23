import pytest

from outlines.inputs import Chat
from outlines.types.dsl import CFG, JsonSchema

from gimkit.contexts import Query, Result
from gimkit.models.utils import (
    get_outlines_output_type,
    infill_responses,
    json_responses_to_gim_response,
    transform_to_outlines,
)
from gimkit.prompts import SYSTEM_PROMPT_MSG, SYSTEM_PROMPT_MSG_JSON
from gimkit.schemas import MaskedTag


def test_get_outlines_output_type():
    query = Query('Hello, <|MASKED id="m_0"|>world<|/MASKED|>!')
    assert get_outlines_output_type(None, query) is None
    assert isinstance(get_outlines_output_type("cfg", query), CFG)
    assert isinstance(get_outlines_output_type("json", query), JsonSchema)
    with pytest.raises(ValueError, match="Invalid output type: xxx"):
        get_outlines_output_type("xxx", query)


def test_transform_to_outlines():
    query = Query('Hello, <|MASKED id="m_0"|><|/MASKED|>!')

    # Test CFG output type without GIM prompt
    model_input, output_type = transform_to_outlines(query, output_type="cfg", use_gim_prompt=False)
    assert isinstance(model_input, str)
    assert isinstance(output_type, CFG)
    assert 'start: "<|GIM_RESPONSE|>" tag0 "<|/GIM_RESPONSE|>"' in output_type.definition

    # Test JSON output type
    model_input, output_type = transform_to_outlines(
        query, output_type="json", use_gim_prompt=False
    )
    assert isinstance(model_input, str)
    assert isinstance(output_type, JsonSchema)

    # Test with GIM prompt and CFG output
    model_input, output_type = transform_to_outlines(query, output_type="cfg", use_gim_prompt=True)
    assert isinstance(model_input, Chat)
    assert model_input.messages[0] == SYSTEM_PROMPT_MSG
    assert isinstance(output_type, CFG)

    # Test with GIM prompt and JSON output
    model_input, output_type = transform_to_outlines(query, output_type="json", use_gim_prompt=True)
    assert isinstance(model_input, Chat)
    assert model_input.messages[0] == SYSTEM_PROMPT_MSG_JSON
    assert isinstance(output_type, JsonSchema)
    assert model_input.messages[2]["content"].startswith('{"m_0": ')
    assert model_input.messages[2]["content"].endswith("}")


def test_json_responses_to_gim_response():
    json_str = '{"m_0": "John", "m_1": "Doe"}'
    expected_gim_str = '<|GIM_RESPONSE|><|MASKED id="m_0"|>John<|/MASKED|><|MASKED id="m_1"|>Doe<|/MASKED|><|/GIM_RESPONSE|>'
    assert json_responses_to_gim_response(json_str) == expected_gim_str

    # Test with invalid key
    with pytest.raises(ValueError, match="Invalid field name in JSON response: m-1"):
        json_responses_to_gim_response('{"m_0": "John", "m-1": "Doe"}')

    # Test non-dict response
    with pytest.raises(ValueError, match="Expected JSON response to be a dictionary, got"):
        json_responses_to_gim_response('["John", "Doe"]')


def test_infill_responses():
    query = Query("Hello, ", MaskedTag(id=0), " and ", MaskedTag(id=1))
    response_str = '<|GIM_RESPONSE|><|MASKED id="m_0"|>world<|/MASKED|><|MASKED id="m_1"|>friend<|/MASKED|><|/GIM_RESPONSE|>'
    result = infill_responses(query, response_str)
    assert isinstance(result, Result)
    assert str(result) == "Hello, world and friend"

    # Test list of responses
    results = infill_responses(query, [response_str, response_str])
    assert isinstance(results, list)
    assert len(results) == 2
    assert str(results[0]) == "Hello, world and friend"

    # Test JSON response
    json_response_str = '{"m_0": "world", "m_1": "friend"}'
    result_from_json = infill_responses(query, json_response_str, json_responses=True)
    assert isinstance(result_from_json, Result)
    assert str(result_from_json) == "Hello, world and friend"

    # Test invalid response type
    with pytest.raises(TypeError, match="Expected responses to be str or list of str, got"):
        infill_responses(query, 123)

    # Test empty list
    with pytest.raises(ValueError, match="Response list is empty"):
        infill_responses(query, [])

    # Test list with non-string items
    with pytest.raises(TypeError, match="All items in the response list must be strings, got"):
        infill_responses(query, ["a", 1])
