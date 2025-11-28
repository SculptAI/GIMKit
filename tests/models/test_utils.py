import pytest

from outlines.inputs import Chat
from outlines.types.dsl import CFG, JsonSchema

from gimkit.contexts import Query, Result
from gimkit.models.utils import (
    get_outlines_model_input,
    get_outlines_output_type,
    infill_responses,
    json_responses_to_gim_response,
)
from gimkit.prompts import SYSTEM_PROMPT_MSG, SYSTEM_PROMPT_MSG_JSON
from gimkit.schemas import MaskedTag


def test_get_outlines_model_input():
    query = Query('Hello, <|MASKED id="m_0"|><|/MASKED|>!')

    # Test without GIM prompt
    model_input = get_outlines_model_input(query, output_type=None, use_gim_prompt=False)
    assert isinstance(model_input, str)
    assert model_input == '<|GIM_QUERY|>Hello, <|MASKED id="m_0"|><|/MASKED|>!<|/GIM_QUERY|>'

    # Test with GIM prompt
    model_input_with_prompt = get_outlines_model_input(query, output_type=None, use_gim_prompt=True)
    assert isinstance(model_input_with_prompt, Chat)
    assert model_input_with_prompt.messages[0] == SYSTEM_PROMPT_MSG
    assert (
        model_input_with_prompt.messages[2]["content"]
        == '<|GIM_RESPONSE|><|MASKED id="m_0"|>nice to meet you<|/MASKED|><|/GIM_RESPONSE|>'
    )

    # Test with JSON mode
    model_input_json = get_outlines_model_input(query, output_type="json", use_gim_prompt=True)
    assert isinstance(model_input_json, Chat)
    assert model_input_json.messages[0] == SYSTEM_PROMPT_MSG_JSON
    assert model_input_json.messages[2]["content"] == '{"m_0": "nice to meet you"}'


def test_get_outlines_output_type():
    query = Query('Hello, <|MASKED id="m_0"|>world<|/MASKED|>!')
    assert get_outlines_output_type(query, None) is None
    assert isinstance(get_outlines_output_type(query, "cfg"), CFG)
    assert isinstance(get_outlines_output_type(query, "json"), JsonSchema)
    with pytest.raises(ValueError, match="Invalid output type: xxx"):
        get_outlines_output_type(query, "xxx")


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


def test_json_responses_to_gim_response_with_valid_json_no_warning(caplog):
    """Test that no warning is emitted when JSON is already valid."""
    import logging

    caplog.set_level(logging.WARNING)

    json_str = '{"m_0": "John", "m_1": "Doe"}'
    expected_gim_str = '<|GIM_RESPONSE|><|MASKED id="m_0"|>John<|/MASKED|><|MASKED id="m_1"|>Doe<|/MASKED|><|/GIM_RESPONSE|>'
    result = json_responses_to_gim_response(json_str)

    assert result == expected_gim_str
    # No warning should be logged for valid JSON
    assert "JSON response required repair" not in caplog.text


def test_json_responses_to_gim_response_with_repaired_json_warning(caplog):
    """Test that a warning is emitted when JSON needs repair."""
    import logging

    caplog.set_level(logging.WARNING)

    # Malformed JSON with missing closing quote
    json_str = '{"m_0": "John, "m_1": "Doe"}'
    expected_gim_str = '<|GIM_RESPONSE|><|MASKED id="m_0"|>John<|/MASKED|><|MASKED id="m_1"|>Doe<|/MASKED|><|/GIM_RESPONSE|>'
    result = json_responses_to_gim_response(json_str)

    assert result == expected_gim_str
    # Warning should be logged when JSON is repaired
    assert "JSON response required repair" in caplog.text
    assert json_str in caplog.text


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
