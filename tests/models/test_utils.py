import json

import pytest

from outlines.inputs import Chat
from outlines.types.dsl import CFG, JsonSchema

from gimkit import guide as g
from gimkit.contexts import Query
from gimkit.models.utils import (
    build_cfg,
    build_json_schema,
    infill_responses,
    json_to_response_string,
    transform_to_outlines,
)
from gimkit.prompts import SYSTEM_PROMPT_MSG


def test_build_cfg():
    query = Query('Hello, <|MASKED id="m_0"|>world<|/MASKED|>!')
    grm = (
        'start: "<|GIM_RESPONSE|>" tag0 "<|/GIM_RESPONSE|>"\n'
        'tag0: "<|MASKED id=\\"m_0\\"|>" /(?s:.)*?/ "<|/MASKED|>"'
    )
    cfg = build_cfg(query)
    assert isinstance(cfg, CFG)
    assert cfg.definition == grm


def test_build_json_schema():
    # Test with basic query
    query = Query('Hello, <|MASKED id="m_0"|>world<|/MASKED|>!')
    schema = build_json_schema(query)
    assert isinstance(schema, JsonSchema)
    
    schema_dict = json.loads(schema.schema)
    assert schema_dict["type"] == "object"
    assert "m_0" in schema_dict["properties"]
    assert schema_dict["properties"]["m_0"]["type"] == "string"
    assert "m_0" in schema_dict["required"]
    assert schema_dict["additionalProperties"] is False


def test_build_json_schema_with_regex():
    # Test with regex pattern
    query = Query('My number: ' + g(name='num', regex=r'\d+'))
    schema = build_json_schema(query)
    
    schema_dict = json.loads(schema.schema)
    assert schema_dict["properties"]["m_0"]["pattern"] == r'\d+'


def test_build_json_schema_with_description():
    # Test with description
    query = Query('Name: ' + g.person_name(name='name'))
    schema = build_json_schema(query)
    
    schema_dict = json.loads(schema.schema)
    assert "description" in schema_dict["properties"]["m_0"]
    assert "person's name" in schema_dict["properties"]["m_0"]["description"].lower()


def test_build_json_schema_multiple_tags():
    # Test with multiple tags
    query = Query(
        'My name is ' + g.person_name(name='name') +
        ' and I like ' + g.select(name='hobby', choices=['reading', 'coding']) + '.'
    )
    schema = build_json_schema(query)
    
    schema_dict = json.loads(schema.schema)
    assert len(schema_dict["properties"]) == 2
    assert "m_0" in schema_dict["properties"]
    assert "m_1" in schema_dict["properties"]
    assert schema_dict["properties"]["m_1"]["pattern"] == "reading|coding"


def test_json_to_response_string():
    # Test basic conversion
    json_response = {"m_0": "Alice"}
    response_str = json_to_response_string(json_response)
    assert response_str == '<|GIM_RESPONSE|><|MASKED id="m_0"|>Alice<|/MASKED|><|/GIM_RESPONSE|>'


def test_json_to_response_string_multiple():
    # Test multiple tags
    json_response = {"m_0": "Alice", "m_1": "reading"}
    response_str = json_to_response_string(json_response)
    expected = (
        '<|GIM_RESPONSE|>'
        '<|MASKED id="m_0"|>Alice<|/MASKED|>'
        '<|MASKED id="m_1"|>reading<|/MASKED|>'
        '<|/GIM_RESPONSE|>'
    )
    assert response_str == expected


def test_json_to_response_string_ordering():
    # Test that tags are ordered correctly even if dict is unordered
    json_response = {"m_2": "third", "m_0": "first", "m_1": "second"}
    response_str = json_to_response_string(json_response)
    expected = (
        '<|GIM_RESPONSE|>'
        '<|MASKED id="m_0"|>first<|/MASKED|>'
        '<|MASKED id="m_1"|>second<|/MASKED|>'
        '<|MASKED id="m_2"|>third<|/MASKED|>'
        '<|/GIM_RESPONSE|>'
    )
    assert response_str == expected


def test_infill_responses_with_json():
    # Test infill with JSON response
    query = Query('Hello, ' + g.person_name(name='name') + '!')
    json_response = {"m_0": "Alice"}
    result = infill_responses(query, json_response)
    assert str(result) == "Hello, Alice!"


def test_infill_responses_with_json_list():
    # Test infill with list of JSON responses
    query = Query('Hello, ' + g.person_name(name='name') + '!')
    json_responses = [{"m_0": "Alice"}, {"m_0": "Bob"}]
    results = infill_responses(query, json_responses)
    assert len(results) == 2
    assert str(results[0]) == "Hello, Alice!"
    assert str(results[1]) == "Hello, Bob!"


def test_infill_responses_with_mixed_list():
    # Test infill with mixed list of string and JSON responses
    query = Query('Hello, ' + g.person_name(name='name') + '!')
    responses = [
        '<|GIM_RESPONSE|><|MASKED id="m_0"|>Alice<|/MASKED|><|/GIM_RESPONSE|>',
        {"m_0": "Bob"}
    ]
    results = infill_responses(query, responses)
    assert len(results) == 2
    assert str(results[0]) == "Hello, Alice!"
    assert str(results[1]) == "Hello, Bob!"


def test_infill_responses_error_handling():
    # Test error handling for invalid response types
    query = Query('Hello, ' + g.person_name(name='name') + '!')
    
    # Invalid type
    with pytest.raises(TypeError, match="Expected responses to be str, dict, or list"):
        infill_responses(query, 123)
    
    # Empty list
    with pytest.raises(ValueError, match="Response list is empty"):
        infill_responses(query, [])
    
    # Invalid items in list
    with pytest.raises(TypeError, match="All items in the response list must be str or dict"):
        infill_responses(query, [123, 456])


def test_transform_to_outlines():
    query = Query('Hello, <|MASKED id="m_0"|>world<|/MASKED|>!')

    # Test CFG output type without GIM prompt
    model_input, output_type = transform_to_outlines(query, output_type="cfg", use_gim_prompt=False)
    assert isinstance(model_input, str)
    assert isinstance(output_type, CFG)
    assert 'start: "<|GIM_RESPONSE|>" tag0 "<|/GIM_RESPONSE|>"' in output_type.definition

    # Test with GIM prompt
    model_input, output_type = transform_to_outlines(query, output_type="cfg", use_gim_prompt=True)
    assert isinstance(model_input, Chat)
    assert model_input.messages[0] == SYSTEM_PROMPT_MSG
    assert isinstance(output_type, CFG)
    
    # Test JSON output type
    model_input, output_type = transform_to_outlines(query, output_type="json", use_gim_prompt=False)
    assert isinstance(model_input, str)
    assert isinstance(output_type, JsonSchema)
    
    # Test JSON output type with GIM prompt
    model_input, output_type = transform_to_outlines(query, output_type="json", use_gim_prompt=True)
    assert isinstance(model_input, Chat)
    assert isinstance(output_type, JsonSchema)
