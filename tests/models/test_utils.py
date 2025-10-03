import json

from outlines.types.dsl import CFG, JsonSchema

from gimkit.contexts import Query
from gimkit.models.utils import build_cfg, build_json_schema, infill_responses


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
    query = Query('Hello, <|MASKED id="m_0" desc="a greeting word"|>world<|/MASKED|>!')
    schema = build_json_schema(query)
    assert isinstance(schema, JsonSchema)
    
    # Parse the schema to verify structure
    schema_dict = json.loads(schema.schema)
    assert schema_dict["type"] == "object"
    assert "tags" in schema_dict["properties"]
    assert "m_0" in schema_dict["properties"]["tags"]["properties"]
    assert schema_dict["properties"]["tags"]["properties"]["m_0"]["description"] == "a greeting word"


def test_build_json_schema_multiple_tags():
    query = Query(
        'Hello, <|MASKED id="m_0" desc="first"|>world<|/MASKED|> and '
        '<|MASKED id="m_1" desc="second"|>universe<|/MASKED|>!'
    )
    schema = build_json_schema(query)
    schema_dict = json.loads(schema.schema)
    
    tags_props = schema_dict["properties"]["tags"]["properties"]
    assert "m_0" in tags_props
    assert "m_1" in tags_props
    assert tags_props["m_0"]["description"] == "first"
    assert tags_props["m_1"]["description"] == "second"
    assert schema_dict["properties"]["tags"]["required"] == ["m_0", "m_1"]


def test_build_json_schema_no_description():
    query = Query('Hello, <|MASKED id="m_0"|>world<|/MASKED|>!')
    schema = build_json_schema(query)
    schema_dict = json.loads(schema.schema)
    
    # Should have a default description
    assert "description" in schema_dict["properties"]["tags"]["properties"]["m_0"]
    assert len(schema_dict["properties"]["tags"]["properties"]["m_0"]["description"]) > 0


def test_infill_responses_json_format():
    query = Query('Hello, <|MASKED id="m_0"|><|/MASKED|>!')
    json_response = '{"tags": {"m_0": "world"}}'
    
    result = infill_responses(query, json_response, output_type="json")
    assert result.tags[0].content == "world"


def test_infill_responses_json_format_multiple_tags():
    query = Query('Hello, <|MASKED id="m_0"|><|/MASKED|> and <|MASKED id="m_1"|><|/MASKED|>!')
    json_response = '{"tags": {"m_0": "world", "m_1": "universe"}}'
    
    result = infill_responses(query, json_response, output_type="json")
    assert result.tags[0].content == "world"
    assert result.tags[1].content == "universe"


def test_infill_responses_cfg_format():
    query = Query('Hello, <|MASKED id="m_0"|><|/MASKED|>!')
    cfg_response = '<|GIM_RESPONSE|><|MASKED id="m_0"|>world<|/MASKED|><|/GIM_RESPONSE|>'
    
    result = infill_responses(query, cfg_response, output_type="cfg")
    assert result.tags[0].content == "world"

