from outlines.inputs import Chat
from outlines.types.dsl import CFG

from gimkit.contexts import Query
from gimkit.models.utils import build_cfg, transform_to_outlines
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


def test_build_cfg_with_regex():
    """Test build_cfg with regex pattern."""
    query = Query('Number: <|MASKED id="m_0" regex="\\d+"|><|/MASKED|>')
    grm = (
        'start: "<|GIM_RESPONSE|>" tag0 "<|/GIM_RESPONSE|>"\n'
        'tag0: "<|MASKED id=\\"m_0\\"|>" /\\d+/ "<|/MASKED|>"'
    )
    cfg = build_cfg(query)
    assert isinstance(cfg, CFG)
    assert cfg.definition == grm


def test_build_cfg_with_grammar():
    """Test build_cfg with EBNF grammar pattern."""
    query = Query('Word: <|MASKED id="m_0" grammar="word: /[a-z]+/"|><|/MASKED|>')
    grm = (
        'start: "<|GIM_RESPONSE|>" tag0 "<|/GIM_RESPONSE|>"\n'
        'tag0: "<|MASKED id=\\"m_0\\"|>" word: /[a-z]+/ "<|/MASKED|>"'
    )
    cfg = build_cfg(query)
    assert isinstance(cfg, CFG)
    assert cfg.definition == grm


def test_build_cfg_with_both_regex_and_grammar():
    """Test that grammar takes precedence over regex when both are provided."""
    query = Query(
        'Item: <|MASKED id="m_0" regex="\\d+" grammar="number: /[0-9]+/"|><|/MASKED|>'
    )
    grm = (
        'start: "<|GIM_RESPONSE|>" tag0 "<|/GIM_RESPONSE|>"\n'
        'tag0: "<|MASKED id=\\"m_0\\"|>" number: /[0-9]+/ "<|/MASKED|>"'
    )
    cfg = build_cfg(query)
    assert isinstance(cfg, CFG)
    assert cfg.definition == grm


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
