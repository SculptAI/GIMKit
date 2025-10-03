import pytest

from gimkit.prompts import (
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    build_few_shot_messages,
    build_few_shot_prompt,
)
from gimkit.schemas import QUERY_PREFIX, QUERY_SUFFIX, RESPONSE_PREFIX, RESPONSE_SUFFIX


def test_system_prompt():
    """Test that the system prompt is defined and non-empty."""
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 0
    assert "GIM" in SYSTEM_PROMPT or "Guided Infilling" in SYSTEM_PROMPT
    assert "MASKED" in SYSTEM_PROMPT


def test_few_shot_examples():
    """Test that few-shot examples are properly formatted."""
    assert isinstance(FEW_SHOT_EXAMPLES, list)
    assert len(FEW_SHOT_EXAMPLES) > 0

    for example in FEW_SHOT_EXAMPLES:
        assert "query" in example
        assert "response" in example
        assert isinstance(example["query"], str)
        assert isinstance(example["response"], str)

        # Check query format
        assert example["query"].startswith(QUERY_PREFIX)
        assert example["query"].endswith(QUERY_SUFFIX)
        assert "<|MASKED" in example["query"]

        # Check response format
        assert example["response"].startswith(RESPONSE_PREFIX)
        assert example["response"].endswith(RESPONSE_SUFFIX)
        assert "<|MASKED" in example["response"]


def test_build_few_shot_messages_openai_format():
    """Test building few-shot messages in OpenAI format."""
    query = f'{QUERY_PREFIX}Hello, <|MASKED id="m_0"|><|/MASKED|>!{QUERY_SUFFIX}'

    # Test with default number of examples (3)
    messages = build_few_shot_messages(query, num_examples=3, message_format="openai")

    # Should have: 1 system + 3 examples (3 user + 3 assistant) + 1 user query = 8 messages
    assert len(messages) == 8

    # First message should be system
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT

    # Next messages should alternate between user and assistant
    for i in range(1, 7, 2):
        assert messages[i]["role"] == "user"
        assert messages[i + 1]["role"] == "assistant"

    # Last message should be the actual query
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == query


def test_build_few_shot_messages_anthropic_format():
    """Test building few-shot messages in Anthropic format."""
    query = f'{QUERY_PREFIX}Hello, <|MASKED id="m_0"|><|/MASKED|>!{QUERY_SUFFIX}'

    messages = build_few_shot_messages(query, num_examples=2, message_format="anthropic")

    # Should have: 1 system + 2 examples (2 user + 2 assistant) + 1 user query = 6 messages
    assert len(messages) == 6

    # First message should be system
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT


def test_build_few_shot_messages_num_examples():
    """Test building few-shot messages with different numbers of examples."""
    query = f'{QUERY_PREFIX}Test <|MASKED id="m_0"|><|/MASKED|>{QUERY_SUFFIX}'

    # Test with 0 examples
    messages = build_few_shot_messages(query, num_examples=0)
    assert len(messages) == 2  # system + query

    # Test with 1 example
    messages = build_few_shot_messages(query, num_examples=1)
    assert len(messages) == 4  # system + 1 example (user + assistant) + query

    # Test with all examples
    max_examples = len(FEW_SHOT_EXAMPLES)
    messages = build_few_shot_messages(query, num_examples=max_examples)
    assert len(messages) == 2 + max_examples * 2  # system + examples + query


def test_build_few_shot_messages_invalid_num_examples():
    """Test that invalid num_examples raises ValueError."""
    query = f'{QUERY_PREFIX}Test <|MASKED id="m_0"|><|/MASKED|>{QUERY_SUFFIX}'

    with pytest.raises(ValueError, match="num_examples must be between"):
        build_few_shot_messages(query, num_examples=-1)

    with pytest.raises(ValueError, match="num_examples must be between"):
        build_few_shot_messages(query, num_examples=len(FEW_SHOT_EXAMPLES) + 1)


def test_build_few_shot_messages_invalid_format():
    """Test that invalid message format raises ValueError."""
    query = f'{QUERY_PREFIX}Test <|MASKED id="m_0"|><|/MASKED|>{QUERY_SUFFIX}'

    with pytest.raises(ValueError, match="Unsupported message format"):
        build_few_shot_messages(query, num_examples=1, message_format="invalid")


def test_build_few_shot_prompt():
    """Test building a single prompt string with few-shot examples."""
    query = f'{QUERY_PREFIX}Hello, <|MASKED id="m_0"|><|/MASKED|>!{QUERY_SUFFIX}'

    # Test with default number of examples (3)
    prompt = build_few_shot_prompt(query, num_examples=3)

    assert isinstance(prompt, str)
    assert SYSTEM_PROMPT in prompt
    assert query in prompt
    assert "Example 1:" in prompt
    assert "Example 2:" in prompt
    assert "Example 3:" in prompt

    # Check that the examples are included
    for i in range(3):
        assert FEW_SHOT_EXAMPLES[i]["query"] in prompt
        assert FEW_SHOT_EXAMPLES[i]["response"] in prompt


def test_build_few_shot_prompt_num_examples():
    """Test building prompt with different numbers of examples."""
    query = f'{QUERY_PREFIX}Test <|MASKED id="m_0"|><|/MASKED|>{QUERY_SUFFIX}'

    # Test with 0 examples
    prompt = build_few_shot_prompt(query, num_examples=0)
    assert SYSTEM_PROMPT in prompt
    assert query in prompt
    assert "Example 1:" not in prompt

    # Test with 1 example
    prompt = build_few_shot_prompt(query, num_examples=1)
    assert "Example 1:" in prompt
    assert "Example 2:" not in prompt


def test_build_few_shot_prompt_invalid_num_examples():
    """Test that invalid num_examples raises ValueError."""
    query = f'{QUERY_PREFIX}Test <|MASKED id="m_0"|><|/MASKED|>{QUERY_SUFFIX}'

    with pytest.raises(ValueError, match="num_examples must be between"):
        build_few_shot_prompt(query, num_examples=-1)

    with pytest.raises(ValueError, match="num_examples must be between"):
        build_few_shot_prompt(query, num_examples=len(FEW_SHOT_EXAMPLES) + 1)


def test_integration_with_query():
    """Test that prompts work with actual Query objects."""
    from gimkit import Query

    query = Query('Hello, <|MASKED id="m_0"|><|/MASKED|>!')
    query_str = str(query)

    # Test messages
    messages = build_few_shot_messages(query_str, num_examples=2)
    assert len(messages) == 6

    # Test prompt
    prompt = build_few_shot_prompt(query_str, num_examples=2)
    assert query_str in prompt
