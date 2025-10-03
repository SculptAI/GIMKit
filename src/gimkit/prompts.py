"""Few-shot examples and system prompts for non-GIM LLMs."""

from typing import Literal

from gimkit.schemas import QUERY_PREFIX, QUERY_SUFFIX, RESPONSE_PREFIX, RESPONSE_SUFFIX


# System prompt explaining the GIM format and task
SYSTEM_PROMPT = """You are a helpful assistant that generates structured responses using the Guided Infilling Modeling (GIM) format.

When given a query wrapped in <|GIM_QUERY|>...<|/GIM_QUERY|> tags, you must:
1. Identify all <|MASKED|> tags in the query
2. Generate appropriate content for each masked section
3. Return your response wrapped in <|GIM_RESPONSE|>...<|/GIM_RESPONSE|> tags
4. Each masked section in your response should have the format: <|MASKED id="m_X"|>content<|/MASKED|>
5. The id must match the corresponding masked tag in the query (m_0, m_1, m_2, etc.)
6. Pay attention to any "desc" attribute in the masked tags as it provides guidance on what to generate

Example format:
Query: <|GIM_QUERY|>Hello, <|MASKED id="m_0" desc="a person's name"|><|/MASKED|>!<|/GIM_QUERY|>
Response: <|GIM_RESPONSE|><|MASKED id="m_0"|>Alice<|/MASKED|><|/GIM_RESPONSE|>"""


# Few-shot examples demonstrating the GIM format
FEW_SHOT_EXAMPLES = [
    {
        "query": f"""{QUERY_PREFIX}I'm <|MASKED id="m_0" desc="A person's name, e.g., John Doe, Alice, Bob, Charlie Brown, etc."|><|/MASKED|>. Hello, <|MASKED id="m_1" desc="A single word without spaces."|><|/MASKED|>!{QUERY_SUFFIX}""",
        "response": f"""{RESPONSE_PREFIX}<|MASKED id="m_0"|>John Smith<|/MASKED|><|MASKED id="m_1"|>world<|/MASKED|>{RESPONSE_SUFFIX}""",
    },
    {
        "query": f"""{QUERY_PREFIX}My favorite hobby is <|MASKED id="m_0" desc="Choose one from the following options: reading, traveling, cooking, swimming."|><|/MASKED|>.{QUERY_SUFFIX}""",
        "response": f"""{RESPONSE_PREFIX}<|MASKED id="m_0"|>reading<|/MASKED|>{RESPONSE_SUFFIX}""",
    },
    {
        "query": f"""{QUERY_PREFIX}Contact: <|MASKED id="m_0" desc="A phone number, e.g., +1-123-456-7890, (123) 456-7890, 123-456-7890, etc."|><|/MASKED|> or <|MASKED id="m_1" desc="An email address, e.g., john.doe@example.com, alice@example.com, etc."|><|/MASKED|>{QUERY_SUFFIX}""",
        "response": f"""{RESPONSE_PREFIX}<|MASKED id="m_0"|>+1-555-123-4567<|/MASKED|><|MASKED id="m_1"|>john.smith@example.com<|/MASKED|>{RESPONSE_SUFFIX}""",
    },
    {
        "query": f"""{QUERY_PREFIX}Question: What is 2 + 2?

Answer: <|MASKED id="m_0"|><|/MASKED|>{QUERY_SUFFIX}""",
        "response": f"""{RESPONSE_PREFIX}<|MASKED id="m_0"|>4<|/MASKED|>{RESPONSE_SUFFIX}""",
    },
    {
        "query": f"""{QUERY_PREFIX}Complete the sentence: The capital of France is <|MASKED id="m_0"|><|/MASKED|>, and the capital of Germany is <|MASKED id="m_1"|><|/MASKED|>.{QUERY_SUFFIX}""",
        "response": f"""{RESPONSE_PREFIX}<|MASKED id="m_0"|>Paris<|/MASKED|><|MASKED id="m_1"|>Berlin<|/MASKED|>{RESPONSE_SUFFIX}""",
    },
]


def build_few_shot_messages(
    query: str,
    num_examples: int = 3,
    message_format: Literal["openai", "anthropic"] = "openai",
) -> list[dict[str, str]]:
    """Build few-shot messages for non-GIM LLMs.

    Args:
        query: The user query in GIM format
        num_examples: Number of few-shot examples to include (default: 3)
        message_format: Message format to use ("openai" or "anthropic")

    Returns:
        List of message dictionaries in the specified format
    """
    if num_examples < 0 or num_examples > len(FEW_SHOT_EXAMPLES):
        raise ValueError(
            f"num_examples must be between 0 and {len(FEW_SHOT_EXAMPLES)}, got {num_examples}"
        )

    messages = []

    # Add system message
    if message_format == "openai":
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    elif message_format == "anthropic":
        # Anthropic handles system prompts separately, but we'll include it here
        # Users can extract it if needed
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    else:
        raise ValueError(f"Unsupported message format: {message_format}")

    # Add few-shot examples
    for example in FEW_SHOT_EXAMPLES[:num_examples]:
        messages.append({"role": "user", "content": example["query"]})
        messages.append({"role": "assistant", "content": example["response"]})

    # Add the actual query
    messages.append({"role": "user", "content": query})

    return messages


def build_few_shot_prompt(query: str, num_examples: int = 3) -> str:
    """Build a single prompt string with few-shot examples for non-GIM LLMs.

    This is useful for models that don't support the chat format.

    Args:
        query: The user query in GIM format
        num_examples: Number of few-shot examples to include (default: 3)

    Returns:
        A single prompt string with system prompt, examples, and query
    """
    if num_examples < 0 or num_examples > len(FEW_SHOT_EXAMPLES):
        raise ValueError(
            f"num_examples must be between 0 and {len(FEW_SHOT_EXAMPLES)}, got {num_examples}"
        )

    prompt_parts = [SYSTEM_PROMPT, ""]

    # Add few-shot examples
    for i, example in enumerate(FEW_SHOT_EXAMPLES[:num_examples], 1):
        prompt_parts.append(f"Example {i}:")
        prompt_parts.append(f"Query: {example['query']}")
        prompt_parts.append(f"Response: {example['response']}")
        prompt_parts.append("")

    # Add the actual query
    prompt_parts.append("Now, generate a response for the following query:")
    prompt_parts.append(f"Query: {query}")
    prompt_parts.append("Response:")

    return "\n".join(prompt_parts)
