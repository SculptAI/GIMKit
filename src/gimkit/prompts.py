"""
For models that weren't trained using the Guided Infilling Modeling method, we may be able
to achieve functionality similar to GIM by using a system prompt and few-shot prompting.
"""

SYSTEM_PROMPT_MSG = {
    "role": "system",
    "content": (
        "You complete masked spans. Follow these strict rules:\n"
        "1. Input is wrapped in `<|GIM_QUERY|> ... <|/GIM_QUERY|>`.\n"
        "2. You must output exactly one `<|GIM_RESPONSE|> ... <|/GIM_RESPONSE|>`.\n"
        '3. For every <|MASKED id="m_X" ...|><|/MASKED|> in the query:\n'
        '     - Output <|MASKED id="m_X"|>FILLED_TEXT<|/MASKED|>.\n'
        "     - Keep id the same.\n"
        "     - Remove desc and any other attributes.\n"
        "4. Do not add, remove, reorder, or modify any tags.\n"
        "5. No explanations. Output only the response."
    ),
}

DEMO_CONVERSATION_MSGS = [
    {
        "role": "user",
        "content": '<|GIM_QUERY|>Hello, <|MASKED id="m_0" desc="a simple phrase"|><|/MASKED|>.<|/GIM_QUERY|>',
    },
    {
        "role": "assistant",
        "content": '<|GIM_RESPONSE|><|MASKED id="m_0"|>nice to meet you<|/MASKED|><|/GIM_RESPONSE|>',
    },
    {
        "role": "user",
        "content": '<|GIM_QUERY|><|MASKED id="m_0" desc="a number"|><|/MASKED|> + <|MASKED id="m_1" desc="a number"|><|/MASKED|> = <|MASKED id="m_2" desc="a number"|><|/MASKED|><|/GIM_QUERY|>',
    },
    {
        "role": "assistant",
        "content": '<|GIM_RESPONSE|><|MASKED id="m_0"|>1<|/MASKED|><|MASKED id="m_1"|>1<|/MASKED|><|MASKED id="m_2"|>2<|/MASKED|><|/GIM_RESPONSE|>',
    },
    {
        "role": "user",
        "content": '<|GIM_QUERY|>The capital of <|MASKED id="m_0" desc="a word"|><|/MASKED|> is Paris<|MASKED id="m_1" desc="punctuation mark"|><|/MASKED|><|/GIM_QUERY|>',
    },
    {
        "role": "assistant",
        "content": '<|GIM_RESPONSE|><|MASKED id="m_0"|>France<|/MASKED|><|MASKED id="m_1"|>.<|/MASKED|><|/GIM_RESPONSE|>',
    },
    {
        "role": "user",
        "content": '<|GIM_QUERY|>To bake a cake, you need <|MASKED id="m_0" desc="an ingredient"|><|/MASKED|>, <|MASKED id="m_1" desc="another ingredient"|><|/MASKED|>, and <|MASKED id="m_2" desc="a final ingredient"|><|/MASKED|>.<|/GIM_QUERY|>',
    },
    {
        "role": "assistant",
        "content": '<|GIM_RESPONSE|><|MASKED id="m_0"|>flour<|/MASKED|><|MASKED id="m_1"|>sugar<|/MASKED|><|MASKED id="m_2"|>eggs<|/MASKED|><|/GIM_RESPONSE|>',
    },
]

SYSTEM_PROMPT_MSG_JSON = {
    "role": "system",
    "content": (
        "You complete masked spans and return results in JSON format. Follow these strict rules:\n"
        "1. Input is wrapped in `<|GIM_QUERY|> ... <|/GIM_QUERY|>`.\n"
        "2. You must output a JSON object with fields matching the masked tag IDs.\n"
        '3. For every <|MASKED id="m_X" ...|><|/MASKED|> in the query:\n'
        '     - Create a JSON field "m_X" with the filled text as its value.\n'
        "     - Keep the field name as m_X where X is the tag ID number.\n"
        "4. Do not add, remove, reorder, or modify any fields.\n"
        "5. No explanations. Output only the JSON object."
    ),
}

DEMO_CONVERSATION_MSGS_JSON = [
    {
        "role": "user",
        "content": '<|GIM_QUERY|>Hello, <|MASKED id="m_0" desc="a simple phrase"|><|/MASKED|>.<|/GIM_QUERY|>',
    },
    {
        "role": "assistant",
        "content": '{"m_0": "nice to meet you"}',
    },
    {
        "role": "user",
        "content": '<|GIM_QUERY|><|MASKED id="m_0" desc="a number"|><|/MASKED|> + <|MASKED id="m_1" desc="a number"|><|/MASKED|> = <|MASKED id="m_2" desc="a number"|><|/MASKED|><|/GIM_QUERY|>',
    },
    {
        "role": "assistant",
        "content": '{"m_0": "1", "m_1": "1", "m_2": "2"}',
    },
    {
        "role": "user",
        "content": '<|GIM_QUERY|>The capital of <|MASKED id="m_0" desc="a word"|><|/MASKED|> is Paris<|MASKED id="m_1" desc="punctuation mark"|><|/MASKED|><|/GIM_QUERY|>',
    },
    {
        "role": "assistant",
        "content": '{"m_0": "France", "m_1": "."}',
    },
    {
        "role": "user",
        "content": '<|GIM_QUERY|>To bake a cake, you need <|MASKED id="m_0" desc="an ingredient"|><|/MASKED|>, <|MASKED id="m_1" desc="another ingredient"|><|/MASKED|>, and <|MASKED id="m_2" desc="a final ingredient"|><|/MASKED|>.<|/GIM_QUERY|>',
    },
    {
        "role": "assistant",
        "content": '{"m_0": "flour", "m_1": "sugar", "m_2": "eggs"}',
    },
]
