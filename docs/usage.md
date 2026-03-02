# Usage Guide

## Creating Masked Tags

Use the `guide` helper (conventionally imported as `g`) to create masked tags:

```python
from gimkit import guide as g

# Basic tag with description
tag = g(name="greeting", desc="A friendly greeting")

# Specialized tags
name_tag  = g.person_name(name="user_name")
email_tag = g.e_mail(name="email")
phone_tag = g.phone_number(name="phone")
word_tag  = g.single_word(name="word")

# Selection from choices
choice_tag = g.select(name="color", choices=["red", "green", "blue"])

# Tag with regex constraint
code_tag = g(name="code", desc="A 4-digit PIN", regex=r"\d{4}")
```

## Building Queries

Masked tags can be embedded directly in Python f-strings:

```python
from gimkit import from_openai, guide as g
from openai import OpenAI

client = OpenAI()
model = from_openai(client, model_name="gpt-4")

query = f"""
Name: {g.person_name(name="name")}
Email: {g.e_mail(name="email")}
Favorite color: {g.select(name="color", choices=["red", "green", "blue"])}
"""

result = model(query, use_gim_prompt=True)
print(result)
```

## Accessing Results

Tags in the result can be accessed by index or by name:

```python
result = model(query, use_gim_prompt=True)

# Iterate over all tags
for tag in result.tags:
    print(f"{tag.name}: {tag.content}")

# Access a specific tag by name
print(result.tags["name"].content)

# Access by index
print(result.tags[0].content)

# Modify tag content
result.tags["email"].content = "REDACTED"
```

## Using vLLM

```python
from gimkit import from_vllm

model = from_vllm(base_url="http://localhost:8000", model_name="your-model")
result = model(query)
```

For offline inference without a running server:

```python
from gimkit import from_vllm_offline

model = from_vllm_offline(model_name="your-model")
result = model(query)
```

!!! note
    `from_vllm` and `from_vllm_offline` require `pip install gimkit[vllm]` on Linux.
