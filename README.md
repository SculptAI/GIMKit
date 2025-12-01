<h1 align="center">GIMKit</h1>

<p align="center">


<a href="https://pypi.org/project/gimkit">
  <img src="https://img.shields.io/pypi/v/gimkit?label=pypi%20package" alt="PyPI Version">
</a>
<a href="https://pypi.org/project/gimkit">
  <img src="https://img.shields.io/pypi/pyversions/gimkit.svg" alt="Supported Python versions">
</a>
<a href="https://pypi.org/project/gimkit">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey" alt="Supported Platforms">
</a>
</p>

## Installation

Install GIMKit using pip:

```bash
pip install gimkit
```

For vLLM support, install with the optional dependency:

```bash
pip install gimkit[vllm]
```

## Quick Start

Here's a simple example using the OpenAI backend:

```python
from openai import OpenAI
from gimkit import from_openai, guide as g

# Initialize the client and model
client = OpenAI()  # Uses OPENAI_API_KEY environment variable
model = from_openai(client, model_name="gpt-4")

# Create a query with masked tags
result = model(f"Hello, {g(desc='a single word')}!", use_gim_prompt=True)
print(result)  # Output: Hello, world!
```

## Usage

### Creating Masked Tags: Use the `guide` helper (imported as `g`) to create masked tags

```python
from gimkit import guide as g

# Basic tag with description
tag = g(name="greeting", desc="A friendly greeting")

# Specialized tags
name_tag = g.person_name(name="user_name")
email_tag = g.e_mail(name="email")
phone_tag = g.phone_number(name="phone")
word_tag = g.single_word(name="word")

# Selection from choices
choice_tag = g.select(name="color", choices=["red", "green", "blue"])

# Tag with regex constraint
custom_tag = g(name="code", desc="A 4-digit code", regex=r"\d{4}")
```

### Building Queries: Combine masked tags with text to build queries

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

### Accessing Results: Access filled tags from the result

```python
result = model(query)

# Iterate over all tags
for tag in result.tags:
    print(f"{tag.name}: {tag.content}")

# Access by name
print(result.tags["name"].content)

# Modify tag content
result.tags["email"].content = "REDACTED"
```

## Design Philosophy

- Stable over feature
- Small open-source model first
