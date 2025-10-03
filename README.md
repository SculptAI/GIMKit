# Guided Infilling Modeling

A Python library for structured text generation using Guided Infilling Modeling (GIM).

## Features

- **Guided Infilling**: Use masked tags to specify exactly what content you want generated
- **Structured Generation**: Control the format and content of LLM outputs
- **Few-Shot Prompting**: Built-in support for non-GIM LLMs with system prompts and examples
- **Multiple Backends**: Support for OpenAI API, vLLM, and more

## Installation

```bash
pip install gimkit
```

For vLLM support:

```bash
pip install gimkit[vllm]
```

## Quick Start

### Using GIM-trained Models

```python
from openai import OpenAI
from gimkit import from_openai
from gimkit import guide as g

# Initialize model
openai_client = OpenAI(api_key="", base_url="http://localhost:8000/v1")
model = from_openai(openai_client, model_name="your-gim-model")

# Create a query with guided infilling
query = f"""I'm {g.person_name(name="name")}. 
My email is {g.e_mail(name="email")}."""

# Generate response
result = model(query)
print(result)
print(result.tags["name"].content)  # Access specific tags
```

### Using Non-GIM LLMs with Few-Shot Prompting

For models not trained with GIM (like GPT-4, Claude, etc.), you can use the built-in few-shot prompts:

```python
from openai import OpenAI
from gimkit import Query, build_few_shot_messages
from gimkit import guide as g

# Create your query
query = f"Hello, {g.person_name()}! Your email is {g.e_mail()}."

# Build few-shot messages
messages = build_few_shot_messages(
    str(Query(query)),
    num_examples=3,  # Include 3 examples
    message_format="openai"
)

# Use with any LLM
client = OpenAI(api_key="your-key")
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages
)

# Parse response
from gimkit.contexts import Response, infill
result = infill(Query(query), Response(response.choices[0].message.content))
print(result)
```

## Examples

See the `examples/` directory for more examples:
- `gimkit_quickstart.py` - Basic usage with GIM-trained models
- `non_gim_llm_example.py` - Using few-shot prompts with non-GIM LLMs
- `cases.ipynb` - Various use cases and examples

## Testing

```bash
pytest tests/
```

## License

MIT
