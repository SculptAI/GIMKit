# OpenAI Client

## Create a model

```python
from openai import OpenAI
from gimkit import from_openai

client = OpenAI()  # reads OPENAI_API_KEY
model = from_openai(client, model_name="gpt-4o")
```

## Prompt recommendation

For OpenAI paths, prefer `use_gim_prompt=True`.

Use natural language with masked tags directly in an f-string:

```python
from gimkit import guide as g

query = f"""
Name: {g.person_name(name="name")}
Email: {g.e_mail(name="email")}
"""

result = model(query, output_type="json", use_gim_prompt=True)
```

If your OpenAI provider does not support JSON constraints, fall back to `output_type=None`:

```python
result = model(query, output_type=None, use_gim_prompt=True)
```

## Output types

### `output_type="json"` (recommended)

Use JSON schema constraints and convert JSON fields back into tag results.

```python
result = model(query, output_type="json", use_gim_prompt=True)
print(result.tags["name"].content)
```

### `output_type=None` (fallback)

Use this when your OpenAI provider does not support JSON constrained output.

```python
result = model(query, output_type=None, use_gim_prompt=True)
print(result.tags["email"].content)
```

## Advanced options

- `include_grammar=True`: inject grammar text into the query string before generation.
- `backend`: pass through to Outlines generator backend selection.
- `**inference_kwargs`: forwarded to the underlying OpenAI call.
