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

## Per-generation error collection

The default `error_mode="raise"` preserves fail-fast behavior. For multiple
candidates, use `error_mode="collect"` to parse each raw response independently:

```python
generations = model(query, n=2, error_mode="collect")

for generation in generations:
    if generation.ok:
        print(generation.result)
    else:
        print(generation.error_type, generation.error_message)
        print(generation.raw_response)
```

`collect` only captures parsing and infill errors after raw text has been
generated. Network, authentication, timeout, model request, and invalid response
container errors still fail the whole call. Async models use the same parameter
through `await model(...)`.

## Advanced options

- `visible_tag_fields`: control which `MaskedTag` fields are visible to the model (e.g. `["id", "name", "desc", "content", "regex"]`). Defaults to `None` (basic fields only: `["id", "desc", "content"]`).
- `backend`: pass through to Outlines generator backend selection.
- `error_mode`: `"raise"` (default) or `"collect"`.
- `**inference_kwargs`: forwarded to the underlying OpenAI call.
