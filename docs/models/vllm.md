# vLLM Client (Server Mode)

## Create a model

`from_vllm` expects an OpenAI-compatible client (pointing to your vLLM server).

```python
from openai import OpenAI
from gimkit import from_vllm

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
model = from_vllm(client, model_name="Qwen/Qwen2.5-7B-Instruct")
```

## Prompt recommendation

For GIM-trained local models, keep `use_gim_prompt=False`.
For non-GIM-trained models, enable `use_gim_prompt=True` as an extra prompt layer.

Example query:

```python
from gimkit import guide as g

query = f"""
Name: {g.person_name(name="name")}
Phone: {g.phone_number(name="phone")}
"""

result = model(query, use_gim_prompt=True)
```

## Output types

### `output_type="cfg"` (default)

vLLM uses CFG constraints by default for strong structure control.

```python
result = model(query, output_type="cfg")
```

### `output_type="json"`

```python
result = model(query, output_type="json", use_gim_prompt=True)
```

## Notes

- GIMKit automatically adds `stop="<|/GIM_RESPONSE|>"` for safer termination.
- You can still pass extra generation args via `**inference_kwargs`.
