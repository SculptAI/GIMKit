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

# GIM-trained model path
result = model(query)

# Non-GIM-trained model path
result_non_gim = model(query, use_gim_prompt=True)
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

## Per-generation error collection

For multiple candidates, pass `error_mode="collect"` to receive one
`GenerationResult` per raw response. Successful items expose `.result`; failed
items retain `.raw_response`, `.error_type`, and `.error_message`. The default
`error_mode="raise"` preserves existing behavior and return types. Model request,
network, and response-container failures still fail the whole call.

Async clients use the same parameter through `await model(...)`.

## Notes

- GIMKit automatically adds `stop="<|/GIM_RESPONSE|>"` for safer termination.
- `error_mode` accepts `"raise"` (default) or `"collect"`.
- You can still pass extra generation args via `**inference_kwargs`.
