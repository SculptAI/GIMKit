# vLLM Offline Client

## Create a model

`from_vllm_offline` expects a `vllm.LLM` instance.

```python
from vllm import LLM
from gimkit import from_vllm_offline

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
model = from_vllm_offline(llm)
```

!!! note
    Install extra dependencies first: `pip install gimkit[vllm]` (Linux).

## Prompt recommendation

For GIM-trained local models, keep `use_gim_prompt=False`.
For non-GIM-trained models, enable `use_gim_prompt=True` as an extra prompt layer.

Example query:

```python
from gimkit import guide as g

query = f"""
Event: {g(name="event", desc="event type")}
Date: {g.datetime(name="date")}
"""

result = model(query, use_gim_prompt=True)
```

## Output types

### `output_type="cfg"` (default)

```python
result = model(query, output_type="cfg")
```

### `output_type="json"`

```python
result = model(query, output_type="json", use_gim_prompt=True)
```

## Notes

- GIMKit ensures `RESPONSE_SUFFIX` is included in vLLM sampling stop conditions.
- You can pass `sampling_params=` and other vLLM generation options via `**inference_kwargs`.
