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

# GIM-trained model path
result = model(query)

# Non-GIM-trained model path
result_non_gim = model(query, use_gim_prompt=True)
```

## Batch inference

`model.batch(...)` wraps Outlines' batch API for vLLM offline.
Each query can use its own GIM-derived structured output schema.

```python
batch_results = model.batch([query, query])
first_result = batch_results[0][0]
```

With `error_mode="collect"`, batch always returns a two-dimensional
`list[list[GenerationResult]]`: the outer list maps to queries and the inner list
maps to candidates.

```python
generation_groups = model.batch(queries, error_mode="collect")

for generation_group in generation_groups:
    for generation in generation_group:
        if generation.ok:
            print(generation.result)
        else:
            print(generation.error_type, generation.error_message)
            print(generation.raw_response)
```

A parsing failure for one candidate does not affect other candidates or queries.
The default `error_mode="raise"` preserves existing return types and fail-fast
behavior. Generation failures, invalid batch shapes, and invalid arguments still
fail the whole call.

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
