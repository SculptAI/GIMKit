# API Reference

## `guide` — Tag Builder

A singleton instance of `Guide` for creating masked tags. Import it as `g` by convention:

```python
from gimkit import guide as g
```

### Methods

| Method | Description |
|---|---|
| `g(name, desc, regex, content)` | Create a generic masked tag with optional attributes. |
| `g.single_word(name)` | A single word without spaces (`\S+`). |
| `g.select(name, choices)` | Choose one value from the given list of options. |
| `g.datetime(name, require_date, require_time)` | A date and/or time string, e.g. `2023-10-05 14:30:00`. |
| `g.person_name(name)` | A person's name, e.g. *John Doe*, *张三*. |
| `g.phone_number(name)` | A phone number, e.g. *+1-123-456-7890*. |
| `g.e_mail(name)` | An email address, e.g. *alice@example.com*. |

### `MaskedTag` attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | `str | None` | Tag name for named access in results. |
| `desc` | `str | None` | Natural-language description sent to the model. |
| `regex` | `str | None` | Regex pattern constraining model output. |
| `content` | `str | None` | Filled content (set after model inference). |

---

## `from_openai` — OpenAI Backend

```python
from gimkit import from_openai
from openai import OpenAI

model = from_openai(client: OpenAI, model_name: str)
result = model(query, use_gim_prompt=True)
```

Returns a callable model. Supports both synchronous and asynchronous calls.

---

## `from_vllm` — vLLM Server Backend

```python
from gimkit import from_vllm

model = from_vllm(base_url: str, model_name: str)
result = model(query)
```

Requires `pip install gimkit[vllm]` on Linux.

---

## `from_vllm_offline` — vLLM Offline Backend

```python
from gimkit import from_vllm_offline

model = from_vllm_offline(model_name: str)
result = model(query)
```

Requires `pip install gimkit[vllm]` on Linux.

---

## `Query` and `Response`

Low-level classes for working with GIM-formatted strings directly:

```python
from gimkit.contexts import Query, Response, infill

query = Query(f"Hello, {g(name='word', desc='a single word')}!")
response = Response(f"<|GIM_RESPONSE|>...<|/GIM_RESPONSE|>")
result = infill(query, response)
```
