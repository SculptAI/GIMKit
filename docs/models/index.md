# Model Usage Overview

This page compares supported clients and explains when to use each mode.

## Client Comparison

| Client | Constructor | Best for |
|---|---|---|
| OpenAI | `from_openai(client, model_name=...)` | Hosted OpenAI-compatible APIs |
| vLLM (Server) | `from_vllm(client, model_name=...)` | OpenAI-compatible vLLM HTTP server |
| vLLM (Offline) | `from_vllm_offline(llm)` | Local offline inference with `vllm.LLM` |

## Support Matrix

| Capability | OpenAI | vLLM (Server) | vLLM (Offline) |
|---|---|---|---|
| `use_gim_prompt=True` | Recommended | Only for non-GIM models | Only for non-GIM models |
| `output_type=None` | Fallback when JSON is unsupported | Available but not recommended | Available but not recommended |
| `output_type="cfg"` | Not available | Recommended | Recommended |
| `output_type="json"` | Yes | Yes | Yes |

## Initialization Differences

- OpenAI and vLLM server mode both take an OpenAI-compatible client object.
- vLLM offline mode takes a `vllm.LLM` instance, not an OpenAI client.
- For vLLM server mode, create the client with `base_url` pointing to your server.

## Prompt Usage Recommendation

- Most local workflows use GIM-trained models: `use_gim_prompt=False` is preferred.
- For non-GIM-trained models, enable `use_gim_prompt=True`.
- For OpenAI paths, prefer `use_gim_prompt=True`.

## Output Type Guide

### OpenAI

- Prefer `output_type="json"`.
- If your OpenAI provider does not support JSON constraints, use `output_type=None`.

### vLLM (Server / Offline)

- Prefer `output_type="cfg"` for both GIM-trained and non-GIM models.
- `output_type="json"` is available when JSON output is specifically needed.

## Common Optional Flags

- `include_grammar=True`: include grammar text in query input.
- `backend`: choose Outlines backend implementation.
- `**inference_kwargs`: pass generation parameters to the underlying backend.
