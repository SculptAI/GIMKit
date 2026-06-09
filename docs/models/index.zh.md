# 模型使用总览

本页用于对比各类 client 的支持差异，并说明何时使用不同模式。

## Client 对比

| Client | 构造方式 | 适用场景 |
|---|---|---|
| OpenAI | `from_openai(client, model_name=...)` | 托管 OpenAI 兼容 API |
| vLLM（服务端） | `from_vllm(client, model_name=...)` | OpenAI 兼容的 vLLM HTTP 服务 |
| vLLM（离线） | `from_vllm_offline(llm)` | 基于 `vllm.LLM` 的本地离线推理 |

## 支持度矩阵

| 能力项 | OpenAI | vLLM（服务端） | vLLM（离线） |
|---|---|---|---|
| `use_gim_prompt=True` | 推荐开启 | 仅非 GIM 训练模型开启 | 仅非 GIM 训练模型开启 |
| `output_type=None` | OpenAI 服务商不支持 JSON 时兜底 | 可用但不推荐 | 可用但不推荐 |
| `output_type="cfg"` | 不支持 | 推荐 | 推荐 |
| `output_type="json"` | 支持 | 支持 | 支持 |

## 初始化差异

- OpenAI 和 vLLM 服务端都接收 OpenAI 兼容 client 对象。
- vLLM 离线模式接收 `vllm.LLM` 实例，而不是 OpenAI client。
- 使用 vLLM 服务端时，需通过 `base_url` 指向你的服务地址。

## 提示词使用建议

- 本地主流场景是 GIM 训练模型：优先 `use_gim_prompt=False`。
- 非 GIM 训练模型建议开启 `use_gim_prompt=True`。
- OpenAI 路径建议优先开启 `use_gim_prompt=True`。

## output_type 选择

### OpenAI

- 推荐 `output_type="json"`。
- 若 OpenAI 服务商不支持 JSON 约束，则使用 `output_type=None`。

### vLLM（服务端 / 离线）

- 无论 GIM 训练模型还是非训练模型，都推荐 `output_type="cfg"`。
- 当你明确需要 JSON 输出时可改用 `output_type="json"`。

## 常用可选参数

- `visible_tag_fields`：控制哪些 `MaskedTag` 字段对模型可见（如 `["id", "name", "desc", "content", "regex"]`）。默认为 `None`（仅基础字段：`["id", "desc", "content"]`）。
- `backend`：选择 Outlines 后端实现。
- `**inference_kwargs`：透传底层后端生成参数。
