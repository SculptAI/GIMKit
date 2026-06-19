# vLLM 离线客户端

## 创建模型

`from_vllm_offline` 需要传入 `vllm.LLM` 实例。

```python
from vllm import LLM
from gimkit import from_vllm_offline

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
model = from_vllm_offline(llm)
```

!!! note
    请先安装扩展依赖：`pip install gimkit[vllm]`（Linux）。

## 提示词建议

对于 GIM 训练的本地模型，建议保持 `use_gim_prompt=False`。
对于非 GIM 训练模型，可额外开启 `use_gim_prompt=True`。

查询示例：

```python
from gimkit import guide as g

query = f"""
Event: {g(name="event", desc="event type")}
Date: {g.datetime(name="date")}
"""

# GIM 训练模型路径
result = model(query)

# 非 GIM 训练模型路径
result_non_gim = model(query, use_gim_prompt=True)
```

## 批量推理

`model.batch(...)` 会包装 Outlines 的 vLLM offline batch API。
每条 query 都可以使用各自从 GIM 推导出的结构化输出 schema。

```python
batch_results = model.batch([query, query])
first_result = batch_results[0][0]
```

使用 `error_mode="collect"` 时，batch 始终返回二维
`list[list[GenerationResult]]`：外层对应 query，内层对应候选。

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

单个候选的解析失败不会影响同一 query 的其他候选或其他 query。默认
`error_mode="raise"` 的返回类型和快速失败行为保持不变。模型生成失败、batch
形状错误和无效参数仍会作为整个调用异常抛出。

## 输出类型

### `output_type="cfg"`（默认）

```python
result = model(query, output_type="cfg")
```

### `output_type="json"`

```python
result = model(query, output_type="json", use_gim_prompt=True)
```

## 说明

- GIMKit 会确保在 vLLM 采样参数中包含 `RESPONSE_SUFFIX` 的 stop 条件。
- 可通过 `sampling_params=` 和其他 `**inference_kwargs` 传递 vLLM 生成参数。
