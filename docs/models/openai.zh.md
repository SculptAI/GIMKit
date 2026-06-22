# OpenAI 客户端

## 创建模型

```python
from openai import OpenAI
from gimkit import from_openai

client = OpenAI()  # 从环境变量读取 OPENAI_API_KEY
model = from_openai(client, model_name="gpt-4o")
```

## 提示词建议

OpenAI 路径建议优先使用 `use_gim_prompt=True`。

直接在 f-string 里写自然语言模板并嵌入标签：

```python
from gimkit import guide as g

query = f"""
Name: {g.person_name(name="name")}
Email: {g.e_mail(name="email")}
"""

result = model(query, output_type="json", use_gim_prompt=True)
```

如果你的 OpenAI 服务商不支持 JSON 约束，则退回 `output_type=None`：

```python
result = model(query, output_type=None, use_gim_prompt=True)
```

## 输出类型

### `output_type="json"`（推荐）

使用 JSON Schema 约束输出，并将 JSON 字段转换回标签结果。

```python
result = model(query, output_type="json", use_gim_prompt=True)
print(result.tags["name"].content)
```

### `output_type=None`（兜底）

当 OpenAI 服务商不支持 JSON 约束输出时使用。

```python
result = model(query, output_type=None, use_gim_prompt=True)
print(result.tags["email"].content)
```

## 逐生成错误收集

默认的 `error_mode="raise"` 保持快速失败行为。多候选生成时，可使用
`error_mode="collect"` 让每条原始响应独立解析：

```python
generations = model(query, n=2, error_mode="collect")

for generation in generations:
    if generation.ok:
        print(generation.result)
    else:
        print(generation.error_type, generation.error_message)
        print(generation.raw_response)
```

`collect` 只收集已经获得原始文本后的解析和 infill 错误。网络、认证、超时、
模型请求失败以及无效的响应容器仍会作为整个调用异常抛出。异步模型使用相同参数，
调用方式为 `await model(...)`。

## 高级参数

- `visible_tag_fields`：控制哪些 `MaskedTag` 字段对模型可见（如 `["id", "name", "desc", "content", "regex"]`）。默认为 `None`（仅基础字段：`["id", "desc", "content"]`）。
- `backend`：透传给 Outlines 生成器后端选择。
- `error_mode`：`"raise"`（默认）或 `"collect"`。
- `**inference_kwargs`：透传到底层 OpenAI 推理参数。
