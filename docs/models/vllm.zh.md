# vLLM 客户端（服务端模式）

## 创建模型

`from_vllm` 需要传入 OpenAI 兼容客户端（指向 vLLM 服务）。

```python
from openai import OpenAI
from gimkit import from_vllm

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
model = from_vllm(client, model_name="Qwen/Qwen2.5-7B-Instruct")
```

## 提示词建议

对于 GIM 训练的本地模型，建议保持 `use_gim_prompt=False`。
对于非 GIM 训练模型，可额外开启 `use_gim_prompt=True`。

查询示例：

```python
from gimkit import guide as g

query = f"""
Name: {g.person_name(name="name")}
Phone: {g.phone_number(name="phone")}
"""

result = model(query, use_gim_prompt=True)
```

## 输出类型

### `output_type="cfg"`（默认）

vLLM 默认使用 CFG 约束，结构控制更强。

```python
result = model(query, output_type="cfg")
```

### `output_type="json"`

```python
result = model(query, output_type="json", use_gim_prompt=True)
```

## 说明

- GIMKit 会自动添加 `stop="<|/GIM_RESPONSE|>"`，确保更稳定停止。
- 可通过 `**inference_kwargs` 继续传递生成参数。
