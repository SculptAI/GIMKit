# 使用指南

## 创建标签

使用 `guide` 辅助函数（通常导入为 `g`）来创建标签：

```python
from gimkit import guide as g

# 带描述的基础标签
tag = g(name="greeting", desc="一句问候语")

# 专用标签
name_tag  = g.person_name(name="user_name")
email_tag = g.e_mail(name="email")
phone_tag = g.phone_number(name="phone")
word_tag  = g.single_word(name="word")

# 从选项中选择
choice_tag = g.select(name="color", choices=["red", "green", "blue"])

# 带正则约束的标签
code_tag = g(name="code", desc="4位PIN码", regex=r"\d{4}")
```

## 构建查询

标签可以直接嵌入 Python f-string 中：

```python
from gimkit import from_openai, guide as g
from openai import OpenAI

client = OpenAI()
model = from_openai(client, model_name="gpt-4")

query = f"""
Name: {g.person_name(name="name")}
Email: {g.e_mail(name="email")}
Favorite color: {g.select(name="color", choices=["red", "green", "blue"])}
"""

result = model(query, use_gim_prompt=True)
print(result)
```

## 访问结果

可以通过索引或名称访问结果中的标签：

```python
result = model(query, use_gim_prompt=True)

# 遍历所有标签
for tag in result.tags:
    print(f"{tag.name}: {tag.content}")

# 按名称访问
print(result.tags["name"].content)

# 按索引访问
print(result.tags[0].content)

# 修改标签内容
result.tags["email"].content = "REDACTED"
```

## 使用 vLLM

```python
from gimkit import from_vllm

model = from_vllm(base_url="http://localhost:8000", model_name="your-model")
result = model(query)
```

离线推理（无需运行服务）：

```python
from gimkit import from_vllm_offline

model = from_vllm_offline(model_name="your-model")
result = model(query)
```

!!! note
    `from_vllm` 和 `from_vllm_offline` 需要在 Linux 上执行 `pip install gimkit[vllm]`。
