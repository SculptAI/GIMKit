# 快速开始

以下是一个使用 OpenAI 后端的最简示例。

## 1. 初始化客户端

```python
from openai import OpenAI
from gimkit import from_openai, guide as g

client = OpenAI()  # 从环境变量 OPENAI_API_KEY 读取密钥
model = from_openai(client, model_name="gpt-4")
```

## 2. 创建带标签的查询

```python
result = model(f"Hello, {g(desc='a single word')}!", use_gim_prompt=True)
print(result)  # Hello, world!
```

## 3. 运行结构化表单

```python
query = f"""
Name: {g.person_name(name="name")}
Email: {g.e_mail(name="email")}
Favorite color: {g.select(name="color", choices=["red", "green", "blue"])}
"""

result = model(query, use_gim_prompt=True)
print(result.tags["name"].content)   # 例如 Alice
print(result.tags["email"].content)  # 例如 alice@example.com
print(result.tags["color"].content)  # red | green | blue
```
