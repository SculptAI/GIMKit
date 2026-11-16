# 快速开始

以下是一个使用 OpenAI 后端的最简示例。

## 1. 创建模型

```python
from openai import OpenAI
from gimkit import from_openai, guide as g

client = OpenAI()  # 从环境变量 OPENAI_API_KEY 读取密钥
model = from_openai(client, model_name="gpt-4")
```

## 2. 创建标签

```python
tag = g(desc="一个单词")
name_tag = g.person_name(name="name")
email_tag = g.e_mail(name="email")
color_tag = g.select(name="color", choices=["red", "green", "blue"])
```

## 3. 构建查询

```python
result = model(f"Hello, {tag}!", use_gim_prompt=True)
print(result)  # Hello, world!

query = f"""
Name: {name_tag}
Email: {email_tag}
Favorite color: {color_tag}
"""

result = model(query, use_gim_prompt=True)
```

## 4. 访问结果

```python
print(result.tags["name"].content)  # 例如 Alice
print(result.tags["email"].content)  # 例如 alice@example.com
print(result.tags["color"].content)  # red | green | blue
```
