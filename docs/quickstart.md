# Quick Start

Here is a minimal example using the OpenAI backend.

## 1. Create a model

```python
from openai import OpenAI
from gimkit import from_openai, guide as g

client = OpenAI()  # reads OPENAI_API_KEY from environment
model = from_openai(client, model_name="gpt-4")
```

## 2. Create masked tags

```python
tag = g(desc="a single word")
name_tag = g.person_name(name="name")
email_tag = g.e_mail(name="email")
color_tag = g.select(name="color", choices=["red", "green", "blue"])
```

## 3. Build a query

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

## 4. Access results

```python
print(result.tags["name"].content)  # e.g. Alice
print(result.tags["email"].content)  # e.g. alice@example.com
print(result.tags["color"].content)  # red | green | blue
```
