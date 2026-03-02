# Quick Start

Here is a minimal example using the OpenAI backend.

## 1. Set up the client

```python
from openai import OpenAI
from gimkit import from_openai, guide as g

client = OpenAI()  # reads OPENAI_API_KEY from environment
model = from_openai(client, model_name="gpt-4")
```

## 2. Create a query with masked tags

```python
result = model(f"Hello, {g(desc='a single word')}!", use_gim_prompt=True)
print(result)  # Hello, world!
```

## 3. Run a structured form

```python
query = f"""
Name: {g.person_name(name="name")}
Email: {g.e_mail(name="email")}
Favorite color: {g.select(name="color", choices=["red", "green", "blue"])}
"""

result = model(query, use_gim_prompt=True)
print(result.tags["name"].content)   # e.g. Alice
print(result.tags["email"].content)  # e.g. alice@example.com
print(result.tags["color"].content)  # red | green | blue
```
