<h1 align="center">GIMKit</h1>

<p align="center">

<a href="https://pypi.org/project/gimkit">
  <img src="https://img.shields.io/pypi/v/gimkit?label=pypi%20package" alt="PyPI Version">
</a>
<a href="https://pypi.org/project/gimkit">
  <img src="https://img.shields.io/pypi/pyversions/gimkit.svg" alt="Supported Python versions">
</a>
<a href="https://pypi.org/project/gimkit">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey" alt="Supported Platforms">
</a>

</p>

**Guided Infilling Modeling Toolkit** — structured text generation and information extraction using language models.

Write a template with typed placeholders. The LLM fills them in. Get structured, named results back.

```python
from gimkit import guide as g

query = f"""Extract from: "Hi, I'm John Smith, reach me at john@gmail.com"

Name: {g.person_name(name="name")}
Email: {g.e_mail(name="email")}"""

result = model(query, use_gim_prompt=True)
result.tags["name"].content  # → "John Smith"
result.tags["email"].content  # → "john@gmail.com"
```

## Installation

```bash
pip install gimkit
```

For vLLM support:

```bash
pip install gimkit[vllm]
```

## What Can You Do With GIMKit?

GIMKit is a **general-purpose information extraction framework**. Write a natural-language template with embedded tags, and the model extracts structured data from any text.

| Use Case | Example |
|----------|---------|
| **Contact extraction** | Parse names, emails, phones from free-form text |
| **Named entity recognition** | Extract orgs, people, locations, dates |
| **Text classification** | Categorize text, assign sentiment labels |
| **Event extraction** | Pull what/where/when/impact from event descriptions |
| **Relation extraction** | Find entities and the relationships between them |
| **Resume parsing** | Extract name, title, education, experience |
| **Review analysis** | Parse product, price, rating, pros/cons |

See the [Classic IE Use Cases](https://sculptai.github.io/GIMKit/use-cases/classic/), [Privacy and PII Use Cases](https://sculptai.github.io/GIMKit/use-cases/privacy-pii/), and [Other Use Cases](https://sculptai.github.io/GIMKit/use-cases/others/) pages for full examples.

## Why GIMKit?

- **Template-driven** — describe what you want in natural language, not label lists
- **Format control** — regex constraints, enumerated choices, type-safe tags
- **Named access** — results are keyed by field name, not token positions
- **Small-model friendly** — works with compact open-source models (4B+)
- **Multiple backends** — OpenAI, vLLM (server and offline)

## Quick Start

```python
from openai import OpenAI
from gimkit import from_openai, guide as g

client = OpenAI()
model = from_openai(client, model_name="gpt-4")

# Simple extraction
result = model(f"Hello, {g(desc='a single word')}!", use_gim_prompt=True)
print(result)  # Hello, world!

# Structured form
query = f"""
Name: {g.person_name(name="name")}
Email: {g.e_mail(name="email")}
Favorite color: {g.select(name="color", choices=["red", "green", "blue"])}
"""
result = model(query, use_gim_prompt=True)
print(result.tags["name"].content)
print(result.tags["email"].content)
print(result.tags["color"].content)
```
