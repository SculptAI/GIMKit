# GIMKit

**Guided Infilling Modeling Toolkit** — structured text generation and information extraction using language models.

GIMKit lets you define placeholders (masked tags) in text and have a language model fill them in. It gives you fine-grained control over model outputs through a typed tag system with optional regex constraints.

[![PyPI Version](https://img.shields.io/pypi/v/gimkit?label=pypi%20package)](https://pypi.org/project/gimkit)
[![Python Versions](https://img.shields.io/pypi/pyversions/gimkit.svg)](https://pypi.org/project/gimkit)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](https://pypi.org/project/gimkit)

---

## What Can You Do With GIMKit?

GIMKit is a **general-purpose information extraction framework**. Write a natural-language template with embedded typed placeholders, and the model extracts structured data from any unstructured text.

| Use Case | Description |
|----------|-------------|
| **Contact extraction** | Parse names, emails, phone numbers from free-form text |
| **Named entity recognition** | Extract organizations, people, locations, dates |
| **Text classification** | Categorize text into labels, assign sentiment |
| **Event extraction** | Pull structured event info (what/where/when/impact) |
| **Relation extraction** | Find entities and the relationships between them |
| **Resume / CV parsing** | Extract candidate name, title, education, experience |
| **Product review analysis** | Parse product, price, rating, pros and cons |
| **Privacy & PII protection** | Extract, classify, redact, and filter PII |

See the [Classic IE Use Cases](use-cases/classic.md), [Privacy and PII Use Cases](use-cases/privacy-pii.md), and [Other Use Cases](use-cases/others.md) pages for full code examples.

---

## Features

- **Masked tag system** — embed typed placeholders directly in f-strings.
- **Regex constraints** — restrict model output to specific patterns.
- **Named access** — retrieve results by tag name or index.
- **Multiple backends** — OpenAI, vLLM (server and offline).
- **Small-model friendly** — designed to work well with compact open-source models.
