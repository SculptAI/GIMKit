# GIMKit

**Guided Infilling Modeling Toolkit** — precise structured text generation using language models.

GIMKit lets you define placeholders (masked tags) in text and have a language model fill them in. It gives you fine-grained control over model outputs through a typed tag system with optional regex constraints.

[![PyPI Version](https://img.shields.io/pypi/v/gimkit?label=pypi%20package)](https://pypi.org/project/gimkit)
[![Python Versions](https://img.shields.io/pypi/pyversions/gimkit.svg)](https://pypi.org/project/gimkit)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](https://pypi.org/project/gimkit)

---

## Features

- **Masked tag system** — embed typed placeholders directly in f-strings.
- **Regex constraints** — restrict model output to specific patterns.
- **Named access** — retrieve results by tag name or index.
- **Multiple backends** — OpenAI, vLLM (server and offline).
- **Small-model friendly** — designed to work well with compact open-source models.

## Design Philosophy

- **Stable over feature** — reliability and correctness are prioritized above new features.
- **Small open-source model first** — designed to work well with small, freely available language models.
