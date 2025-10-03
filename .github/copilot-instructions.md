# Copilot Instructions for GIM (Guided Infilling Modeling)

## Project Overview

GIM (Guided Infilling Modeling) is a Python toolkit for structured text generation using language models. It enables precise control over LLM outputs through a masked tag system that guides the model to fill in specific content.

## Architecture

### Core Components

- **MaskedTag**: The fundamental building block (`src/gimkit/schemas.py`)
  - Defines placeholders in text with optional id, name, desc, and content
  - Uses special markers: `<|MASKED|>...<|/MASKED|>`
  - Supports HTML-escaped attributes for safety

- **Guide**: Helper class for creating common tag types (`src/gimkit/guides.py`)
  - Provides convenience methods like `single_word()`, `person_name()`, `phone_number()`, etc.
  - Uses mixins for organization (BaseMixin, FormMixin, PersonalInfoMixin)

- **Contexts**: Query and Response classes (`src/gimkit/contexts.py`)
  - Query: Input with masked tags to be filled
  - Response: Output with filled masked tags
  - Support wrapping with special prefixes/suffixes

- **Models**: Adapters for different LLM backends (`src/gimkit/models/`)
  - OpenAI client support
  - vLLM support (both server and offline modes)
  - Unified interface across backends

## Code Style and Quality

### Linting and Formatting

- **Ruff**: Used for both linting and formatting
  - Configuration in `pyproject.toml`
  - Line length: 100 characters
  - Target: Python 3.10+
  - Run: `make lint` to check, `make lint-fix` to auto-fix

- **MyPy**: Type checking
  - Run as part of `make lint`
  - Configuration in `pyproject.toml`

- **Pre-commit hooks**: Configured in `.pre-commit-config.yaml`
  - Run: `make pre-commit` or `uv run pre-commit run --all-files`

### Testing

- **pytest**: Test framework with coverage tracking
  - Minimum coverage target: 97%
  - Run: `make test` or `uv run pytest tests --cov=gimkit --cov-report=term-missing:skip-covered -vv`
  - Tests in `tests/` directory mirror `src/gimkit/` structure
  - Mock external dependencies (OpenAI, vLLM)
  - Skip tests requiring optional dependencies (e.g., vLLM offline tests)

## Development Workflow

### Package Manager

- **uv**: Fast Python package manager
  - Install dependencies: `uv sync --locked --dev`
  - Run commands: `uv run <command>`

### Common Commands

```bash
make install       # Install package in editable mode
make build        # Build package
make lint         # Run linters (ruff, mypy)
make lint-fix     # Auto-fix linting issues
make test         # Run tests with coverage
make pre-commit   # Run all pre-commit hooks
make clean        # Clean build artifacts
```

## Coding Conventions

### General Guidelines

1. **Type hints**: Always use type hints for function signatures
   - Use modern syntax: `list[str]` instead of `List[str]`
   - Use `str | None` instead of `Optional[str]`
   - Use `TypeAlias` for complex types

2. **Dataclasses**: Prefer dataclasses for data structures
   - Use `@dataclass` decorator
   - Implement validation in `__post_init__`

3. **Error handling**:
   - Custom exceptions in `src/gimkit/exceptions.py`
   - Use descriptive error messages

4. **String formatting**:
   - Use f-strings for string interpolation
   - Escape HTML in tag attributes using `html.escape()`

5. **Regular expressions**:
   - Compile patterns as module-level constants
   - Use named groups for clarity
   - Use `re.DOTALL` for multiline matching when needed

### Testing Conventions

1. **File structure**: `tests/test_<module>.py` for `src/gimkit/<module>.py`
2. **Test naming**: `test_<function>_<scenario>`
3. **Fixtures**: Use pytest fixtures for common setup
4. **Mocking**: Mock external dependencies (OpenAI, vLLM) to avoid network calls
5. **Coverage**: Add `# pragma: no cover` only for methods not meant to be tested directly

### Documentation

1. **Docstrings**: Use Google-style docstrings
2. **Examples**: Provide usage examples in `examples/` directory
3. **Type information**: Include `py.typed` marker for PEP 561 compliance

## Key Patterns

### Creating Masked Tags

```python
from gimkit import guide as g

# Basic tag
tag = g(name="my_tag", desc="A description")

# Specialized tags
name_tag = g.person_name(name="user_name")
email_tag = g.e_mail(name="email")
```

### Building Queries

```python
from gimkit import Query

query_text = f"Hello, {g.person_name(name='name')}!"
query = Query(query_text)
```

### Model Integration

```python
from gimkit import from_openai
from openai import OpenAI

client = OpenAI(api_key="...")
model = from_openai(client, model_name="gpt-4")
result = model(query)
```

## Dependencies

### Core Dependencies
- `outlines[openai]>=1.2.5`: Structured generation framework

### Optional Dependencies
- `vllm>=0.10.2`: For vLLM backend support

### Development Dependencies
- Testing: pytest, pytest-cov, pytest-asyncio
- Linting: ruff, mypy
- Tools: pre-commit, ipykernel

## Important Notes

- **Magic strings**: Defined in `src/gimkit/schemas.py` as module constants
- **Tag IDs**: Must be sequential (0, 1, 2, ...) when present
- **Content restrictions**: MaskedTag content cannot contain magic strings (except TAG_OPEN_RIGHT)
- **Async support**: Models support both sync and async calls
- **Python versions**: Supports Python 3.10, 3.11, 3.12, 3.13
