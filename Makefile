build:
	uv build

install:
	uv pip install -e .

lint:
	uv run ruff check src tests
	uv run ruff format src tests --diff
	uv run mypy src

lint-fix:
	uv run ruff check src tests --fix
	uv run ruff format src tests

test:
	uv run pytest tests

pre-commit:
	uv run pre-commit run --all-files

clean:
	rm -rf dist
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf unsloth_compiled_cache
