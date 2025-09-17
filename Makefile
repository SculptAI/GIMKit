build:
	uv build

install:
	uv pip install -e .

lint:
	uvx ruff check src tests
	uvx ruff format src tests --diff
	uvx mypy src

lint-fix:
	uvx ruff check src tests --fix
	uvx ruff format src tests

test:
	uvx pytest tests

pre-commit:
	uvx pre-commit run --all-files

clean:
	rm -rf dist
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf unsloth_compiled_cache
