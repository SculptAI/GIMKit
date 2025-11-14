build:
	uv build

install:
	uv pip install -e .

install-dev:
	uv sync --all-groups --all-extras

lint:
	uv run ruff check
	uv run ruff format --diff
	uv run mypy --config-file pyproject.toml src tests examples evals

lint-fix:
	uv run ruff check --fix
	uv run ruff format

test:
	uv run pytest tests --cov=gimkit --cov-report=term-missing:skip-covered -vv --durations=10

pre-commit:
	uv run pre-commit run --all-files

clean:
	rm -rf .coverage
	rm -rf dist
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
