build:
	uv build

install:
	uv pip install -e .

serve:
	@if [ -z "$(model_path)" ]; then \
		echo "make serve model_path=/path/to/model"; \
		exit 1; \
	fi
	uv run vllm serve $(model_path) --max_model_len 10240

lint:
	uv run ruff check
	uv run ruff format --diff
	uv run mypy --config-file pyproject.toml src tests examples dataset

lint-fix:
	uv run ruff check --fix
	uv run ruff format

test:
	uv run pytest --cov=gimkit --cov-report=term-missing:skip-covered tests

pre-commit:
	uv run pre-commit run --all-files

clean:
	rm -rf .coverage
	rm -rf dist
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf unsloth_compiled_cache
