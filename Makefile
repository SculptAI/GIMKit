build:
	uv build

install:
	uv pip install -e .

serve:
	@if [ -z "$(model_path)" ]; then \
		echo "make serve model_path=/path/to/model"; \
		exit 1; \
	fi
	vllm serve $(model_path) --max_model_len 10240

lint:
	uv run ruff check src tests examples
	uv run ruff format src tests examples --diff
	uv run mypy src tests examples

lint-fix:
	uv run ruff check src tests examples --fix
	uv run ruff format src tests examples

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
