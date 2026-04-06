.PHONY: help test format lint run clean install

help:
	@echo "ManifoldX Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make install   - Install dependencies with uv"
	@echo "  make test     - Run tests with pytest"
	@echo "  make format   - Format code with ruff"
	@echo "  make lint     - Lint code with ruff"
	@echo "  make run      - Run an example (EXAMPLE=examples.pbr_demo)"
	@echo "  make clean    - Clean build artifacts"

install:
	uv sync --all-extras

test:
	uv run pytest tests/ -v --tb=short

format:
	uv run ruff format src/

lint:
	uv run ruff check src/

run:
	@if [ -z "$(EXAMPLE)" ]; then \
		echo "Usage: make run EXAMPLE=examples.pbr_demo"; \
	else \
		uv run python -m $(EXAMPLE); \
	fi

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
