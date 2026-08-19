# LitKG-Integrate Makefile
# Provides convenient commands for development and deployment

.PHONY: help install install-dev setup test test-all test-integration test-slow test-gpu test-coverage test-specific test-env-check test-env-setup test-env-cleanup test-report lint format clean run-phase1 evaluate evaluate-years train-lp rag rag-coverage build-queryset eval-retrieval discover run-examples run-phase3 run-langchain run-discovery run-ollama run-agents

# Default target
help:
	@echo "LitKG-Integrate Development Commands"
	@echo "====================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install      Install dependencies with uv"
	@echo "  install-dev  Install with development dependencies"
	@echo "  setup        Setup models and environment"
	@echo "  env          Copy environment template"
	@echo ""
	@echo "Development Commands:"
	@echo "  test         Run unit tests"
	@echo "  test-all     Run all test suites"
	@echo "  test-integration Run integration tests"
	@echo "  test-slow    Run slow/performance tests"
	@echo "  test-gpu     Run GPU-specific tests"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  test-specific MODULE=<name> Run tests for specific module"
	@echo "  test-env-check Check test environment setup"
	@echo "  test-report  Generate comprehensive test report"
	@echo "  lint         Run linting (black, isort, flake8)"
	@echo "  format       Format code (black, isort)"
	@echo "  typecheck    Run type checking (mypy)"
	@echo ""
	@echo "Run Commands:"
	@echo "  run-phase1   Run Phase 1 integration pipeline"
	@echo "  evaluate     Temporal-holdout link prediction evaluation"
	@echo "  train-lp     Train the GNN/hybrid link predictors and compare"
	@echo "  rag          Ask a question over the corpus (Q=\"...\")"
	@echo "  rag-coverage Report RAG index coverage without calling the LLM"
	@echo "  build-queryset Build the CIVIC-judged retrieval query set"
	@echo "  eval-retrieval Score retrieval against the judged queries"
	@echo "  discover     End to end: rank candidates and gather their evidence"
	@echo "  evaluate-years Show year distribution for choosing a cutoff"
	@echo "  run-examples Run all example scripts"
	@echo "  run-lit      Run literature processing example"
	@echo "  run-kg       Run KG preprocessing example"
	@echo "  run-link     Run entity linking example"
	@echo "  run-ml       Run ML/HuggingFace integration example"
	@echo "  run-phase2   Run Phase 2 hybrid GNN architecture demo"
	@echo "  run-phase3   Run Phase 3 confidence scoring demo"
	@echo "  run-langchain Run LangChain integration demo"
	@echo "  run-discovery Run complete novel discovery system demo"
	@echo "  run-ollama   Run Ollama local LLM integration demo"
	@echo "  run-agents   Run conversational agents and RAG systems demo"
	@echo ""
	@echo "CLI Commands:"
	@echo "  cli-help     Show CLI help"
	@echo "  cli-setup    Setup models and environment"
	@echo "  cli-phase1   Run Phase 1 pipeline"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean        Clean cache and temporary files"
	@echo "  docs         Build documentation"
	@echo "  lock         Update uv.lock file"

# Installation
install:
	uv sync

install-dev:
	uv sync --extra dev

setup: install-dev
	uv run python scripts/setup_models.py

env:
	@if [ ! -f .env ]; then \
		cp env.template .env; \
		echo "Created .env file from template. Please edit it with your API keys."; \
	else \
		echo ".env file already exists. Skipping."; \
	fi

# Development
test:
	uv run python tests/test_runner.py --suite unit --verbose

test-all:
	uv run python tests/test_runner.py --suite all --verbose

test-integration:
	uv run python tests/test_runner.py --suite integration --verbose

test-slow:
	uv run python tests/test_runner.py --suite slow --verbose

test-gpu:
	uv run python tests/test_runner.py --suite gpu --verbose

test-coverage:
	uv run python tests/test_runner.py --suite unit --verbose
	@echo "Coverage report available at test_reports/htmlcov/index.html"

test-specific:
	@echo "Usage: make test-specific MODULE=<module_name>"
	@echo "Example: make test-specific MODULE=utils"
ifdef MODULE
	uv run python tests/test_runner.py --module $(MODULE) --verbose
endif

test-env-check:
	uv run python tests/test_runner.py --check-env

test-env-setup:
	uv run python tests/test_runner.py --setup-env

test-env-cleanup:
	uv run python tests/test_runner.py --cleanup

test-report:
	uv run python tests/test_runner.py --report

lint:
	uv run black --check src/ scripts/ tests/
	uv run isort --check-only src/ scripts/ tests/
	uv run flake8 src/ scripts/ tests/

format:
	uv run black src/ scripts/ tests/
	uv run isort src/ scripts/ tests/

typecheck:
	uv run mypy src/

# Run commands
run-phase1:
	PYTHONPATH=$(PWD)/src uv run python scripts/phase1_integration.py

run-examples: run-lit run-kg run-link run-ml run-phase2 run-phase3 run-langchain run-discovery run-ollama run-agents

run-lit:
	PYTHONPATH=$(PWD)/src uv run python scripts/example_literature_processing.py

run-kg:
	PYTHONPATH=$(PWD)/src uv run python scripts/example_kg_preprocessing.py

run-link:
	PYTHONPATH=$(PWD)/src uv run python scripts/example_entity_linking.py

run-ml:
	PYTHONPATH=$(PWD)/src uv run python scripts/example_ml_integration.py

run-phase2:
	PYTHONPATH=$(PWD)/src uv run python scripts/example_phase2_hybrid_gnn.py

run-phase3:
	PYTHONPATH=$(PWD)/src uv run python scripts/example_phase3_confidence_scoring.py

run-langchain:
	PYTHONPATH=$(PWD)/src uv run python scripts/example_langchain_integration.py

run-discovery:
	PYTHONPATH=$(PWD)/src uv run python scripts/example_novel_discovery_system.py

run-ollama:
	PYTHONPATH=$(PWD)/src uv run python scripts/example_ollama_integration.py

run-agents:
	PYTHONPATH=$(PWD)/src uv run python scripts/example_conversational_agents.py

# CLI commands (working)
cli-setup:
	uv run python scripts/litkg_cli.py setup

cli-phase1:
	uv run python scripts/litkg_cli.py phase1 --queries "BRCA1 cancer" "TP53 mutation"

cli-help:
	uv run python scripts/litkg_cli.py --help

# Utilities
clean:
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf src/**/__pycache__/
	rm -rf scripts/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

docs:
	uv run mkdocs build

docs-serve:
	uv run mkdocs serve

lock:
	uv lock

# Docker commands (future)
docker-build:
	docker build -t litkg-integrate .

docker-run:
	docker run -it --rm -v $(PWD)/data:/app/data litkg-integrate

# Jupyter notebook
notebook:
	uv run jupyter lab notebooks/

# Quick start for new users
quickstart: env install-dev setup
	@echo ""
	@echo "🎉 LitKG-Integrate setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit .env file with your API keys"
	@echo "2. Run: make run-phase1"
	@echo ""
	@echo "For help: make help"

# Evaluation
evaluate:
	python scripts/evaluate_link_prediction.py --cutoff $(or $(CUTOFF),2016) --degree-matched

evaluate-years:
	python scripts/evaluate_link_prediction.py --list-years

train-lp:
	python scripts/train_link_prediction.py --cutoff $(or $(CUTOFF),2016) --seeds $(or $(SEEDS),5)

rag:
	PYTHONPATH=$(PWD)/src uv run python scripts/run_rag.py $(if $(Q),"$(Q)",--coverage) $(if $(HOPS),--hops $(HOPS),)

rag-coverage:
	PYTHONPATH=$(PWD)/src uv run python scripts/run_rag.py --coverage

build-queryset:
	PYTHONPATH=$(PWD)/src uv run python scripts/build_retrieval_queryset.py --queries $(or $(N),60)

eval-retrieval:
	PYTHONPATH=$(PWD)/src uv run python scripts/evaluate_retrieval.py $(if $(SWEEP),--sweep,)

discover:
	PYTHONPATH=$(PWD)/src uv run python scripts/discover.py --top $(or $(TOP),20) --explain $(or $(EXPLAIN),5) $(if $(CUTOFF),--cutoff $(CUTOFF),)
