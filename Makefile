.PHONY: install lint type test check run api docker index eval

install:        ## Install package + dev dependencies
	pip install -e ".[dev]"

lint:           ## Run the linter
	ruff check src tests scripts

type:           ## Run the type checker
	mypy

test:           ## Run tests with coverage
	pytest

check: lint type test  ## Run all quality gates

run:            ## Run the CLI (pass ARGS="--pdf ... --question ...")
	python scripts/run_qa.py $(ARGS)

index:          ## Index a PDF for RAG (pass PDF=path/to.pdf)
	python scripts/index_document.py --pdf $(PDF)

eval:           ## Run the evaluation harness against the gold set
	python scripts/run_evaluation.py --eval-set data/evaluation_set.json --output eval_results.json

api:            ## Run the API locally with autoreload
	uvicorn llm_qa.api.main:app --reload --port 8000

docker:         ## Build the Docker image
	docker build -t llm-qa-pipeline:latest .
