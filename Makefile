.PHONY: install dev lint test smoke bench docker clean

install: ; pip install -e .
dev: ; pip install -e ".[dev]"
lint: ; ruff check src tests
test: ; pytest --cov=stgnn --cov-report=term-missing
smoke: ; WANDB_MODE=disabled stgnn-bench --config configs/smoke.yaml --demo --model all
bench: ; stgnn-bench --config configs/default.yaml
docker: ; docker build -t stgnn-bench:latest .
clean: ; rm -rf build dist *.egg-info .pytest_cache .ruff_cache mlruns wandb
