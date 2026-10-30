.PHONY: install run test lint train backtest icons docker

install:
	python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

run:
	.venv/bin/uvicorn app.main:app --reload --port 8000

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

train:
	.venv/bin/python -m training.train

backtest:
	.venv/bin/python -m training.backtest

icons:
	.venv/bin/python scripts/make_icons.py

docker:
	docker build -t bourse-score . && docker run --rm -p 10000:10000 bourse-score
