.PHONY: help install precompute rank serve test lint fmt migrate up down logs clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## install python deps
	pip install -r requirements.txt

precompute:  ## build offline artifacts
	python precompute.py --candidates ./candidates.jsonl --jd ./data/job_description.txt

rank:  ## produce submission.csv (the graded step)
	python rank.py --candidates ./candidates.jsonl --out ./submission.csv

serve:  ## run the API locally (sqlite + in-memory cache)
	uvicorn api.main:app --reload --port 8000

test:  ## run the test suite
	python -m pytest -q

lint:  ## ruff + mypy
	ruff check . && mypy core pipeline ml output services api db

fmt:  ## auto-format
	ruff check --fix . && ruff format .

migrate:  ## apply DB migrations
	alembic upgrade head

up:  ## bring up the full stack
	docker compose up --build -d

down:  ## tear down the stack
	docker compose down

logs:  ## tail api logs
	docker compose logs -f api1 api2

clean:  ## remove caches + local db
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ *.db *.db-wal *.db-shm
