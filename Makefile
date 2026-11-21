.PHONY: dev down test test-models lint fmt secrets create-admin logs setup

setup:                 ## install dev tools and git hooks (run once)
	python3.11 -m venv .venv
	.venv/bin/pip install -U pip ruff pytest pre-commit -e libs/privface_core
	.venv/bin/pre-commit install

dev:                   ## full stack locally (needs Docker; see docs/SPEC_AMENDMENTS.md section A)
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
down:
	docker compose down
test:                  ## all Python tests (no model-dependent tests)
	pytest libs/privface_core services/vault services/api services/worker -m "not models" -q
test-models:           ## slow tests that need the ML models
	pytest services/ml -q
lint:
	ruff check . && ruff format --check .
	@if [ -f web/package.json ]; then cd web && npm run lint; fi
fmt:
	ruff check --fix . && ruff format .
	@if [ -f web/package.json ]; then cd web && npx prettier -w src; fi
secrets:
	python deploy/scripts/gen_secrets.py
create-admin:
	docker compose exec api python -m app.scripts.create_admin
logs:
	docker compose logs -f --tail=100