.PHONY: help install dev clean test lint format docker-up docker-down init-db airflow-init

help:
	@echo "ConformAI - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make dev            Install development dependencies"
	@echo "  make init           Initialize project (databases, collections)"
	@echo ""
	@echo "Development:"
	@echo "  make run-api        Run API Gateway locally"
	@echo "  make test           Run tests"
	@echo "  make lint           Run linters"
	@echo "  make format         Format code"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up      Start all services"
	@echo "  make docker-down    Stop all services"
	@echo "  make docker-logs    View logs"
	@echo "  make docker-ps      List running containers"
	@echo ""
	@echo "Airflow:"
	@echo "  make airflow-init   Initialize Airflow database"
	@echo "  make airflow-user   Create Airflow admin user"
	@echo "  make airflow-web    Open Airflow web UI"
	@echo ""
	@echo "Database:"
	@echo "  make db-migrate     Run database migrations"
	@echo "  make db-upgrade     Upgrade database schema"

# Installation
install:
	uv pip install -e .

dev:
	uv pip install -e ".[dev]"

# Initialization
init:
	python scripts/init_project.py

# Development
run-api:
	cd services/api-gateway && uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --cov=services --cov-report=html

lint:
	ruff check .
	mypy services/ shared/

format:
	ruff check --fix .
	black .
	isort .

# Docker
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-ps:
	docker-compose ps

docker-rebuild:
	docker-compose up -d --build

# Airflow
airflow-init:
	docker-compose exec airflow-webserver airflow db migrate

airflow-user:
	docker-compose exec airflow-webserver airflow users create \
		--username admin \
		--firstname Admin \
		--lastname User \
		--role Admin \
		--email admin@conformai.com \
		--password admin

airflow-web:
	@echo "Opening Airflow UI at http://localhost:8080"
	@open http://localhost:8080 || xdg-open http://localhost:8080

# Database
db-migrate:
	alembic revision --autogenerate -m "Auto migration"

db-upgrade:
	alembic upgrade head

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build

# Quick start for new developers
quickstart: dev init docker-up airflow-init airflow-user
	@echo ""
	@echo "✅ ConformAI is ready!"
	@echo ""
	@echo "Services running:"
	@echo "  - API Gateway:    http://localhost:8000/docs"
	@echo "  - Airflow:        http://localhost:8080 (admin/admin)"
	@echo "  - Qdrant:         http://localhost:6333/dashboard"
	@echo "  - MinIO Console:  http://localhost:9001 (minioadmin/minioadmin)"
	@echo ""
