PYTHON  ?= python
VENV    ?= .venv
COMPOSE := docker compose -f docker/docker-compose.yml

ifeq ($(OS),Windows_NT)
  BIN := $(VENV)/Scripts
else
  BIN := $(VENV)/bin
endif
PY := $(BIN)/python

.DEFAULT_GOAL := help

## install: create a venv and install the package with dev extras
install:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

## test: run the full test suite
test:
	$(PY) -m pytest

## test-fast: skip tests that use real time
test-fast:
	$(PY) -m pytest -m "not slow"

## lint: ruff check and format check
lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

## format: apply ruff formatting and autofixes
format:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

## typecheck: mypy over the package
typecheck:
	$(PY) -m mypy

## up: start the stack (skein, ollama, jaeger, prometheus, grafana)
up:
	$(COMPOSE) up -d --build

## down: stop the stack
down:
	$(COMPOSE) down

## logs: follow the skein service logs
logs:
	$(COMPOSE) logs -f skein

## serve: run the API locally against an already-running Ollama
serve:
	$(PY) -m uvicorn skein.api.app:app --host 0.0.0.0 --port 8000 --reload

## demo: submit the example workflow to a running API
demo:
	$(PY) -m skein.cli submit examples/research.yaml --follow

## k3d-up: create the local cluster
k3d-up:
	k3d cluster create --config k8s/k3d-config.yaml

## k3d-deploy: build, import and apply the manifests
k3d-deploy:
	docker build -f docker/Dockerfile -t skein:local .
	k3d image import skein:local -c skein
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
	kubectl apply -f k8s/hpa.yaml

## k3d-down: delete the local cluster
k3d-down:
	k3d cluster delete skein

.PHONY: install test test-fast lint format typecheck up down logs serve demo \
        k3d-up k3d-deploy k3d-down help

## help: list targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'
