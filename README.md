# ML Inference Orchestration Platform

A production-style ML inference orchestration service built with FastAPI, containerized with Docker, deployed on Kubernetes, and shipped through two Azure deployment paths — a custom AKS cluster and a managed Azure ML Studio online endpoint.

## What it does

Routes real-time prediction requests to two independently-trained models through a single orchestration layer:

- **Employee Attrition Risk** — predicts whether an employee is likely to leave
- **Promotion Readiness** — predicts internal-mobility/promotion readiness

Both models are RandomForest classifiers trained on synthetic HR data, served behind a common FastAPI service with request validation, retry/timeout resilience, structured logging, and Prometheus metrics.

## Architecture

```
Client
  │
  ▼
FastAPI (app/api/main.py)
  ├── POST /predict/attrition
  ├── POST /predict/promotion
  ├── GET  /health
  └── GET  /metrics (Prometheus)
       │
       ▼
model_registry.py — loads both models once, tenacity retry/timeout wrapper, routes by endpoint
       │
       ▼
schemas.py (Pydantic validation) → attrition_model.joblib / promotion_model.joblib
```

## Tech stack

- **API:** FastAPI, Pydantic, Uvicorn
- **ML:** scikit-learn (RandomForestClassifier + OneHotEncoder pipelines), joblib
- **Resilience:** tenacity (retry/timeout)
- **Observability:** prometheus-fastapi-instrumentator
- **Dependency management:** uv
- **Containerization:** Docker (multi-stage build)
- **Orchestration:** Kubernetes (validated locally via `kind`, deployed to Azure AKS)
- **CI/CD:** GitHub Actions → GHCR
- **Cloud:** Azure AKS (custom) + Azure ML Studio (managed online endpoint)
- **Testing:** pytest, FastAPI TestClient

## Project structure

```
app/
├── models/           # data generators, training scripts, saved joblib artifacts
├── api/main.py       # FastAPI app, routes, Prometheus instrumentation
└── core/
    ├── schemas.py         # Pydantic request models
    └── model_registry.py  # model loading, retry/timeout, routing
tests/                # integration tests (pytest + TestClient)
k8s/                  # Deployment + Service manifests
deploy/               # Azure ML Studio endpoint/deployment configs + score.py
.github/workflows/    # CI/CD pipeline
Dockerfile
```

## Running locally

```bash
uv sync
uv run uvicorn app.api.main:app --reload
```

```bash
curl -X POST http://localhost:8000/predict/attrition \
  -H "Content-Type: application/json" \
  -d '{
    "age": 29,
    "department": "Engineering",
    "work_mode": "Hybrid",
    "tenure_months": 18,
    "monthly_salary": 85000,
    "months_since_last_hike": 14,
    "productivity_score": 6.5,
    "leaves_last_90_days": 3,
    "manager_change_count": 1,
    "performance_rating": 3
  }'
```

## Running with Docker

```bash
docker build -t ml-inference-orchestration:latest .
docker run -d -p 8000:8000 ml-inference-orchestration:latest
```

## Testing

```bash
uv run pytest
```

## CI/CD

Every push to `main` runs the test suite, then builds and publishes a Docker image to GitHub Container Registry (GHCR) if tests pass.

## Deployment

Two deployment paths are supported, both serving the same models:

1. **Kubernetes (AKS)** — `k8s/deployment.yaml` + `k8s/service.yaml`, exposed via a `LoadBalancer` service
2. **Azure ML Studio managed online endpoint** — `deploy/endpoint.yml` + `deploy/deployment.yml` + `deploy/score.py`

See manifests in `k8s/` and `deploy/` for full configuration.

## Observability

- `/health` — liveness/readiness probe endpoint
- `/metrics` — Prometheus-format request counts, latency histograms, and status codes per route
- Structured logs with a per-request `request_id` for traceability