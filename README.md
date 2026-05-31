# NexusAI — Intelligent Model Serving Platform

> Production-grade AI inference platform serving medical imaging models as containerised microservices, orchestrated with Kubernetes, monitored with Prometheus + Grafana.

![Architecture](docs/architecture.png)

## What is NexusAI?

NexusAI is a cloud-native AI model serving platform that wraps trained deep learning models into independent, scalable microservices. Built to production standards — not a notebook, not a demo.

**Three core services run behind an API gateway:**
- `brain-tumor-service` — EfficientNetV2 MRI classification (97.68% accuracy)
- `retina-service` — Diabetic retinopathy grading (5 severity classes)
- `gateway-service` — Request routing, rate limiting, observability

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         NexusAI Platform             │
                    │                                      │
  Client ──────────▶  API Gateway (:8000)                 │
                    │   ├── Rate Limiting (100 req/min)    │
                    │   ├── Request Routing                │
                    │   └── Prometheus Metrics             │
                    │         │                            │
                    │    ┌────┴────┐                       │
                    │    │        │                        │
                    │ Brain    Retina                      │
                    │ Tumor    Service                     │
                    │ (:8001)  (:8002)                     │
                    │                                      │
                    │ Prometheus ◄──── All Services        │
                    │ Grafana ◄─────── Prometheus          │
                    └─────────────────────────────────────┘
```

---

## Stack

| Layer | Technology |
|---|---|
| ML Framework | TensorFlow 2.x, EfficientNetV2 |
| API | FastAPI, Uvicorn |
| Containerisation | Docker, Docker Compose |
| Orchestration | Kubernetes (HPA, Rolling Updates) |
| Monitoring | Prometheus, Grafana |
| CI/CD | GitHub Actions |
| Languages | Python 3.11, TypeScript |

---

## Quick Start (Local)

**Prerequisites:** Docker Desktop installed

```bash
# Clone the repo
git clone https://github.com/edashah474-spec/nexusai
cd nexusai

# Start the entire platform
docker-compose up --build

# Services available at:
# Gateway:    http://localhost:8000
# Brain Tumor: http://localhost:8001
# RetinaScan:  http://localhost:8002
# Prometheus:  http://localhost:9090
# Grafana:     http://localhost:3001 (admin / nexusai2025)
```

---

## API Usage

### Brain Tumor Detection

```bash
curl -X POST http://localhost:8000/predict/brain-tumor \
  -F "file=@mri_scan.jpg"
```

Response:
```json
{
  "service": "brain-tumor-detection",
  "prediction": "Tumor Detected",
  "confidence": 97.32,
  "latency_ms": 142.5,
  "model_version": "EfficientNetV2-B0",
  "timestamp": "2025-05-26T10:23:11"
}
```

### Retinopathy Grading

```bash
curl -X POST http://localhost:8000/predict/retina \
  -F "file=@retinal_fundus.jpg"
```

Response:
```json
{
  "service": "retinascan-ai",
  "prediction": "Moderate DR",
  "severity_grade": 2,
  "confidence": 91.7,
  "clinical_recommendation": "Specialist referral within 6 months",
  "latency_ms": 189.3
}
```

---

## Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace nexusai

# Deploy all services
kubectl apply -f k8s/deployments/

# Check pods
kubectl get pods -n nexusai

# Watch auto-scaling
kubectl get hpa -n nexusai --watch
```

---

## Monitoring

Grafana dashboards included for:
- Request rate per service
- Inference latency (p50, p95, p99)
- Model confidence distribution
- Auto-scaling events
- Error rates

---

## Built by

**Safa Hundekar** — AI Engineering Undergraduate, KLE Technological University  
GitHub: [edashah474-spec](https://github.com/edashah474-spec)  
Portfolio: [safahu.dev](https://safahu.dev)
