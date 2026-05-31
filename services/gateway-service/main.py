"""
NexusAI — API Gateway Service
Central routing, authentication, rate limiting, and observability
"""

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import httpx
import time
import logging
import os
from datetime import datetime
from collections import defaultdict
import asyncio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] GATEWAY - %(message)s")
logger = logging.getLogger("gateway")

# ── METRICS ──
GATEWAY_REQUESTS = Counter("gateway_requests_total", "Total gateway requests", ["service", "status"])
GATEWAY_LATENCY = Histogram("gateway_latency_seconds", "Gateway latency", ["service"])
ACTIVE_CONNECTIONS = Gauge("gateway_active_connections", "Active connections")
SERVICE_HEALTH = Gauge("gateway_service_health", "Service health status", ["service"])

app = FastAPI(
    title="NexusAI — API Gateway",
    description="Intelligent routing gateway for NexusAI model serving platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SERVICE REGISTRY ──
SERVICES = {
    "brain-tumor": os.getenv("BRAIN_TUMOR_URL", "http://brain-tumor-service:8001"),
    "retina": os.getenv("RETINA_URL", "http://retina-service:8002"),
}

# ── RATE LIMITING ──
request_counts = defaultdict(list)
RATE_LIMIT = 100  # requests per minute

def check_rate_limit(client_ip: str):
    now = time.time()
    minute_ago = now - 60
    request_counts[client_ip] = [t for t in request_counts[client_ip] if t > minute_ago]
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 100 requests/minute.")
    request_counts[client_ip].append(now)

START_TIME = datetime.utcnow()

# ── ROUTES ──
@app.get("/health")
async def health():
    uptime = (datetime.utcnow() - START_TIME).total_seconds()
    service_status = {}
    async with httpx.AsyncClient(timeout=3.0) as client:
        for name, url in SERVICES.items():
            try:
                r = await client.get(f"{url}/health")
                service_status[name] = "healthy" if r.status_code == 200 else "degraded"
                SERVICE_HEALTH.labels(service=name).set(1 if r.status_code == 200 else 0)
            except:
                service_status[name] = "unreachable"
                SERVICE_HEALTH.labels(service=name).set(0)

    all_healthy = all(v == "healthy" for v in service_status.values())
    return {
        "status": "healthy" if all_healthy else "degraded",
        "gateway": "online",
        "services": service_status,
        "uptime_seconds": round(uptime, 2),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/services")
async def list_services():
    return {
        "available_services": [
            {
                "id": "brain-tumor",
                "name": "Brain Tumor Detection",
                "endpoint": "/predict/brain-tumor",
                "method": "POST",
                "input": "MRI image file",
                "accuracy": "97.68%"
            },
            {
                "id": "retina",
                "name": "RetinaScan AI",
                "endpoint": "/predict/retina",
                "method": "POST",
                "input": "Retinal fundus image",
                "grades": 5
            }
        ],
        "total": len(SERVICES)
    }

@app.post("/predict/{service_name}")
async def proxy_predict(service_name: str, request: Request):
    client_ip = request.client.host
    check_rate_limit(client_ip)

    if service_name not in SERVICES:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found. Available: {list(SERVICES.keys())}")

    start = time.time()
    ACTIVE_CONNECTIONS.inc()

    try:
        url = f"{SERVICES[service_name]}/predict"
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, content=body, headers=headers)

        latency = time.time() - start
        GATEWAY_REQUESTS.labels(service=service_name, status=str(response.status_code)).inc()
        GATEWAY_LATENCY.labels(service=service_name).observe(latency)

        result = response.json()
        result["gateway_latency_ms"] = round(latency * 1000, 2)
        result["routed_by"] = "NexusAI Gateway v1.0"
        return result

    except httpx.TimeoutException:
        GATEWAY_REQUESTS.labels(service=service_name, status="timeout").inc()
        raise HTTPException(status_code=504, detail=f"Service '{service_name}' timed out")
    except Exception as e:
        GATEWAY_REQUESTS.labels(service=service_name, status="error").inc()
        logger.error(f"Gateway error for {service_name}: {e}")
        raise HTTPException(status_code=502, detail="Service temporarily unavailable")
    finally:
        ACTIVE_CONNECTIONS.dec()

@app.get("/stats")
async def stats():
    return {
        "gateway": "NexusAI API Gateway",
        "version": "1.0.0",
        "uptime_seconds": round((datetime.utcnow() - START_TIME).total_seconds(), 2),
        "rate_limit": f"{RATE_LIMIT} req/min",
        "services_registered": len(SERVICES),
        "built_by": "Safa Hundekar"
    }
