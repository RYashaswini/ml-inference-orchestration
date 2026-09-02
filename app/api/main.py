"""
FastAPI orchestration service.

Endpoints:
  POST /predict/attrition   - Employee Attrition Risk model
  POST /predict/promotion   - Promotion Readiness model
  GET  /health               - liveness/readiness probe target for K8s

Each request gets a request_id for tracing/audit, and is logged
(structured logging - JD's "logging, monitoring, tracing, audit" requirement).
"""
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException
from app.core.schemas import AttritionRequest, PromotionRequest, PredictionResponse
from app.core.model_registry import MODEL_ROUTER, ModelTimeoutError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("orchestration-service")

app = FastAPI(title="ML Inference Orchestration Platform")


@app.get("/health")
def health():
    return {"status": "ok"}


def _handle_prediction(model_name: str, features: dict) -> PredictionResponse:
    request_id = str(uuid.uuid4())
    start = time.time()

    predict_fn = MODEL_ROUTER.get(model_name)
    if predict_fn is None:
        logger.warning(f"request_id={request_id} unknown model '{model_name}'")
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_name}'")

    try:
        result = predict_fn(features)
    except ModelTimeoutError as e:
        logger.error(f"request_id={request_id} model={model_name} timeout: {e}")
        raise HTTPException(status_code=504, detail="Model call timed out")
    except Exception as e:
        logger.error(f"request_id={request_id} model={model_name} error: {e}")
        raise HTTPException(status_code=500, detail="Internal model error")

    elapsed_ms = round((time.time() - start) * 1000, 2)
    logger.info(
        f"request_id={request_id} model={model_name} "
        f"prediction={result['prediction']} probability={result['probability']} "
        f"latency_ms={elapsed_ms}"
    )

    return PredictionResponse(
        model_name=model_name,
        prediction=result["prediction"],
        probability=result["probability"],
        request_id=request_id,
    )


@app.post("/predict/attrition", response_model=PredictionResponse)
def predict_attrition(payload: AttritionRequest):
    return _handle_prediction("attrition", payload.model_dump())


@app.post("/predict/promotion", response_model=PredictionResponse)
def predict_promotion(payload: PromotionRequest):
    return _handle_prediction("promotion", payload.model_dump())