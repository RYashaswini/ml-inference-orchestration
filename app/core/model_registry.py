"""
Model registry + orchestration logic.

- Loads both model artifacts once at startup (not per-request).
- Exposes one predict() per model, wrapped with tenacity for
  retry/timeout resilience (JD requirement).
- In a real platform, "calling the model" might mean an HTTP call to
  a hosted model service. We keep that same shape here (a function
  that could fail transiently) so the resilience layer is real,
  even though today it's an in-process joblib call.
"""
import time
import joblib
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

ATTRITION_MODEL_PATH = "app/models/attrition_model.joblib"
PROMOTION_MODEL_PATH = "app/models/promotion_model.joblib"

_attrition_model = joblib.load(ATTRITION_MODEL_PATH)
_promotion_model = joblib.load(PROMOTION_MODEL_PATH)


class ModelTimeoutError(Exception):
    """Raised if a model call takes too long - simulated here,
    but this is the same pattern used for a real remote model call."""
    pass


def _run_with_timeout(fn, *args, timeout_seconds=2.0, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - start
    if elapsed > timeout_seconds:
        raise ModelTimeoutError(f"Model call exceeded {timeout_seconds}s")
    return result


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.2, min=0.2, max=2),
    retry=retry_if_exception_type(ModelTimeoutError),
    reraise=True,
)
def predict_attrition(features: dict) -> dict:
    X = pd.DataFrame([features])
    proba = _run_with_timeout(_attrition_model.predict_proba, X)[0][1]
    prediction = int(proba >= 0.5)
    return {"prediction": prediction, "probability": round(float(proba), 4)}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.2, min=0.2, max=2),
    retry=retry_if_exception_type(ModelTimeoutError),
    reraise=True,
)
def predict_promotion(features: dict) -> dict:
    X = pd.DataFrame([features])
    proba = _run_with_timeout(_promotion_model.predict_proba, X)[0][1]
    prediction = int(proba >= 0.5)
    return {"prediction": prediction, "probability": round(float(proba), 4)}


# Orchestration/routing table: maps a model_name in the request path
# to the right predict function. This is the "route request to the
# appropriate model endpoint" logic from the JD.
MODEL_ROUTER = {
    "attrition": predict_attrition,
    "promotion": predict_promotion,
}