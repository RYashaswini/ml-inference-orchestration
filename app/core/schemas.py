"""
Request/response schemas for both models.
Pydantic handles the JD's "request validation" requirement automatically -
if a request doesn't match these shapes, FastAPI rejects it before it
ever reaches our code.
"""
from pydantic import BaseModel, Field
from typing import Literal


class AttritionRequest(BaseModel):
    age: int = Field(ge=18, le=70)
    department: Literal["Engineering", "Sales", "Support", "HR", "Finance", "Marketing"]
    work_mode: Literal["Onsite", "Hybrid", "Remote"]
    tenure_months: int = Field(ge=0, le=600)
    monthly_salary: float = Field(gt=0)
    months_since_last_hike: int = Field(ge=0, le=120)
    productivity_score: float = Field(ge=0, le=10)
    leaves_last_90_days: int = Field(ge=0)
    manager_change_count: int = Field(ge=0)
    performance_rating: int = Field(ge=1, le=5)


class PromotionRequest(BaseModel):
    department: Literal["Engineering", "Sales", "Support", "HR", "Finance", "Marketing"]
    current_level: Literal["Junior", "Mid", "Senior"]
    tenure_in_role_months: int = Field(ge=0, le=600)
    avg_performance_rating_last_4q: float = Field(ge=1, le=5)
    goals_completed_pct: float = Field(ge=0, le=100)
    peer_review_score: float = Field(ge=0, le=10)
    training_hours_last_year: int = Field(ge=0)
    cross_team_projects: int = Field(ge=0)
    manager_recommendation: Literal[0, 1]


class PredictionResponse(BaseModel):
    model_name: str
    prediction: int
    probability: float
    request_id: str