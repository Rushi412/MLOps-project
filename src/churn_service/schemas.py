from typing import Literal

from pydantic import BaseModel, Field


class ChurnRequest(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(gt=0, le=500)
    total_charges: float = Field(ge=0)
    support_tickets: int = Field(ge=0, le=50)
    contract_type: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["dsl", "fiber", "none"]


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    churn_risk: Literal["low", "high"]
    model_version: str

