"""Request/response models. Pydantic validates every payload at the edge."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .config import DEPLOY_THRESHOLD

Severity = Literal["Warning", "Minor", "Major", "Critical"]
RaiseOrClear = Literal["Raise", "Clear"]
DeviceRole = Literal["access", "core", "edge"]


class AlarmEvent(BaseModel):
    hours_ago: float = Field(
        ..., ge=0, le=24, description="Hours before now that this alarm fired (0-24)."
    )
    alarm_type: str = Field(..., description="e.g. equipmentAlarm, communicationsAlarm")
    severity: Severity = "Warning"
    raise_or_clear: RaiseOrClear = "Raise"


class DeviceHistory(BaseModel):
    n_prior_faults: int = Field(0, ge=0)
    n_prior_hw_faults: int = Field(0, ge=0)
    n_prior_sw_faults: int = Field(0, ge=0)
    has_pending_fault: int = Field(0, ge=0, le=1)


class PredictRequest(BaseModel):
    device_id: str | None = None
    device_role: DeviceRole = "access"
    events: list[AlarmEvent] = Field(default_factory=list)
    device_history: DeviceHistory = Field(default_factory=DeviceHistory)
    threshold: float | None = Field(
        None, ge=0, le=1, description=f"Override the deploy threshold ({DEPLOY_THRESHOLD})."
    )
    include_explanation: bool = False


class FeatureContribution(BaseModel):
    feature: str
    label: str
    value: float
    contribution: float


class Explanation(BaseModel):
    base_value: float
    contributions: list[FeatureContribution]


class PredictResponse(BaseModel):
    device_id: str | None = None
    fault_probability: float
    threshold: float
    fault_predicted: bool
    fault_type: str | None = None
    type_confidence: float | None = None
    action: str
    action_detail: str
    features: dict[str, float]
    explanation: Explanation | None = None


class ScenarioRequest(BaseModel):
    """Two named scenarios scored side by side — the live version of the
    chronic-vs-healthy probe from the training notebook."""

    label_a: str = "Scenario A"
    scenario_a: PredictRequest
    label_b: str = "Scenario B"
    scenario_b: PredictRequest


class ScenarioResponse(BaseModel):
    label_a: str
    result_a: PredictResponse
    label_b: str
    result_b: PredictResponse
    delta: float
    verdict: str


class FleetDevice(BaseModel):
    device_id: str
    device_role: str
    fault_probability: float
    fault_predicted: bool
    fault_type: str | None = None
    action: str
    n_events_window: int
    n_prior_faults: int
    risk_band: str


class FleetResponse(BaseModel):
    analyzed_at: str
    n_devices: int
    n_at_risk: int
    threshold: float
    devices: list[FleetDevice]


class HistoryPoint(BaseModel):
    timestamp: str
    fault_probability: float
    n_events_window: int
    n_prior_faults: int


class DeviceHistoryResponse(BaseModel):
    device_id: str
    device_role: str
    n_events_total: int
    first_seen: str
    last_seen: str
    n_fault_episodes: int
    n_hw_faults: int
    n_sw_faults: int
    health_assessment: str
    current: PredictResponse
    timeline: list[HistoryPoint]
    alarm_type_breakdown: dict[str, int]
    severity_breakdown: dict[str, int]


class ChatRequest(BaseModel):
    """One question from the on-call engineer about a raised alert."""
    message: str = Field(..., min_length=1, max_length=2000,
                         description="What the engineer wants to know about this incident.")
