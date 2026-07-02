from typing import Literal

from pydantic import BaseModel


class CreateJobResponse(BaseModel):
    job_id: str
    status: str


class JobResponse(BaseModel):
    job_id: str
    status: str
    assessment: dict | None = None


class ApproveRequest(BaseModel):
    action: Literal["approve", "edit", "reject"] = "approve"
    edits: dict = {}


class TraceStep(BaseModel):
    step: int
    node: str


class TraceResponse(BaseModel):
    job_id: str
    steps: list[TraceStep]
