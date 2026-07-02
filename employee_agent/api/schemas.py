from pydantic import BaseModel


class CreateJobResponse(BaseModel):
    job_id: str
    status: str


class JobResponse(BaseModel):
    job_id: str
    status: str
    assessment: dict | None = None
