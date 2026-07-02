from dataclasses import dataclass


@dataclass
class JobRecord:
    job_id: str
    owner: str
    role: str
    status: str


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}

    def create(self, job_id: str, owner: str, role: str, status: str = "running") -> JobRecord:
        rec = JobRecord(job_id=job_id, owner=owner, role=role, status=status)
        self._jobs[job_id] = rec
        return rec

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def set_status(self, job_id: str, status: str) -> None:
        rec = self._jobs.get(job_id)
        if rec is not None:
            rec.status = status
