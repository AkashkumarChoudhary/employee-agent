import httpx


class EmployeeAgentClient:
    def __init__(self, base_url: str = "http://localhost:8000",
                 api_key: str = "demo-key", client: httpx.Client | None = None):
        self._client = client or httpx.Client(base_url=base_url, timeout=60.0)
        self._headers = {"x-api-key": api_key}

    def create_job(self, *, job_description: str, role: str, filename: str,
                   content: bytes, content_type: str = "text/plain") -> dict:
        r = self._client.post(
            "/jobs", headers=self._headers,
            data={"job_description": job_description, "role": role},
            files={"resume": (filename, content, content_type)},
        )
        r.raise_for_status()
        return r.json()

    def get_job(self, job_id: str) -> dict:
        r = self._client.get(f"/jobs/{job_id}", headers=self._headers)
        r.raise_for_status()
        return r.json()

    def approve(self, job_id: str, action: str, edits: dict | None = None) -> dict:
        r = self._client.post(
            f"/jobs/{job_id}/approve", headers=self._headers,
            json={"action": action, "edits": edits or {}},
        )
        r.raise_for_status()
        return r.json()

    def trace(self, job_id: str) -> list[dict]:
        r = self._client.get(f"/jobs/{job_id}/trace", headers=self._headers)
        r.raise_for_status()
        return r.json()["steps"]
