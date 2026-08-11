FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; wheels cover the rest.
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Dependencies come from the lockfile (exact, tested versions) so the image is
# reproducible and cannot drift; copied before the source for layer caching.
COPY requirements.lock.txt ./
RUN pip install --upgrade pip && pip install -r requirements.lock.txt

COPY employee_agent ./employee_agent

EXPOSE 8000
CMD ["uvicorn", "employee_agent.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
