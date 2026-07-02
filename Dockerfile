FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; wheels cover the rest.
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install \
    "pydantic>=2.6" "pydantic-settings>=2.2" "google-genai>=0.3" "ollama>=0.3" \
    "langchain-text-splitters>=0.2" "pypdf>=4.0" "chromadb>=0.5" \
    "langgraph>=0.2" "langgraph-checkpoint-sqlite>=2.0" \
    "fastapi>=0.110" "python-multipart>=0.0.9" "slowapi>=0.1.9" "uvicorn>=0.29" \
    "mcp>=1.0" "streamlit>=1.30"

COPY employee_agent ./employee_agent

EXPOSE 8000
CMD ["uvicorn", "employee_agent.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
