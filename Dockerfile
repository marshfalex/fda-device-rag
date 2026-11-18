# Multi-arch image (linux/amd64 + linux/arm64 for OCI Ampere nodes).
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface

# Pick up Debian security fixes (Trivy gates on fixable CRITICAL/HIGH).
RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 app
WORKDIR /app

# requirements.txt begins with "." (installs this package, as Streamlit
# Cloud needs), so pyproject.toml and src/ must be present before pip runs.
COPY requirements.txt pyproject.toml ./
COPY src/ src/
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

COPY app.py .

USER 10001
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health').status==200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
