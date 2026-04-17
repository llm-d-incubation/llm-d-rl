FROM python:3.11-slim

WORKDIR /app

# Install py-inference-scheduler (scheduling package only — no Ray required)
# and the server dependencies.
RUN pip install --no-cache-dir \
    "git+https://github.com/llm-d-incubation/py-inference-scheduler.git" \
    "fastapi>=0.100" \
    "uvicorn[standard]>=0.20" \
    "httpx>=0.24"

# Copy the integration source.
COPY integration/py-is/ /app/

# Default config path — override with ROUTER_CONFIG_PATH env var or mount a
# ConfigMap over /etc/scheduler/scheduler.yaml in Kubernetes.
ENV ROUTER_CONFIG_PATH=/app/scheduler.yaml

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
