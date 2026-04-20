FROM python:3.11-slim

WORKDIR /app

# Install server dependencies and git (needed to clone py-inference-scheduler).
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "fastapi>=0.100" \
    "uvicorn[standard]>=0.20" \
    "httpx>=0.24"

# Clone py-inference-scheduler and copy the scheduling package directly.
# pip install via git+... only installs the top-level package (pyproject.toml
# lists packages = ["scheduling"] without subpackages), so we clone and copy instead.
RUN git clone --depth=1 https://github.com/llm-d-incubation/py-inference-scheduler.git /tmp/py-is \
    && cp -r /tmp/py-is/scheduling /app/scheduling \
    && rm -rf /tmp/py-is

# Copy the integration source.
COPY integration/py-is/ /app/

# Default config path — override with ROUTER_CONFIG_PATH env var or mount a
# ConfigMap over /etc/scheduler/scheduler.yaml in Kubernetes.
ENV ROUTER_CONFIG_PATH=/app/scheduler.yaml

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
