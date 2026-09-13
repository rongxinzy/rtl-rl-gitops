# Build from repository root; providers.py is the exact coordinator implementation.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}
WORKDIR /app
COPY research/ops_agent/agent.py /app/agent.py
COPY research/brain/providers.py /app/providers.py
USER 65532:65532
ENTRYPOINT ["python3", "/app/agent.py"]
