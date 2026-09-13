FROM rtl-operator:20260913-v3
WORKDIR /app
COPY client.py task.py bridge.py /app/
USER 65532:65532
ENTRYPOINT ["python3", "/app/bridge.py"]
