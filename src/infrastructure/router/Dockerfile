FROM rtl-operator:20260913-v4 AS timezone-data
FROM created-at-shim:latest
COPY --from=timezone-data /usr/share/zoneinfo /usr/share/zoneinfo
COPY router.py traffic_policy.py /app/
USER 65532:65532
ENTRYPOINT ["python3", "/app/router.py"]
