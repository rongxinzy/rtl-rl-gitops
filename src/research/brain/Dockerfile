FROM rtl-operator:20260913-v3
WORKDIR /app
COPY providers.py state.py workflow.py loop.py guardian.py observer.py /app/
USER 65532:65532
ENTRYPOINT ["python3","/app/loop.py"]
