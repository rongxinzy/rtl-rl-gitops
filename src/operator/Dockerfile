FROM rtl-python-base:20260913
RUN apk add --no-cache openssh-client tzdata && addgroup -g 65532 rtl && adduser -D -H -u 65532 -G rtl rtl
WORKDIR /app
COPY controller.py /app/controller.py
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER 65532:65532
ENTRYPOINT ["python", "/app/controller.py"]
