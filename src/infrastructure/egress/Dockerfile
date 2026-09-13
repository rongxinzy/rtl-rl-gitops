FROM created-at-shim:latest
RUN apk add --no-cache wireguard-tools iproute2 iptables curl
COPY egress.py /app/egress.py
ENTRYPOINT ["python3", "/app/egress.py"]
