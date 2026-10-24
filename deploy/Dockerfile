# LightOS API image. Multi-stage so the runtime layer carries no build tools.
FROM python:3.11-slim AS build

WORKDIR /build
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime

# Run unprivileged: the API accepts untrusted input from the internet.
RUN useradd --create-home --uid 10001 lightos
WORKDIR /app

COPY --from=build /install /usr/local
COPY lightos/ ./lightos/
COPY main.py task_config.yaml ./

USER lightos
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
    sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', \
    timeout=3).status == 200 else 1)"

CMD ["python", "-m", "lightos.server"]
