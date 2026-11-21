# Runtime image for the proxy, simulator, load generator and alert sink.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY inferobs ./inferobs
RUN pip install --no-cache-dir . "uvloop>=0.19"
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["inferobs"]
