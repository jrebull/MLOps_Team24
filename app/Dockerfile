FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libpq-dev libssl-dev libffi-dev python3-dev curl ca-certificates git cargo && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt /app/requirements-prod.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements-prod.txt

COPY app /app
EXPOSE 8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]