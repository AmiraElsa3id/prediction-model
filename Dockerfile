FROM python:3.14-slim

# libgomp1 is the OpenMP runtime LightGBM's Linux wheel links against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps before copying app code so this layer is cached between
# code-only rebuilds (the layer only invalidates when requirements.txt changes).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data/ holds the generated synthetic dataset and the optional model/registry
# stores.  Mount it as a volume so files survive container restarts.
RUN mkdir -p data

# 8200, not uvicorn's default 8000: RestoMindAPI's AiClientService defaults
# AI_SERVICE_URL to http://127.0.0.1:8200 (Common/Services/ai-client.service.ts) --
# matching it here means the backend can reach this container with zero config.
EXPOSE 8200

# Training (~20s) runs inside the lifespan on first request, not at image build
# time, so the image stays lean and cold starts are predictable.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8200"]
