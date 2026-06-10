# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt requirements.txt
# Create and activate virtual environment
RUN python3 -m venv venv
RUN . venv/bin/activate && pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip \
    . venv/bin/activate && pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130

RUN --mount=type=cache,target=/root/.cache/pip \
    . venv/bin/activate && pip install --upgrade --force-reinstall \
    torch torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple

RUN . venv/bin/activate && python -c "\
import torch; \
assert '+cu130' in torch.__version__, torch.__version__; \
assert torch.version.cuda and torch.version.cuda.startswith('13'), torch.version.cuda"

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the API (deps are baked in; no run.sh — that is for local dev / venv bootstrap)
# HOST and PORT match docker-compose environment defaults.
CMD ["/bin/bash", "-c", "exec /app/venv/bin/uvicorn main:app --host \"${HOST:-0.0.0.0}\" --port \"${PORT:-8000}\""]
