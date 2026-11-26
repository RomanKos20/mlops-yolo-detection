FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp/ultralytics
ENV MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app

# System libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install CPU PyTorch
RUN python -m pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install inference dependencies
COPY configs/inference/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# Copy Python inference script
COPY scripts/inference_docker.py /app/inference.py

# Create non-root user
RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app/models /app/input /app/output && \
    chown -R appuser:appuser /app/output
USER appuser
ENTRYPOINT ["python", "/app/inference.py"]
CMD ["--model", "/app/models/best.pt", "--source", "/app/input", "--output", "/app/output"]
