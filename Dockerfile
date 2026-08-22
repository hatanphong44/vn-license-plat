# LPR Runtime Dockerfile
# Multi-stage build for GPU-enabled inference
# CUDA 13.0 compatible (supports both PyTorch and PaddlePaddle)

# ============================================================
# Stage 1: Builder
# Install GPU-enabled Python packages using CUDA-enabled indexes
# Uses CUDA 13.0 for maximum compatibility with latest GPUs
# ============================================================
FROM nvidia/cuda:13.2.devel-ubuntu22.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    python3.12-venv \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# Create virtual environment for GPU packages
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first
COPY requirements.txt /tmp/requirements.txt

# Install PyTorch with CUDA 13.0 support FIRST
# PyTorch 2.6.x officially supports CUDA 13.0 (cu130)
RUN pip install --no-cache-dir \
    "torch==2.6.0" \
    "torchvision==0.21.0" \
    --index-url https://download.pytorch.org/whl/cu130

# Install PaddlePaddle GPU with CUDA 13.0
# IMPORTANT: Use -i (index-url) so pip finds the correct CUDA build
# Do NOT use .post130 suffix when using -i flag
RUN pip install --no-cache-dir \
    paddlepaddle-gpu==3.3.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu130/

# Install PaddleOCR and dependencies that need paddlepaddle (before other packages)
# Use --no-deps for packages that might re-install torch
RUN pip install --no-cache-dir \
    paddlex>=3.0.0 \
    paddleocr>=2.9.0 \
    triton>=3.0.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu130/

# Install ultralytics with --no-deps to prevent re-installing torch/torchvision
RUN pip install --no-cache-dir \
    "ultralytics>=8.0.0" \
    --no-deps

# Install remaining Python dependencies
# Skip GPU packages (already installed), skip ultralytics (installed with --no-deps)
RUN grep -v -E "^(torch|torchvision|paddlepaddle-gpu|paddleocr|paddlex|triton|ultralytics)" /tmp/requirements.txt > /tmp/requirements_filtered.txt && \
    pip install --no-cache-dir -r /tmp/requirements_filtered.txt

# ============================================================
# Stage 2: Production Runtime
# Uses CUDA runtime image with all dependencies
# ============================================================
FROM nvidia/cuda:13.2.runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics

# Install runtime dependencies
# FFmpeg and OpenCV system libraries for RTSP camera support
# Using packages compatible with nvidia/cuda base image
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    python3.12-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    ffmpeg \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# Create virtual environment matching builder stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy virtual environment from builder (includes all GPU packages)
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Copy configuration files needed at runtime
COPY pyproject.toml ./

# Create models directory (single root for all models)
RUN mkdir -p /models

# Environment defaults - consistent model paths
ENV YOLO_MODEL_PATH=/models/Plate.pt
ENV YOLO_DEVICE=0
ENV PADDLE_DEVICE=gpu:0
ENV PADDLE_MODEL_DIR=/models
ENV OCR_UPSCALE=4
ENV TEXT_PADDING=10
ENV CAMERA_SOURCE=0
ENV INFERENCE_FPS=5.0
ENV LOG_LEVEL=INFO
ENV PYTHONPATH=/app
ENV VISUALIZE=false

# Healthcheck using /ready endpoint (verifies models and GPU)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request, json; r=urllib.request.urlopen('http://localhost:8000/ready', timeout=5); data=json.loads(r.read()); exit(0 if data.get('ready') else 1)" || exit 1

# Expose port
EXPOSE 8000

# Run application with graceful SIGTERM/SIGINT handling
CMD ["python", "-m", "src.main", "--mode", "api"]
