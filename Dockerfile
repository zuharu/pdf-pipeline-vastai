# ── PDF Ingestion Pipeline — Docker Image for Vast.ai ────────────────────────
# Multi-stage: builder (warm Surya cache) → runner (slim)
# Base: PyTorch 2.5.1 + CUDA 12.4 (Ampere RTX 3090 + Ada RTX 4090)
# VLM: Qwen2.5-VL-7B-Instruct via Ollama (6.5 GB VRAM Q4_K_M)
#
# Usage on Vast.ai:
#   vastai create instance OFFER_ID --image ghcr.io/zuharu/pdf-pipeline-vastai:latest --disk 50 --ssh --direct
#   scp -P PORT input/*.pdf root@HOST:/workspace/input/
#   ssh -p PORT root@HOST 'bash /workspace/run_pipeline.sh'

# ── Stage 1: Builder (warm Surya cache) ─────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git openssh-server supervisor \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps — pin transformers for Surya compatibility
RUN pip install --no-cache-dir \
    marker-pdf \
    "transformers>=4.45.2,<5.0.0" \
    "regex<2025.0.0,>=2024.4.28" \
    opencv-python-headless \
    Pillow \
    rich \
    openai \
    ollama \
    python-dotenv

# Warm Surya model cache (prevents 17s cold-start on instance boot)
RUN python3 -c "from marker.converters.pdf import PdfConverter; \
    converter = PdfConverter(artifact_dict={}); \
    print('[BUILD] Surya models cached successfully')"

# ── Stage 2: Runner (slim) ──────────────────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS runner

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_HOST=0.0.0.0

# System deps (runtime only — no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git openssh-server supervisor \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    marker-pdf \
    "transformers>=4.45.2,<5.0.0" \
    "regex<2025.0.0,>=2024.4.28" \
    opencv-python-headless \
    Pillow \
    rich \
    openai \
    ollama \
    python-dotenv

# Copy Surya cache from builder
COPY --from=builder /root/.cache /root/.cache

# Install Ollama binary (GPU-aware runner)
RUN curl -fsSL https://ollama.com/install.sh | sh

# ── SSHd setup (Vast.ai connectivity) ──────────────────────────────────────
RUN mkdir -p /var/run/sshd /root/.ssh && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# ── Workspace directories ───────────────────────────────────────────────────
RUN mkdir -p /workspace/input /workspace/staging /workspace/output /var/log/supervisor

# ── Supervisor config ──────────────────────────────────────────────────────
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# ── Pipeline scripts ───────────────────────────────────────────────────────
COPY run_pipeline.sh /workspace/run_pipeline.sh
COPY vlm_prompt_gen.py /workspace/vlm_prompt_gen.py
COPY vlm_describe.py /workspace/vlm_describe.py
COPY validate_container.py /workspace/validate_container.py
RUN chmod +x /workspace/*.sh /workspace/*.py

# ── Entry point ────────────────────────────────────────────────────────────
EXPOSE 22
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]