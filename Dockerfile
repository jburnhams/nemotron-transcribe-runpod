FROM nvidia/cuda:13.5.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/runpod-volume/huggingface \
    PATH="/opt/nemo/.venv/bin:/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# NeMo Speech with the vLLM plugin, as in the model card
ARG NEMO_REF=main
RUN git clone https://github.com/NVIDIA-NeMo/Speech.git /opt/nemo \
    && cd /opt/nemo && git checkout ${NEMO_REF} \
    && uv sync --extra asr --extra vllm --locked --no-install-project \
    && uv pip install --no-deps -e . \
    && uv pip install "av==19.0.2" "opencv-python-headless==5.0.0.93" runpod

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
