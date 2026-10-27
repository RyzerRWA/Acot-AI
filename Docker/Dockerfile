FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Embeddings run on CPU in this service; keep CUDA out of the image.
ENV CUDA_VISIBLE_DEVICES=""

COPY requirements.txt .

# 1) CPU torch first
# 2) app deps (sentence-transformers will reuse installed torch if compatible)
# 3) force CPU torch again in case a dep pulled the CUDA wheel from PyPI
# Without this, Linux builds pull multi-GB NVIDIA packages and the image
# will not fit / pull cleanly on a ~30GB VM root disk.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --force-reinstall --no-deps torch \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip uninstall -y \
        nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
        nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand \
        nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 \
        nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx \
        cuda-toolkit cuda-bindings cuda-pathfinder triton \
        2>/dev/null || true \
    && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

COPY . .

ENV PORT=8080

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
