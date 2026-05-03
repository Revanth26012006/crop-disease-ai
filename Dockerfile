FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# ✅ Core deps
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    opencv-python \
    pillow \
    requests \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    pyyaml \
    python-multipart

# ✅ Torch (CPU)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ✅ ONNX (optional but fine)
RUN pip install --no-cache-dir onnxruntime

# 🔥 LOCK NUMPY (VERY IMPORTANT)
RUN pip install --no-cache-dir --force-reinstall numpy==1.26.4

# ❌ Prevent ultralytics config issues
ENV YOLO_CONFIG_DIR=/tmp
ENV ULTRALYTICS_DISABLE=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]