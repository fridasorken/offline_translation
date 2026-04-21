FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PRODUCT_SOURCE_LANG=en
ENV PRODUCT_TARGET_LANG=nob
ENV PRODUCT_MODEL_QUANTIZATION=int8
ENV PRODUCT_MODEL_CACHE_DIR=/models/product_ct2
ENV PRODUCT_API_HOST=0.0.0.0
ENV HF_HOME=/models/huggingface
ENV PRODUCT_LOCAL_FILES_ONLY=false
ENV PRODUCT_RUN_MODE=interactive

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/config.py app/config.py
COPY app/inference.py app/inference.py
COPY preload_models.py .
RUN mkdir -p /models && python preload_models.py

COPY . /app/
ENV PRODUCT_LOCAL_FILES_ONLY=true

CMD ["python", "main.py"]
