FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PRODUCT_SOURCE_LANG=en
ENV PRODUCT_TARGET_LANG=nob
ENV PRODUCT_MODEL_QUANTIZATION=int8
ENV PRODUCT_MODEL_CACHE_DIR=/opt/product-models/ct2
ENV HF_HOME=/opt/product-models/huggingface
ENV PRODUCT_LOCAL_FILES_ONLY=false
ENV PRODUCT_RUN_MODE=interactive

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py inference.py preload_models.py ./
RUN mkdir -p /opt/product-models && python preload_models.py

COPY . /app/
ENV PRODUCT_LOCAL_FILES_ONLY=true

EXPOSE 8080

CMD ["python", "main.py"]
