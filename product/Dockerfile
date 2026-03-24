FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PRODUCT_SOURCE_LANG=en
ENV PRODUCT_TARGET_LANG=nob
ENV PRODUCT_MODEL_QUANTIZATION=int8
ENV PRODUCT_MODEL_CACHE_DIR=/models/product_ct2
ENV PRODUCT_RUN_MODE=interactive

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 8080

CMD ["python", "main.py"]
