FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/src/main/python \
    /app/data/bronze \
    /app/data/silver \
    /app/data/gold \
    /app/logs \
    /app/mlruns

COPY src/main/python/ /app/src/main/python/
COPY spark-defaults.conf /app/

ENV PYTHONPATH=/app
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV SPARK_HOME=/usr/local/lib/python3.9/site-packages/pyspark

RUN echo '#!/bin/bash\n\
python -m src.main.python.bronze_layer && \
python -m src.main.python.silver_layer && \
python -m src.main.python.gold_layer' > /app/run_pipeline.sh && \
chmod +x /app/run_pipeline.sh

CMD ["/app/run_pipeline.sh"] 