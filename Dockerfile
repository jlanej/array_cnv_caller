FROM python:3.12-slim

LABEL maintainer="jlanej"
LABEL description="array_cnv_caller – Deep-learning CNV caller for Illumina array data"

# Install system dependencies required by pysam (htslib)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libc6-dev \
        zlib1g-dev \
        libbz2-dev \
        liblzma-dev \
        libcurl4-openssl-dev \
        tabix \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_ml.txt .
RUN pip install --no-cache-dir -r requirements_ml.txt

COPY scripts/ scripts/
COPY resources/ resources/

ENTRYPOINT ["python"]
