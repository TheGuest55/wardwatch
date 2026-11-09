# Analysis image for Tasks 4-5: feature extraction, labelling, dataset build and data checks.
# Pinned to Python 3.11 because the VM host Python (3.14) has no reliable NFStream wheels.
# This image is independent of the lab's compose project; scripts/analysis.sh runs it with
# `docker run` (never compose) and mounts data/raw read-only.
FROM python:3.11.9-slim-bookworm

# libpcap for scapy/nfstream pcap reading; libgomp for scikit-learn/nfstream threading.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libpcap0.8 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Core, hard dependencies (the pipeline must run on these alone).
RUN pip install --no-cache-dir \
      numpy==1.26.4 \
      pandas==2.2.2 \
      pyarrow==16.1.0 \
      scikit-learn==1.5.1 \
      scapy==2.6.1

# NFStream is an optional enrichment (flow-level table + a per-window flow-count feature). It
# pulls heavy transitive deps and occasionally conflicts; the build must not fail without it, and
# the extractor guards the import and logs a [DATA] note when it is absent. Set WITH_NFSTREAM=0 to
# skip it for a fast, light build (useful while a capture session is recording); the pipeline then
# leaves the nf_* columns NaN.
ARG WITH_NFSTREAM=1
RUN if [ "$WITH_NFSTREAM" = "1" ]; then \
      pip install --no-cache-dir nfstream==6.5.4 || echo "WARN: nfstream unavailable; flow-level enrichment disabled"; \
    else \
      echo "skipping nfstream (WITH_NFSTREAM=0)"; \
    fi

# Model stack (review N5): pinned versions for the three families + tracking + explanations. torch is
# the CPU-ONLY wheel from download.pytorch.org (allowlist-verified) — never the default PyPI wheel,
# which drags in multi-GB CUDA/GPU packages. mlflow/xgboost/shap come from PyPI.
RUN pip install --no-cache-dir \
      mlflow==2.16.2 \
      xgboost==2.1.1 \
      shap==0.46.0
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.4.1

WORKDIR /work
ENV PYTHONPATH=/work
CMD ["python", "-c", "print('wardwatch-analysis image ready')"]
