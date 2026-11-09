# Analysis image for Tasks 4-5: feature extraction, labelling, dataset build, data checks and the
# model harness. Pinned to Python 3.11 (NFStream has no reliable 3.14 wheels). Independent of the
# lab's compose project; run with `docker run` (never compose), mounting data/raw read-only.
FROM python:3.11.9-slim-bookworm

# Build provenance: the git commit this image was built from, as a build-arg baked into an OCI label
# and an env var — so a run can record exactly which code built its image, and a stale image reused
# under the same tag is caught with `docker image inspect` (.Config.Labels / .revision).
ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.title="wardwatch-analysis" \
      org.opencontainers.image.revision="$GIT_COMMIT"
# GIT_PYTHON_REFRESH=quiet silences mlflow/GitPython's "Bad git executable" probe (git is not in the
# image on purpose; run provenance is passed in via WARDWATCH_GIT_* env, see src/models/tracking.py).
ENV WARDWATCH_IMAGE_GIT_COMMIT=$GIT_COMMIT \
    PYTHONPATH=/work \
    GIT_PYTHON_REFRESH=quiet

# libpcap for scapy/nfstream pcap reading; libgomp for scikit-learn/nfstream threading.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libpcap0.8 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# requirements.lock is a `pip freeze` of a known-good build — every direct AND indirect package
# pinned. Passed as -c constraints to every install below so indirect deps cannot drift on a rebuild.
COPY requirements.lock /tmp/requirements.lock

# Core, hard dependencies (the pipeline must run on these alone).
RUN pip install --no-cache-dir -c /tmp/requirements.lock \
      numpy==1.26.4 pandas==2.2.2 pyarrow==16.1.0 scikit-learn==1.5.1 scapy==2.6.1

# NFStream: optional flow-level enrichment, OFF the model path (review B1). The build must not fail
# without it. Set WITH_NFSTREAM=0 for a lighter build.
ARG WITH_NFSTREAM=1
RUN if [ "$WITH_NFSTREAM" = "1" ]; then \
      pip install --no-cache-dir -c /tmp/requirements.lock nfstream==6.5.4 \
        || echo "WARN: nfstream unavailable; flow-level enrichment disabled"; \
    else echo "skipping nfstream (WITH_NFSTREAM=0)"; fi

# Model stack: the three families + tracking + explanations. CPU-ONLY torch from the pytorch cpu
# index (never the default PyPI wheel, which pulls multi-GB CUDA packages).
RUN pip install --no-cache-dir -c /tmp/requirements.lock \
      mlflow==2.16.2 xgboost==2.1.1 shap==0.46.0
RUN pip install --no-cache-dir -c /tmp/requirements.lock \
      --index-url https://download.pytorch.org/whl/cpu torch==2.4.1

WORKDIR /work
CMD ["python", "-c", "print('wardwatch-analysis image ready')"]
