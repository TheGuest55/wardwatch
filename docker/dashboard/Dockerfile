# wardwatch-dashboard:0.1.0 — read-only results dashboard. CPU only, no torch.
# Kept separate from wardwatch-analysis:0.2.0 so the digest recorded in run records never changes.
FROM python:3.11.9-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 STREAMLIT_SERVER_PORT=8501 \
    WARDWATCH_DATA_ROOT=/work/data/processed/v2 PYTHONPATH=/work
COPY requirements.txt /tmp/requirements.txt
RUN pip install --timeout 120 --retries 8 -r /tmp/requirements.txt && rm /tmp/requirements.txt
WORKDIR /work
EXPOSE 8501
CMD ["streamlit", "run", "src/dashboard/app.py"]
