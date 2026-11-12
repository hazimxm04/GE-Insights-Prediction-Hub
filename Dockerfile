# MYRamalan dashboard (includes the analyst agent page).
# Build:  docker build -t myramalan .
# Run:    docker run --rm -p 8501:8501 -e GROQ_API_KEY=your_key_here myramalan
# Then open http://localhost:8501
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first, so editing code does not reinstall them.
# xgboost-cpu is the same xgboost 2.1.1 without the 336 MB GPU (NCCL) download that the plain
# package pulls in on Linux; this app only predicts on CPU. Retries cover a dropped download.
COPY requirements.txt .
RUN sed -i 's/^xgboost==/xgboost-cpu==/' requirements.txt \
 && pip install --default-timeout=120 --retries 5 -r requirements.txt

COPY . .

# Run as a normal user, not root.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "streamlit_app/Home.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]