FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY app.py train_model.py ./
COPY templates ./templates
COPY static ./static

# Build-time training creates model/spam_classifier.joblib for the web app.
RUN python train_model.py \
    && rm -rf data

EXPOSE 5000

CMD ["python", "app.py"]
