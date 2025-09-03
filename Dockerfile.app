FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# copia tu app y los artefactos del modelo que usa la app
COPY src/app ./src/app
COPY src/artifacts ./src/artifacts

EXPOSE 8501
ENV PYTHONPATH=/app
# La app usará esta URL para llamar a la API en docker-compose
ENV API_URL=http://api:8000

# si tu entrada es src/app/streamlit_app.py, déjalo así
CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
