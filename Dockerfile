# Dockerfile - Streamlit + TF (CPU)
FROM python:3.10-slim

# Avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system deps for Pillow, OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# copy app
COPY . .

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "src/app_streamlit.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
