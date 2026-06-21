FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt waitress

COPY . .

# Create required directories
RUN mkdir -p uploads instance model

EXPOSE 8000

CMD ["python", "wsgi.py"]
