FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# set dir for your code
WORKDIR /app/

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app
COPY . .

# Streamlit port
EXPOSE 8501

ENTRYPOINT [ "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]