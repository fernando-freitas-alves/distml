# TODO: #0 user a cleaner image / make the final image smaller
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ENV PATH="/home/appuser/.local/bin:$PATH"
ENV PYTHONPATH=src

RUN apt-get update && apt-get install -y \
    wget  \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY ./app /app

RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

CMD ["/bin/bash", "-c", "scripts/download-data.sh; scripts/start.sh"]
