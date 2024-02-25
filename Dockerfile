FROM python:3.9-slim as python-base

WORKDIR /app

COPY . /app

RUN apt-get update -y && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    curl \
    && pip install --upgrade pip && pip install -r requirements.txt

RUN curl -L https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_S.gguf -o models/zephyr-7b-beta.Q4_K_S.gguf

CMD ["python", "main.py"]
