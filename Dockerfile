FROM python:3.9-slim as python-base

WORKDIR /app

COPY . /app

RUN apt-get update -y && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    curl \
    libgl1-mesa-glx \
    libglib2.0-dev \
    git-lfs \
    && pip install --upgrade pip && pip install -r requirements.txt

RUN git lfs install --force && \
    git clone -b 'main' --single-branch --depth 1 https://github.com/octadion/visionllm.git && \
    cd visionllm && \
    git lfs pull

RUN mkdir -p models/llm
RUN curl -L https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_S.gguf -o models/llm/zephyr-7b-beta.Q4_K_S.gguf

CMD ["python", "main.py"]