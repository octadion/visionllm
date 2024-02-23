FROM python:3.9-slim as python-base

WORKDIR /app
COPY . /app

RUN apt update -y

RUN apt-get update && apt-get install && pip install -r requirements.txt

CMD ["python", "streamlit.py"]