FROM python:3.9-slim as python-base

# WORKDIR /app

# COPY . /app

# RUN apt-get update -y && apt-get install -y \
#     gcc \
#     g++ \
#     python3-dev

# RUN pip install --upgrade pip && pip install -r requirements.txt

# # Atur perintah default
# CMD ["python", "streamlit.py"]
