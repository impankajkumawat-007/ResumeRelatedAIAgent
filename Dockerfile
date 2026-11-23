# Hugging Face Space: sdk = docker, app_port = 7860
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-docker.txt

COPY app.py fastapi_ui.py ./
COPY me/ me/
COPY static/ static/

ENV CHAT_UI=fastapi
ENV PORT=7860
ENV USE_LOCAL_LLM=0

EXPOSE 7860

CMD ["python", "app.py"]