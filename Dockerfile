# Listens on 0.0.0.0:$PORT (8080). Fly's proxy refuses the instance unless
# the process binds that address — 127.0.0.1 and port 7860 are unreachable.
# Hugging Face Docker Space: set app_port to the same value in README.md.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-docker.txt

COPY app.py fastapi_ui.py ./
COPY me/ me/
COPY static/ static/

ENV CHAT_UI=fastapi
ENV HOST=0.0.0.0
ENV PORT=8080
ENV USE_LOCAL_LLM=0

EXPOSE 8080

CMD ["python", "app.py"]