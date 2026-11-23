"""
Lightweight chat UI (HTML + FastAPI) for Hugging Face Docker Spaces — no Gradio.

Run locally:
  CHAT_UI=fastapi uv run python app.py
  # or: uvicorn fastapi_ui:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import MY_NAME, chat

STATIC = Path(__file__).resolve().parent / "static"
COOKIE = "professional_you_sid"

app = FastAPI(title=f"Chat with {MY_NAME}")


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/api/chat")
async def api_chat(body: ChatIn, request: Request):
    sid = request.cookies.get(COOKIE) or uuid.uuid4().hex

    async def sse():
        last = ""
        async for chunk in chat(body.message, history=[], session_key=sid):
            last = chunk or last
            payload = json.dumps({"text": last, "done": False})
            yield f"data: {payload}\n\n"
        yield f"data: {json.dumps({'text': last, 'done': True})}\n\n"

    response = StreamingResponse(sse(), media_type="text/event-stream")
    response.set_cookie(
        COOKIE,
        sid,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,
    )
    return response