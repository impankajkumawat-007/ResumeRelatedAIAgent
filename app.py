"""
Professional You - deployable agent for Hugging Face Spaces.

This is Notebook 3, rewritten as a script. Edit the PROFILE block below,
put your LinkedIn PDF at me/linkedin.pdf, then upload this file plus
requirements.txt and the me/ folder to a Gradio Space.

Keys come from two places, and this file never contains them:
  - Running locally  -> the .env file at the project root (see .env.example)
  - Running on Spaces -> Settings -> Secrets

Either way the names are the same:
  OPENAI_API_KEY                    (required on Spaces / cloud; optional locally if
                                     LOCAL_LLM_BASE_URL is set — see .env.example)
  PUSHOVER_USER, PUSHOVER_TOKEN     (phone notifications via Pushover)
  FCM_PROJECT_ID, FCM_DEVICE_TOKEN, FCM_SERVICE_ACCOUNT_PATH
                                    (Android via Firebase Cloud Messaging)
  FCM_SERVICE_ACCOUNT_JSON          (same JSON as the file — for HF Secrets / no disk file)
  FIREBASE_SERVICE_ACCOUNT          (alias for FCM_SERVICE_ACCOUNT_JSON)
  SENDGRID_API_KEY                  (emails + calendar invites)
  SENDER_EMAIL                      (must be a VERIFIED SendGrid sender)
  MY_INBOX                          (where lead alerts land)

Never hardcode a key in this file - Spaces are public.
"""

from __future__ import annotations

import base64
import csv
import inspect
import json
import os
import re
import sys
import traceback
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import requests
from agents import (
    Agent,
    RunConfig,
    Runner,
    SQLiteSession,
    WebSearchTool,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
)
from agents.model_settings import ModelSettings
from agents.stream_events import RunItemStreamEvent
from agents.tracing import set_tracing_disabled
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from pypdf import PdfReader

# Load the project's .env when running locally. Note override=False: anything
# already in the real environment wins. On Spaces there is no .env file, so the
# Secrets you set there are used and this call quietly does nothing.
# (The notebook uses override=True instead, because there you want an edited
# .env to take effect the moment you re-run the cell.)
load_dotenv()

PUSH_TOOLS = frozenset({"notify_me", "record_unknown_question", "log_lead", "book_meeting"})


def push_log(message: str) -> None:
    """Notification debug lines on stderr (Gradio does not show these in the browser)."""
    print(f"[notifications] {message}", flush=True, file=sys.stderr)


_PUSH_CALLS_THIS_TURN = 0


def _reset_push_turn_counter() -> None:
    global _PUSH_CALLS_THIS_TURN
    _PUSH_CALLS_THIS_TURN = 0


def _mask(s: str, show: int = 6) -> str:
    s = (s or "").strip()
    if not s:
        return "(not set)"
    if len(s) <= show:
        return "***"
    return f"{s[:show]}…({len(s)} chars)"


_FCM_SA_JSON_ENV_NAMES = ("FCM_SERVICE_ACCOUNT_JSON", "FIREBASE_SERVICE_ACCOUNT")
_FCM_SCOPES = ["https://www.googleapis.com/auth/firebase.messaging"]


def _fcm_service_account_from_env() -> dict | None:
    """Service account JSON from HF Secrets (no file on disk)."""
    for name in _FCM_SA_JSON_ENV_NAMES:
        raw = os.getenv(name, "").strip()
        if not raw:
            continue
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            push_log(f"[fcm] {name} is not valid JSON: {exc}")
            return None
    return None


def _fcm_service_account_from_file() -> dict | None:
    path = os.getenv("FCM_SERVICE_ACCOUNT_PATH") or os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        push_log(f"[fcm] cannot read service account file {path}: {exc}")
        return None


def _fcm_service_account_info() -> dict | None:
    """Env JSON first (Spaces), then file path (local)."""
    return _fcm_service_account_from_env() or _fcm_service_account_from_file()


def _fcm_service_account_source_label() -> str:
    for name in _FCM_SA_JSON_ENV_NAMES:
        if os.getenv(name, "").strip():
            return f"{name} (secret JSON)"
    path = os.getenv("FCM_SERVICE_ACCOUNT_PATH") or os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )
    if path:
        return f"{path} (file)"
    return ""


def _fcm_access_token() -> str:
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    info = _fcm_service_account_info()
    if info is None:
        raise ValueError(
            "FCM service account missing — set FCM_SERVICE_ACCOUNT_JSON, "
            "FIREBASE_SERVICE_ACCOUNT, or FCM_SERVICE_ACCOUNT_PATH"
        )
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=_FCM_SCOPES
    )
    creds.refresh(Request())
    return creds.token


def log_notification_config() -> None:
    """Print once at import/startup so you know whether FCM/Pushover will actually fire."""
    po_ok = bool(os.getenv("PUSHOVER_TOKEN") and os.getenv("PUSHOVER_USER"))
    project_id = os.getenv("FCM_PROJECT_ID", "").strip()
    device_token = os.getenv("FCM_DEVICE_TOKEN", "").strip()
    sa_source = _fcm_service_account_source_label()
    fcm_ok = bool(
        project_id and device_token and _fcm_service_account_info() is not None
    )
    push_log("--- push notification config (startup) ---")
    push_log(f"Pushover enabled: {po_ok}")
    push_log(f"FCM enabled: {fcm_ok}")
    if project_id or device_token or sa_source:
        push_log(f"FCM_PROJECT_ID: {project_id or '(not set)'}")
        push_log(f"FCM_DEVICE_TOKEN: {_mask(device_token)}")
        push_log(f"FCM service account: {sa_source or '(not set)'}")
        if (
            sa_source
            and sa_source.endswith("(file)")
            and not _fcm_service_account_info()
        ):
            path = os.getenv("FCM_SERVICE_ACCOUNT_PATH") or os.getenv(
                "GOOGLE_APPLICATION_CREDENTIALS"
            )
            push_log(f"FCM service account file missing on disk: {path}")
    if not po_ok and not fcm_ok:
        push_log(
            "No push channel active — notify_me will only print [push] lines until env is set."
        )
    push_log("--- (watch this same terminal for [fcm] / [pushover] when chatting) ---")


def _local_n_ctx_hint() -> str:
    return os.getenv("LOCAL_LLM_N_CTX", "4096")


def log_agent_backend() -> None:
    if USE_LOCAL_LLM:
        push_log(f"AGENT BACKEND: LOCAL LLM | model={MODEL!r}")
        push_log(
            f"Expect llama server --n_ctx {_local_n_ctx_hint()} "
            "(restart scripts/start_local_llm.sh after changing .env)."
        )
        if _local_compact_prompt():
            push_log(
                "Local compact prompt ON (12B) — shorter system text for n_ctx=4096. "
                "Set LOCAL_LLM_COMPACT_OFF=1 for full prompt + higher n_ctx."
            )
        push_log(
            "Local mode: fewer tools, tool_choice=required on urgent "
            "visitor text (disable with LOCAL_LLM_TOOL_REQUIRED_OFF=1)."
        )
        push_log(
            "If tools stay empty, NOTIFY_FALLBACK still fires push() for urgent messages."
        )
    else:
        push_log(f"AGENT BACKEND: OpenAI cloud | model={MODEL!r}")


def _local_run_config(high_pri: bool) -> RunConfig | None:
    if not USE_LOCAL_LLM:
        return None
    temp = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.2"))
    tool_choice = None
    if high_pri and not _env_flag("LOCAL_LLM_TOOL_REQUIRED_OFF"):
        tool_choice = "required"
    elif _env_flag("LOCAL_LLM_TOOL_REQUIRED_ALWAYS"):
        tool_choice = "required"
    max_tokens = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "384"))
    return RunConfig(
        model_settings=ModelSettings(
            temperature=temp,
            parallel_tool_calls=False,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
        )
    )


_GEMMA_TOOL_BLOCK = re.compile(
    r"<\|tool_call\>.*?(?:<\/?tool_call\|>|<\|\/tool_call\|>|</tool_call>)",
    re.DOTALL | re.IGNORECASE,
)
_GEMMA_INLINE_CALL = re.compile(
    r"call:\s*[a-z_][a-z0-9_]*\s*\{.*?\}",
    re.DOTALL | re.IGNORECASE,
)
def _looks_like_leaked_tool_text(text: str) -> bool:
    """Local models sometimes stream raw tool syntax into visible assistant content."""
    t = (text or "").strip()
    if not t:
        return False
    if t[0] in "{[":
        return True
    lower = t.lower()
    if "tool_calls" in lower or "tool_call" in lower:
        return True
    if "<|tool_call>" in lower or "<tool_call|>" in lower:
        return True
    if re.search(r"call:\s*(?:notify_me|log_lead|book_meeting)\b", t, re.I):
        return True
    if '"arguments"' in t and ('"name"' in t or '"function"' in lower):
        return True
    if lower.startswith("```json"):
        return True
    return False


def _sanitize_visitor_text(text: str) -> str:
    """Strip Gemma / llama tool-call tokens; keep any real prose the model wrote."""
    t = (text or "").strip()
    if not t:
        return ""
    t = _GEMMA_TOOL_BLOCK.sub("", t)
    t = _GEMMA_INLINE_CALL.sub("", t)
    t = re.sub(r"<\|\"\|>", "", t)
    t = t.strip()
    if not t or _looks_like_leaked_tool_text(t):
        return ""
    return t


def _infer_tool_names_from_text(text: str) -> list[str]:
    """When the local model writes tool syntax into content instead of API tool_calls."""
    names: list[str] = []
    if not text:
        return names
    for m in re.finditer(r"call:\s*([a-z_][a-z0-9_]*)\s*[\{\(]", text, re.I):
        name = m.group(1).lower()
        if name not in names:
            names.append(name)
    for m in re.finditer(r'["\']name["\']\s*:\s*["\']([a-z_][a-z0-9_]*)', text, re.I):
        name = m.group(1).lower()
        if name not in names:
            names.append(name)
    return names


def _coerce_final_output_text(final) -> str:
    if final is None:
        return ""
    if isinstance(final, str):
        return final.strip()
    return str(final).strip()


def _local_reply_when_only_tools(tool_names: list[str]) -> str:
    """Short visitor-facing line when the local model never produced natural language."""
    names = set(tool_names)
    if "book_meeting" in names:
        return (
            "You're booked — check your inbox for the calendar invite. "
            f"If anything looks off, say so here and I'll fix it."
        )
    if "log_lead" in names:
        return (
            f"Thanks — I've saved your details and {MY_NAME} will follow up soon."
        )
    if "offer_meeting_slots" in names:
        return "Here are a few slots that work on my side — which one suits you, and what's the best email for the invite?"
    if "notify_me" in names:
        return (
            f"Thanks — I've flagged this to {MY_NAME} right away. "
            "What else should I know while we're here?"
        )
    if names:
        return f"Thanks — I've taken care of that on my side. Anything else I can help with?"
    return ""


def _merge_tool_names_from_turn(
    sdk_tools: list[str], streamed: str, final_s: str, push_calls: int
) -> list[str]:
    merged = list(sdk_tools)
    for name in _infer_tool_names_from_text(streamed) + _infer_tool_names_from_text(final_s):
        if name not in merged:
            merged.append(name)
    if not merged and push_calls > 0:
        merged.append("notify_me")
    return merged


def _pick_visitor_reply(
    streamed: str, final, tool_names: list[str], *, push_calls: int = 0
) -> str:
    """What the visitor should see — hide leaked tool syntax from local models."""
    streamed = (streamed or "").strip()
    final_s = _coerce_final_output_text(final)
    tool_names = _merge_tool_names_from_turn(
        tool_names, streamed, final_s, push_calls
    )

    for candidate in (final_s, streamed):
        clean = _sanitize_visitor_text(candidate)
        if clean:
            return clean

    if USE_LOCAL_LLM and tool_names:
        fallback = _local_reply_when_only_tools(tool_names)
        if fallback:
            return fallback

    if USE_LOCAL_LLM and (
        _looks_like_leaked_tool_text(streamed)
        or _looks_like_leaked_tool_text(final_s)
        or push_calls > 0
        or tool_names
    ):
        return _local_reply_when_only_tools(tool_names or ["notify_me"])

    if _looks_like_leaked_tool_text(streamed) or _looks_like_leaked_tool_text(final_s):
        return ""

    return _sanitize_visitor_text(streamed or final_s) or (streamed or final_s)


def _looks_high_priority(text: str) -> bool:
    t = text.lower()
    return any(
        w in t
        for w in (
            "hiring",
            "urgent",
            "recruit",
            "trainer",
            "interview",
            "job",
            "asap",
            "deadline",
            "partnership",
            "consult",
        )
    )


def _tool_name_from_run_item(item) -> str:
    if item.type != "tool_call_item":
        return "?"
    raw = item.raw_item
    name = getattr(raw, "name", None)
    if name:
        return name
    fn = getattr(raw, "function", None)
    if fn is not None:
        return getattr(fn, "name", None) or str(fn)
    return str(getattr(raw, "type", raw))


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


_want_local_llm = _env_flag("USE_LOCAL_LLM")
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "").strip().rstrip("/")
if _want_local_llm and not LOCAL_LLM_BASE_URL:
    default_port = os.getenv("LOCAL_LLM_PORT", "8000")
    LOCAL_LLM_BASE_URL = f"http://127.0.0.1:{default_port}"
if _want_local_llm and LOCAL_LLM_BASE_URL and not LOCAL_LLM_BASE_URL.endswith("/v1"):
    LOCAL_LLM_BASE_URL = f"{LOCAL_LLM_BASE_URL}/v1"
USE_LOCAL_LLM = _want_local_llm and bool(LOCAL_LLM_BASE_URL)
if not USE_LOCAL_LLM and os.getenv("LOCAL_LLM_BASE_URL", "").strip():
    print(
        "[local llm] LOCAL_LLM_BASE_URL is set but USE_LOCAL_LLM=0 — using OpenAI cloud.",
        flush=True,
    )


def _local_compact_prompt() -> bool:
    """Shrink system prompt so Gemma 12B fits in n_ctx=4096 on 16GB Macs."""
    if not USE_LOCAL_LLM or _env_flag("LOCAL_LLM_COMPACT_OFF"):
        return False
    if _env_flag("LOCAL_LLM_COMPACT"):
        return True
    return "12b" in os.getenv("LOCAL_GGUF_PATH", "").lower()


if not USE_LOCAL_LLM and not os.getenv("OPENAI_API_KEY"):
    raise SystemExit(
        "OPENAI_API_KEY is missing.\n"
        "  Locally: copy .env.example to .env and add your key,\n"
        "           or set LOCAL_LLM_BASE_URL for an offline model (scripts/start_local_llm.sh).\n"
        "  On Spaces: Settings -> Secrets -> add OPENAI_API_KEY."
    )

# ============================ PROFILE - EDIT THIS ============================
MY_NAME = "Pankaj Kumawat"
MY_ROLE = "Senior Mobile Engineer"
MY_CITY = "Bengaluru"

# Only real links go here - share_links() hands these straight to visitors.
# Add "GitHub" and "Portfolio" entries once you have those URLs.
MY_LINKS = {
    "LinkedIn": "https://www.linkedin.com/in/pankaj-kumawat-job/",
}

# gpt-4.1-mini is a non-reasoning model: ~2.3x faster end to end than
# gpt-5-mini here, because it does not spend time on hidden reasoning tokens
# before answering. For a website chat that mostly recites a profile and calls
# tools, that reasoning buys nothing.
if USE_LOCAL_LLM:
    _local_client = AsyncOpenAI(
        base_url=LOCAL_LLM_BASE_URL,
        api_key=os.getenv("LOCAL_LLM_API_KEY", "local"),
    )
    # Do not send traces to OpenAI when running offline (would bill / fail on dead keys).
    set_default_openai_client(_local_client, use_for_tracing=False)
    set_tracing_disabled(True)
    set_default_openai_api("chat_completions")
    _local_model_name = os.getenv("LOCAL_LLM_MODEL", "qwen2.5-7b-instruct")
    MODEL = OpenAIChatCompletionsModel(_local_model_name, _local_client)
    print(
        f"[local llm] Using chat completions at {LOCAL_LLM_BASE_URL} "
        f"(model id {_local_model_name!r}; WebSearchTool disabled)."
    )

    def _check_local_llm_server() -> None:
        root = LOCAL_LLM_BASE_URL[: -len("/v1")] if LOCAL_LLM_BASE_URL.endswith("/v1") else LOCAL_LLM_BASE_URL
        last_error = None
        for path in ("/health", "/v1/models"):
            url = f"{root}{path}"
            try:
                response = requests.get(url, timeout=5)
                if response.status_code < 500:
                    print(f"[local llm] Server OK ({url})")
                    return
            except requests.RequestException as exc:
                last_error = exc
        raise SystemExit(
            "Cannot reach the local LLM server at "
            f"{LOCAL_LLM_BASE_URL}.\n"
            f"  ({last_error})\n"
            "  Start it in another terminal and leave it running:\n"
            "    bash scripts/start_local_llm.sh\n"
            "  If the server uses another port, set LOCAL_LLM_BASE_URL and LOCAL_LLM_PORT to match."
        )

    _check_local_llm_server()
else:
    MODEL = os.getenv("MODEL", "gpt-4.1-mini")

log_notification_config()
log_agent_backend()

WORKING_HOURS = [11, 16]
MEETING_MINUTES = 30
MEETING_TITLE = f"Intro call with {MY_NAME}"
MEETING_LINK = ""
# ============================================================================

IST = timezone(timedelta(hours=5, minutes=30))

SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
SENDER_NAME = f"{MY_NAME} (AI assistant)"
# Use "or", not getenv's default: a blank MY_INBOX= line in .env sets the
# variable to an empty string, so the default would never fire.
MY_INBOX = os.getenv("MY_INBOX") or SENDER_EMAIL

LEADS_FILE = "leads.csv"
LEAD_COLUMNS = ["when", "name", "email", "company", "intent", "score", "reason"]


# --------------------------------------------------------------------------
# Profile loading
# --------------------------------------------------------------------------
def load_summary():
    """Read me/summary.txt, ignoring the shipped placeholder.

    The template file starts with REPLACE THIS FILE... - if that is still there,
    the agent would happily recite those setup instructions to visitors as if they
    were facts about you. Treat it as missing instead, and say so in the logs.
    """
    try:
        with open("me/summary.txt", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        text = ""

    if "REPLACE THIS FILE" in text.upper() or not text.strip():
        print("[warning] me/summary.txt is still the placeholder - the agent is "
              "running on the LinkedIn PDF alone. Write a real summary before the demo.")
        return f"I am {MY_NAME}, a {MY_ROLE} based in {MY_CITY}."
    return text


def load_linkedin():
    text = ""
    try:
        reader = PdfReader("me/linkedin.pdf")
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except FileNotFoundError:
        text = "(No LinkedIn PDF provided.)"
    return text


MY_SUMMARY = load_summary()
LINKEDIN_TEXT = load_linkedin()


if USE_LOCAL_LLM:
    _gguf_path = os.getenv("LOCAL_GGUF_PATH", "").lower()
    if _local_compact_prompt():
        _sum_cap = int(os.getenv("LOCAL_LLM_SUMMARY_CHARS", "500"))
        if len(MY_SUMMARY) > _sum_cap:
            MY_SUMMARY = MY_SUMMARY[:_sum_cap].rstrip() + "\n[Summary truncated for local LLM.]"
        LINKEDIN_TEXT = "(Omitted in local compact mode — use Summary only.)"
    else:
        _default_cap = "1000" if "12b" in _gguf_path else "2500"
        _profile_cap = int(os.getenv("LOCAL_LLM_PROFILE_CHARS", _default_cap))
        if len(LINKEDIN_TEXT) > _profile_cap:
            LINKEDIN_TEXT = (
                LINKEDIN_TEXT[:_profile_cap].rstrip()
                + "\n\n[LinkedIn text truncated for local context limit.]"
            )


# --------------------------------------------------------------------------
# Real-world plumbing
# --------------------------------------------------------------------------
def _push_pushover(title, message):
    token = os.getenv("PUSHOVER_TOKEN")
    user = os.getenv("PUSHOVER_USER")
    if not token or not user:
        return None

    push_log(f"[pushover] sending | {title}")
    response = requests.post(
        "https://api.pushover.net/1/messages.json",
        data={"token": token, "user": user, "title": title, "message": message},
        timeout=10,
    )
    if response.status_code == 200:
        push_log("[pushover] sent OK")
        return "Pushover: notification sent"
    push_log(f"[pushover] failed | {response.status_code} {response.text[:200]}")
    return f"Pushover failed: {response.text}"


def _push_fcm(title, message):
    project_id = os.getenv("FCM_PROJECT_ID")
    device_token = os.getenv("FCM_DEVICE_TOKEN")
    sa_info = _fcm_service_account_info()
    fcm_touched = any(
        os.getenv(k)
        for k in (
            "FCM_PROJECT_ID",
            "FCM_DEVICE_TOKEN",
            "FCM_SERVICE_ACCOUNT_PATH",
            "GOOGLE_APPLICATION_CREDENTIALS",
            *_FCM_SA_JSON_ENV_NAMES,
        )
    )
    if not project_id or not device_token or sa_info is None:
        if fcm_touched:
            missing = []
            if not project_id:
                missing.append("FCM_PROJECT_ID")
            if not device_token:
                missing.append("FCM_DEVICE_TOKEN")
            if sa_info is None:
                missing.append(
                    "FCM_SERVICE_ACCOUNT_JSON / FIREBASE_SERVICE_ACCOUNT / "
                    "FCM_SERVICE_ACCOUNT_PATH"
                )
            push_log(f"[fcm] skipped — missing: {', '.join(missing)}")
        return None

    push_log(f"[fcm] WILL CALL FCM HTTP API | project={project_id} | token={_mask(device_token)}")
    push_log(f"[fcm] POST messages:send | title={title!r}")
    try:
        access_token = _fcm_access_token()
        push_log("[fcm] OAuth access token obtained OK")
    except Exception as exc:
        push_log(f"[fcm] failed (auth) | {exc}")
        return f"FCM failed (auth): {exc}"

    url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={
            "message": {
                "token": device_token,
                "notification": {"title": title, "body": message[:4000]},
            }
        },
        timeout=15,
    )
    if response.status_code == 200:
        push_log("[fcm] sent OK — Android notification accepted by Firebase")
        return "FCM: notification sent to Android"
    push_log(f"[fcm] failed | {response.status_code} {response.text[:200]}")
    return f"FCM failed ({response.status_code}): {response.text[:200]}"


def push(title, message):
    """Send a notification to the owner (Pushover and/or FCM), or print if none set."""
    global _PUSH_CALLS_THIS_TURN
    _PUSH_CALLS_THIS_TURN += 1
    caller = traceback.extract_stack()[-2]
    push_log(
        f"[push] triggered | from {caller.name} @ {os.path.basename(caller.filename)}:"
        f"{caller.lineno} | {title} | {message}"
    )
    if _env_flag("PUSH_DEBUG"):
        push_log("[push] PUSH_DEBUG stack:\n" + "".join(traceback.format_stack()[-6:-1]))

    channels = []
    if os.getenv("PUSHOVER_TOKEN") and os.getenv("PUSHOVER_USER"):
        channels.append("pushover")
    if (
        os.getenv("FCM_PROJECT_ID")
        and os.getenv("FCM_DEVICE_TOKEN")
        and _fcm_service_account_info() is not None
    ):
        channels.append("fcm")
    push_log(f"[push] channels to try: {channels or ['(none — log only)']}")

    outcomes = []
    pushover = _push_pushover(title, message)
    if pushover:
        outcomes.append(pushover)
    else:
        push_log("[pushover] not configured — skipped")
    fcm = _push_fcm(title, message)
    if fcm:
        outcomes.append(fcm)
    elif not (
        os.getenv("FCM_PROJECT_ID")
        or os.getenv("FCM_DEVICE_TOKEN")
        or os.getenv("FCM_SERVICE_ACCOUNT_PATH")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        or any(os.getenv(k) for k in _FCM_SA_JSON_ENV_NAMES)
    ):
        push_log("[fcm] not configured — skipped")

    if not outcomes:
        push_log("[push] no channel configured (set Pushover and/or FCM env vars)")
        return "printed to logs (Pushover and FCM not configured)"

    summary = "; ".join(outcomes)
    push_log(f"[push] done | {summary}")
    return summary


def send_email(to_email, subject, html, attachments=None):
    """Send an email through SendGrid. Returns a short status string."""
    api_key = os.getenv("SENDGRID_API_KEY")
    if not api_key or not SENDER_EMAIL:
        print(f"[email] to {to_email} | {subject}")
        return "printed to logs (SendGrid not configured)"

    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": SENDER_EMAIL, "name": SENDER_NAME},
        "subject": subject,
        "content": [{"type": "text/html", "value": html}],
    }
    if attachments:
        payload["attachments"] = attachments

    response = requests.post(
        "https://api.sendgrid.com/v3/mail/send",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=20,
    )

    if response.status_code == 202:
        return f"email sent to {to_email}"
    if response.status_code == 403:
        return ("email REJECTED: SENDER_EMAIL is not a verified sender in SendGrid. "
                "Settings > Sender Authentication > Verify a Single Sender.")
    return f"email failed ({response.status_code}): {response.text[:200]}"


def upcoming_slots(count=3):
    """Return the next few free weekday slots as (id, datetime) pairs."""
    slots = []
    day = datetime.now(IST) + timedelta(days=1)
    while len(slots) < count:
        if day.weekday() < 5:
            for hour in WORKING_HOURS:
                if len(slots) < count:
                    start = day.replace(hour=hour, minute=0, second=0, microsecond=0)
                    slots.append((f"S{len(slots) + 1}", start))
        day += timedelta(days=1)
    return slots


def calendar_link(start, end, title, details):
    """Build a Google Calendar 'add event' URL. No login or API needed."""
    fmt = "%Y%m%dT%H%M%SZ"
    return (
        "https://calendar.google.com/calendar/render?action=TEMPLATE"
        f"&text={quote(title)}"
        f"&dates={start.astimezone(timezone.utc).strftime(fmt)}"
        f"/{end.astimezone(timezone.utc).strftime(fmt)}"
        f"&details={quote(details)}"
        f"&location={quote(MEETING_LINK)}"
    )


def ics_invite(start, end, title, description, organiser, attendee):
    """Build a .ics calendar file that any calendar app can open."""
    fmt = "%Y%m%dT%H%M%SZ"
    now = datetime.now(timezone.utc).strftime(fmt)
    lines = [
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Professional You//EN",
        "METHOD:REQUEST", "BEGIN:VEVENT",
        f"UID:{now}-{abs(hash(attendee))}@professional-you",
        f"DTSTAMP:{now}",
        f"DTSTART:{start.astimezone(timezone.utc).strftime(fmt)}",
        f"DTEND:{end.astimezone(timezone.utc).strftime(fmt)}",
        f"SUMMARY:{title}",
        f"DESCRIPTION:{description}",
        f"LOCATION:{MEETING_LINK}",
        f"ORGANIZER;CN={MY_NAME}:mailto:{organiser}",
        f"ATTENDEE;CN=Guest;RSVP=TRUE:mailto:{attendee}",
        "END:VEVENT", "END:VCALENDAR",
    ]
    return "\r\n".join(lines)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
@function_tool
def notify_me(reason: str, details: str) -> str:
    """Send an instant push notification to the owner's phone. Highest-priority tool.

    Call this IMMEDIATELY, in the same turn, before writing your reply, whenever the
    visitor shows high-priority intent: hiring or recruiting, a job or interview, paid
    work, training or consulting, a partnership, press or a podcast, anything urgent or
    with a deadline, an unhappy visitor, or anything important you cannot answer.
    Do not wait until you have their email - fire this on intent alone.

    reason: short label, e.g. 'Meeting request', 'Hiring - GenAI trainer', 'Urgent'.
    details: who they are and exactly what they want, including their email if known.
    """
    push_log(f"TOOL notify_me invoked | reason={reason!r} | details={details[:200]!r}")
    result = push(f"🔔 {reason}", details)
    push_log(f"TOOL notify_me returned | {result}")
    return result


@function_tool
def record_unknown_question(question: str) -> str:
    """Record a question you could not answer from the profile, so the owner can add
    that information later. Call this the moment you have to say 'I don't know', and
    then tell the visitor honestly that you have passed the question on."""
    return push("❓ Unanswered question", question)


def send_lead_card(name, email, company, purpose, notes=""):
    """Email the owner a lead card and buzz their phone. Plain function, not a tool,
    so log_lead can call it directly - a hot lead must never depend on the model
    remembering to fire a second tool."""
    html = f"""
    <div style="font-family:system-ui,sans-serif;max-width:520px">
      <h2 style="color:#b45309;margin-bottom:4px">New lead from your website</h2>
      <p style="color:#666;margin-top:0">Captured by your Professional You agent</p>
      <table cellpadding="8" style="border-collapse:collapse;width:100%">
        <tr><td style="background:#fff7ed"><b>Name</b></td><td>{name}</td></tr>
        <tr><td style="background:#fff7ed"><b>Email</b></td><td>{email}</td></tr>
        <tr><td style="background:#fff7ed"><b>Company</b></td><td>{company}</td></tr>
        <tr><td style="background:#fff7ed"><b>Wants</b></td><td>{purpose}</td></tr>
        <tr><td style="background:#fff7ed"><b>Notes</b></td><td>{notes}</td></tr>
      </table>
      <p><a href="mailto:{email}"
            style="background:#d97706;color:#fff;padding:10px 18px;
                   border-radius:8px;text-decoration:none">Reply to {name}</a></p>
    </div>
    """
    email_status = send_email(MY_INBOX, f"🔥 New lead: {name} ({company})", html)
    push_status = push("🔥 New lead", f"{name} from {company} - {purpose}")
    return f"{email_status}; {push_status}"


@function_tool
def email_lead(name: str, email: str, company: str, purpose: str, notes: str = "") -> str:
    """Email the owner a full lead card AND buzz their phone. Use for real opportunities.

    Call this as soon as you know the visitor's name and email AND they have real
    work intent - hiring, a project, training, consulting, a partnership. Send it in
    the same turn you learn the missing piece; never save it for later.
    Use the name and email the visitor gave you earlier in this conversation - do not
    ask for them a second time, and never invent them.
    A visitor you scored 'hot' in log_lead has already had this card sent automatically,
    so do not call this for them a second time.

    company: their company, or 'not given'.
    purpose: what they want, e.g. 'hiring for a GenAI trainer role'.
    notes: anything else useful - budget, timeline, team size, what they asked about.
    """
    return send_lead_card(name, email, company, purpose, notes)


@function_tool
def offer_meeting_slots() -> str:
    """Show the visitor the next available meeting slots.

    Call this as soon as anyone asks for a call, a meeting, a demo or 'some time'.
    Then list the slots in your reply exactly as returned, keeping the S1/S2/S3 ids,
    and ask which one suits them. Do not invent times - only offer what this returns.
    """
    lines = [
        f"{slot_id}: {start.strftime('%A %d %B at %I:%M %p')} IST"
        for slot_id, start in upcoming_slots()
    ]
    return "Available slots:\n" + "\n".join(lines)


@function_tool
def book_meeting(slot_id: str, visitor_name: str, visitor_email: str, topic: str) -> str:
    """Book one of the offered slots: emails the invite, files it, and buzzes the phone.

    Call this the moment you have BOTH a slot id (like S1) and the visitor's email -
    even if they were given several messages apart. Scroll back through the
    conversation for anything you are missing before asking again; ask only for the
    piece that was genuinely never given.
    Never guess an email address and never book a slot id you did not offer.

    visitor_name: their name, or 'Guest' if they truly never gave one.
    topic: what the call is about, in a few words.
    """
    chosen = dict(upcoming_slots()).get(slot_id.strip().upper())
    if chosen is None:
        return f"'{slot_id}' is not one of the offered slots. Offer the slots again."

    end = chosen + timedelta(minutes=MEETING_MINUTES)
    when = chosen.strftime("%A %d %B at %I:%M %p") + " IST"
    details = f"{topic}\n\nJoin: {MEETING_LINK}"
    gcal = calendar_link(chosen, end, MEETING_TITLE, details)

    invite = ics_invite(chosen, end, MEETING_TITLE, topic, SENDER_EMAIL, visitor_email)
    attachment = [{
        "content": base64.b64encode(invite.encode()).decode(),
        "type": "text/calendar",
        "filename": "invite.ics",
        "disposition": "attachment",
    }]

    html = f"""
    <div style="font-family:system-ui,sans-serif;max-width:520px">
      <h2 style="color:#b45309">Meeting confirmed</h2>
      <p><b>{when}</b> ({MEETING_MINUTES} minutes)</p>
      <p><b>With:</b> {MY_NAME}<br><b>Topic:</b> {topic}</p>
      <p><a href="{MEETING_LINK}">Join link</a></p>
      <p><a href="{gcal}"
            style="background:#d97706;color:#fff;padding:10px 18px;
                   border-radius:8px;text-decoration:none">Add to Google Calendar</a></p>
      <p style="color:#888;font-size:13px">The attached invite works in any calendar app.</p>
    </div>
    """

    to_visitor = send_email(visitor_email, f"Confirmed: {MEETING_TITLE} - {when}",
                            html, attachment)
    to_me = send_email(MY_INBOX, f"📅 {visitor_name} booked {when}", html, attachment)
    buzz = push("📅 Meeting booked", f"{visitor_name} ({visitor_email}) - {when}")

    return (f"Booked {when}. {to_visitor}; {to_me}; {buzz}. "
            f"Now tell the visitor it is confirmed, give the day and time, and say the "
            f"invite is in their inbox with an Add to Google Calendar button. "
            f"Do NOT put a calendar link in the chat - the email already has it.")


@function_tool
def log_lead(name: str, email: str, company: str, intent: str,
             score: str, reason: str) -> str:
    """Save a visitor to the owner's CRM and score how promising they are.

    Call this once, as soon as you know their name and email. Call it a second time
    only if something material changed - a corrected email, a new company, or much
    stronger intent. Do not re-log the same person on every message.

    score: exactly one of hot, warm, cold.
      hot  = clear hiring or paid-work intent, with a real company.
      warm = genuine interest but no concrete offer yet.
      cold = students, curiosity, or general questions.
    Score honestly. Do not flatter a polite visitor into 'hot'.
    reason: one short sentence explaining the score.
    """
    # Models improvise here - "high", "strong", "urgent" all show up instead of the
    # three words we asked for. Without this the CSV fills with junk scores and the
    # hot-lead push below never fires, which is the whole point of the tool.
    raw = (score or "").strip().lower()
    aliases = {
        "high": "hot", "strong": "hot", "urgent": "hot", "very hot": "hot", "🔥": "hot",
        "medium": "warm", "moderate": "warm", "mid": "warm",
        "low": "cold", "weak": "cold",
    }
    score = raw if raw in {"hot", "warm", "cold"} else aliases.get(raw, "warm")

    is_new = not os.path.exists(LEADS_FILE)
    with open(LEADS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(LEAD_COLUMNS)
        writer.writerow([
            datetime.now(IST).strftime("%Y-%m-%d %H:%M"),
            name, email, company, intent, score, reason,
        ])

    if score == "hot":
        # A hot lead is the whole reason this agent exists, so the email is sent here
        # rather than left to a second tool call the model might skip.
        delivery = send_lead_card(name, email, company, intent, reason)
        return (f"Logged {name} ({email}) as a HOT lead. Lead card already emailed to "
                f"the owner and pushed to their phone ({delivery}). Do NOT call "
                f"email_lead for this visitor, and do not log them again unless their "
                f"details change.")
    return (f"Logged {name} ({email}) as a {score} lead. "
            f"Do not log this visitor again unless their details change.")


@function_tool
def share_links() -> str:
    """Give the visitor the owner's real portfolio, GitHub and LinkedIn links.

    Call this whenever someone asks where to see your work, code, projects or profile.
    Return the links exactly as given - never type a URL from memory.
    """
    return "\n".join(f"{name}: {url}" for name, url in MY_LINKS.items())


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------
MEMORY_RULES = """
## Conversation memory - read this before every reply

The entire conversation so far is in front of you, including what you did with your
tools. Before you answer anything, silently re-read it and rebuild this visitor sheet:

    name · email · company · what they want · slot they picked · tools already fired

Then follow these rules without exception:

- Details the visitor gave you EARLIER are still true NOW. An email from three messages
  ago is just as usable as one from this message.
- NEVER ask for something you have already been given. No "could you remind me of your
  email", no "what was your name again". Scroll back and use it.
- If the visitor asks what you know about them ("do you have my email?", "what's my
  name?"), answer with the exact value from the sheet.
- If they correct a detail, the newest value wins and you use that from then on.
- Ask for at most ONE missing thing per reply, and only if it was genuinely never given.
- Never invent a name, email or company. If it was never given, say so and ask once.
- When you have enough for a tool, fire it immediately - do not wait for a tidier moment.
"""

_local_news_rule = (
    "6. INDUSTRY NEWS OR CURRENT EVENTS -> you have no live web access; answer from general "
    "knowledge if you can, otherwise call record_unknown_question.\n"
    if USE_LOCAL_LLM
    else "6. INDUSTRY NEWS OR CURRENT EVENTS -> web search, and mention that you looked it up.\n"
)

LOCAL_MEMORY_RULES = """
## Memory
Keep visitor name, email, company, and intent across turns. Never re-ask for details they already gave.
Fire tools as soon as you have enough facts.
"""

LOCAL_TOOL_RULES = f"""
## Tools — call via API (not prose)
1. Urgent / hiring / job / trainer → notify_me, then log_lead when name+email known.
2. name+email → log_lead (hot/warm/cold).
3. Call request → offer_meeting_slots; slot id + email → book_meeting.
4. Cannot answer from Summary → record_unknown_question.
5. Want links → share_links.
{_local_news_rule.strip()}
"""

TOOL_RULES = f"""
## Before every single reply, run this checklist

Silently ask yourself these four questions and act on any that come back yes. Facts
arrive in any order - the email may come three messages before the intent does - so
check the WHOLE conversation, not just the last message.

  1. Do I now know their name AND email, and have I not yet called log_lead? -> call it.
  2. Do they want real work (hiring, project, training, consulting, partnership) AND do
     I have their name and email? -> log_lead scored hot sends the lead card by itself.
     If you scored them warm and the work is real, call email_lead as well.
  3. Is this high priority and have I not yet called notify_me for it? -> call it.
  4. Do I hold a slot id AND an email, and have I not yet called book_meeting? -> call it.

A yes that you noticed two turns late is still a yes. Fire the tool now.

## Tool playbook - what fires when

Work down this list. More than one can apply in a single turn; fire all that apply.

1. HIGH PRIORITY -> notify_me, immediately, in the same turn, before you write a word.
   High priority = hiring or recruiting, a job or interview, paid work, training or
   consulting, a partnership, press or a podcast, anything urgent or with a deadline,
   an unhappy visitor, or something important you cannot answer.
   Fire it on intent alone - you do not need their email first.
2. NAME + EMAIL known -> log_lead once, scored hot / warm / cold (those three words
   only, never "high" or "strong"). Do this even for a curious student - that is simply
   a cold lead. Scoring someone hot ALSO emails me their lead card automatically, so
   never follow a hot log_lead with email_lead - that would send it twice. Call
   email_lead yourself only when you scored someone warm and the work is genuinely
   real. If you have a name and email and have not logged them yet, do it now, even
   several turns late.
3. WANTS A CALL -> offer_meeting_slots straight away, list the slots with their S1/S2/S3
   ids in your reply, and ask which suits them plus anything still missing.
   The moment you hold a slot id AND an email -> book_meeting. If one of those came two
   messages ago, that still counts - use it.
4. CANNOT ANSWER from the profile -> record_unknown_question, then say honestly that you
   have passed it on.
5. ASKED FOR WORK, CODE OR PROFILE -> share_links.
{_local_news_rule}
Tool discipline:
- Doing beats promising. Never say "I'll pass this on", "I'll let him know" or "I'll send
  that over" unless you called the tool in this same turn.
- Do not announce a tool before using it. Use it, then say plainly what happened:
  "Pinged Rohit and sent you the invite - check your inbox."
- Do not fire the same tool twice for the same facts. Already logged this visitor?
  Only log again if something material changed.
- If a tool returns an error or a REJECTED status, say so honestly and offer the fallback
  (share an email address, or offer to note it down). Never pretend it worked.

## Formatting
- Never paste a long raw URL into the chat. Write links as markdown: [Add to Google
  Calendar](url), [LinkedIn](url). The chat renders markdown, so a bare URL just looks
  broken.
- No headings, no tables, no bullet walls. Short paragraphs, like a person typing.
- Meeting slots are the one exception: list them on their own lines with their ids.
"""

LOCAL_LLM_TOOL_PREAMBLE = """
## CRITICAL — function tools (local model)

You MUST emit real tool_calls via the API. Never only describe actions in plain text.
- Hiring, urgent, job, trainer, interview, partnership → call notify_me FIRST this turn.
- Visitor name + email known → call log_lead (hot/warm/cold).
- Wants a meeting → offer_meeting_slots; with slot id + email → book_meeting.
If any line above applies, your turn MUST include at least one tool call before chatting.

"""

_INSTRUCTIONS_BODY = f"""You are {MY_NAME} - {MY_ROLE}, based in {MY_CITY} - talking to visitors
on your own website. They may be recruiters, clients, students or collaborators, and they
are talking to you because they want something: information, your time, or your work.

## Voice
- First person, as {MY_NAME}. Warm, direct, professional. No corporate filler.
- Two to four sentences by default. Expand only when genuinely asked for detail.
- One question per reply, maximum.
- Never claim to be an AI unless the visitor asks directly - if they ask, be honest.

## What you may state as fact
- Facts ABOUT ME (experience, skills, projects, availability, rates) come only from the
  Summary and LinkedIn Profile below. If something is not there, say you are not sure and
  offer to find out - do not invent it.
- Facts ABOUT THE VISITOR come from this conversation. Whatever they tell you - name,
  email, company, role, what they need, which slot they chose - is true and yours to use
  for the rest of the chat. The "only use my profile" rule NEVER applies to their details.
{MEMORY_RULES}
{TOOL_RULES}

## How a good conversation looks
Visitor: "Can we get on a call this week?"
You: call offer_meeting_slots, then -> "Sure. I've got S1 Tue 11:00, S2 Tue 16:00,
S3 Wed 11:00 IST. Which works, and what's the best email for the invite?"
Visitor: "S1. I'm Priya from Infosys, priya@infosys.com - hiring a GenAI trainer."
You: call notify_me, log_lead (hot - which emails me the lead card by itself) and
book_meeting, then -> "Booked S1, Tuesday 11:00 IST - the invite is on its way to
priya@infosys.com. I've flagged this to my phone too, so I'll come back to you myself."
Visitor (two messages later): "Sorry, which email did you send it to?"
You: "priya@infosys.com - the one you gave me. Want me to resend it?"
Never: "Could you remind me of your email?"

## Summary
{MY_SUMMARY}

## Links
{MY_LINKS}

## LinkedIn Profile
{LINKEDIN_TEXT}
"""

if USE_LOCAL_LLM and _local_compact_prompt():
    INSTRUCTIONS = f"""You are {MY_NAME} - {MY_ROLE} in {MY_CITY}. Website chat for visitors.
Voice: first person, warm, brief (2-4 sentences). Use tools when rules apply.
{LOCAL_LLM_TOOL_PREAMBLE}
{LOCAL_MEMORY_RULES}
{LOCAL_TOOL_RULES}

## Summary
{MY_SUMMARY}

## Links
{MY_LINKS}
"""
elif USE_LOCAL_LLM:
    INSTRUCTIONS = LOCAL_LLM_TOOL_PREAMBLE + _INSTRUCTIONS_BODY
else:
    INSTRUCTIONS = _INSTRUCTIONS_BODY

if USE_LOCAL_LLM:
    # email_lead duplicates hot log_lead — fewer tools helps small models choose correctly.
    AGENT_TOOLS = [
        notify_me,
        record_unknown_question,
        offer_meeting_slots,
        book_meeting,
        log_lead,
        share_links,
    ]
else:
    AGENT_TOOLS = [
        notify_me,
        record_unknown_question,
        email_lead,
        offer_meeting_slots,
        book_meeting,
        log_lead,
        share_links,
        WebSearchTool(),
    ]

_local_model_settings = (
    ModelSettings(
        temperature=float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.2")),
        parallel_tool_calls=False,
        max_tokens=int(os.getenv("LOCAL_LLM_MAX_TOKENS", "384")),
    )
    if USE_LOCAL_LLM
    else ModelSettings()
)

professional_you = Agent(
    name="Professional You",
    model=MODEL,
    instructions=INSTRUCTIONS,
    tools=AGENT_TOOLS,
    model_settings=_local_model_settings,
)


def to_agent_input(message, history):
    """Turn Gradio's chat history into clean input for the Agents SDK.

    Gradio's messages carry extra UI keys (options, metadata, duration) and
    sometimes non-text content for file uploads. The API rejects all of that,
    so we keep only the plain role + text pairs.
    """
    conversation = []
    for turn in history:
        # Old Gradio ("tuples" format) hands us [user_text, assistant_text].
        if isinstance(turn, (list, tuple)):
            user_text, assistant_text = (list(turn) + [None, None])[:2]
            for role, content in (("user", user_text), ("assistant", assistant_text)):
                if isinstance(content, str) and content.strip():
                    conversation.append({"role": role, "content": content})
            continue
        if not isinstance(turn, dict):  # a gr.ChatMessage object
            turn = {"role": turn.role, "content": turn.content}
        role, content = turn.get("role"), turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            conversation.append({"role": role, "content": content})
    conversation.append({"role": "user", "content": message})
    return conversation


# One Session per browser tab. Gradio's `history` only contains what is on screen,
# so passing it as the input silently throws away every tool call and tool result -
# the agent could not see that it had already logged a lead or booked a slot, and had
# to re-derive everything from its own prose. A Session stores the real conversation
# (messages AND tool traffic) and replays all of it on the next turn, which is what
# makes "you gave me that three messages ago" work.
SESSIONS = {}
MAX_SESSIONS = 200  # plain dict, so cap it - a public Space gets a lot of tabs


def session_for_key(key: str | None):
    """SQLite session for one visitor tab (tool calls + messages persist across turns)."""
    if not key:
        return None
    if key not in SESSIONS:
        if len(SESSIONS) >= MAX_SESSIONS:
            SESSIONS.pop(next(iter(SESSIONS)))  # drop the oldest tab
        SESSIONS[key] = SQLiteSession(f"visitor_{key}")
    return SESSIONS[key]


def session_for(request=None, session_key: str | None = None):
    """Gradio passes request.session_hash; FastAPI passes session_key from a cookie."""
    key = session_key
    if key is None and request is not None:
        key = getattr(request, "session_hash", None)
    if not key:
        return None  # no request object (a script or a test) -> fall back to history
    return session_for_key(key)


async def chat(message, history=None, request=None, session_key: str | None = None):
    """Stream the reply token by token.

    Runner.run() waits for the whole answer before returning anything, so the
    visitor stares at a spinner for the full response time. run_streamed()
    hands us text deltas as they arrive; yielding the reply-so-far makes
    Gradio type it out live. Same total time, but the visitor sees the first
    words in roughly a third of it.

    Gradio fills in `request` by itself because of the gr.Request annotation.
    """
    try:
        _reset_push_turn_counter()
        user_text = (message or "").strip()
        push_log(f"CHAT TURN START | user={user_text[:240]!r}")
        high_pri = _looks_high_priority(user_text)
        if high_pri:
            push_log("CHAT TURN | user text looks high-priority → expect notify_me this turn")

        run_config = _local_run_config(high_pri)
        if run_config and run_config.model_settings and run_config.model_settings.tool_choice == "required":
            push_log("CHAT TURN | LOCAL_LLM tool_choice=required for this turn")

        session = session_for(request, session_key=session_key)
        history = history if history is not None else []
        if session is not None:
            # The Session already holds every earlier turn, so send only what is new.
            result = Runner.run_streamed(
                professional_you, message, session=session, run_config=run_config
            )
        else:
            result = Runner.run_streamed(
                professional_you,
                to_agent_input(message, history),
                run_config=run_config,
            )

        reply = ""
        tools_this_turn: list[str] = []
        stream_event_count = 0
        async for event in result.stream_events():
            stream_event_count += 1
            if _env_flag("PUSH_DEBUG"):
                push_log(
                    f"STREAM #{stream_event_count} | {type(event).__name__} | "
                    f"{getattr(event, 'name', getattr(event, 'type', ''))}"
                )
            if isinstance(event, RunItemStreamEvent):
                if event.name == "tool_called" and event.item.type == "tool_call_item":
                    tool_name = _tool_name_from_run_item(event.item)
                    tools_this_turn.append(tool_name)
                    push_log(f"AGENT tool_called | {tool_name}")
                    if tool_name in PUSH_TOOLS:
                        push_log(
                            f"AGENT → may trigger FCM/Pushover via push() when {tool_name} runs"
                        )
                elif event.name == "tool_output" and event.item.type == "tool_call_output_item":
                    out = str(event.item.output)[:400]
                    push_log(f"AGENT tool_output | {out}")
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                reply += event.data.delta
                if not USE_LOCAL_LLM:
                    yield reply

        if USE_LOCAL_LLM and not tools_this_turn:
            push_log(
                "CHAT TURN: local model returned ZERO SDK tools — if this persists, verify "
                "LOCAL_LLM_CHAT_FORMAT=llama-3, restart llama server, or use cloud "
                "(USE_LOCAL_LLM=0). NOTIFY_FALLBACK may still push on urgent text."
            )
        if _PUSH_CALLS_THIS_TURN == 0:
            push_log(
                "CHAT TURN: push() was NOT called → neither Pushover nor FCM ran this turn."
            )
            if high_pri and not _env_flag("NOTIFY_FALLBACK_OFF"):
                push_log(
                    "CHAT TURN: NOTIFY_FALLBACK — firing push() because the model skipped "
                    "all tools (stream_events="
                    f"{stream_event_count} with no tool_called events)."
                )
                push(
                    "🔔 High-priority visitor (server fallback)",
                    f"Visitor said (no notify_me from model): {user_text[:900]}",
                )
        else:
            push_log(
                "CHAT TURN: push() ran — look above for [pushover] / [fcm] lines."
            )

        visitor_reply = _pick_visitor_reply(
            reply,
            result.final_output,
            tools_this_turn,
            push_calls=_PUSH_CALLS_THIS_TURN,
        )
        if USE_LOCAL_LLM and visitor_reply and (
            visitor_reply != reply.strip() or _looks_like_leaked_tool_text(reply)
        ):
            push_log(
                "CHAT TURN: using scrubbed/fallback assistant text for the visitor"
            )
        if USE_LOCAL_LLM:
            if visitor_reply:
                yield visitor_reply
        elif not reply.strip() and visitor_reply:
            yield visitor_reply
        elif _looks_like_leaked_tool_text(reply) and visitor_reply:
            yield visitor_reply

        push_related = [t for t in tools_this_turn if t in PUSH_TOOLS]
        push_log(
            f"CHAT TURN END | stream_events={stream_event_count} | "
            f"tools={tools_this_turn or '(none)'} | push_tools={push_related or '(none)'} | "
            f"push() calls={_PUSH_CALLS_THIS_TURN} | reply_chars={len(reply)}"
        )
        if high_pri and "notify_me" not in tools_this_turn:
            other_push = [t for t in tools_this_turn if t in ("log_lead", "book_meeting")]
            if _PUSH_CALLS_THIS_TURN > 0 and other_push:
                push_log(
                    "CHAT TURN NOTE: high-priority message — notify_me was skipped, but "
                    f"{other_push} called push() ({_PUSH_CALLS_THIS_TURN}x). "
                    "FCM/Pushover should still have fired (check [fcm] / [pushover] above). "
                    "Instructions want notify_me *before* reply for instant urgent ping; "
                    "log_lead only pushes once with the lead card."
                )
            elif _PUSH_CALLS_THIS_TURN > 0:
                push_log(
                    "CHAT TURN NOTE: notify_me was not called but push() ran "
                    f"({_PUSH_CALLS_THIS_TURN}x) — see [push] lines above for source."
                )
            else:
                push_log(
                    "CHAT TURN WARNING: high-priority message and notify_me was NOT "
                    "called; push() calls=0 unless NOTIFY_FALLBACK just ran. "
                    "The model should call notify_me on urgent hiring intent. "
                    + (
                        "Local models often skip tools — try a fuller message with "
                        "name, email, and company, or USE_LOCAL_LLM=0 for stricter tool use."
                        if USE_LOCAL_LLM
                        else "Include name, email, and company so the model calls "
                        "notify_me and log_lead."
                    )
                )
        elif tools_this_turn and not any(t in PUSH_TOOLS for t in tools_this_turn):
            push_log(
                "CHAT TURN NOTE: tools ran but none were push-related "
                f"({tools_this_turn}) — no FCM unless notify_me/log_lead/book_meeting."
            )

        if not visitor_reply:
            if USE_LOCAL_LLM and (
                tools_this_turn
                or _looks_like_leaked_tool_text(reply)
                or _PUSH_CALLS_THIS_TURN > 0
            ):
                yield (
                    f"Thanks — I've noted that and flagged it for {MY_NAME}. "
                    "What else should I know?"
                )
            else:
                yield (
                    "(No reply from the model. If you just raised --n_ctx, restart "
                    "scripts/start_local_llm.sh and try again.)"
                )
    except Exception as exc:
        message = str(exc)
        ctx_too_small = (
            "context_length" in message.lower()
            or "context length" in message.lower()
            or "exceed context window" in message.lower()
        )
        if ctx_too_small:
            n_ctx = _local_n_ctx_hint()
            yield (
                "The local llama server context is too small for this turn "
                f"(.env wants LOCAL_LLM_N_CTX={n_ctx}).\n\n"
                "1. Stop the old server on port 8000 (it may still be on n_ctx=2048).\n"
                "2. Open a **new** Gradio tab (fresh session).\n"
                "3. Restart:\n\n"
                "  bash scripts/start_local_llm.sh\n\n"
                "Then restart app.py. Compact prompt for 12B is on by default."
            )
        else:
            yield f"Sorry, something went wrong: {message}"


def _launch_gradio() -> None:
    import gradio as gr

    # Gradio 5 still defaults to the old [user, assistant] list format, so it
    # needs type="messages" to hand us role/content dicts. Later versions drop
    # the argument because that format is the only one left - pass it only when
    # this installed version still accepts it. (to_agent_input copes with both
    # shapes either way.)
    chat_kwargs = {}
    if "type" in inspect.signature(gr.ChatInterface.__init__).parameters:
        chat_kwargs["type"] = "messages"

    # Bind every interface. Gradio's default is 127.0.0.1, which Fly's proxy
    # (and any other process outside this container) cannot connect to.
    server_name = os.getenv("GRADIO_SERVER_NAME") or os.getenv("HOST") or "0.0.0.0"
    launch_kwargs = {"server_name": server_name}
    if os.getenv("PORT"):
        launch_kwargs["server_port"] = int(os.environ["PORT"])
    elif os.getenv("GRADIO_SERVER_PORT"):
        launch_kwargs["server_port"] = int(os.environ["GRADIO_SERVER_PORT"])

    print(f"Starting Gradio on http://{server_name}:{launch_kwargs.get('server_port', 7860)}", flush=True)
    gr.ChatInterface(
        chat,
        **chat_kwargs,
        title=f"Chat with {MY_NAME}",
        description="Ask about my work, or book a call - my agent can actually do it.",
        examples=[
            "What kind of work do you do?",
            "Can we schedule a call this week?",
            "Where can I see your code?",
        ],
    ).launch(**launch_kwargs)


if __name__ == "__main__":
    ui = os.getenv("CHAT_UI", "gradio").strip().lower()
    if ui == "fastapi":
        import uvicorn

        from fastapi_ui import app as fastapi_app

        # Fly's proxy dials 0.0.0.0:8080. PORT overrides this; never bind loopback.
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8080"))
        print(f"Starting FastAPI on http://{host}:{port}", flush=True)
        uvicorn.run(fastapi_app, host=host, port=port)
    else:
        _launch_gradio()
