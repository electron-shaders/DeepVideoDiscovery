"""
mcp_server.py — Deep Video Discovery MCP server.

Extends the original server with a ``run_dvd_query`` tool that supports:
  - Local video files (not just YouTube URLs)
  - Per-video JIT database build (captions + vector index)
  - Full config injection via environment variables so the server can be
    started by dvd_adapter.py with the DVD venv and talk to any vLLM endpoint

Environment variables read at startup
--------------------------------------
    DVD_BASE_URL          VLM server, e.g. "http://localhost:8000/v1"
    DVD_EMBED_BASE_URL    Embedding server, e.g. "http://localhost:8001/v1"
    DVD_API_KEY           API key (default: "EMPTY")
    DVD_VLM_MODEL         VLM model name
    DVD_EMBED_MODEL       Embedding model name (default: "BAAI/bge-m3")
    DVD_EMBED_DIM         Embedding vector dim (default: 1024)
    DVD_LITE_MODE         "true"/"false" (default: "true")
    DVD_MAX_ITERATIONS    int (default: 15)
    DVD_CLIP_SECS         int (default: 10)
    DVD_VIDEO_FPS         float (default: 2.0)
    DVD_GLOBAL_BROWSE_TOPK int (default: 300)
    DVD_DB_DIR            root dir for per-video databases (default: ./.dvd_dbs)
"""

import base64
import copy
import json
import os
import sys
import threading
import traceback

# Redirect sys.stdout to sys.stderr for print() calls so they don't corrupt
# the JSON-RPC stdio stream, while preserving sys.stdout.buffer for FastMCP.
class StderrRedirector:
    def __init__(self, original_stdout):
        self._original_stdout = original_stdout
    @property
    def buffer(self):
        return self._original_stdout.buffer
    def write(self, s):
        return sys.stderr.write(s)
    def flush(self):
        sys.stderr.flush()
    def __getattr__(self, name):
        return getattr(self._original_stdout, name)

sys.stdout = StderrRedirector(sys.stdout)

import dvd.config as config
from dvd.utils import extract_answer
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Read config from environment and patch dvd.config
# ---------------------------------------------------------------------------

_base_url      = os.environ.get("DVD_BASE_URL", "http://localhost:8000/v1")
_embed_base_url = os.environ.get("DVD_EMBED_BASE_URL", "") or _base_url
_api_key       = os.environ.get("DVD_API_KEY", "EMPTY")
_vlm_model     = os.environ.get("DVD_VLM_MODEL", "")
_embed_model   = os.environ.get("DVD_EMBED_MODEL", "BAAI/bge-m3")
_embed_dim     = int(os.environ.get("DVD_EMBED_DIM", "1024"))
_lite_mode     = os.environ.get("DVD_LITE_MODE", "true").lower() not in ("false", "0", "no")
_max_iter      = int(os.environ.get("DVD_MAX_ITERATIONS", "15"))
_clip_secs     = int(os.environ.get("DVD_CLIP_SECS", "10"))
_video_fps     = float(os.environ.get("DVD_VIDEO_FPS", "2.0"))
_browse_topk   = int(os.environ.get("DVD_GLOBAL_BROWSE_TOPK", "300"))
_db_dir        = os.environ.get("DVD_DB_DIR", "./.dvd_dbs")

# Patch dvd.config before any DVD code imports it further
config.OPENAI_API_KEY                  = _api_key
config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST = [_base_url]
config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME   = _vlm_model
config.AOAI_TOOL_VLM_ENDPOINT_LIST        = [_base_url]
config.AOAI_TOOL_VLM_MODEL_NAME           = _vlm_model
config.AOAI_CAPTION_VLM_ENDPOINT_LIST     = [_base_url]
config.AOAI_CAPTION_VLM_MODEL_NAME        = _vlm_model
config.AOAI_EMBEDDING_RESOURCE_LIST       = [_embed_base_url]
config.AOAI_EMBEDDING_LARGE_MODEL_NAME    = _embed_model
config.AOAI_EMBEDDING_LARGE_DIM           = _embed_dim
config.LITE_MODE                          = _lite_mode
config.MAX_ITERATIONS                     = _max_iter
config.CLIP_SECS                          = _clip_secs
config.VIDEO_FPS                          = _video_fps
config.VIDEO_DATABASE_FOLDER              = _db_dir
config.GLOBAL_BROWSE_TOPK                 = _browse_topk

# Patch raw-HTTP calls → openai SDK so they reach the vLLM server
import openai as _openai
from tenacity import (
    retry as _retry,
)
from tenacity import (
    retry_if_exception_type as _retry_if,
)
from tenacity import (
    stop_after_attempt as _stop,
)
from tenacity import (
    wait_exponential as _wait,
)


def _image_path_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"

def _call_openai_compat(
    messages, endpoints=None, model_name=None, api_key=None,
    tools=None, image_paths=None, max_tokens=4096, temperature=0.0,
    tool_choice="auto", return_json=False,
):
    payload_messages = copy.deepcopy(messages)
    if image_paths:
        payload_messages.append({"role": "user", "content": []})
        for p in image_paths:
            payload_messages[-1]["content"].append(
                {"type": "image_url", "image_url": {"url": _image_path_to_data_url(p)}}
            )
    client = _openai.OpenAI(base_url=_base_url, api_key=_api_key)
    kwargs = dict(
        model=model_name or _vlm_model,
        messages=payload_messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    if return_json:
        kwargs["response_format"] = {"type": "json_object"}

    @_retry(
        retry=_retry_if((_openai.APIConnectionError, _openai.APITimeoutError)),
        wait=_wait(multiplier=1, min=2, max=60),
        stop=_stop(8),
        reraise=True,
    )
    def _call():
        return client.chat.completions.create(**kwargs)

    try:
        response = _call()
    except Exception as exc:
        print(f"[dvd_mcp_server] LLM call failed: {exc}", file=sys.stderr)
        return None

    msg = response.choices[0].message
    if msg.tool_calls:
        return {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        }
    return {"content": (msg.content or "").strip(), "tool_calls": None}


def _get_embeddings_compat(endpoints=None, model_name=None, input_text=None, api_key=None):
    client = _openai.OpenAI(base_url=_embed_base_url, api_key=_api_key)
    response = client.embeddings.create(model=model_name or _embed_model, input=input_text)
    return [{"embedding": item.embedding} for item in response.data]


import dvd.build_database as _bdb
import dvd.utils as _du

_du.call_openai_model_with_tools = _call_openai_compat
_du.AzureOpenAIEmbeddingService.get_embeddings = staticmethod(_get_embeddings_compat)
_bdb.call_openai_model_with_tools = _call_openai_compat
_bdb.AzureOpenAIEmbeddingService.get_embeddings = staticmethod(_get_embeddings_compat)
try:
    import dvd.frame_caption as _fc
    _fc.call_openai_model_with_tools = _call_openai_compat
except ImportError:
    pass

# Per-video build lock
_build_locks: dict[str, threading.Lock] = {}
_build_locks_meta = threading.Lock()

def _get_build_lock(video_id: str) -> threading.Lock:
    with _build_locks_meta:
        if video_id not in _build_locks:
            _build_locks[video_id] = threading.Lock()
        return _build_locks[video_id]

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("dvd_agent")


def get_video_id(video_url: str) -> str:
    if "v=" in video_url:
        video_id = video_url.split("v=")[1].split("&")[0]
    else:
        video_id = os.path.splitext(os.path.basename(video_url))[0]
    return video_id


@mcp.tool()
def query_video(video_url: str, question: str) -> str:
    """
    Process a video from a URL and answer a question about it.

    This tool will download the video, decode it into frames, generate captions,
    and then use the DVDCoreAgent to answer a question about the video.

    Args:
        video_url: The URL of the video to process.
        question: The question to ask about the video.

    Returns:
        The answer to the question from the DVDCoreAgent.
    """
    from dvd.dvd_core import DVDCoreAgent
    from dvd.frame_caption import process_video
    from dvd.video_utils import decode_video_to_frames, load_video

    video_id = get_video_id(video_url)

    video_path = os.path.join(config.VIDEO_DATABASE_FOLDER, "raw", f"{video_id}.mp4")
    frames_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, video_id, "frames")
    captions_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, video_id, "captions")
    video_db_path = os.path.join(config.VIDEO_DATABASE_FOLDER, video_id, "database.json")

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(captions_dir, exist_ok=True)

    if not os.path.exists(video_path):
        load_video(video_url, video_path)

    if not os.path.exists(frames_dir) or not os.listdir(frames_dir):
        decode_video_to_frames(video_path)

    caption_file = os.path.join(captions_dir, "captions.json")
    if not os.path.exists(caption_file):
        process_video(frames_dir, captions_dir)

    agent = DVDCoreAgent(video_db_path, caption_file, config.MAX_ITERATIONS)
    msgs = agent.run(question)
    return extract_answer(msgs[-1])


@mcp.tool()
def run_dvd_query(
    video_path: str,
    question: str,
    dvd_db_dir: str = "",
    srt_path: str = "",
) -> str:
    """
    Run the DVD agent on a local video file and return the answer.

    Supports pre-extracted databases (database.json only, no captions.json).
    Builds the per-video database on first call and caches it under dvd_db_dir.
    Concurrent calls for the same video are serialized via a per-video lock.

    Args:
        video_path: Absolute path to the local video file.
        question:   The question to ask about the video.
        dvd_db_dir: Root directory for per-video databases.
                    Falls back to the DVD_DB_DIR env var or ./.dvd_dbs.
        srt_path:   Optional path to an SRT subtitle file for the video.

    Returns:
        The DVD agent's answer string.
    """
    from dvd.build_database import init_single_video_db
    from dvd.dvd_core import DVDCoreAgent, StopException
    from dvd.frame_caption import process_video, process_video_lite
    from nano_vectordb import NanoVectorDB

    db_dir = dvd_db_dir or _db_dir
    config.VIDEO_DATABASE_FOLDER = db_dir
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    lock = _get_build_lock(video_id)

    video_dir    = os.path.join(db_dir, video_id)
    caption_path = os.path.join(video_dir, "captions", "captions.json")
    db_path      = os.path.join(video_dir, "database.json")

    with lock:
        db_exists = os.path.isfile(db_path)

        if not db_exists:
            # --- Full build from scratch ---
            os.makedirs(os.path.join(video_dir, "captions"), exist_ok=True)
            have_srt = srt_path and os.path.isfile(srt_path)

            if _lite_mode:
                if have_srt:
                    process_video_lite(os.path.join(video_dir, "captions"), srt_path)
                else:
                    with open(caption_path, "w") as f:
                        json.dump({"subject_registry": {}}, f)
            else:
                from dvd.video_utils import decode_video_to_frames
                frames_dir = os.path.join(video_dir, "frames")
                if not os.path.isdir(frames_dir) or not os.listdir(frames_dir):
                    decode_video_to_frames(video_path)
                process_video(
                    frames_dir,
                    os.path.join(video_dir, "captions"),
                    subtitle_file_path=srt_path if have_srt else None,
                )
            init_single_video_db(caption_path, db_path, _embed_dim)

        elif not os.path.isfile(caption_path):
            # --- Pre-extracted DB: database.json exists but captions.json is absent.
            #     Synthesise a minimal captions.json from the already-loaded vector DB
            #     so DVDCoreAgent can initialise without rebuilding anything. ---
            os.makedirs(os.path.join(video_dir, "captions"), exist_ok=True)
            vdb = NanoVectorDB(_embed_dim, storage_file=db_path)
            additional = vdb.get_additional_data() or {}
            subject_registry = additional.get("subject_registry", {})
            # Reconstruct caption entries from the stored vector records
            captions_dict = {"subject_registry": subject_registry}
            for record in (vdb._data if hasattr(vdb, "_data") else []):
                t0 = record.get("time_start_secs", 0)
                t1 = record.get("time_end_secs", t0)
                key = f"{int(t0)}_{int(t1)}"
                captions_dict[key] = {"caption": record.get("caption", "")}
            with open(caption_path, "w") as f:
                json.dump(captions_dict, f)

        # --- Patch video_file_root in the loaded DB so frame_inspect_tool can
        #     locate frames relative to the correct local directory.
        # ---
        if not _lite_mode:
            vdb = NanoVectorDB(_embed_dim, storage_file=db_path)
            additional = vdb.get_additional_data() or {}
            local_video_dir = os.path.dirname(video_path)
            if additional.get("video_file_root") != local_video_dir:
                additional["video_file_root"] = local_video_dir
                vdb.store_additional_data(**additional)
                vdb.save()

    agent = DVDCoreAgent(db_path, caption_path, _max_iter)
    try:
        msgs = agent.run(question)
    except StopException as exc:
        return str(exc)
    except Exception as exc:
        print(f"[dvd_mcp_server] run_dvd_query failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise

    if not msgs:
        return ""
    for msg in reversed(msgs):
        answer = extract_answer(msg)
        if answer:
            return answer
    return ""


if __name__ == "__main__":
    mcp.run(transport="stdio")
