"""
dvd_adapter.py — MCP client adapter between lmms-eval and Deep Video Discovery.

The DVD agent runs in a separate subprocess under the DVD venv
(deepvideodiscovery/.venv) via the MCP stdio protocol.  This file manages:
  - Spawning and keeping alive one persistent MCP server subprocess.
  - Forwarding run_dvd_query() calls to it from any thread.

No DVD dependencies are imported here; all DVD code runs in the subprocess.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from concurrent.futures import Future
from typing import Optional

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_init_lock = threading.Lock()
_initialized = False

# Config captured at init time — passed as env vars to the subprocess
_server_env: dict[str, str] = {}
_dvd_venv_python: str = "python"
_mcp_server_script: str = ""

# Background asyncio thread that owns the MCP session
_bg_thread: Optional[threading.Thread] = None
_bg_loop: Optional[asyncio.AbstractEventLoop] = None

# Async-side resources (only accessed from _bg_loop)
_exit_stack: Optional[contextlib.AsyncExitStack] = None
_session = None   # mcp.ClientSession


# ---------------------------------------------------------------------------
# Public init
# ---------------------------------------------------------------------------

def init_dvd_instance(
    dvd_path: str,
    base_url: str,
    api_key: str,
    vlm_model: str,
    embed_model: str = "BAAI/bge-m3",
    embed_dim: int = 1024,
    embed_base_url: Optional[str] = None,
    lite_mode: bool = True,
    max_iterations: int = 15,
    clip_secs: int = 10,
    video_fps: float = 2.0,
    global_browse_topk: int = 300,
    dvd_db_dir: str = "./.dvd_dbs",
    dvd_venv_python: str = "python",
) -> None:
    """
    Configure the DVD MCP session.  Must be called once on the main thread
    before any run_dvd_query() calls.

    Parameters
    ----------
    dvd_path : str
        Path to the deepvideodiscovery repo root (contains mcp_server.py).
    base_url : str
        VLM server base URL, e.g. "http://localhost:8000/v1".
    api_key : str
        API key for vLLM (use "EMPTY" if not set).
    vlm_model : str
        VLM model name for orchestrator + tool VLM calls.
    embed_model : str
        Embedding model name.
    embed_dim : int
        Embedding vector dimension (must match the deployed embed_model).
    embed_base_url : str, optional
        Embedding server base URL.  Defaults to base_url if omitted.
    lite_mode : bool
        Skip frame extraction; use SRT subtitles only.
    max_iterations : int
        Max DVD agent reasoning iterations per question.
    clip_secs : int
        Clip duration in seconds for video segmentation.
    video_fps : float
        Target FPS for frame extraction (non-lite mode).
    global_browse_topk : int
        Clips retrieved by global_browse_tool.
    dvd_db_dir : str
        Root directory for per-video databases.
    dvd_venv_python : str
        Absolute path to the Python interpreter in the DVD venv, e.g.
        "./deepvideodiscovery/.venv/bin/python".
    """
    global _initialized, _server_env, _dvd_venv_python, _mcp_server_script

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        _dvd_venv_python = dvd_venv_python
        _mcp_server_script = os.path.join(dvd_path, "mcp_server.py")

        if not os.path.isfile(_mcp_server_script):
            raise FileNotFoundError(
                f"[dvd_adapter] mcp_server.py not found at '{_mcp_server_script}'.\n"
                f"Set dvd_path to the deepvideodiscovery repo root."
            )

        # All config is passed to the subprocess via environment variables.
        _server_env = {
            **os.environ,
            "DVD_BASE_URL":           base_url,
            "DVD_EMBED_BASE_URL":     embed_base_url or base_url,
            "DVD_API_KEY":            api_key,
            "DVD_VLM_MODEL":          vlm_model,
            "DVD_EMBED_MODEL":        embed_model,
            "DVD_EMBED_DIM":          str(embed_dim),
            "DVD_LITE_MODE":          "true" if lite_mode else "false",
            "DVD_MAX_ITERATIONS":     str(max_iterations),
            "DVD_CLIP_SECS":          str(clip_secs),
            "DVD_VIDEO_FPS":          str(video_fps),
            "DVD_GLOBAL_BROWSE_TOPK": str(global_browse_topk),
            "DVD_DB_DIR":             dvd_db_dir,
        }

        _initialized = True


# ---------------------------------------------------------------------------
# MCP Session Management
# ---------------------------------------------------------------------------

_session_lock: Optional[asyncio.Lock] = None

async def _get_session():
    global _session, _exit_stack, _session_lock
    if _session is not None:
        return _session

    if _session_lock is None:
        _session_lock = asyncio.Lock()

    async with _session_lock:
        if _session is not None:
            return _session

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=_dvd_venv_python,
            args=[_mcp_server_script],
            env=_server_env,
        )

        _exit_stack = contextlib.AsyncExitStack()
        read, write = await _exit_stack.enter_async_context(stdio_client(server_params))
        _session = await _exit_stack.enter_async_context(ClientSession(read, write))
        await _session.initialize()
        return _session


# ---------------------------------------------------------------------------
# Public query API
# ---------------------------------------------------------------------------

async def run_dvd_query(
    video_path: str,
    question: str,
    dvd_db_dir: str = "",
    max_iterations: int = None,
    lite_mode: bool = None,
    srt_path: Optional[str] = None,
) -> str:
    """
    Run the DVD agent for a (video, question) pair via the MCP subprocess.

    The per-video database is built and cached on first call inside the
    MCP server process (DVD venv).  Must be awaited from an asyncio event loop.

    Returns
    -------
    str
        The DVD agent's final answer, or "" on failure.
    """
    if not _initialized:
        raise RuntimeError(
            "[dvd_adapter] init_dvd_instance() must be called before run_dvd_query()."
        )

    # Resolve SRT path if not provided
    if srt_path is None:
        candidate = os.path.splitext(video_path)[0] + ".srt"
        srt_path = candidate if os.path.isfile(candidate) else ""

    session = await _get_session()

    try:
        result = await session.call_tool(
            "run_dvd_query",
            {
                "video_path": video_path,
                "question": question,
                "dvd_db_dir": dvd_db_dir,
                "srt_path": srt_path,
            },
        )
        if result and result.content:
            return result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])
        return ""
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ""
