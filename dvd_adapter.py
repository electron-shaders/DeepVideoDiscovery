"""
dvd_adapter.py — MCP client adapter between lmms-eval and Deep Video Discovery.

The DVD agent runs as a persistent FastMCP streamable-HTTP server in the DVD
venv (deepvideodiscovery/.venv).  This adapter manages:

  - Launching the server subprocess once (init_dvd_instance).
  - Issuing each run_dvd_query() call over a fresh, independent HTTP connection
    so that all concurrent callers proceed in parallel — unlike the stdio
    transport there is no shared session gate.

No DVD dependencies are imported here; all DVD code runs in the subprocess.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Optional

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_init_lock   = threading.Lock()
_initialized = False

# Config captured at init time
_server_env: dict[str, str] = {}
_dvd_venv_python: str = "python"
_mcp_server_script: str = ""
_mcp_url: str = ""          # e.g. "http://127.0.0.1:9002/mcp"
_mcp_client: Optional[Client] = None

# Server subprocess handle
_server_proc: Optional[subprocess.Popen] = None


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
    dvd_host: str = "127.0.0.1",
    dvd_port: int = 9002,
) -> None:
    """
    Launch the DVD MCP HTTP server as a background subprocess and wait for it
    to be ready.  Must be called once before any run_dvd_query() calls.

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
        Absolute path to the Python interpreter in the DVD venv.
    dvd_host : str
        Host for the DVD MCP HTTP server (default: 127.0.0.1).
    dvd_port : int
        Port for the DVD MCP HTTP server (default: 9002).
    """
    global _initialized, _server_env, _dvd_venv_python, _mcp_server_script
    global _mcp_url, _mcp_client, _server_proc

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        _dvd_venv_python  = dvd_venv_python
        _mcp_server_script = os.path.join(dvd_path, "mcp_server.py")
        _mcp_url           = f"http://{dvd_host}:{dvd_port}/mcp"

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
            "DVD_HOST":               dvd_host,
            "DVD_PORT":               str(dvd_port),
        }

        # Launch the DVD MCP server as a background subprocess.
        _server_proc = subprocess.Popen(
            [_dvd_venv_python, _mcp_server_script],
            env=_server_env,
            stdout=subprocess.DEVNULL,
            stderr=None,         # inherit stderr so logs are visible
        )

        # Wait for the HTTP server to accept connections.
        import httpx
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if _server_proc.poll() is not None:
                raise RuntimeError(
                    f"[dvd_adapter] MCP server subprocess exited early "
                    f"(returncode={_server_proc.returncode})."
                )
            try:
                httpx.get(_mcp_url.replace("/mcp", "/"), timeout=1.0)
                break
            except Exception:
                time.sleep(0.5)
        else:
            _server_proc.kill()
            raise RuntimeError(
                f"[dvd_adapter] DVD MCP server at {_mcp_url} did not become "
                f"ready within 60 s."
            )

        _mcp_client = Client(StreamableHttpTransport(_mcp_url))
        _initialized = True


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
    Run the DVD agent for a (video, question) pair via the MCP HTTP server.

    Each call opens its own independent HTTP session, so concurrent calls for
    different videos proceed in parallel with no shared lock.

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

    try:
        async with _mcp_client:
            result = await _mcp_client.call_tool(
                "run_dvd_query",
                {
                    "video_path": video_path,
                    "question":   question,
                    "dvd_db_dir": dvd_db_dir,
                    "srt_path":   srt_path,
                },
            )
        if result and result.content:
            return (
                result.content[0].text
                if hasattr(result.content[0], "text")
                else str(result.content[0])
            )
        return ""
    except Exception:
        import traceback
        traceback.print_exc()
        return ""
