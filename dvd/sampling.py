"""Qwen3.5-4B sampling at the DVD request boundary.

DVD runs in its own environment, so this deliberately has no lmms-eval
dependency. The presets match Qwen's general-task recommendations.
"""

from __future__ import annotations

import copy
import os
import re
from contextvars import ContextVar

generation_context: ContextVar[dict | None] = ContextVar("dvd_generation_kwargs", default=None)


def _as_bool(value) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1", "yes"}:
        return True
    if str(value).lower() in {"false", "0", "no"}:
        return False
    raise ValueError("enable_thinking must be a boolean")


def qwen35_request_kwargs(model_name: str, *, enable_thinking=None, generation=None) -> dict:
    """Return SDK kwargs, overriding stale greedy defaults for this model only."""
    if not re.search(r"qwen3[._-]?5[-_]4b(?:$|[^a-z0-9])", str(model_name).lower()):
        return {}

    generation = generation_context.get() if generation is None else generation
    generation = generation or {}
    extra = copy.deepcopy(generation.get("extra_body") or {})
    template = {**(generation.get("chat_template_kwargs") or {}), **(extra.get("chat_template_kwargs") or {})}
    thinking = os.environ.get("DVD_ENABLE_THINKING", "true")
    thinking = generation.get("enable_thinking", thinking)
    thinking = template.get("enable_thinking", thinking)
    if enable_thinking is not None:
        thinking = enable_thinking
    thinking = _as_bool(thinking)
    template["enable_thinking"] = thinking

    # Extra-body fields are merged last by the SDK: remove conflicting
    # standard keys so they cannot silently override the selected preset.
    for name in ("temperature", "top_p", "presence_penalty"):
        extra.pop(name, None)
    extra.update(top_k=20, min_p=0.0, repetition_penalty=1.0, chat_template_kwargs=template)
    return {
        "temperature": 1.0 if thinking else 0.7,
        "top_p": 0.95 if thinking else 0.8,
        "presence_penalty": 1.5,
        "extra_body": extra,
    }
