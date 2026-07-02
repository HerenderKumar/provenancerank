"""One provider-agnostic LLM call. Auto order: GLM (your key) -> local Ollama ->
Gemini -> nothing.

Every LLM touchpoint in the project (JD parsing, evidence summaries, pseudo-label
grading, NL-query parsing) goes through `complete()` / `complete_json()`, so the
provider is chosen in exactly one place.

    export GLM_API_KEY=...              # Z.ai / Zhipu GLM, used first when set
    export OLLAMA_MODEL=llama3.1:8b     # local fallback (run `ollama serve`)
    # force one provider with LLM_PROVIDER=glm|ollama|gemini|none

GLM is OpenAI-compatible, so `_glm` is really an OpenAI chat-completions call —
point `GLM_BASE_URL` at any compatible endpoint to use a different one. If no
provider is reachable, both functions return None and every caller falls back to
its deterministic heuristic, so the pipeline always runs offline.
"""

from __future__ import annotations

import json as _json
import re

from core.config import get_settings
from core.logging import get_logger

log = get_logger("core.llm")

# Tri-state reachability cache so we probe Ollama once per process, not on every
# call (pseudo-label grading alone can make dozens of calls).
_ollama_up: bool | None = None


def _ollama_reachable(base_url: str) -> bool:
    global _ollama_up
    if _ollama_up is None:
        try:
            import httpx

            httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.0)
            _ollama_up = True
            log.info("llm.ollama_reachable", base_url=base_url)
        except Exception:
            _ollama_up = False
            log.info("llm.ollama_unreachable", base_url=base_url)
    return _ollama_up


def reset_reachability() -> None:
    """Test hook / call after starting Ollama mid-process."""
    global _ollama_up
    _ollama_up = None


def _ollama(prompt: str, want_json: bool, timeout: float) -> str | None:
    import httpx

    s = get_settings()
    payload: dict = {"model": s.ollama_model, "prompt": prompt, "stream": False}
    if want_json:
        payload["format"] = "json"  # constrains the model to emit valid JSON
    resp = httpx.post(
        f"{s.ollama_base_url.rstrip('/')}/api/generate", json=payload, timeout=timeout
    )
    resp.raise_for_status()
    return resp.json().get("response")


def _glm(prompt: str, want_json: bool, timeout: float) -> str | None:
    """Z.ai / Zhipu GLM via its OpenAI-compatible chat-completions endpoint."""
    import httpx

    s = get_settings()
    body: dict = {
        "model": s.glm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    if want_json:
        body["response_format"] = {"type": "json_object"}
    resp = httpx.post(
        f"{s.glm_base_url.rstrip('/')}/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {s.glm_api_key}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _gemini(prompt: str, want_json: bool, timeout: float) -> str | None:
    s = get_settings()
    import google.generativeai as genai

    genai.configure(api_key=s.google_api_key)
    model = genai.GenerativeModel(s.summariser_model)
    cfg = {"response_mime_type": "application/json"} if want_json else {}
    resp = model.generate_content(
        prompt, generation_config=cfg, request_options={"timeout": timeout}
    )
    return resp.text


def complete(prompt: str, *, json: bool = False, timeout: float | None = None) -> str | None:
    """Raw text from the active provider, or None if none is available."""
    s = get_settings()
    timeout = timeout or float(getattr(s, "llm_timeout", 60))
    provider = (getattr(s, "llm_provider", "auto") or "auto").lower()
    if provider == "none":
        return None
    # auto = GLM (your key) first, then local Ollama, then Gemini.
    order = [provider] if provider in ("glm", "ollama", "gemini") else ["glm", "ollama", "gemini"]

    for p in order:
        try:
            if p == "glm":
                if not s.glm_api_key:
                    continue
                out = _glm(prompt, json, timeout)
            elif p == "ollama":
                if not _ollama_reachable(s.ollama_base_url):
                    continue
                out = _ollama(prompt, json, timeout)
            else:  # gemini
                if not s.google_api_key:
                    continue
                out = _gemini(prompt, json, timeout)
            if out:
                return out
        except Exception as exc:
            log.warning("llm.provider_failed", provider=p, reason=str(exc)[:140])
    return None


def _extract_json(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):  # strip ```json ... ``` fences some models add
        raw = raw.strip("`")
        nl = raw.find("\n")
        if nl != -1 and raw[:nl].strip().lower() in ("json", ""):
            raw = raw[nl + 1 :]
    try:
        return _json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)  # last resort: first {...} block
        if m:
            try:
                return _json.loads(m.group(0))
            except Exception:
                return None
    return None


def complete_json(prompt: str, *, timeout: float | None = None) -> dict | None:
    """Parsed JSON object from the active provider, or None. Robust to code fences
    and chatty wrappers, so a heuristic caller can rely on `is None` to fall back."""
    raw = complete(prompt, json=True, timeout=timeout)
    return _extract_json(raw) if raw else None


def active_provider() -> str:
    """Which provider would be used right now (for logging / health)."""
    s = get_settings()
    provider = (getattr(s, "llm_provider", "auto") or "auto").lower()
    if provider == "none":
        return "none"
    if provider in ("glm", "auto") and s.glm_api_key:
        return "glm"
    if provider in ("ollama", "auto") and _ollama_reachable(s.ollama_base_url):
        return "ollama"
    if provider in ("gemini", "auto") and s.google_api_key:
        return "gemini"
    return "none"
