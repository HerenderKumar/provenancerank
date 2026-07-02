"""core.llm - provider selection + robust JSON parsing. No real LLM needed: the
Ollama HTTP call is monkeypatched, so this proves the wiring (request shape,
response parsing, fallback) without a running model."""

from __future__ import annotations

from core import llm
from core.config import get_settings


def test_extract_json_plain_fenced_and_chatty():
    assert llm._extract_json('{"a": 1}') == {"a": 1}
    assert llm._extract_json('```json\n{"a": 1}\n```') == {"a": 1}   # code fence
    assert llm._extract_json('Sure! {"a": 1} hope that helps') == {"a": 1}  # chatty wrapper
    assert llm._extract_json("not json at all") is None


def test_provider_none_short_circuits():
    s = get_settings()
    original = s.llm_provider
    try:
        s.llm_provider = "none"
        assert llm.complete("anything") is None
        assert llm.complete_json("anything") is None
    finally:
        s.llm_provider = original


def test_ollama_path_with_mocked_http(monkeypatch):
    import httpx

    class _Resp:
        def __init__(self, payload):
            self._p = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._p

    # reachability probe + the generate call both succeed
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp({"models": []}))
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp({"response": '{"grade": 5}'}))

    s = get_settings()
    original = s.llm_provider
    try:
        s.llm_provider = "ollama"
        llm.reset_reachability()
        assert llm.complete_json("grade this candidate") == {"grade": 5}
        assert llm.active_provider() == "ollama"
    finally:
        s.llm_provider = original
        llm.reset_reachability()


def test_glm_is_preferred_when_key_set(monkeypatch):
    import httpx

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            # OpenAI-compatible chat-completions shape that GLM returns
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp())

    s = get_settings()
    orig_provider, orig_key = s.llm_provider, s.glm_api_key
    try:
        s.llm_provider = "auto"     # GLM should win the auto order when keyed
        s.glm_api_key = "test-key"
        llm.reset_reachability()
        assert llm.active_provider() == "glm"
        assert llm.complete_json("anything") == {"ok": True}
    finally:
        s.llm_provider, s.glm_api_key = orig_provider, orig_key
        llm.reset_reachability()


def test_ollama_unreachable_falls_through(monkeypatch):
    import httpx

    def _boom(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", _boom)

    s = get_settings()
    original = s.llm_provider
    try:
        s.llm_provider = "ollama"   # no gemini key in tests -> nothing reachable
        llm.reset_reachability()
        assert llm.complete_json("hi") is None      # clean None, callers fall back
        assert llm.active_provider() == "none"
    finally:
        s.llm_provider = original
        llm.reset_reachability()
