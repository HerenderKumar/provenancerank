"""Pick the fastest compute we can actually use for the transformer models, and
optionally the ONNX/int8 backend - both best-effort, both with safe fallbacks.

torch may be absent (the sandbox), MPS/CUDA may be missing, onnxruntime may not
be installed. Nothing here raises: device detection falls back to "cpu", and the
ONNX path is offered as a *list of attempts* (most-optimised first) that the
model constructors walk until one works, ending at plain torch. Same graceful-
degradation contract as the rest of the pipeline - we never silently drop to the
hashing/lexical floor just because an optimisation was unavailable.
"""

from __future__ import annotations

from functools import lru_cache

from core.config import get_settings
from core.logging import get_logger

log = get_logger("core.device")

# Quantised ONNX weights published alongside these models on the Hub. Used only
# as a hint - if the file isn't there, the constructor falls back a rung.
_KNOWN_QUANTIZED = {
    "all-MiniLM-L6-v2": "onnx/model_qint8_avx512_vnni.onnx",
    "sentence-transformers/all-MiniLM-L6-v2": "onnx/model_qint8_avx512_vnni.onnx",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "onnx/model_qint8_avx512_vnni.onnx",
}


@lru_cache(maxsize=1)
def best_device() -> str:
    """'mps', 'cuda', or 'cpu'. Honours an explicit ``compute_device`` override;
    otherwise probes torch (CUDA first, then Apple MPS)."""
    pref = (getattr(get_settings(), "compute_device", "auto") or "auto").lower()
    if pref in {"cpu", "cuda", "mps"}:
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _onnx_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


def transformer_attempts(model_name: str, allow_onnx: bool = True) -> list[tuple[str, dict]]:
    """Constructor kwargs to try, fastest first, for a SentenceTransformer or
    CrossEncoder. Each is ``(label, kwargs)``; the caller tries them in order and
    keeps the first that loads. Always ends with a plain-torch attempt so a
    transformer model is still used even when ONNX isn't.
    """
    device = best_device()
    attempts: list[tuple[str, dict]] = []
    s = get_settings()
    if allow_onnx and getattr(s, "use_onnx", False) and _onnx_available():
        onnx = {"device": device, "backend": "onnx"}
        if getattr(s, "onnx_quantized", True) and model_name in _KNOWN_QUANTIZED:
            attempts.append(
                ("onnx-int8", {**onnx, "model_kwargs": {"file_name": _KNOWN_QUANTIZED[model_name]}})
            )
        attempts.append(("onnx", onnx))
    attempts.append(("torch", {"device": device}))
    return attempts


def construct(cls, model_name: str, *, allow_onnx: bool = True, **extra):
    """Build ``cls(model_name, ...)`` walking the optimised->plain attempts.

    Returns ``(instance, runtime_label)``. Raises only if even the plain-torch
    attempt fails - that's a genuine "the library isn't usable" signal the caller
    turns into its own floor (hashing embedder / lexical reranker)."""
    last_exc: Exception | None = None
    for label, kwargs in transformer_attempts(model_name, allow_onnx=allow_onnx):
        try:
            inst = cls(model_name, **extra, **kwargs)
            if label != "torch":
                log.info("device.transformer_backend", model=model_name, runtime=label)
            return inst, label
        except Exception as exc:
            last_exc = exc
            log.warning("device.attempt_failed", runtime=label, reason=str(exc)[:120])
    raise last_exc if last_exc else RuntimeError("no constructor attempt ran")
