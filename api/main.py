"""Uvicorn entry point: `uvicorn api.main:app` (or `python -m api.main`)."""

from __future__ import annotations

from api.app import app  # noqa: F401  (re-exported for uvicorn)


def main() -> None:
    import uvicorn

    from core.config import get_settings

    s = get_settings()
    uvicorn.run(
        "api.main:app",
        host=s.api_host,
        port=s.api_port,
        reload=not s.is_production,
        log_level=s.log_level.lower(),
    )


if __name__ == "__main__":
    main()
