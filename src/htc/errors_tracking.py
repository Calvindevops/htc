"""Sentry error tracking — env-gated, off unless explicitly configured.

`init_error_tracking()` is a no-op unless `HTC_SENTRY_DSN` is set AND
`sentry-sdk` is installed (it's an optional `[telemetry]` extra, not a core
dependency). `send_default_pii=False` — HTC never forwards user data to
Sentry. `capture_exception()` forwards to Sentry only when active, and never
raises itself.
"""

from __future__ import annotations

import os

_active = False


def init_error_tracking() -> bool:
    """Initialize Sentry if configured; return whether it is now active."""
    global _active
    dsn = os.environ.get("HTC_SENTRY_DSN")
    if not dsn:
        _active = False
        return False
    try:
        import sentry_sdk
    except ImportError:
        _active = False
        return False
    try:
        sentry_sdk.init(dsn=dsn, send_default_pii=False)
        _active = True
    except Exception:
        _active = False
    return _active


def capture_exception(exc: BaseException) -> None:
    """Forward `exc` to Sentry if active; no-op otherwise. Never raises."""
    if not _active:
        return
    try:
        import sentry_sdk

        sentry_sdk.capture_exception(exc)
    except Exception:
        pass
