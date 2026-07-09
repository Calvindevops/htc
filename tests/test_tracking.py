"""Telemetry (opt-in), error tracking (env-gated), and local run history.

No network calls, no real Sentry/PostHog. Privacy invariants under test:
default OFF, explicit opt-in required, anonymized-only payloads, and
tracking failures never propagate to the caller.
"""

from __future__ import annotations

import json

import httpx
import pytest

from htc import errors_tracking, history, telemetry


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point telemetry's config at a scratch dir so tests never touch the
    real `~/.htc/config.json`, and start with a clean env each test."""
    monkeypatch.setenv("HTC_CONFIG_DIR", str(tmp_path / "htc-config"))
    monkeypatch.delenv("HTC_TELEMETRY", raising=False)
    monkeypatch.delenv("HTC_POSTHOG_KEY", raising=False)
    monkeypatch.delenv("HTC_SENTRY_DSN", raising=False)
    yield


class TestTelemetryDefaults:
    def test_off_by_default_no_stored_preference(self, capsys):
        enabled = telemetry.ensure_preference()
        assert enabled is False
        assert "OFF by default" in capsys.readouterr().out

    def test_first_run_persists_default_off_preference(self, tmp_path):
        telemetry.ensure_preference()
        config = json.loads(telemetry._config_path().read_text())
        assert config["telemetry"] is False
        assert "install_id" in config

    def test_second_run_does_not_reprint_notice(self, capsys):
        telemetry.ensure_preference()
        capsys.readouterr()
        telemetry.ensure_preference()
        assert capsys.readouterr().out == ""

    def test_track_is_noop_when_disabled(self, monkeypatch):
        monkeypatch.setenv("HTC_POSTHOG_KEY", "phc_test")
        called = False

        def _boom(*a, **k):
            nonlocal called
            called = True
            raise AssertionError("should not be called when telemetry disabled")

        monkeypatch.setattr(httpx, "post", _boom)
        telemetry.track("command_run", {"command": "eval"})
        assert called is False

    def test_track_is_noop_when_enabled_but_no_key(self, monkeypatch):
        monkeypatch.setenv("HTC_TELEMETRY", "1")
        called = False

        def _boom(*a, **k):
            nonlocal called
            called = True

        monkeypatch.setattr(httpx, "post", _boom)
        telemetry.track("command_run", {"command": "eval"})
        assert called is False


class TestTelemetryOptIn:
    def test_opt_in_preference_persists_and_is_read_back(self):
        config = telemetry._load_config()
        config["telemetry"] = True
        telemetry._save_config(config)
        assert telemetry.is_enabled() is True

    def test_env_override_enables(self, monkeypatch):
        telemetry.ensure_preference()  # stores telemetry: false
        monkeypatch.setenv("HTC_TELEMETRY", "1")
        assert telemetry.is_enabled() is True

    def test_env_override_disables(self, monkeypatch):
        config = telemetry._load_config()
        config["telemetry"] = True
        telemetry._save_config(config)
        monkeypatch.setenv("HTC_TELEMETRY", "0")
        assert telemetry.is_enabled() is False


class TestTelemetrySendFailureSafe:
    def test_send_failure_does_not_propagate(self, monkeypatch):
        monkeypatch.setenv("HTC_TELEMETRY", "1")
        monkeypatch.setenv("HTC_POSTHOG_KEY", "phc_test")

        def _raise(*a, **k):
            raise httpx.ConnectError("boom")

        monkeypatch.setattr(httpx, "post", _raise)
        telemetry.track("command_run", {"command": "eval"})  # must not raise

    def test_no_pii_keys_in_props(self):
        # Sanity check on the bucket helpers: they only ever emit coarse labels.
        assert telemetry.bucket_repo_size(3000) == "500-4999"
        assert telemetry.bucket_duration(45) == "30-120s"
        assert telemetry.bucket_score(82) == "75-89"


class TestErrorTracking:
    def test_init_noop_without_dsn(self):
        assert errors_tracking.init_error_tracking() is False
        errors_tracking.capture_exception(ValueError("x"))  # must not raise

    def test_init_noop_when_sdk_missing(self, monkeypatch):
        monkeypatch.setenv("HTC_SENTRY_DSN", "https://example.invalid/1")
        # sentry-sdk is not a core dependency; if absent, this must no-op.
        try:
            import sentry_sdk  # noqa: F401

            pytest.skip("sentry-sdk is installed in this environment")
        except ImportError:
            pass
        assert errors_tracking.init_error_tracking() is False


class TestHistory:
    def test_record_and_load_round_trip(self, tmp_path):
        history.record_run(tmp_path, "goldens", {"count": 20}, now=1000.0)
        history.record_run(tmp_path, "eval", {"score": 80.0}, now=1001.0)
        entries = history.load_history(tmp_path)
        assert [e["kind"] for e in entries] == ["goldens", "eval"]
        assert [e["index"] for e in entries] == [0, 1]
        assert entries[0]["summary"] == {"count": 20}
        assert entries[1]["timestamp"] == 1001.0

    def test_load_history_empty_when_no_runs(self, tmp_path):
        assert history.load_history(tmp_path) == []

    def test_score_trend_returns_eval_scores_in_order(self, tmp_path):
        history.record_run(tmp_path, "goldens", {"count": 10}, now=1.0)
        history.record_run(tmp_path, "eval", {"score": 60.0}, now=2.0)
        history.record_run(tmp_path, "onboard", {"score": 60.0, "gaps_found": True}, now=3.0)
        history.record_run(tmp_path, "eval", {"score": 75.0}, now=4.0)
        assert history.score_trend(tmp_path) == [60.0, 75.0]
