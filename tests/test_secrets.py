"""`secrets.py` — API keys are never written to plaintext `~/.htc/config.json`.
Keyring is preferred (mocked here — no real OS credential store or network);
a `0600` `~/.htc/credentials.json` file is the fallback. `tests/conftest.py`
isolates `secrets_module._CREDENTIALS_PATH` to a tmp path for every test."""

from __future__ import annotations

import json
import os
import sys
import types

from htc.world_model.memory import local as local_module
from htc.world_model.memory import secrets as secrets_module


def _install_fake_keyring(monkeypatch, store: dict, *, raises: bool = False):
    fake = types.ModuleType("keyring")

    def set_password(service, name, value):
        if raises:
            raise RuntimeError("no keyring backend available")
        store[(service, name)] = value

    def get_password(service, name):
        if raises:
            raise RuntimeError("no keyring backend available")
        return store.get((service, name))

    fake.set_password = set_password
    fake.get_password = get_password
    monkeypatch.setitem(sys.modules, "keyring", fake)


class TestKeyringBackend:
    def test_save_and_load_round_trip_via_keyring(self, monkeypatch):
        store: dict = {}
        _install_fake_keyring(monkeypatch, store)

        method = secrets_module.save_secret("embedding_api_key", "sk-test-123")

        assert method == "keyring"
        assert store == {("htc", "embedding_api_key"): "sk-test-123"}
        assert secrets_module.load_secret("embedding_api_key") == "sk-test-123"


class TestFileFallback:
    def test_falls_back_to_0600_file_when_keyring_raises(self, monkeypatch):
        _install_fake_keyring(monkeypatch, {}, raises=True)

        method = secrets_module.save_secret("embedding_api_key", "sk-test-456")

        assert method == "file"
        path = secrets_module._CREDENTIALS_PATH
        assert path.is_file()
        assert oct(os.stat(path).st_mode)[-3:] == "600"
        assert json.loads(path.read_text()) == {"embedding_api_key": "sk-test-456"}

    def test_load_from_file_when_keyring_raises(self, monkeypatch):
        _install_fake_keyring(monkeypatch, {}, raises=True)
        secrets_module.save_secret("embedding_api_key", "sk-test-789")

        assert secrets_module.load_secret("embedding_api_key") == "sk-test-789"

    def test_falls_back_when_keyring_not_installed(self, monkeypatch):
        # `sys.modules["keyring"] = None` makes `import keyring` raise
        # ImportError, simulating the package not being installed at all.
        monkeypatch.setitem(sys.modules, "keyring", None)

        method = secrets_module.save_secret("embedding_api_key", "sk-test-000")

        assert method == "file"
        assert secrets_module.load_secret("embedding_api_key") == "sk-test-000"

    def test_load_secret_returns_none_when_absent(self, monkeypatch):
        _install_fake_keyring(monkeypatch, {})

        assert secrets_module.load_secret("does_not_exist") is None


class TestWizardNeverWritesKeyToConfig:
    def test_cloud_wizard_choice_saves_key_encrypted_not_in_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HTC_EMBED_BASE_URL", raising=False)
        monkeypatch.delenv("HTC_EMBED_API_KEY", raising=False)
        monkeypatch.delenv("HTC_EMBED_MODEL", raising=False)
        monkeypatch.delenv("HTC_EMBED_NONINTERACTIVE", raising=False)
        monkeypatch.setattr(local_module.sys.stdin, "isatty", lambda: True)

        store: dict = {}
        _install_fake_keyring(monkeypatch, store)

        api_key = "sk-super-secret-value"
        answers = iter(["b", "https://api.example.com/v1", "text-embedding-3-small", api_key])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

        local_module._maybe_run_wizard(False)

        config = local_module._load_global_config()
        assert config["embed_backend"] == "cloud"
        assert config["embed_base_url"] == "https://api.example.com/v1"
        assert config["embed_model"] == "text-embedding-3-small"
        # The API key must never appear anywhere in the plaintext config.
        assert api_key not in json.dumps(config)

        # But it's retrievable via the encrypted secret store.
        assert secrets_module.load_secret("embedding_api_key") == api_key
        assert store == {("htc", "embedding_api_key"): api_key}

    def test_embed_config_resolves_cloud_key_from_secret_store(self, monkeypatch):
        monkeypatch.delenv("HTC_EMBED_BASE_URL", raising=False)
        monkeypatch.delenv("HTC_EMBED_API_KEY", raising=False)
        monkeypatch.delenv("HTC_EMBED_MODEL", raising=False)
        _install_fake_keyring(monkeypatch, {})

        local_module._save_global_config(
            {
                "embed_backend": "cloud",
                "embed_base_url": "https://api.example.com/v1",
                "embed_model": "text-embedding-3-small",
            }
        )
        secrets_module.save_secret("embedding_api_key", "sk-from-store")

        assert local_module._embed_config() == (
            "https://api.example.com/v1",
            "sk-from-store",
            "text-embedding-3-small",
        )

    def test_env_api_key_takes_precedence_over_secret_store(self, monkeypatch):
        _install_fake_keyring(monkeypatch, {})
        secrets_module.save_secret("embedding_api_key", "sk-from-store")

        monkeypatch.setenv("HTC_EMBED_BASE_URL", "https://api.example.com/v1")
        monkeypatch.setenv("HTC_EMBED_MODEL", "text-embedding-3-small")
        monkeypatch.setenv("HTC_EMBED_API_KEY", "sk-from-env")

        assert local_module._embed_config() == (
            "https://api.example.com/v1",
            "sk-from-env",
            "text-embedding-3-small",
        )
