import pytest

from rsi import config
from rsi.llm.echo import EchoProvider
from rsi.llm.base import LLMConfigError, load_provider


@pytest.fixture(autouse=True)
def _reset():
    config.set_config_path(None)
    yield
    config.set_config_path(None)


def test_registry_defaults():
    assert config.get("llm.provider") == "echo"
    assert config.get_int("engine.cycles") == 6


def test_unknown_key_rejected():
    with pytest.raises(config.ConfigError):
        config.get("nope.nope")


def test_credential_keys_unreadable():
    with pytest.raises(config.ConfigError):
        config.get("api_key")
    with pytest.raises(config.ConfigError):
        config.get("llm.api_key")


def test_file_overrides_registry(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("llm:\n  provider: null\nengine:\n  cycles: 3\n")
    config.set_config_path(f)
    assert config.get("engine.cycles") == 3


def test_file_credential_rejected(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("llm:\n  api_key: xyz\n")
    config.set_config_path(f)
    with pytest.raises(config.ConfigError):
        config.get("llm.provider")


def test_load_provider_echo_default():
    assert isinstance(load_provider(), EchoProvider)


def test_load_provider_unknown():
    with pytest.raises(LLMConfigError):
        load_provider({"llm.provider": "bogus"})


def test_echo_deterministic_and_keyless(monkeypatch):
    monkeypatch.delenv("RSI_OPENAI_API_KEY", raising=False)
    p = EchoProvider()
    a = p.compose("choose:\n<options>\n1. first\n2. second\n</options>", 64)
    b = p.compose("choose:\n<options>\n1. first\n2. second\n</options>", 64)
    assert a == b and a in ("first", "second")
    assert p.analyze("anything", 8).startswith("[echo:")
    assert p.rerank("q", ["a", "b", "c"]) == p.rerank("q", ["a", "b", "c"])


def test_echo_rerank_stable_order():
    p = EchoProvider()
    out = p.rerank("query", ["x", "y", "z"])
    assert sorted(out) == ["x", "y", "z"]  # permutation, never loss
