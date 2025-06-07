import pytest
from codin.cli.repl import ReplSession


def test_load_project_instructions(monkeypatch):
    monkeypatch.setattr(
        "codin.cli.repl.load_agents_instructions",
        lambda: "Sample instructions",
    )
    session = ReplSession(verbose=False, debug=False)
    assert session.load_project_instructions() == "Sample instructions"


def test_load_project_instructions_disabled(monkeypatch):
    session = ReplSession(verbose=False, debug=False)
    session.config.enable_rules = False
    assert session.load_project_instructions() is None
