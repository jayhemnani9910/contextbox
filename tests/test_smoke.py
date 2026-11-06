"""Smoke tests: store a sample context, find it again, and drive the CLI."""

from click.testing import CliRunner

from contextbox import ContextBox
from contextbox.cli import cli, extract_urls_enhanced


def test_capture_and_search(tmp_path):
    app = ContextBox({"database": {"db_path": str(tmp_path / "test.db")}})
    context_id = app.capture_context(
        {"active_window": {"title": "Terminal"}, "clipboard": "fix login timeout"}
    )

    assert app.get_context(context_id) is not None
    assert app.database.search_contexts("login")
    assert not app.database.search_contexts("nothing-like-this")


def test_extract_urls():
    text = "See https://docs.python.org/3/library/sqlite3.html for details"
    assert extract_urls_enhanced(text) == ["https://docs.python.org/3/library/sqlite3.html"]


def test_cli_version_and_empty_list(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "ContextBox" in result.output

    result = runner.invoke(cli, ["--log-level", "error", "list"])
    assert result.exit_code == 0
