from click.testing import CliRunner

from antio._commands.main import run


def test_main():
    """Test the main package entry-point."""
    runner = CliRunner()
    result = runner.invoke(run, ["--help"])
    assert result.exit_code == 0
    assert "Main package entry-point" in result.output
    assert "Options:" in result.output
    assert "Commands:" in result.output
