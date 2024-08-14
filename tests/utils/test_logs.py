from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from antio.utils._logs import _use_log_level, logger, verbose, warn

if TYPE_CHECKING:
    from typing import Optional, Union


def test_default_log_level(caplog: pytest.LogCaptureFixture):
    """Test the default log level."""
    with _use_log_level("WARNING"):  # set to default
        caplog.clear()
        logger.debug("101")
        assert "101" not in caplog.text

        caplog.clear()
        logger.info("101")
        assert "101" not in caplog.text

        caplog.clear()
        logger.warning("101")
        assert "101" in caplog.text

        caplog.clear()
        logger.error("101")
        assert "101" in caplog.text

        caplog.clear()
        logger.critical("101")
        assert "101" in caplog.text


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_logger(level: str, caplog: pytest.LogCaptureFixture):
    """Test basic logger functionalities."""
    level_functions = {
        "DEBUG": logger.debug,
        "INFO": logger.info,
        "WARNING": logger.warning,
        "ERROR": logger.error,
        "CRITICAL": logger.critical,
    }
    level_int = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    with _use_log_level(level):
        for level_, function in level_functions.items():
            caplog.clear()
            function("101")
            if level_int[level] <= level_int[level_]:
                assert "101" in caplog.text
            else:
                assert "101" not in caplog.text


def test_verbose(caplog: pytest.LogCaptureFixture):
    """Test verbose decorator."""

    # function
    @verbose
    def foo(verbose: Optional[Union[bool, str, int]] = None):
        """Foo function."""
        logger.debug("101")

    assert foo.__doc__ == "Foo function."
    assert foo.__name__ == "foo"
    with _use_log_level("INFO"):
        caplog.clear()
        foo()
        assert "101" not in caplog.text

    for level in (20, 25, 30, True, False, "WARNING", "ERROR"):
        caplog.clear()
        foo(verbose=level)
        assert "101" not in caplog.text

    caplog.clear()
    foo(verbose="DEBUG")
    assert "101" in caplog.text

    # method
    class Foo:
        def __init__(self):
            pass

        @verbose
        def foo(self, verbose: Optional[Union[bool, str, int]] = None):
            logger.debug("101")

        @staticmethod
        @verbose
        def foo2(verbose: Optional[Union[bool, str, int]] = None):
            logger.debug("101")

    foo = Foo()
    with _use_log_level("INFO"):
        caplog.clear()
        foo.foo()
        assert "101" not in caplog.text
        caplog.clear()
        foo.foo(verbose="DEBUG")
        assert "101" in caplog.text

        # static method
        caplog.clear()
        Foo.foo2()
        assert "101" not in caplog.text
        caplog.clear()
        Foo.foo2(verbose="DEBUG")
        assert "101" in caplog.text


def test_warn():
    """Test warning functions."""
    with _use_log_level("ERROR"):
        warn("This is a warning.", RuntimeWarning)
    with (
        _use_log_level("WARNING"),
        pytest.warns(RuntimeWarning, match="This is a warning."),
    ):
        warn("This is a warning.", RuntimeWarning)
