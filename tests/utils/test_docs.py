import re
from pathlib import Path

import pytest

from antio.utils._docs import docdict, fill_doc
from antio.utils._logs import verbose


def test_fill_doc_function():
    """Test decorator to fill docstring on functions."""

    # test filling docstring
    @fill_doc
    def foo(verbose):
        """My doc.

        Parameters
        ----------
        %(verbose)s
        """

    assert "verbose : int | str | bool | None" in foo.__doc__

    # test filling empty-docstring
    @fill_doc
    def foo():
        pass

    assert foo.__doc__ is None

    # test filling docstring with invalid key
    with pytest.raises(RuntimeError, match="Error documenting"):

        @fill_doc
        def foo(verbose):
            """My doc.

            Parameters
            ----------
            %(invalid_key)s
            """

    # test filling docstring of decorated function
    @fill_doc
    @verbose
    def foo(verbose=None):
        """My doc.

        Parameters
        ----------
        %(verbose)s
        """

    assert "verbose : int | str | bool | None" in foo.__doc__


def test_fill_doc_class():
    """Test decorator to fill docstring on classes."""

    @fill_doc
    class Foo:
        """My doc.

        Parameters
        ----------
        %(verbose)s
        """

        def __init__(self, verbose=None):
            pass

        @fill_doc
        def method(self, verbose=None):
            """My method doc.

            Parameters
            ----------
            %(verbose)s
            """

        @fill_doc
        @verbose
        def method_decorated(self, verbose=None):
            """My method doc.

            Parameters
            ----------
            %(verbose)s
            """

        @staticmethod
        @fill_doc
        @verbose
        def method_decorated_static(verbose=None):
            """My method doc.

            Parameters
            ----------
            %(verbose)s
            """

    assert "verbose : int | str | bool | None" in Foo.__doc__
    assert "verbose : int | str | bool | None" in Foo.method.__doc__
    assert "verbose : int | str | bool | None" in Foo.method_decorated.__doc__
    assert "verbose : int" in Foo.method_decorated_static.__doc__
    foo = Foo()
    assert "verbose : int | str | bool | None" in foo.__doc__
    assert "verbose : int | str | bool | None" in foo.method.__doc__
    assert "verbose : int | str | bool | None" in foo.method_decorated.__doc__


def test_docdict_order():
    """Test that docdict is alphabetical."""
    # read the file as text, and get entries via regex
    docs_path = Path(__file__).parents[1] / "_docs.py"
    assert docs_path.is_file()
    with open(docs_path, encoding="UTF-8") as fid:
        docs = fid.read()
    entries = re.findall(r'docdict\[(?:\n    )?["\'](.+)["\']\n?\] = ', docs)
    # test length, uniqueness and order
    assert len(docdict) == len(entries)
    assert sorted(entries) == entries
