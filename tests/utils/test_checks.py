from pathlib import Path

import pytest

from antio.utils._checks import ensure_path


def test_ensure_path():
    """Test ensure_path checker."""
    # valids
    cwd = Path.cwd()
    path = ensure_path(cwd, must_exist=False)
    assert isinstance(path, Path)
    path = ensure_path(cwd, must_exist=True)
    assert isinstance(path, Path)
    path = ensure_path(str(cwd), must_exist=False)
    assert isinstance(path, Path)
    path = ensure_path(str(cwd), must_exist=True)
    assert isinstance(path, Path)
    path = ensure_path("101", must_exist=False)
    assert isinstance(path, Path)

    with pytest.raises(FileNotFoundError, match="does not exist."):
        ensure_path("101", must_exist=True)

    # invalids
    with pytest.raises(TypeError, match="'101' is invalid"):
        ensure_path(101, must_exist=False)

    class Foo:
        def __str__(self):
            pass

    with pytest.raises(TypeError, match="path is invalid"):
        ensure_path(Foo(), must_exist=False)
