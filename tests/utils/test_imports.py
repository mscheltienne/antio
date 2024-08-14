import pytest

from antio.utils._imports import import_optional_dependency


def test_import_optional_dependency():
    """Test the import of optional dependencies."""
    import_optional_dependency("numpy")
    with pytest.raises(ImportError, match="Missing optional dependency"):
        import_optional_dependency("non_existing_pkg")
    with pytest.raises(ImportError, match="blabla"):  # test extra parameter
        import_optional_dependency("non_existing_pkg", extra="blabla")
