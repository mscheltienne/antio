import pytest

from antio.utils._imports import import_optional_dependency


def test_import_optional_dependency():
    """Test the import of optional dependencies."""
    # Test import of present package
    import_optional_dependency("numpy")

    # Test import of absent package
    with pytest.raises(ImportError, match="Missing optional dependency"):
        import_optional_dependency("non_existing_pkg")

    # Test import of absent package without raise
    import_optional_dependency("non_existing_pkg")

    # Test extra
    with pytest.raises(ImportError, match="blabla"):
        import_optional_dependency("non_existing_pkg", extra="blabla")
