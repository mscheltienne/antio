import pytest

from antio.io import read_raw_ant


def test_read_raw_ant():
    """Test antio.io.read_raw_ant deprecation."""
    pytest.importorskip("mne", "1.9")
    with pytest.warns(DeprecationWarning, match="is deprecated"):
        read_raw_ant()
