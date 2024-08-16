from pathlib import Path

import pytest

from antio.datasets import ca_208
from antio.io import read_raw_ant


@pytest.fixture
def fname() -> Path:
    """Path to a test file from the CA_208 dataset."""
    directory = ca_208.data_path(Path(__file__).parent / "data")
    return directory / "test_CA_208.cnt"


def test_read_raw_ant(fname):
    """Test antio.io.read_raw_ant deprecation."""
    pytest.importorskip("mne", "1.9")
    with pytest.warns(DeprecationWarning, match="is deprecated"):
        read_raw_ant(fname)
