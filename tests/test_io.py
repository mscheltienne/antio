from __future__ import annotations

from pathlib import Path

import pytest

from antio.io import read_raw_ant


@pytest.fixture()
def datafiles() -> Path:
    """Return the path to a .cnt file."""
    return Path(__file__).parent / "data" / "test_test_2023-06-07_18-54-58.cnt"


@pytest.mark.filterwarnings("ignore:Omitted .* annotation.*:RuntimeWarning")
def test_io(datafiles):
    """Test loading of .cnt file."""
    pytest.importorskip("mne")
    read_raw_ant(datafiles)
