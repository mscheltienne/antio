from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING

import pytest
from mne.io import read_raw_brainvision

from antio.io import read_raw_ant

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def datafiles() -> dict[str, Path]:
    """Return a dict of cnt and brainvision files from the same recording."""
    cnt = files("antio.tests") / "data" / "cnt" / "test_test_2023-06-07_18-54-58.cnt"
    bv = (
        files("antio.tests")
        / "data"
        / "brainvision"
        / "test_test_2023-06-07_18-54-58.vhdr"
    )
    return dict(cnt=cnt, bv=bv)


@pytest.mark.filterwarnings("ignore:Limited.*annotation.*:RuntimeWarning")
def test_io(datafiles):
    """Test loading of .cnt file."""
    raw1 = read_raw_ant(datafiles["cnt"])
    raw2 = read_raw_brainvision(datafiles["bv"])
    # assert_allclose(raw1.get_data(), raw2.get_data())
