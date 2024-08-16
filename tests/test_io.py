from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from mne.io import read_raw_brainvision
from numpy.testing import assert_allclose

from antio.datasets import ca_208
from antio.io import read_raw_ant

if TYPE_CHECKING:
    from mne.io import BaseRaw


def read_raw_bv(fname: Path) -> BaseRaw:
    """Read a brainvision file exported from eego."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Limited .* annotation.*outside the data range.",
            category=RuntimeWarning,
        )
        raw_bv = read_raw_brainvision(fname)
    return raw_bv


@pytest.fixture
def fname() -> dict[str, Path]:
    """Path to a test file from the CA_208 dataseti CNT and BV format."""
    directory = ca_208.data_path(Path(__file__).parent / "data")
    return {"cnt": directory / "test_CA_208.cnt", "bv": directory / "test_CA_208.vhdr"}


def test_read_raw_ant(fname):
    """Test antio.io.read_raw_ant deprecation."""
    pytest.importorskip("mne", "1.9")
    raw_cnt = read_raw_ant(fname["cnt"])
    raw_bv = read_raw_bv(fname["bv"])
    assert raw_cnt.ch_names == raw_bv.ch_names
    assert raw_cnt.info["sfreq"] == raw_bv.info["sfreq"]
    assert_allclose(raw_cnt.get_data(), raw_bv.get_data(), atol=1e-8)
