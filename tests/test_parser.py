from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from mne.io import read_raw_brainvision
from numpy.testing import assert_allclose

from antio import read_cnt
from antio.datasets import ca_208
from antio.parser import read_data, read_info

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


def test_read_info(fname):
    """Test parsing channel information."""
    cnt = read_cnt(fname["cnt"])
    ch_names, ch_units, ch_refs = read_info(cnt)
    raw = read_raw_bv(fname["bv"])
    assert ch_names == raw.ch_names
    assert ch_units == ["uv"] * len(ch_names)
    assert ch_refs == ["CPz"] * 64 + [""] * 24
    assert len(ch_names) == len(ch_units)
    assert len(ch_names) == len(ch_refs)


def test_read_data(fname):
    """Test reading the data array."""
    cnt = read_cnt(fname["cnt"])
    data = read_data(cnt)
    data *= 1e-6  # convert from uV to V
    raw = read_raw_bv(fname["bv"])
    assert data.shape == raw.get_data().shape
    assert_allclose(data, raw.get_data(), atol=1e-8)