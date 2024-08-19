from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from mne.io import read_raw_brainvision
from mne.utils import check_version
from numpy.testing import assert_allclose

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


def test_read_raw_ant(ca_208):
    """Test antio.io.read_raw_ant deprecation."""
    # TODO: replace with pytest.importorskip("mne", "1.9") when MNE 1.9 is released.
    if not check_version("mne", "1.9"):
        pytest.skip("Requires MNE 1.9+")
    raw_cnt = read_raw_ant(ca_208["cnt"]["short"])
    raw_bv = read_raw_bv(ca_208["bv"]["short"])
    assert raw_cnt.ch_names == raw_bv.ch_names
    assert raw_cnt.info["sfreq"] == raw_bv.info["sfreq"]
    assert_allclose(raw_cnt.get_data(), raw_bv.get_data(), atol=1e-8)
