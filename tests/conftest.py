from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from mne.io import read_raw_brainvision

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Union

    from mne.io import BaseRaw


@pytest.fixture(scope="session")
def ca_208() -> dict[str, Union[dict[str, Path], str, int]]:
    """Return the paths and info to the CA_208 dataset."""
    directory = Path(__file__).parent / "data" / "CA_208"
    cnt = {
        "short": directory / "test_CA_208.cnt",
        "amp-dc": directory / "test_CA_208_amp_disconnection.cnt",
        "start-stop": directory / "test_CA_208_start_stop.cnt",
    }
    bv = {key: value.with_suffix(".vhdr") for key, value in cnt.items()}
    return {
        "cnt": cnt,
        "bv": bv,
        "ch_ref": "CPz",
        "ch_unit": "uv",
        "n_channels": 64,
        "n_bips": 24,
    }


@pytest.fixture(scope="session")
def andy_101() -> dict[str, Union[dict[str, Path], str, int]]:
    """Return the path and info to the andy_101 dataset."""
    directory = Path(__file__).parent / "data" / "andy_101"
    cnt = {
        "short": directory / "Andy_101-raw.cnt",
    }
    bv = {key: value.with_suffix(".vhdr") for key, value in cnt.items()}
    return {
        "cnt": cnt,
        "bv": bv,
        "ch_ref": "Z3",
        "ch_unit": "uv",
        "n_channels": 128,
        "n_bips": 0,
    }


@pytest.fixture(scope="session")
def read_raw_bv() -> Callable[[Path], BaseRaw]:
    """Fixture to read a brainvision file exported from eego."""

    def _read_raw_bv(fname: Path) -> BaseRaw:
        """Read a brainvision file exported from eego."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Limited .* annotation.*outside the data range.",
                category=RuntimeWarning,
            )
            raw_bv = read_raw_brainvision(fname)
        return raw_bv

    return _read_raw_bv
