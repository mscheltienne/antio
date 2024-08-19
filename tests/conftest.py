from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from mne.io import read_raw_brainvision

if TYPE_CHECKING:
    from collections.abc import Callable

    from mne.io import BaseRaw


@pytest.fixture(scope="session")
def ca_208() -> dict[str, dict[str, Path]]:
    """Return the paths to the CA_208 dataset containing 64 channel gel recordings."""
    directory = Path(__file__).parent / "data" / "CA_208"
    cnt = {
        "short": directory / "test_CA_208.cnt",
        "amp-dc": directory / "test_CA_208_amp_disconnection.cnt",
        "start-stop": directory / "test_CA_208_start_stop.cnt",
    }
    bv = {key: value.with_suffix(".vhdr") for key, value in cnt.items()}
    return {"cnt": cnt, "bv": bv}


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
