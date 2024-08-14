from __future__ import annotations

from pathlib import Path

import pytest

import antio


@pytest.fixture()
def ca_208() -> dict[str, dict[str, Path]]:
    """Return the paths to the CA_208 dataset containing 64 channel gel recordings."""
    directory = Path(__file__).parent / "data" / "CA_208"
    cnt = {
        "short": directory / "test_CA_208.cnt",
        "amp-dc": directory / "test_CA_208_amp_disconnection.cnt",
        "start-stop": directory / "test_CA_208_start_stop.cnt",
    }
    bv = {key: value.with_suffix(".eeg") for key, value in cnt.items()}
    return {"cnt": cnt, "bv": bv}


@pytest.mark.filterwarnings("ignore:Omitted .* annotation.*:RuntimeWarning")
def test_io(datafiles):
    """Test loading of .cnt file."""
    pytest.importorskip("mne")
    antio.io.read_raw_ant(ca_208["cnt"]["short"])
