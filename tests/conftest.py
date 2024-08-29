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
        "ch_ref": "CPz",  # common ref on all channels
        "ch_unit": "uv",  # common unit on all channels
        "n_channels": 64,
        "n_bips": 24,
        "sfreq": 1000,
        "meas_date": "2024-08-14-10-44-47+0000",
        "patient_info": {
            "name": "antio test",
            "id": "",
            "birthday": "2024-08-14",
            "sex": "",
        },
        # TODO: Investigate why the serial number is missing.
        "machine_info": ("eego", "EE_225", ""),
        "hospital": "",
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
        "ch_ref": "Z3",  # common ref on all channels
        "ch_unit": "uv",  # common unit on all channels
        "n_channels": 128,
        "n_bips": 0,
        "sfreq": 2000,
        "meas_date": "2024-08-19-16-17-07+0000",
        "patient_info": {
            "name": "Andy test_middle_name EEG_Exam",
            "id": "test_subject_code",
            "birthday": "2024-08-19",
            "sex": "F",
        },
        # TODO: Investigate why the serial number is missing.
        "machine_info": ("eego", "EE_226", ""),
        "hospital": "",
    }


@pytest.fixture(scope="session")
def user_annotations() -> dict[str, Union[dict[str, Path], str, int]]:
    """Return the path to a dataset containing user annotations with floating pins."""
    directory = Path(__file__).parent / "data" / "user_annotations"
    cnt = {
        "short": directory / "test-user-annotation.cnt",
        "legacy": directory / "test-user-annotation-legacy.cnt",
    }
    bv = {
        "short": cnt["short"].with_suffix(".vhdr"),
    }
    return {
        "cnt": cnt,
        "bv": bv,
        "ch_ref": "5Z",
        "ch_unit": "uv",
        "n_channels": 64,
        "n_bips": 0,
        "sfreq": 500,
        "meas_date": "2024-08-29-16-15-44+0000",
        "patient_info": {
            "name": "test test",
            "id": "",
            "birthday": "2024-02-06",
            "sex": "",
        },
        # TODO: Investigate why the serial number is missing.
        "machine_info": ("eego", "EE_226", ""),
        "hospital": "",
    }


@pytest.fixture(scope="session")
def meas_date_format() -> str:
    """Return the format of the measurement date."""
    return "%Y-%m-%d-%H-%M-%S%z"


@pytest.fixture(scope="session")
def birthday_format() -> str:
    """Return the format of the birthday."""
    return "%Y-%m-%d%z"


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
