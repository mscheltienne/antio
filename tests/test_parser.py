from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from mne.io import read_raw_brainvision
from numpy.testing import assert_allclose

from antio import read_cnt
from antio.datasets import ca_208
from antio.parser import read_data, read_info, read_triggers

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
def ca_208_short() -> dict[str, Path]:
    """Path to a test file from the CA_208 dataseti CNT and BV format."""
    directory = ca_208.data_path(Path(__file__).parent / "data")
    return {"cnt": directory / "test_CA_208.cnt", "bv": directory / "test_CA_208.vhdr"}


def test_read_info(ca_208_short):
    """Test parsing channel information."""
    cnt = read_cnt(ca_208_short["cnt"])
    ch_names, ch_units, ch_refs = read_info(cnt)
    raw = read_raw_bv(ca_208_short["bv"])
    assert ch_names == raw.ch_names
    assert ch_units == ["uv"] * len(ch_names)
    assert ch_refs == ["CPz"] * 64 + [""] * 24
    assert len(ch_names) == len(ch_units)
    assert len(ch_names) == len(ch_refs)


def test_read_data(ca_208_short):
    """Test reading the data array."""
    cnt = read_cnt(ca_208_short["cnt"])
    data = read_data(cnt)
    data *= 1e-6  # convert from uV to V
    raw = read_raw_bv(ca_208_short["bv"])
    assert data.shape == raw.get_data().shape
    assert_allclose(data, raw.get_data(), atol=1e-8)


def test_read_triggers(ca_208_short):
    """Test reading the triggers from a cnt file."""
    onsets, durations, descriptions, impedances, disconnect = read_triggers(
        read_cnt(ca_208_short["cnt"])
    )
    assert len(disconnect["start"]) == len(disconnect["stop"])
    assert len(disconnect["start"]) == 0  # no disconnect in this test file
    assert len(impedances) == len([elt for elt in descriptions if elt == "impedance"])
    assert len(onsets) == len(durations)
    assert len(onsets) == len(descriptions)
    assert all([0 <= elt for elt in onsets])
    assert all([0 <= elt for elt in durations])
    # compare with brainvision file
    raw = read_raw_bv(ca_208_short["bv"])
    idx = np.where(raw.annotations.description != "New Segment/")[0]
    assert len(raw.annotations[idx]) == len(onsets)
    # give a bit of jitter as the trigger might not land on the exact same sample
    assert_allclose(
        raw.annotations[idx][0]["onset"], onsets[0] / raw.info["sfreq"], atol=2e-3
    )
    assert_allclose(
        raw.annotations[idx][-1]["onset"], onsets[-1] / raw.info["sfreq"], atol=2e-3
    )


@pytest.fixture
def ca_208_disconnect() -> dict[str, Path]:
    """Path to a test file from the CA_208 dataseti CNT and BV format."""
    directory = ca_208.data_path(Path(__file__).parent / "data")
    return {
        "cnt": directory / "test_CA_208_amp_disconnection.cnt",
        "bv": directory / "test_CA_208_amp_disconnection.vhdr",
    }


def test_read_triggers_disconnet(ca_208_disconnect):
    """Test reading the triggers from a cnt file with amplifier disconnect."""
    onsets, durations, descriptions, impedances, disconnect = read_triggers(
        read_cnt(ca_208_disconnect["cnt"])
    )
    assert len(disconnect["start"]) == len(disconnect["stop"])
    assert len(disconnect["start"]) == 1
    assert disconnect["start"][0] < disconnect["stop"][0]
    # compare with brainvision file
    raw = read_raw_bv(ca_208_disconnect["bv"])
    idx = np.where(raw.annotations.description != "New Segment/")[0]
    # look for 9001 and 9002 which corresponds to the amplifier disconnect/reconnect
    annotations = raw.annotations[idx]
    idx_9001 = np.where(annotations.description == "Stimulus/s9001")[0]
    assert_allclose(
        annotations[idx_9001].onset,
        disconnect["start"][0] / raw.info["sfreq"],
        atol=2e-3,
    )
    idx_9002 = np.where(annotations.description == "Stimulus/s9002")[0]
    assert_allclose(
        annotations[idx_9002].onset,
        disconnect["stop"][0] / raw.info["sfreq"],
        atol=2e-3,
    )
    # look for others
    idx = np.where(
        (annotations.description != "Stimulus/s9001")
        & (annotations.description != "Stimulus/s9002")
    )[0]
    annotations = annotations[idx]
    assert len(annotations) == len(onsets)
    assert len(annotations) == len(durations)
    assert len(annotations) == len(descriptions)
    assert all([0 <= elt for elt in onsets])
    assert all([0 <= elt for elt in durations])
    for onset1, onset2 in zip(annotations.onset, onsets):
        assert_allclose(onset1, onset2 / raw.info["sfreq"], atol=2e-3)
