from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from antio import read_cnt
from antio.parser import read_data, read_info, read_triggers


def test_read_info(ca_208, read_raw_bv):
    """Test parsing basic channel information."""
    cnt = read_cnt(ca_208["cnt"]["short"])
    ch_names, ch_units, ch_refs, _, _ = read_info(cnt)
    raw = read_raw_bv(ca_208["bv"]["short"])
    assert ch_names == raw.ch_names
    assert ch_units == ["uv"] * len(ch_names)
    assert ch_refs == ["CPz"] * 64 + [""] * 24
    assert len(ch_names) == len(ch_units)
    assert len(ch_names) == len(ch_refs)


def test_read_info_status_types():
    """Test parsing channel status and types."""
    # TODO: Placeholder for when we have a test file with channel status and types


def test_read_data(ca_208, read_raw_bv):
    """Test reading the data array."""
    cnt = read_cnt(ca_208["cnt"]["short"])
    data = read_data(cnt)
    data *= 1e-6  # convert from uV to V
    raw = read_raw_bv(ca_208["bv"]["short"])
    assert data.shape == raw.get_data().shape
    assert_allclose(data, raw.get_data(), atol=1e-8)


def test_read_triggers(ca_208, read_raw_bv):
    """Test reading the triggers from a cnt file."""
    onsets, durations, descriptions, impedances, disconnect = read_triggers(
        read_cnt(ca_208["cnt"]["short"])
    )
    assert len(disconnect["start"]) == len(disconnect["stop"])
    assert len(disconnect["start"]) == 0  # no disconnect in this test file
    assert len(impedances) == len([elt for elt in descriptions if elt == "impedance"])
    assert len(onsets) == len(durations)
    assert len(onsets) == len(descriptions)
    assert all([0 <= elt for elt in onsets])
    assert all([0 <= elt for elt in durations])
    # compare with brainvision file
    raw = read_raw_bv(ca_208["bv"]["short"])
    idx = np.where(raw.annotations.description != "New Segment/")[0]
    assert len(raw.annotations[idx]) == len(onsets)
    # give a bit of jitter as the trigger might not land on the exact same sample
    assert_allclose(
        raw.annotations[idx][0]["onset"], onsets[0] / raw.info["sfreq"], atol=2e-3
    )
    assert_allclose(
        raw.annotations[idx][-1]["onset"], onsets[-1] / raw.info["sfreq"], atol=2e-3
    )


def test_read_triggers_disconnet(ca_208, read_raw_bv):
    """Test reading the triggers from a cnt file with amplifier disconnect."""
    onsets, durations, descriptions, impedances, disconnect = read_triggers(
        read_cnt(ca_208["cnt"]["amp-dc"])
    )
    assert len(disconnect["start"]) == len(disconnect["stop"])
    assert len(disconnect["start"]) == 1
    assert disconnect["start"][0] < disconnect["stop"][0]
    # compare with brainvision file
    raw = read_raw_bv(ca_208["bv"]["amp-dc"])
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
