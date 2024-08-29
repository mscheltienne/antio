from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from antio import read_cnt
from antio.parser import (
    read_data,
    read_device_info,
    read_info,
    read_meas_date,
    read_subject_info,
    read_triggers,
)


@pytest.mark.parametrize("dataset", ["andy_101", "ca_208", "user_annotations"])
def test_read_info(dataset, read_raw_bv, request):
    """Test parsing basic channel information."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    ch_names, ch_units, ch_refs, _, _ = read_info(cnt)
    raw = read_raw_bv(dataset["bv"]["short"])
    assert ch_names == raw.ch_names
    assert len(ch_names) == dataset["n_channels"] + dataset["n_bips"]
    assert ch_units == [dataset["ch_unit"]] * len(ch_names)
    assert (
        ch_refs
        == [dataset["ch_ref"]] * dataset["n_channels"] + [""] * dataset["n_bips"]
    )
    assert len(ch_names) == len(ch_units)
    assert len(ch_names) == len(ch_refs)


def test_read_info_status_types():
    """Test parsing channel status and types."""
    # TODO: Placeholder for when we have a test file with channel status and types


@pytest.mark.parametrize("dataset", ["andy_101", "ca_208", "user_annotations"])
def test_read_subject_info(dataset, birthday_format, request):
    """Test reading the data array."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    his_id, name, sex, birthday = read_subject_info(cnt)
    assert his_id == dataset["patient_info"]["id"]
    assert name == dataset["patient_info"]["name"]
    assert ("", "M", "F")[sex] == dataset["patient_info"]["sex"]
    assert birthday.strftime(birthday_format) == dataset["patient_info"]["birthday"]


@pytest.mark.parametrize("dataset", ["andy_101", "ca_208", "user_annotations"])
def test_read_device_info(dataset, request):
    """Test reading the data array."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    *machine_info, site = read_device_info(cnt)
    assert tuple(machine_info) == dataset["machine_info"]
    assert site == dataset["hospital"]


@pytest.mark.parametrize("dataset", ["andy_101", "ca_208", "user_annotations"])
def test_read_meas_date(dataset, meas_date_format, request):
    """Test reading the data array."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    meas_date = read_meas_date(cnt)
    assert meas_date.strftime(meas_date_format) == dataset["meas_date"]


@pytest.mark.parametrize("dataset", ["andy_101", "ca_208"])
def test_read_data(dataset, read_raw_bv, request):
    """Test reading the data array."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    data = read_data(cnt)
    data *= 1e-6  # convert from uV to V
    raw = read_raw_bv(dataset["bv"]["short"])
    assert data.shape == raw.get_data().shape
    assert_allclose(data, raw.get_data(), atol=1e-8)


@pytest.mark.parametrize("dataset", ["andy_101", "ca_208"])
def test_read_triggers(dataset, read_raw_bv, request):
    """Test reading the triggers from a cnt file."""
    dataset = request.getfixturevalue(dataset)
    onsets, durations, descriptions, impedances, disconnect = read_triggers(
        read_cnt(dataset["cnt"]["short"])
    )
    assert len(disconnect["start"]) == len(disconnect["stop"])
    assert len(disconnect["start"]) == 0  # no disconnect in this test file
    assert len(impedances) == len([elt for elt in descriptions if elt == "impedance"])
    assert len(onsets) == len(durations)
    assert len(onsets) == len(descriptions)
    assert all([0 <= elt for elt in onsets])
    assert all([0 <= elt for elt in durations])
    # compare with brainvision file
    raw = read_raw_bv(dataset["bv"]["short"])
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
