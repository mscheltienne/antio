from __future__ import annotations

from datetime import date, datetime
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose

from antio.libeep import read_cnt

DATASETS: list[str] = [
    "andy_101",
    "ca_208",
    "user_annotations",
    "na_271",
    "na_271_bips",
]


@pytest.mark.parametrize("dataset", DATASETS)
def test_get_channel_information(dataset, read_raw_bv, request):
    """Test getting channel information from a CNT file."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    assert cnt.get_channel_count() == dataset["n_channels"] + dataset["n_bips"]
    assert cnt.get_sample_frequency() == dataset["sfreq"]
    raw = read_raw_bv(dataset["bv"]["short"])
    for k in range(dataset["n_channels"]):
        label, unit, ref, status, ch_type = cnt.get_channel(k)
        assert label == raw.ch_names[k]
        assert unit.lower() == dataset["ch_unit"]
        assert ref == dataset["ch_ref"]
        assert status == ""
        assert ch_type == ""


def test_get_channel_information_custom_reference(ca_208_refs, read_raw_bv):
    """Test getting channel information where custom montage was applied."""
    cnt = read_cnt(ca_208_refs["cnt"]["short"])
    assert cnt.get_channel_count() == ca_208_refs["n_channels"] + ca_208_refs["n_bips"]
    assert cnt.get_sample_frequency() == ca_208_refs["sfreq"]
    raw = read_raw_bv(ca_208_refs["bv"]["short"])
    for k in range(ca_208_refs["n_channels"]):
        label, unit, ref, status, ch_type = cnt.get_channel(k)
        assert label == raw.ch_names[k]
        assert unit.lower() == ca_208_refs["ch_unit"]
        assert status == ""
        assert ch_type == ""
        if label in ("Fp1", "Fpz", "Fp2"):
            assert ref == "Fz"
        elif label in ("CP3", "CP4"):
            assert ref == "Cz"
        else:
            assert ref == "CPz"


@pytest.mark.parametrize("dataset", DATASETS + ["ca_208_refs"])
def test_get_invalid_channel(dataset, request):
    """Test getting an invalid channel."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    n_channels = cnt.get_channel_count()
    assert n_channels == dataset["n_channels"] + dataset["n_bips"]
    with pytest.raises(RuntimeError, match="exceeds total channel count"):
        cnt.get_channel(n_channels + 1)
    with pytest.raises(RuntimeError, match="exceeds total channel count"):
        cnt.get_channel(n_channels)
    with pytest.raises(RuntimeError, match="cannot be negative"):
        cnt.get_channel(-1)
    info = cnt.get_channel(n_channels - 1)
    assert isinstance(info, tuple)
    assert len(info) != 0


def test_read_invalid_cnt(tmp_path):
    """Test reading of an invalid file."""
    with open(tmp_path / "test.txt", "w") as fid:
        fid.write("invalid file")
    with pytest.raises(RuntimeError, match="Unsupported file extension"):
        read_cnt(tmp_path / "test.txt")


@pytest.mark.parametrize("dataset", DATASETS + ["ca_208_refs"])
def test_read_meas_date(dataset, meas_date_format, request):
    """Test reading the measurement date."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    start_time = cnt.get_start_time()
    assert isinstance(start_time, datetime)
    assert start_time.strftime(meas_date_format) == dataset["meas_date"]
    start_time_fraction = cnt.get_start_time_and_fraction()
    assert isinstance(start_time_fraction, datetime)
    assert start_time_fraction.strftime(meas_date_format) == dataset["meas_date"]
    assert start_time != start_time_fraction


@pytest.mark.parametrize("dataset", DATASETS + ["ca_208_refs"])
def test_get_sample_count(dataset, request):
    """Test getting the sample count from a CNT file."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    assert 0 < cnt.get_sample_count()


@pytest.mark.parametrize("dataset", DATASETS + ["ca_208_refs"])
def test_get_samples(dataset, read_raw_bv, request):
    """Test retrieving samples from a CNT file."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    n_channels = cnt.get_channel_count()
    assert n_channels == dataset["n_channels"] + dataset["n_bips"]
    for k, size in product(range(5), (1, 5)):
        samples = cnt.get_samples(k, k + size)
        assert len(samples) == n_channels * size
        samples_np = cnt.get_samples_as_nparray(k, k + size)
        assert samples_np.shape == (n_channels, size)
        for i in range(size):
            assert_allclose(
                samples_np[:, i], samples[n_channels * i : n_channels * (i + 1)]
            )
    # compare with brainvision files
    raw = read_raw_bv(dataset["bv"]["short"])
    for start, stop in ((10, 25), (100, 200)):
        if raw.times.size < stop:
            pytest.skip("Raw file is too short to run this test")
        raw_data = raw.get_data(start=start, stop=stop)
        samples = (
            np.array(cnt.get_samples(start, stop)).reshape((stop - start), -1).T * 1e-6
        )
        assert_allclose(raw_data, samples, atol=1e-8)
        samples_np = cnt.get_samples_as_nparray(start, stop) * 1e-6
        assert_allclose(raw_data, samples_np, atol=1e-8)


@pytest.mark.parametrize("dataset", DATASETS + ["ca_208_refs"])
def test_get_invalid_samples(dataset, request):
    """Test retrieving samples outside of the range of the file."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    n_samples = cnt.get_sample_count()
    # end index exceeds total sample count
    with pytest.raises(RuntimeError, match="exceeds total sample count"):
        cnt.get_samples(0, n_samples + 1)
    with pytest.raises(RuntimeError, match="exceeds total sample count."):
        cnt.get_samples_as_nparray(0, n_samples + 1)
    # negative values
    with pytest.raises(RuntimeError, match="cannot be negative."):
        cnt.get_samples(-1, n_samples - 1)
    with pytest.raises(RuntimeError, match="cannot be negative."):
        cnt.get_samples(0, -1)
    with pytest.raises(RuntimeError, match="cannot be negative."):
        cnt.get_samples_as_nparray(-1, n_samples - 1)
    with pytest.raises(RuntimeError, match="cannot be negative."):
        cnt.get_samples_as_nparray(0, -1)
    # maximum range
    samples = cnt.get_samples(0, n_samples)
    samples_np = cnt.get_samples_as_nparray(0, n_samples)
    assert samples_np.size == len(samples)


@pytest.mark.parametrize("dataset", DATASETS + ["ca_208_refs"])
def test_get_patient_information(dataset, birthday_format, request):
    """Test reading the patient information."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    name, patient_id, sex, birthday = cnt.get_patient_info()
    assert name == dataset["patient_info"]["name"]
    assert patient_id == dataset["patient_info"]["id"]
    assert sex == dataset["patient_info"]["sex"]
    assert isinstance(birthday, date)
    assert birthday.strftime(birthday_format) == dataset["patient_info"]["birthday"]


@pytest.mark.parametrize("dataset", DATASETS + ["ca_208_refs"])
def test_get_hospital_field(dataset, request):
    """Test getting the hospital field."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    assert cnt.get_hospital() == dataset["hospital"]


@pytest.mark.parametrize("dataset", DATASETS + ["ca_208_refs"])
def test_get_machine_information(dataset, request):
    """Test getting the machine information."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    # TODO: Investigate why the serial number is missing in both datasets.
    assert cnt.get_machine_info() == dataset["machine_info"]


@pytest.mark.parametrize("dataset", ["andy_101", "ca_208"])
def test_get_base_triggers(dataset, request):
    """Test getting trigger information from a file with basic triggers."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    assert 0 < cnt.get_trigger_count()
    n_channels = cnt.get_channel_count()
    assert n_channels == dataset["n_channels"] + dataset["n_bips"]
    assert len(cnt.get_trigger(0)) == 6
    assert cnt.get_trigger(0)[4] == "Impedance"
    assert len(cnt.get_trigger(0)[5].split(" ")) == n_channels


@pytest.mark.parametrize("dataset", DATASETS)
def test_get_invalid_triggers(dataset, request):
    """Test getting triggers from an invalid index."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    with pytest.raises(RuntimeError, match="exceeds total trigger count"):
        cnt.get_trigger(cnt.get_trigger_count())
    with pytest.raises(RuntimeError, match="cannot be negative"):
        cnt.get_trigger(-1)
    trigger = cnt.get_trigger(cnt.get_trigger_count() - 1)
    assert isinstance(trigger, tuple)
    assert len(trigger) != 0
