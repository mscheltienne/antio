from __future__ import annotations

from datetime import datetime
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose

from antio.libeep import read_cnt


def test_InputCNT1(ca_208):
    """Test the methods in InputCNT."""
    cnt = read_cnt(ca_208["cnt"]["short"])
    assert cnt.get_channel_count() == 88  # 64 EEG + 24 BIP
    assert cnt.get_channel(0) == ("Fp1", "uV", "CPz", "", "")
    assert cnt.get_channel(87) == ("BIP24", "uV", "", "", "")
    with pytest.raises(RuntimeError, match="exceeds total channel count"):
        cnt.get_channel(88)
    assert cnt.get_sample_frequency() == 1000
    assert 0 < cnt.get_sample_count()
    assert 0 < cnt.get_trigger_count()
    assert len(cnt.get_trigger(0)) == 6
    assert cnt.get_trigger(0)[4] == "Impedance"
    assert len(cnt.get_trigger(0)[5].split(" ")) == 88  # 64 EEG + 24 BIP
    with pytest.raises(RuntimeError, match="exceeds total trigger count"):
        cnt.get_trigger(7)


def test_InputCNT2(andy_101):
    """Test the methods in InputCNT."""
    cnt = read_cnt(andy_101["cnt"]["short"])
    assert cnt.get_channel_count() == 128  # 128 EEG
    assert cnt.get_channel(0) == ("Lm", "uV", "Z3", "", "")
    assert cnt.get_channel(127) == ("LE4", "uV", "Z3", "", "")
    with pytest.raises(RuntimeError, match="exceeds total channel count"):
        cnt.get_channel(128)
    assert cnt.get_sample_frequency() == 2000
    assert 0 < cnt.get_sample_count()
    assert 0 < cnt.get_trigger_count()
    assert len(cnt.get_trigger(0)) == 6
    assert cnt.get_trigger(0)[4] == "Impedance"
    assert len(cnt.get_trigger(0)[5].split(" ")) == 128  # 128 EEG
    assert len(cnt.get_trigger(1)) == 6
    assert cnt.get_trigger(1)[4] == "Impedance"
    assert len(cnt.get_trigger(1)[5].split(" ")) == 128  # 128 EEG

    with pytest.raises(RuntimeError, match="exceeds total trigger count"):
        cnt.get_trigger(7)


def test_read_invalid_cnt(tmp_path):
    """Test reading of an invalid file."""
    with open(tmp_path / "test.txt", "w") as fid:
        fid.write("invalid file")
    with pytest.raises(RuntimeError, match="Unsupported file extension"):
        read_cnt(tmp_path / "test.txt")


@pytest.mark.parametrize("dataset", ["andy_101", "ca_208"])
def test_read_meas_date(dataset, meas_date_format, request):
    """Test reading the measurement date."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    start_time = cnt.get_start_time()
    assert isinstance(start_time, datetime)
    assert start_time.strftime(meas_date_format) == dataset["meas_date"]


@pytest.mark.parametrize("dataset", ["andy_101", "ca_208"])
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
            pytest.skip("Raw file is too short to run this test.")
        raw_data = raw.get_data(start=start, stop=stop)
        samples = (
            np.array(cnt.get_samples(start, stop)).reshape((stop - start), -1).T * 1e-6
        )
        assert_allclose(raw_data, samples, atol=1e-8)
        samples_np = cnt.get_samples_as_nparray(start, stop) * 1e-6
        assert_allclose(raw_data, samples_np, atol=1e-8)


@pytest.mark.parametrize("dataset", ["andy_101", "ca_208"])
def test_get_invalid_samples(dataset, request):
    """Test retrieving samples outside of the range of the file."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    n_samples = cnt.get_sample_count()
    # end index exceeds total sample count
    with pytest.raises(RuntimeError, match="End index exceeds total sample count."):
        cnt.get_samples(0, n_samples + 1)
    with pytest.raises(RuntimeError, match="End index exceeds total sample count."):
        cnt.get_samples_as_nparray(0, n_samples + 1)
    # negative values
    with pytest.raises(RuntimeError, match="Start/Stop index cannot be negative."):
        cnt.get_samples(-1, n_samples - 1)
    with pytest.raises(RuntimeError, match="Start/Stop index cannot be negative."):
        cnt.get_samples(0, -1)
    with pytest.raises(RuntimeError, match="Start/Stop index cannot be negative."):
        cnt.get_samples_as_nparray(-1, n_samples - 1)
    with pytest.raises(RuntimeError, match="Start/Stop index cannot be negative."):
        cnt.get_samples_as_nparray(0, -1)
    # maximum range
    samples = cnt.get_samples(0, n_samples)
    samples_np = cnt.get_samples_as_nparray(0, n_samples)
    assert samples_np.size == len(samples)


@pytest.mark.parametrize("dataset", ["andy_101", "ca_208"])
def test_get_patient_information(dataset, birthday_format, request):
    """Test reading the patient information."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    name, patient_id, sex, birthday = cnt.get_patient_info()
    assert name == dataset["patient_info"]["name"]
    assert patient_id == dataset["patient_info"]["id"]
    assert isinstance(birthday, datetime)
    assert birthday.strftime(birthday_format) == dataset["patient_info"]["birthday"]


@pytest.mark.parametrize("dataset", ["andy_101", "ca_208"])
def test_get_empty_hospital_field(dataset, request):
    """Test getting the hospital field on datasets without this information."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    assert cnt.get_hospital() == ""


@pytest.mark.parametrize("dataset", ["andy_101", "ca_208"])
def test_get_machine_information(dataset, request):
    """Test getting the machine information."""
    dataset = request.getfixturevalue(dataset)
    cnt = read_cnt(dataset["cnt"]["short"])
    # TODO: Investigate why the serial number is missing in both datasets.
    assert cnt.get_machine_info() == dataset["machine_info"]
