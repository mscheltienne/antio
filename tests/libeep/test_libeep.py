from __future__ import annotations

import numpy as np
import pytest

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
    assert len(cnt.get_samples(0, 1)) == 88
    assert len(cnt.get_samples(0, 2)) == 88 * 2
    assert cnt.get_samples_as_nparray(0, 1).shape == (88, 1)
    assert cnt.get_samples_as_nparray(0, 2).shape == (88, 2)
    np.testing.assert_allclose(
        cnt.get_samples(0, 1), cnt.get_samples_as_nparray(0, 1)[:, 0]
        )
    np.testing.assert_allclose(
        cnt.get_samples_as_nparray(0, 1), cnt.get_samples_as_nparray(0, 2)[:, :1]
        )
    np.testing.assert_allclose(
        cnt.get_samples_as_nparray(1, 2), cnt.get_samples_as_nparray(0, 2)[:, 1:]
        )
    assert 0 < cnt.get_trigger_count()
    assert len(cnt.get_trigger(0)) == 6
    assert cnt.get_trigger(0)[4] == "Impedance"
    assert len(cnt.get_trigger(0)[5].split(" ")) == 88  # 64 EEG + 24 BIP
    with pytest.raises(RuntimeError, match="exceeds total trigger count"):
        cnt.get_trigger(7)

    assert cnt.get_hospital() == ""
    name, patient_id, sex, birthday = cnt.get_patient_info()
    assert name == "antio test"
    assert patient_id == ""
    assert sex == ""
    assert birthday.day == 14
    assert birthday.month == 8
    assert birthday.year == 2024

    meas_date = cnt.get_start_time()
    assert meas_date.day == 14
    assert meas_date.month == 8
    assert meas_date.year == 2024
    assert meas_date.hour == 10
    assert meas_date.minute == 44
    assert meas_date.second == 47

    assert cnt.get_machine_info() == ("eego", "EE_225", "")


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
    assert len(cnt.get_samples(0, 1)) == 128
    assert len(cnt.get_samples(0, 2)) == 128 * 2
    assert cnt.get_samples_as_nparray(0, 1).shape == (128, 1)
    assert cnt.get_samples_as_nparray(0, 2).shape == (128, 2)
    np.testing.assert_allclose(
        cnt.get_samples(0, 1), cnt.get_samples_as_nparray(0, 1)[:, 0]
        )
    np.testing.assert_allclose(
        cnt.get_samples_as_nparray(0, 1), cnt.get_samples_as_nparray(0, 2)[:, :1]
        )
    np.testing.assert_allclose(
        cnt.get_samples_as_nparray(1, 2), cnt.get_samples_as_nparray(0, 2)[:, 1:]
        )
    assert 0 < cnt.get_trigger_count()
    assert len(cnt.get_trigger(0)) == 6
    assert cnt.get_trigger(0)[4] == "Impedance"
    assert len(cnt.get_trigger(0)[5].split(" ")) == 128  # 128 EEG
    assert len(cnt.get_trigger(1)) == 6
    assert cnt.get_trigger(1)[4] == "Impedance"
    assert len(cnt.get_trigger(1)[5].split(" ")) == 128  # 128 EEG

    with pytest.raises(RuntimeError, match="exceeds total trigger count"):
        cnt.get_trigger(7)

    assert cnt.get_hospital() == ""
    name, patient_id, sex, birthday = cnt.get_patient_info()
    assert name == "Andy test_middle_name EEG_Exam"
    assert patient_id == "test_subject_code"
    assert sex == "F"
    assert birthday.day == 19
    assert birthday.month == 8
    assert birthday.year == 2024

    meas_date = cnt.get_start_time()
    assert meas_date.day == 19
    assert meas_date.month == 8
    assert meas_date.year == 2024
    assert meas_date.hour == 16
    assert meas_date.minute == 17
    assert meas_date.second == 6

    assert cnt.get_machine_info() == ("eego", "EE_226", "")


def test_read_invalid_cnt(tmp_path):
    """Test reading of an invalid file."""
    with open(tmp_path / "test.txt", "w") as fid:
        fid.write("invalid file")
    with pytest.raises(RuntimeError, match="Unsupported file extension"):
        read_cnt(tmp_path / "test.txt")
