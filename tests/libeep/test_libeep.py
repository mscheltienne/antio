from __future__ import annotations

import pytest

from antio.libeep import read_cnt


def test_InputCNT(ca_208):
    """Test the methods in InputCNT."""
    cnt = read_cnt(ca_208["cnt"]["short"])
    assert cnt.get_channel_count() == 88  # 64 EEG + 24 BIP
    assert cnt.get_channel(0) == ("Fp1", "uV", "CPz")
    assert cnt.get_channel(87) == ("BIP24", "uV", "")
    assert cnt.get_sample_frequency() == 1000
    assert 0 < cnt.get_sample_count()
    assert len(cnt.get_samples(0, 1)) == 88
    assert len(cnt.get_samples(0, 2)) == 88 * 2
    assert 0 < cnt.get_trigger_count()
    assert len(cnt.get_trigger(0)) == 6
    assert cnt.get_trigger(0)[4] == "Impedance"


def test_read_invalid_cnt(tmp_path):
    """Test reading of an invalid file."""
    with open(tmp_path / "test.txt", "w") as fid:
        fid.write("invalid file")
    with pytest.raises(RuntimeError, match="Unsupported file extension"):
        read_cnt(tmp_path / "test.txt")
