from __future__ import annotations

import pytest
from mne.utils import check_version
from numpy.testing import assert_allclose

from antio.io import read_raw_ant


@pytest.mark.parametrize("dataset", ["andy_101", "ca_208"])
def test_read_raw_ant(dataset, read_raw_bv, request):
    """Test antio.io.read_raw_ant."""
    # TODO: replace with pytest.importorskip("mne", "1.9") when MNE 1.9 is released.
    if not check_version("mne", "1.9"):
        pytest.skip("Requires MNE 1.9+")
    dataset = request.getfixturevalue(dataset)
    raw_cnt = read_raw_ant(dataset["cnt"]["short"])
    raw_bv = read_raw_bv(dataset["bv"]["short"])
    assert raw_cnt.ch_names == raw_bv.ch_names
    assert raw_cnt.info["sfreq"] == raw_bv.info["sfreq"]
    assert_allclose(raw_cnt.get_data(), raw_bv.get_data(), atol=1e-8)
