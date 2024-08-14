from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from mne import Annotations, create_info
from mne._fiff.constants import FIFF
from mne.io import BaseRaw, read_raw_brainvision
from mne.utils import logger, warn
from numpy.testing import assert_allclose

from antio.io import read_data, read_info, read_triggers
from antio.libeep import read_cnt
from antio.utils._checks import check_type, ensure_path

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Optional, Union


units = {"uv": 1e-6}


class RawANT(BaseRaw):
    r"""Reader for Raw ANT files in .cnt format."""

    _extra_attributes = ("impedances",)

    def __init__(
        self,
        fname: Union[str, Path],
        eog: Optional[str],
        misc: Optional[str],
        bipolars: Optional[Union[list[str], tuple[str, ...]]],
        impedance_annotation: str,
        *,
        verbose=None,
    ) -> None:
        logger.info("Reading ANT file %s", fname)
        fname = ensure_path(fname, must_exist=True)
        check_type(eog, (str, None), "eog")
        check_type(misc, (str, None), "misc")
        check_type(bipolars, (list, tuple, None), "bipolar")
        check_type(impedance_annotation, (str,), "impedance_annotation")
        if len(impedance_annotation) == 0:
            raise ValueError("The impedance annotation cannot be an empty string.")
        cnt = read_cnt(str(fname))
        # parse channels, sampling frequency, and create info
        ch_names, ch_units, ch_refs, ch_types = read_info(cnt, eog, misc)
        if bipolars is not None:  # handle bipolar channels
            bipolars_idx = _handle_bipolar_channels(ch_names, ch_refs, bipolars)
            for idx, ch in zip(bipolars_idx, bipolars):
                if ch_types[idx] != "eeg":
                    warn(
                        f"Channel {ch} was not parsed as an EEG channel, changing to "
                        "EEG channel type since bipolar EEG was requested."
                    )
                ch_names[idx] = ch
                ch_types[idx] = "eeg"
        info = create_info(
            ch_names, sfreq=cnt.get_sample_frequency(), ch_types=ch_types
        )
        if bipolars is not None:
            with info._unlock():
                for idx in bipolars_idx:
                    info["chs"][idx]["coil_type"] = FIFF.FIFFV_COIL_EEG_BIPOLAR
        data = read_data(cnt, ch_units)  # read data array
        super().__init__(info, preload=data, filenames=[fname], verbose=verbose)
        # look for annotations (called trigger by ant)
        onsets, durations, descriptions, impedances = read_triggers(
            cnt, impedance_annotation
        )
        onsets = np.array(onsets) / self.info["sfreq"]
        durations = np.array(durations) / self.info["sfreq"]
        annotations = Annotations(onsets, duration=durations, description=descriptions)
        self.set_annotations(annotations)
        # set impedance similarly as for brainvision files
        self._impedances = [
            {ch: imp[k] for k, ch in enumerate(ch_names)} for imp in impedances
        ]

    @property
    def impedances(self) -> list[dict[str, float]]:
        """Impedances for each impedance measurement event.

        The measurements are ordered as in the attached :class:`~mne.Annotations`.
        """
        return self._impedances


def _handle_bipolar_channels(
    ch_names: list[str], ch_refs: list[str], bipolars: Union[list[str], tuple[str, ...]]
) -> list[int]:
    """Handle bipolar channels."""
    bipolars_idx = []
    for ch in bipolars:
        check_type(ch, (str,), "bipolar_channel")
        if "-" not in ch:
            raise ValueError(
                "Bipolar channels should be provided as 'anode-cathode' or "
                f"'label-reference'. '{ch}' is not valid."
            )
        anode, cathode = ch.split("-")
        if anode not in ch_names:
            raise ValueError(f"Anode channel {anode} not found in the channels.")
        idx = ch_names.index(anode)
        if cathode != ch_refs[idx]:
            raise ValueError(
                f"Reference electrode for {anode} is {ch_refs[idx]}, not {cathode}."
            )
        # store idx for later FIFF coil type change
        bipolars_idx.append(idx)
    return bipolars_idx


def read_raw_ant(
    fname: Union[str, Path],
    eog: Optional[str] = None,
    misc: Optional[str] = r"BIP\d+",
    bipolars: Optional[Union[list[str], tuple[str, ...]]] = None,
    impedance_annotation: str = "impedance",
    *,
    verbose=None,
) -> RawANT:
    """Read raw to MNE."""
    return RawANT(
        fname,
        eog=eog,
        misc=misc,
        bipolars=bipolars,
        impedance_annotation=impedance_annotation,
        verbose=verbose,
    )


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


@pytest.fixture()
def ca_208() -> dict[str, dict[str, Path]]:
    """Return the paths to the CA_208 dataset containing 64 channel gel recordings."""
    directory = Path(__file__).parent / "data" / "CA_208"
    cnt = {
        "short": directory / "test_CA_208.cnt",
        "amp-dc": directory / "test_CA_208_amp_disconnection.cnt",
        "start-stop": directory / "test_CA_208_start_stop.cnt",
    }
    bv = {key: value.with_suffix(".vhdr") for key, value in cnt.items()}
    return {"cnt": cnt, "bv": bv}


def test_io_data(ca_208: dict[str, dict[str, Path]]) -> None:
    """Test loading of .cnt file."""
    raw_cnt = read_raw_ant(ca_208["cnt"]["short"])
    raw_bv = read_raw_bv(ca_208["bv"]["short"])
    cnt = raw_cnt.get_data()
    bv = raw_bv.get_data()
    assert cnt.shape == bv.shape
    assert_allclose(raw_cnt.get_data(), raw_bv.get_data(), atol=1e-8)


def test_io_info(ca_208: dict[str, dict[str, Path]]) -> None:
    """Test the info loaded from a .cnt file."""
    raw_cnt = read_raw_ant(ca_208["cnt"]["short"])
    raw_bv = read_raw_bv(ca_208["bv"]["short"])
    assert raw_cnt.ch_names == raw_bv.ch_names
    assert raw_cnt.info["sfreq"] == raw_bv.info["sfreq"]
    assert raw_cnt.get_channel_types() == ["eeg"] * 64 + ["misc"] * 24
    with pytest.warns(
        RuntimeWarning,
        match="All EEG channels are not referenced to the same electrode.",
    ):
        raw_cnt = read_raw_ant(ca_208["cnt"]["short"], misc=None)
    assert raw_cnt.get_channel_types() == ["eeg"] * len(raw_cnt.ch_names)
    raw_cnt = read_raw_ant(ca_208["cnt"]["short"], eog="EOG")
    idx = raw_cnt.ch_names.index("EOG")
    ch_types = ["eeg"] * 64 + ["misc"] * 24
    ch_types[idx] = "eog"
    assert raw_cnt.get_channel_types() == ch_types


def test_io_amp_disconnection(ca_208: dict[str, dict[str, Path]]) -> None:
    """Test loading of .cnt file with amplifier disconnection."""
    raw_cnt = read_raw_ant(ca_208["cnt"]["amp-dc"])
    raw_bv = read_raw_bv(ca_208["bv"]["amp-dc"])
    assert_allclose(raw_cnt.get_data(), raw_bv.get_data(), atol=1e-8)
    assert (
        raw_cnt.get_data(reject_by_annotation="omit").shape != raw_bv.get_data().shape
    )
    # create annotation on the BV file
    idx = [
        k
        for k, elt in enumerate(raw_bv.annotations.description)
        if any(code in elt for code in ("9001", "9002"))
    ]
    assert len(idx) == 2
    start = raw_bv.annotations.onset[idx[0]]
    stop = raw_bv.annotations.onset[idx[1]]
    annotations = Annotations(
        onset=start,
        duration=stop - start + 1 / raw_bv.info["sfreq"],  # estimate is 1 sample short
        description="BAD_segment",
    )
    raw_bv.set_annotations(annotations)
    assert_allclose(
        raw_cnt.get_data(reject_by_annotation="omit"),
        raw_bv.get_data(reject_by_annotation="omit"),
        atol=1e-8,
    )


@pytest.mark.parametrize("description", ["impedance", "test"])
def test_io_impedance(ca_208: dict[str, dict[str, Path]], description: str) -> None:
    """Test loading of impedances from a .cnt file."""
    raw_cnt = read_raw_ant(ca_208["cnt"]["amp-dc"], impedance_annotation=description)
    assert isinstance(raw_cnt.impedances, list)
    for elt in raw_cnt.impedances:
        assert isinstance(elt, dict)
        assert list(elt) == raw_cnt.ch_names
        assert all(isinstance(val, float) for val in elt.values())
    annotations = [
        annot for annot in raw_cnt.annotations if annot["description"] == description
    ]
    assert len(annotations) == len(raw_cnt.impedances)


def test_io_segments(ca_208: dict[str, dict[str, Path]]) -> None:
    """Test reading a .cnt file with segents (start/stop)."""
    raw_cnt = read_raw_ant(ca_208["cnt"]["start-stop"])
    raw_bv = read_raw_bv(ca_208["bv"]["start-stop"])
    assert_allclose(raw_cnt.get_data(), raw_bv.get_data(), atol=1e-8)
