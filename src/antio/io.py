from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from .libeep import read_cnt
from .utils._checks import check_type, ensure_path
from .utils._imports import import_optional_dependency

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Optional, Union

    from numpy.typing import NDArray

    from .libeep import InputCNT

import_optional_dependency("mne")

from mne import Annotations, create_info  # noqa: E402
from mne._fiff.constants import FIFF  # noqa: E402
from mne.io import BaseRaw  # noqa: E402
from mne.utils import copy_doc, fill_doc, logger, verbose, warn  # noqa: E402

units = {"uv": 1e-6}


@fill_doc
class RawANT(BaseRaw):
    r"""Reader for Raw ANT files in .cnt format.

    Parameters
    ----------
    fname : str | Path
        Path to the ANT raw file to load. The file should have the extension ``.cnt``.
    eog : str | None
        Regex pattern to find EOG channel labels. If None, no EOG channels are
        automatically detected.
    misc : str | None
        Regex pattern to find miscellaneous channels. If None, no miscellaneous channels
        are automatically detected. The default pattern ``"BIP\d+"`` will mark all
        bipolar channels as ``misc``.

        .. note::

            A bipolar channel might actually contain ECG, EOG or other signal types
            which might have a dedicated channel type in MNE-Python. In this case, use
            :meth:`mne.io.Raw.set_channel_types` to change the channel type of the
            channel.
    bipolars : list of str | tuple of str | None
        The list of channels to treat as bipolar EEG channels. Each element should be
        a string of the form ``'anode-cathode'`` or in ANT terminology as ``'label-
        reference'``. If None, all channels are interpreted as ``'eeg'`` channels
        referenced to the same reference electrode. Bipolar channels are treated
        as EEG channels with a special coil type in MNE-Python, see also
        :func:`mne.set_bipolar_reference`

        .. warning::

            Do not provide auxiliary channels in this argument, provide them in the
            ``eog`` and ``misc`` arguments.
    %(verbose)s
    """

    _extra_attributes = ("impedances",)

    @verbose
    def __init__(
        self,
        fname: Union[str, Path],
        eog: Optional[str],
        misc: Optional[str],
        bipolars: Optional[Union[list[str], tuple[str, ...]]],
        verbose=None,
    ) -> None:
        logger.info("Reading ANT file %s", fname)
        fname = ensure_path(fname, must_exist=True)
        check_type(eog, (str, None), "eog")
        check_type(misc, (str, None), "misc")
        check_type(bipolars, (list, tuple, None), "bipolar")
        cnt = read_cnt(str(fname))
        # parse channels, sampling frequency, and create info
        ch_names, ch_units, ch_refs, ch_types = _parse_channels(cnt, eog, misc)
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
        data = _parse_data(cnt, ch_units)  # read data array
        super().__init__(info, preload=data, filenames=[fname], verbose=verbose)
        # look for annotations (called trigger by ant)
        onsets, durations, descriptions, impedances = _parse_triggers(cnt)
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


def _parse_channels(
    cnt: InputCNT, eog: Optional[str], misc: Optional[str]
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Parse the channel names annd attempt to find channel type."""
    ch_names, ch_units, ch_refs, ch_types = [], [], [], []
    eog = re.compile(eog) if eog is not None else None
    misc = re.compile(misc) if misc is not None else None
    for k in range(cnt.get_channel_count()):
        ch_curr = cnt.get_channel(k)
        ch_names.append(ch_curr[0])
        ch_units.append(ch_curr[1].lower())  # always lower the unit for mapping
        ch_refs.append(ch_curr[2])
        if eog is not None and re.fullmatch(eog, ch_curr[0]):
            ch_types.append("eog")
        elif misc is not None and re.fullmatch(misc, ch_curr[0]):
            ch_types.append("misc")
        else:
            ch_types.append("eeg")
    eeg_refs = [ch_refs[k] for k, elt in enumerate(ch_types) if elt == "eeg"]
    if len(set(eeg_refs)) == 1:
        logger.info(
            "All %i EEG channels are referenced to %s.", len(eeg_refs), eeg_refs[0]
        )
    else:
        warn("All EEG channels are not referenced to the same electrode.")
    return ch_names, ch_units, ch_refs, ch_types


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


def _parse_data(cnt: InputCNT, ch_units: list[str]) -> NDArray[np.float64]:
    """Parse the data array."""
    n_samples = cnt.get_sample_count()  # sample = (n_channels,)
    data = cnt.get_samples(0, n_samples)
    data = np.array(data).reshape(n_samples, -1).T  # (n_channels, n_samples)
    # apply scalings to SI units if able
    units_index = defaultdict(list)
    for idx, unit in enumerate(ch_units):
        units_index[unit].append(idx)
    for unit, value in units_index.items():
        if unit in units:
            data[np.array(value, dtype=np.int16), :] *= units[unit]
        else:
            warn(f"Unit {unit} not recognized, not scaling.")
    return data


def _parse_triggers(
    cnt: InputCNT,
) -> tuple[list[int], list[int], list[str], list[list[float]]]:
    """Parse triggers into annotations."""
    n_triggers = cnt.get_trigger_count()
    onsets, durations, descriptions, impedances = [], [], [], []
    disconnect = dict(start=[], stop=[])
    for k in range(n_triggers):
        code, idx, duration, condition, description, impedance = cnt.get_trigger(k)
        # detect impedance measurements
        if (
            description is not None
            and description.lower() == "impedance"
            and impedance is not None
        ):
            impedances.append([float(elt) for elt in impedance.split(" ")])
            # create an impedance annotation to mark the measurement
            onsets.append(idx)
            durations.append(duration)
            descriptions.append("impedance")
            continue
        # detect amplifier disconnection
        if condition is not None and condition.lower() == "amplifier disconnected":
            disconnect["start"].append(idx)
            continue
        elif condition is not None and condition.lower() == "amplifier reconnected":
            disconnect["stop"].append(idx)
            continue
        # treat all the other triggers as regular event annotations
        onsets.append(idx)
        durations.append(duration)
        if description is not None:
            descriptions.append(description)
        else:
            descriptions.append(code)
    # create BAD_disconnection annotations, don't bother with all the special cases, if
    # the annotations look weird, just add the bare 9001 and 9002 disconnect/reconnect
    # annotations.
    if (
        len(disconnect["start"]) == len(disconnect["stop"])
        and len(disconnect["start"]) != 0
        and disconnect["start"][0] < disconnect["stop"][0]
        and disconnect["start"][-1] < disconnect["stop"][-1]
    ):
        for start, stop in zip(disconnect["start"], disconnect["stop"]):
            onsets.append(start)
            durations.append(stop - start)
            descriptions.append("BAD_disconnection")
    else:
        for elt in disconnect["start"]:
            onsets.append(elt)
            durations.append(0)
            descriptions.append("Amplifier disconnected")
        for elt in disconnect["stop"]:
            onsets.append(elt)
            durations.append(0)
            descriptions.append("Amplifier reconnected")
    return onsets, durations, descriptions, impedances


@copy_doc(RawANT)
def read_raw_ant(
    fname: Union[str, Path],
    eog: Optional[str] = None,
    misc: Optional[str] = r"BIP\d+",
    bipolars: Optional[Union[list[str], tuple[str, ...]]] = None,
    verbose=None,
) -> RawANT:
    """

    Returns
    -------
    RawANT
        The ANT raw object containing the channel information, data and relevant
        :class:`~mne.Annotations`.
    """
    return RawANT(fname, eog=eog, misc=misc, bipolars=bipolars, verbose=verbose)
