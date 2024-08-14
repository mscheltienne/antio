from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from .utils.logs import logger, verbose, warn

if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import NDArray

    from .libeep import InputCNT


units = {"uv": 1e-6}


@verbose
def read_info(cnt: InputCNT, eog: Optional[str], misc: Optional[str], *, verbose):
    """Parse the channel information from the cnt file.

    Parameters
    ----------
    cnt : InputCNT
        The cnt object from which the information is read.
    eog : str | None
        Regex pattern to find EOG channel labels. If None, no EOG channels are
        automatically detected.
    misc : str | None
        Regex pattern to find MISC channel labels. If None, no MISC channels are
        automatically detected.
    %(verbose)s
    """
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


@verbose
def read_data(cnt: InputCNT, ch_units: list[str], *, verbose) -> NDArray[np.float64]:
    """Read the data array.

    Parameters
    ----------
    cnt : InputCNT
        The cnt object from which the data is read.
    ch_units : list[str]
        List of units for each channel.
    %(verbose)s

    Returns
    -------
    data : array of shape (n_channels, n_samples)
        The numpy array containing the data, scaled to SI units if able.
    """
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


@verbose
def read_triggers(
    cnt: InputCNT, impedance_annotation: str, *, verbose
) -> tuple[list[int], list[int], list[str], list[list[float]]]:
    """Read triggers into the attribute of MNE's annotation.

    Parameters
    ----------
    cnt : InputCNT
        The cnt object from which the triggers are read.
    impedance_annotation : str
        The description of the annotation for impedance measurements.
    %(verbose)s

    Returns
    -------
    onsets : list of int
        List of onsets of the triggers, in sample.
    durations : list of int
        List of duration of the triggers, in sample.
    descriptions : list of str
        List of descriptions of the triggers.
    impedances : list of list of float
        List of impedances for each channel, for each impedance trigger.
    """
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
            descriptions.append(impedance_annotation)
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
