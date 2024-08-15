from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .libeep import InputCNT


def read_info(cnt: InputCNT) -> tuple[list[str], list[str], list[str], list[str]]:
    """Parse the channel information from the cnt file.

    Parameters
    ----------
    cnt : InputCNT
        The cnt object from which the information is read.

    Returns
    -------
    ch_names : list of str
        List of channel names.
    ch_units : list of str
        List of human-readable units for each channel.
    ch_refs : list of str
        List of channel reference electrodes.
    """
    ch_names, ch_units, ch_refs = [], [], []
    for k in range(cnt.get_channel_count()):
        ch_curr = cnt.get_channel(k)
        ch_names.append(ch_curr[0])
        ch_units.append(ch_curr[1].lower())  # always lower the unit for mapping
        ch_refs.append(ch_curr[2])
    return ch_names, ch_units, ch_refs


def read_data(cnt: InputCNT) -> NDArray[np.float64]:
    """Read the data array.

    Parameters
    ----------
    cnt : InputCNT
        The cnt object from which the data is read.

    Returns
    -------
    data : array of shape (n_channels, n_samples)
        The numpy array containing the data, scaled to SI units if able.
    """
    n_samples = cnt.get_sample_count()  # sample = (n_channels,)
    data = cnt.get_samples(0, n_samples)
    return np.array(data).reshape(n_samples, -1).T  # (n_channels, n_samples)


def read_triggers(
    cnt: InputCNT, impedance_annotation: str
) -> tuple[list[int], list[int], list[str], list[list[float]]]:
    """Read triggers into the attribute of MNE's annotation.

    Parameters
    ----------
    cnt : InputCNT
        The cnt object from which the triggers are read.
    impedance_annotation : str
        The description of the annotation for impedance measurements.

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
