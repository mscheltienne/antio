from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .libeep import InputCNT


def read_info(
    cnt: InputCNT,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
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
    ch_status : list of str
        List of channel status. Added in version 0.3.0.
    ch_types : list of str
        List of channel types. Added in version 0.3.0.
    """
    ch_names, ch_units, ch_refs, ch_status, ch_types = [], [], [], [], []
    for k in range(cnt.get_channel_count()):
        channel = cnt.get_channel(k)
        ch_names.append(channel[0])
        ch_units.append(channel[1].lower())  # always lower the unit for mapping
        ch_refs.append(channel[2])
        ch_status.append(channel[3])
        ch_types.append(channel[4])
    return ch_names, ch_units, ch_refs, ch_status, ch_types


def read_data(cnt: InputCNT) -> NDArray[np.float64]:
    """Read the data array.

    Parameters
    ----------
    cnt : InputCNT
        The cnt object from which the data is read.

    Returns
    -------
    data : array of shape (n_channels, n_samples)
        The numpy array containing the data.
    """
    n_samples = cnt.get_sample_count()  # sample = (n_channels,)
    data = cnt.get_samples(0, n_samples)
    return np.array(data).reshape(n_samples, -1).T  # (n_channels, n_samples)


def read_triggers(cnt: InputCNT) -> tuple[list, list, list, list, dict[str, list[int]]]:
    """Read triggers into the attribute of MNE's annotation.

    Parameters
    ----------
    cnt : InputCNT
        The cnt object from which the triggers are read.

    Returns
    -------
    onsets : list of int
        List of onsets of the trigger in samples.
    durations : list of int
        List of durations of the trigger in samples.
    descriptions : list of str
        List of descriptions of the trigger.
    impedances : list of list of float
        List of impedance measurements, one value per channel.
    disconnect : dict
        Dictionary with keys 'start' and 'stop' containing the onsets of the amplifier
        disconnection and reconnection.
    """
    onsets, durations, descriptions, impedances = [], [], [], []
    disconnect = dict(start=[], stop=[])
    for k in range(cnt.get_trigger_count()):
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
    return onsets, durations, descriptions, impedances, disconnect
