from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from datetime import date, datetime
    from typing import Optional

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


def read_subject_info(cnt: InputCNT) -> tuple[str, str, int, date]:
    """Parse the subject information from the cnt file.

    Parameters
    ----------
    cnt : InputCNT
        The cnt object from which the information is read.

    Returns
    -------
    his_id : str
        String subject identifier.
    name : str
        Name.
    sex : int
        Subject sex (0=unknown, 1=male, 2=female).
    birthday : datetime.date
        The subject birthday.
    """
    name, his_id, sex, birthday = cnt.get_patient_info()
    sex = {"": 0, "M": 1, "F": 2}[sex]
    return his_id, name, sex, birthday


def read_device_info(cnt: InputCNT) -> tuple[str, str, str, str]:
    """Parse the machine information from the cnt file.

    Parameters
    ----------
    cnt : InputCNT
        The cnt object from which the information is read.

    Returns
    -------
    make : str
        Device type.
    model : str
        Device model.
    serial : str
        Device serial.
    site : str
        Device site.
    """
    make, mode, serial = cnt.get_machine_info()
    site = cnt.get_hospital()
    return make, mode, serial, site


def read_meas_date(cnt: InputCNT) -> Optional[datetime]:
    """Parse the measurement from the cnt file.

    Parameters
    ----------
    cnt : InputCNT
        The cnt object from which the information is read.

    Returns
    -------
    meas_date : datetime | None
        The measurement time of the recording (in the UTC timezone).
    """
    return cnt.get_start_time_and_fraction()


def read_data(
    cnt: InputCNT, first_samp: int = 0, last_samp: Optional[int] = None
) -> NDArray[np.float64]:
    """Read the data array.

    Parameters
    ----------
    cnt : InputCNT
        The cnt object from which the data is read.
    first_samp : int
        Start index.
    last_samp : int
        End index.

    Returns
    -------
    data : array of shape (n_channels, n_samples)
        The numpy array containing the data.

    Notes
    -----
    The type casting makes the output array writeable.
    """
    if last_samp is None:
        last_samp = cnt.get_sample_count()  # sample = (n_channels,)
    return cnt.get_samples_as_nparray(first_samp, last_samp).astype("float64")


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
