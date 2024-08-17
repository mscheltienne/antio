from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils._checks import ensure_path
from . import pyeep

if TYPE_CHECKING:
    from typing import Optional, Union


class BaseCNT:
    """Object representing a CNT file."""

    def __init__(self, handle: int) -> None:
        self._handle = handle
        if self._handle == -1:
            raise RuntimeError("Not a valid libeep handle.")

    def __del__(self) -> None:  # noqa: D105
        if self._handle != -1:
            pyeep.close(self._handle)


class InputCNT(BaseCNT):
    """Object representing reading a CNT file."""

    def __init__(self, handle: int) -> None:
        BaseCNT.__init__(self, handle)

    def get_channel_count(self) -> int:
        """Get the total number of channels.

        Returns
        -------
        n_channels : int
            Number of channels.
        """
        return pyeep.get_channel_count(self._handle)

    def get_channel(self, index: int) -> tuple[str, str, str]:
        """Get the channel information at a given index.

        Parameters
        ----------
        index : int
            Index of the channel.

        Returns
        -------
        channel : tuple of shape (3,)
            The tuple contains the following elements:
            - 0: label
            - 1: unit
            - 2: reference
            - 3: status
            - 4: type
        """
        if index < self.get_channel_count():
            return (
                pyeep.get_channel_label(self._handle, index),
                pyeep.get_channel_unit(self._handle, index),
                pyeep.get_channel_reference(self._handle, index),
                pyeep.get_channel_status(self._handle, index),
                pyeep.get_channel_type(self._handle, index),
            )
        else:
            raise RuntimeError(
                f"Channel index exceeds total channel count"
                f", {self.get_channel_count()}."
            )

    def get_sample_frequency(self) -> int:
        """Get the sampling frequency of the recording in Hz.

        Returns
        -------
        sfreq : int
            Sampling frequency in Hz.
        """
        return pyeep.get_sample_frequency(self._handle)

    def get_sample_count(self) -> int:
        """Get the total number of samples.

        Returns
        -------
        n_samples : int
            Number of samples, a sample being of shape (n_channels, ).
        """
        return pyeep.get_sample_count(self._handle)

    def get_samples(self, fro: int, to: int) -> list[float]:
        """Get samples between 2 index.

        Parameters
        ----------
        fro : int
            Start index.
        to : int
            End index.

        Returns
        -------
        samples : list of shape (n_channels * n_samples)
            List of retrieved samples, ordered by (n_channels, ) samples.
        """
        return pyeep.get_samples(self._handle, fro, to)

    def _get_start_time(self):
        """Get start time in UNIX format.

        Returns
        -------
        time_t : int
            start time.
        """
        return pyeep.get_start_time(self._handle)

    def get_start_time(self):
        """Get start time in datetime format.

        Returns
        -------
        time_t : datetime.datetime
            start time.
        """
        return datetime.datetime.fromtimestamp(self._get_start_time(), datetime.UTC)

    def get_hospital(self):
        """Get hospital name of the recording.

        Returns
        -------
        Hospital : str
            hospital name.
        """
        return pyeep.get_hospital(self._handle)

    def get_machine_info(self):
        """Get machine information.

        Returns
        -------
        machine_info : tuple of shape (3,)
            The tuple contains the following elements:
            - 0: machine make
            - 1: machine model
            - 2: machine serial number
        """
        return (
            pyeep.get_machine_make(self._handle),
            pyeep.get_machine_model(self._handle),
            pyeep.get_machine_serial_number(self._handle),
        )

    def get_patient_info(self):
        """Get patient info.

        Returns
        -------
        pt_info : tuple of shape (3,)
            The tuple contains the following elements:
            - 0: patient name
            - 1: patient id
            - 2: patient sex
            - 3: patient date of birth
        """
        return (
            pyeep.get_patient_name(self._handle),
            pyeep.get_patient_id(self._handle),
            ""
            if pyeep.get_patient_sex(self._handle) is None
            else pyeep.get_patient_sex(self._handle),
            self._get_date_of_birth(),
        )

    def _get_date_of_birth(self):
        """Get date of birth of the patient.

        Returns
        -------
        dob : datetime.datetime
            date of birth in datetime format.
        """
        year, month, date = pyeep.get_date_of_birth(self._handle)
        return datetime.datetime(year=year, month=month, day=date)

    def get_trigger_count(self) -> int:
        """Get the total number of triggers (annotations).

        Returns
        -------
        n_triggers : int
            Number of triggers (annotations).
        """
        return pyeep.get_trigger_count(self._handle)

    def get_trigger(
        self, index: int
    ) -> tuple[str, int, int, Optional[str], Optional[str], Optional[str]]:
        """Get the trigger (annotation) at a given index.

        Parameters
        ----------
        index : int
            Index of the trigger.

        Returns
        -------
        trigger : tuple of shape (6,)
            The tuple contains the following elements:
            - 0: code
            - 1: sample index
            - 2: duration in samples
            - 3: condition
            - 4: description
            - 5: impedance, as a string separated by spaces.
        """
        if index < self.get_trigger_count():
            return pyeep.get_trigger(self._handle, index)
        else:
            raise RuntimeError(
                "Trigger index exceeds total trigger count"
                f", {self.get_trigger_count()}."
            )


def read_cnt(fname: Union[str, Path]) -> InputCNT:
    """Read a CNT file.

    Parameters
    ----------
    filename : str | Path
        Path to the .cnt file.

    Returns
    -------
    cnt : InputCNT
        An object representing the CNT file.
    """
    fname = ensure_path(fname, must_exist=True)
    if fname.suffix != ".cnt":
        raise RuntimeError(f"Unsupported file extension '{fname.suffix}'.")
    return InputCNT(pyeep.read(str(fname)))
