from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..utils._checks import ensure_path
from . import pyeep

if TYPE_CHECKING:
    from collections.abc import ByteString
    from typing import Optional, Union

    from numpy.typing import NDArray


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

    def get_channel(self, index: int) -> tuple[str, str, str, str, str]:
        """Get the channel information at a given index.

        Parameters
        ----------
        index : int
            Index of the channel.

        Returns
        -------
        channel : tuple of shape (5,)
            The tuple contains the following elements:
            - 0: label
            - 1: unit
            - 2: reference
            - 3: status
            - 4: type
        """
        n_channels = self.get_channel_count()
        if index < n_channels:
            return (
                pyeep.get_channel_label(self._handle, index),
                pyeep.get_channel_unit(self._handle, index),
                pyeep.get_channel_reference(self._handle, index),
                pyeep.get_channel_status(self._handle, index),
                pyeep.get_channel_type(self._handle, index),
            )
        else:
            raise RuntimeError(
                f"Channel index exceeds total channel count, {n_channels}."
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
            List of retrieved samples, ordered by (n_channels,) samples.
        """
        return pyeep.get_samples(self._handle, fro, to)

    def get_samples_as_nparray(self, fro: int, to: int) -> NDArray[np.float32]:
        """Get samples between 2 index as numpy array.

        Parameters
        ----------
        fro : int
            Start index.
        to : int
            End index.

        Returns
        -------
        samples : array of shape (n_channels, n_samples)
            List of retrieved samples as 2-dimensional numpy array.

        Notes
        -----
        This array is read-only.
        """
        buffer = self._get_samples_as_buffer(fro, to)
        return np.frombuffer(buffer, dtype=np.float32).reshape((to - fro, -1)).T

    def _get_samples_as_buffer(self, fro: int, to: int) -> ByteString:
        """Get samples between 2 index as memoryview.

        Parameters
        ----------
        fro : int
            Start index.
        to : int
            End index.

        Returns
        -------
        samples : buffer of shape (n_channels * n_samples)
            List of retrieved samples, ordered by (n_channels,) samples.

        Notes
        -----
        This buffer is read-only.
        """
        return pyeep.get_samples_as_buffer(self._handle, fro, to)

    def _get_start_time(self):
        """Get start time in UNIX format.

        Returns
        -------
        start_time : int
            Acquisition start time.
        """
        return pyeep.get_start_time(self._handle)

    def get_start_time(self) -> datetime.datetime:
        """Get start time in datetime format.

        Returns
        -------
        start_time : datetime.datetime
            Acquisition start time.
        """
        return datetime.datetime.fromtimestamp(
            self._get_start_time(), datetime.timezone.utc
        )

    def get_hospital(self) -> str:
        """Get hospital name of the recording.

        Returns
        -------
        hospital : str
            Hospital name.
        """
        return pyeep.get_hospital(self._handle)

    def get_machine_info(self) -> tuple[str, str, str]:
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

    def get_patient_info(self) -> tuple[str, str, str, datetime.datetime]:
        """Get patient info.

        Returns
        -------
        pt_info : tuple of shape (4,)
            The tuple contains the following elements:
            - 0: patient name
            - 1: patient id
            - 2: patient sex
            - 3: patient date of birth
        """
        sex = pyeep.get_patient_sex(self._handle)
        return (
            pyeep.get_patient_name(self._handle),
            pyeep.get_patient_id(self._handle),
            "" if sex == "\x00" else sex,
            self._get_date_of_birth(),
        )

    def _get_date_of_birth(self) -> datetime.datetime:
        """Get date of birth of the patient.

        Returns
        -------
        dob : datetime.datetime
            date of birth in datetime format.
        """
        year, month, date = pyeep.get_date_of_birth(self._handle)
        return datetime.datetime(
            year=year, month=month, day=date, tzinfo=datetime.timezone.utc
        )

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
        n_triggers = self.get_trigger_count()
        if index < n_triggers:
            return pyeep.get_trigger(self._handle, index)
        else:
            raise RuntimeError(
                f"Trigger index exceeds total trigger count, {n_triggers}."
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
