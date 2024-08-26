from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..utils._checks import ensure_path
from . import pyeep

if TYPE_CHECKING:
    from collections.abc import ByteString
    from datetime import date
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
        if index < 0:
            raise RuntimeError(f"Channel index {index} cannot be negative.")
        n_channels = self.get_channel_count()
        if n_channels <= index:
            raise RuntimeError(
                f"Channel index {index} exceeds total channel count {n_channels}."
            )
        return (
            pyeep.get_channel_label(self._handle, index),
            pyeep.get_channel_unit(self._handle, index),
            pyeep.get_channel_reference(self._handle, index),
            pyeep.get_channel_status(self._handle, index),
            pyeep.get_channel_type(self._handle, index),
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
        if fro < 0 or to < 0:
            raise RuntimeError(f"Start/Stop index {fro}/{to} cannot be negative.")
        if self.get_sample_count() < to:
            raise RuntimeError(f"End index {to} exceeds total sample count.")
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
        if fro < 0 or to < 0:
            raise RuntimeError(f"Start/Stop index {fro}/{to} cannot be negative.")
        if self.get_sample_count() < to:
            raise RuntimeError(f"End index {to} exceeds total sample count.")
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

    def get_start_time(self) -> datetime:
        """Get start time.

        Returns
        -------
        start_time : datetime
            Acquisition start time.
        """
        start_time = pyeep.get_start_time(self._handle)
        return datetime.fromtimestamp(start_time, timezone.utc)

    def get_start_time_and_fraction(self) -> Optional[datetime]:
        """Get start time with second fraction.

        Returns
        -------
        start_time : datetime | None
            The measurement time of the recording (in the UTC timezone). None if the
            start date parsed was not a valid EXCEL format start date.
        """
        start_date, start_fraction = pyeep.get_start_date_and_fraction(self._handle)
        # start date is in EXCEL format
        if start_date >= 27538 and start_date <= 2958464:
            start_date = np.round(start_date * 3600.0 * 24.0) - 2209161600
            return datetime.fromtimestamp(start_date + start_fraction, timezone.utc)

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

    def get_patient_info(self) -> tuple[str, str, str, datetime]:
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

    def _get_date_of_birth(self) -> date:
        """Get date of birth of the patient.

        Returns
        -------
        dob : datetime.datetime
            date of birth in datetime format.
        """
        year, month, day = pyeep.get_date_of_birth(self._handle)
        return datetime(year=year, month=month, day=day, tzinfo=timezone.utc).date()

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
        if index < 0:
            raise RuntimeError(f"Trigger index {index} cannot be negative.")
        n_triggers = self.get_trigger_count()
        if n_triggers <= index:
            raise RuntimeError(
                f"Trigger index {index} exceeds total trigger count {n_triggers}."
            )
        return pyeep.get_trigger(self._handle, index)


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
