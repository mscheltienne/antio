from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from . import pyeep

if TYPE_CHECKING:
    from typing import Optional, Union


class BaseCNT:
    """Object representing a CNT file."""

    def __init__(self, handle: int) -> None:
        self._handle = handle
        if self._handle == -1:
            raise RuntimeError("Not a valid libeep handle.")

    def __del__(self) -> None:
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
        """
        return (
            pyeep.get_channel_label(self._handle, index),
            pyeep.get_channel_unit(self._handle, index),
            pyeep.get_channel_reference(self._handle, index),
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
    fname = Path(fname)
    if fname.suffix != ".cnt":
        raise RuntimeError(f"Unsupported file extension '{fname.suffix}'.")
    if not fname.exists():
        raise FileNotFoundError(f"File {fname} not found.")
    return InputCNT(pyeep.read(str(fname)))
