from __future__ import annotations

from . import pyeep


class cnt_base:
    def __init__(self, handle):
        self._handle = handle
        if self._handle == -1:
            raise RuntimeError("Not a valid libeep handle.")

    def __del__(self):
        if self._handle != -1:
            pyeep.close(self._handle)


class cnt_in(cnt_base):
    def __init__(self, handle):
        cnt_base.__init__(self, handle)

    def get_channel_count(self):
        return pyeep.get_channel_count(self._handle)

    def get_channel(self, index):
        return (
            pyeep.get_channel_label(self._handle, index),
            pyeep.get_channel_unit(self._handle, index),
            pyeep.get_channel_reference(self._handle, index),
        )

    def get_sample_frequency(self):
        return pyeep.get_sample_frequency(self._handle)

    def get_sample_count(self):
        return pyeep.get_sample_count(self._handle)

    def get_samples(self, fro, to):
        return pyeep.get_samples(self._handle, fro, to)

    def get_trigger_count(self):
        return pyeep.get_trigger_count(self._handle)

    def get_trigger(self, index):
        return pyeep.get_trigger(self._handle, index)


class cnt_out(cnt_base):
    def __init__(self, handle, channel_count):
        cnt_base.__init__(self, handle)
        self._channel_count = channel_count

    def add_samples(self, samples):
        return pyeep.add_samples(self._handle, samples, self._channel_count)


def read_cnt(filename: str) -> cnt_in:
    """Read a CNT file.

    Parameters
    ----------
    filename : str
        Path to the .cnt file.
    """
    if not filename.endswith(".cnt"):
        raise RuntimeError("Unsupported file extension.")
    return cnt_in(pyeep.read(filename))


def write_cnt(
    filename: str, rate: float, channels: list[tuple], rf64: int = 0
) -> cnt_out:
    """Create an object for writing a .cnt file.

    Parameters
    ----------
    filename : str
        Path to the .cnt file.
    rate : float
        Sampling rate in Hz.
    channels : list of tuple
        List of channel names.
    rf64 : int
        If 0, creates default 32-bit CNT data, otherwise 64 bit (for larger than 2GB
        files).
    """
    if not filename.endswith(".cnt"):
        raise RuntimeError("Unsupported file extension.")
    channels_handle = pyeep.create_channel_info()
    for c in channels:
        pyeep.add_channel(channels_handle, c[0], c[1], c[2])
    rv = cnt_out(pyeep.write_cnt(filename, rate, channels_handle, rf64), len(channels))
    pyeep.close_channel_info(channels_handle)
    return rv
