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
