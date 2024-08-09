from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from mne import Annotations, create_info
from mne.io import RawArray

from . import libeep
from .utils._checks import ensure_path

if TYPE_CHECKING:
    from typing import Union

    from mne.io import BaseRaw


def read_raw_ant(fname: Union[Path, str]) -> BaseRaw:
    """Read a raw ANT file in *.cnt format.

    Parameters
    ----------
    fname : str | Path
        Path to the file name to read.
    """
    fname = ensure_path(fname, must_exist=True)
    cnt = libeep.read_cnt(str(fname))

    # read data array
    samples = []
    samples_per_channel = cnt.get_sample_count()
    for k in range(samples_per_channel):
        samples.append(cnt.get_samples(k, (k + 1)))
    samples = np.array(samples).T

    # parse channel names
    ch_names = []
    for k in range(cnt.get_channel_count()):
        ch_curr = cnt.get_channel(k)
        ch_names.append(f"{ch_curr[0]} - {ch_curr[2]}")

    # transform data for MNE
    info = create_info(ch_names, sfreq=cnt.get_sample_frequency(), ch_types="eeg")
    raw = RawArray(data=samples * 1e-6, info=info)

    # look for annotations (called trigger by ant)
    trigger_count = cnt.get_trigger_count()
    labels = []
    onsets = []
    for k in range(trigger_count):
        label, idx, _, _, _, _ = cnt.get_trigger(k)
        labels.append(label)
        onsets.append(idx)
    onsets = np.array(onsets) / raw.info["sfreq"]
    annotations = Annotations(
        onsets, duration=np.zeros_like(onsets), description=labels
    )
    raw.set_annotations(annotations)
    return raw
