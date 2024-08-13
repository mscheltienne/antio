from __future__ import annotations

from collections import defaultdict

import nump as np

from .libeep import read_cnt
from .utils._checks import ensure_path
from .utils._imports import import_optional_dependency

import_optional_dependency("mne")

from mne import Annotations, create_info  # noqa: E402
from mne.io import BaseRaw  # noqa: E402
from mne.utils import logger, verbose  # noqa: E402

units = {"uv": 1e-6}


class RawANT(BaseRaw):
    @verbose
    def __init__(self, fname, verbose=None):
        logger.info("Reading ANT file %s", fname)
        fname = ensure_path(fname, must_exist=True)
        cnt = read_cnt(str(fname))
        # parse channels
        n_channels = cnt.get_channel_count()
        ch_names, ch_units, ch_refs = [], [], []
        for k in range(n_channels):
            ch_curr = cnt.get_channel(k)
            ch_names.append(ch_curr[0])
            ch_units.append(ch_curr[1].lower())  # always lower the unit for mapping
            ch_refs.append(ch_curr[2])
        if len(set(ch_refs)) == 1:
            logger.info(
                "All %i channels are referenced to %s.", len(ch_refs), ch_refs[0]
            )
        # parse sampling frequency and create info
        info = create_info(ch_names, sfreq=cnt.get_sample_frequency(), ch_types="eeg")
        # read data array
        n_samples = cnt.get_sample_count()  # of shape (n_channels,)
        data = cnt.get_samples(0, n_samples)
        data = np.array(data).reshape(n_channels, -1).T  # (n_channels, n_samples)
        # apply scalings to SI units if able
        units_index = defaultdict(list)
        for idx, unit in enumerate(ch_units):
            units_index[unit].append(idx)
        for unit, value in units_index.items():
            if unit in units:
                data[np.array(value, dtype=np.int16), :] *= units[unit]
            else:
                logger.warning("Unit %s not recognized, not scaling.", unit)
        super().__init__(info, preload=data, filenames=[fname], verbose=verbose)
        # look for annotations (called trigger by ant)
        n_triggers = cnt.get_trigger_count()
        labels, onsets, durations = [], [], []
        for k in range(n_triggers):
            label, idx, duration, _, _, _ = cnt.get_trigger(k)
            labels.append(label)
            onsets.append(idx)
            durations.append(duration)  # TODO: handle duration
        onsets = np.array(onsets) / self.info["sfreq"]
        annotations = Annotations(
            onsets, duration=np.zeros_like(onsets), description=labels
        )
        self.set_annotations(annotations)
