from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from mne.io.ant.ant import RawANT


def read_raw_ant(
    fname: str | Path,
    eog: str | None = None,
    misc: str | None = r"BIP\d+",
    bipolars: list[str] | tuple[str, ...] | None = None,
    impedance_annotation: str = "impedance",
    *,
    verbose=None,
) -> RawANT:
    """Reader for Raw ANT files in .cnt format."""
    if importlib.util.find_spec("mne") is None:
        raise ImportError("Missing 'mne'. Use pip or conda to install 'mne'.")

    from mne.utils import check_version

    if not check_version("mne", "1.9"):
        raise RuntimeError(
            "The ANT-Neuro reader was added to MNE-Python 1.9. Either upgrade MNE "
            "or use 'antio' version 0.1.0 to read the CNT file to a Raw object."
        )

    from mne.io import read_raw_ant

    return read_raw_ant(
        fname=fname,
        eog=eog,
        misc=misc,
        bipolars=bipolars,
        impedance_annotation=impedance_annotation,
        verbose=verbose,
    )
