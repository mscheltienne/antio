from __future__ import annotations

import importlib
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import pooch

from ..utils._checks import ensure_path
from ._fetch import fetch_dataset

if TYPE_CHECKING:
    from typing import Optional, Union

_REGISTRY: Path = files("antio.datasets") / "ca_208-registry.txt"


def _make_registry(
    folder: Union[str, Path], output: Optional[Union[str, Path]] = None
) -> None:  # pragma: no cover
    """Create the registry file for the dataset.

    Parameters
    ----------
    folder : path-like
        Path to the sample dataset.
    output : path-like
        Path to the output registry file.
    """
    folder = ensure_path(folder, must_exist=True)
    output = _REGISTRY if output is None else output
    pooch.make_registry(folder, output=output, recursive=True)


def data_path(path: str | Path | None = None) -> Path:  # pragma: no cover
    """Return the path to the dataset, downloaded if needed.

    Parameters
    ----------
    path : str | Path | None
        Path where the dataset should be downloaded. If None, the path in the
        environment variable ``MNE_DATA`` or ``"~/mne_data/`` is used.

    Returns
    -------
    path : Path
        Path to the sample dataset, by default in ``"~/mne_data/antio-data"``.
    """
    if importlib.util.find_spec("mne") is None:
        raise ImportError("Missing 'mne'. Use pip or conda to install 'mne'.")

    from mne.utils import get_config

    if path is None:
        path = (
            Path(get_config("MNE_DATA", Path.home() / "mne_data")).expanduser()
            / "antio-data"
        )
    else:
        path = ensure_path(path, must_exist=True)
    base_url = "https://github.com/mscheltienne/antio/raw/main/tests/data/CA_208"
    return fetch_dataset(path / "CA_208", base_url, _REGISTRY)
