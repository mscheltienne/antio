from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from mne.datasets import testing
from mne.utils import check_version

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture(scope="module")
def mne_testing_dataset() -> Path:
    """Check if the MNE testing dataset is available."""
    # TODO: replace with pytest.importorskip("mne", "1.9") when MNE 1.9 is released.
    if not check_version("mne", "1.9"):
        pytest.skip("Requires MNE 1.9+")
    path = testing.data_path(download=False)
    if str(path) == ".":  # dataset is not available
        pytest.skip("MNE testing dataset not available")
    return path / "antio"


def sha256sum(fname: Path) -> str:
    """Efficiently hash a file."""
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(fname, "rb", buffering=0) as file:
        while n := file.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


def walk(path: Path) -> Generator[Path, None, None]:
    """Walk recursively through a directory tree and yield the existing files."""
    if not path.is_dir():
        raise RuntimeError(
            f"The provided path '{path}' is not a directory. It can not be walked."
        )
    for entry in path.iterdir():
        if entry.is_dir():
            yield from walk(entry)
        else:
            yield entry


def test_data_synchronization(mne_testing_dataset):
    """Test that MNE's testing dataset does mirror the data folder in this project."""
    directory = Path(__file__).parent / "data"
    for file in walk(directory):
        relative = file.relative_to(directory)
        mne_file = mne_testing_dataset / relative
        assert mne_file.exists(), f"File '{mne_file}' is missing."
        assert sha256sum(file) == sha256sum(
            mne_file
        ), f"File '{mne_file}' is different."
