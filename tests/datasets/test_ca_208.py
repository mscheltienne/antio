import hashlib
from pathlib import Path

from antio.datasets import ca_208
from antio.datasets.ca_208 import _REGISTRY


def sha256sum(fname) -> str:
    """Efficiently hash a file."""
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(fname, "rb", buffering=0) as file:
        while n := file.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


def test_ca_208():
    data_path = ca_208.data_path(Path(__file__).parent.parent / "data")
    assert data_path.exists()
    for elt in (
        "test_CA_208.cnt",
        "test_CA_208_amp_disconnection.cnt",
        "test_CA_208_start_stop.cnt",
    ):
        for suffix in (".cnt", ".vhdr", ".eeg", ".vmrk", ".evt"):
            assert (data_path / elt).with_suffix(suffix).exists()


def test_make_registry(tmp_path):
    """Test the registry"""
    ca_208._make_registry(
        Path(__file__).parent.parent / "data" / "CA_208",
        output=tmp_path / "registry.txt",
    )
    assert (tmp_path / "registry.txt").exists()
    assert sha256sum(tmp_path / "registry.txt") == sha256sum(_REGISTRY)
