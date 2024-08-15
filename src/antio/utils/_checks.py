from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


def ensure_path(item: Any, must_exist: bool) -> Path:
    """Ensure a variable is a Path.

    Parameters
    ----------
    item : Any
        Item to check.
    must_exist : bool
        If True, the path must resolve to an existing file or directory.

    Returns
    -------
    path : Path
        Path validated and converted to a pathlib.Path object.
    """
    try:
        item = Path(item)
    except TypeError:
        try:
            str_ = f"'{str(item)}' "
        except Exception:
            str_ = ""
        raise TypeError(
            f"The provided path {str_}is invalid and can not be converted. Please "
            f"provide a str, an os.PathLike or a pathlib.Path object, not {type(item)}."
        )
    if must_exist and not item.exists():
        raise FileNotFoundError(f"The provided path '{str(item)}' does not exist.")
    return item
