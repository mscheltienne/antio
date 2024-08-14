"""Handle optional dependency imports.

Inspired from pandas: https://pandas.pydata.org/
"""

from __future__ import annotations

import importlib

# A mapping from import name to package name (on PyPI) when the package name
# is different.
_INSTALL_MAPPING: dict[str, str] = {}


def import_optional_dependency(name: str, extra: str = "") -> None:
    """Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice message will be
    raised.

    Parameters
    ----------
    name : str
        The module name.
    extra : str
        Additional text to include in the ImportError message.
    """
    package_name = _INSTALL_MAPPING.get(name)
    install_name = package_name if package_name is not None else name

    if importlib.util.find_spec(name) is None:
        raise ImportError(
            f"Missing optional dependency '{install_name}'. {extra} Use pip or "
            f"conda to install {install_name}."
        )
