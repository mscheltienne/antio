[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![codecov](https://codecov.io/gh/mscheltienne/antio/graph/badge.svg?token=ebC07d0dyM)](https://codecov.io/gh/mscheltienne/antio)
[![ci](https://github.com/mscheltienne/antio/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/mscheltienne/antio/actions/workflows/ci.yaml)
[![PyPI version](https://badge.fury.io/py/antio.svg)](https://badge.fury.io/py/antio)
[![Downloads](https://static.pepy.tech/badge/antio)](https://pepy.tech/project/antio)

# ANT I/O

Python package to handle I/O with the CNT format from ANT Neuro.
This software uses the [LIBEEP Library](http://libeep.sourceforge.net).

The version `0.1.0` has a function `antio.io.read_raw_ant` which can load a CNT file
with most MNE-Python version. However, for advance parsing and features such as file
preloading, `antio` version 0.3.0 and MNE-Python version 1.9 are required. MNE-Python
1.9 includes support for CNT files in `mne.io.read_raw` and `mne.io.read_raw_ant`.
