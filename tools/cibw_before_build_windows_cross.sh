#!/bin/bash
set -eo pipefail

pip install delvewheel
echo "CMAKE_GENERATOR=Visual Studio 17 2022" | tee -a $GITHUB_ENV
echo "CMAKE_GENERATOR_PLATFORM=$CIBW_ARCHS" | tee -a $GITHUB_ENV
PY_VER=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
FT=$(python -c "import sysconfig; print(int(bool(sysconfig.get_config_var('Py_GIL_DISABLED'))))")
ARCH="${CIBW_ARCHS,,}"
NUGET_CACHE="C:\\Users\\runneradmin\\AppData\\Local\\pypa\\cibuildwheel\\Cache\\nuget-cpython"
if [ "$FT" = "1" ]; then
  # cp3XYt cross-build: point CMake at the ARM64 free-threaded python lib,
  # otherwise CMake autodiscovery picks up the host x64 lib and the linker fails.
  PY_TAG=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}t')")
  Python3_LIBRARY="${NUGET_CACHE}\\python${ARCH}-freethreaded.${PY_VER}\\tools\\libs\\python${PY_TAG}.lib"
  echo "Python3_LIBRARY=$Python3_LIBRARY" | tee -a $GITHUB_ENV
else
  Python3_SABI_LIBRARY="${NUGET_CACHE}\\python${ARCH}.${PY_VER}\\tools\\libs\\python3.lib"
  echo "Python3_SABI_LIBRARY=$Python3_SABI_LIBRARY" | tee -a $GITHUB_ENV
fi
