#!/bin/bash
set -eo pipefail

pip install delvewheel
echo "CMAKE_GENERATOR=Visual Studio 17 2022" | tee -a $GITHUB_ENV
echo "CMAKE_GENERATOR_PLATFORM=$CIBW_ARCHS" | tee -a $GITHUB_ENV
PY_VER=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
Python3_SABI_LIBRARY="C:\\Users\\runneradmin\\AppData\\Local\\pypa\\cibuildwheel\\Cache\\nuget-cpython\\python${CIBW_ARCHS,,}.${PY_VER}\\tools\\libs\\python3.lib"
echo "Python3_SABI_LIBRARY=$Python3_SABI_LIBRARY" | tee -a $GITHUB_ENV
