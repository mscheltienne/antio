import os
import platform
import subprocess
import sys
from pathlib import Path
from shutil import move
from tempfile import TemporaryDirectory

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel

# set the platform-specific files, libeep first, pyeep second.
if platform.system() == "Linux":
    lib_files = ["libEep.so", "pyeep.abi3.so"]
elif platform.system() == "Windows":
    lib_files = ["Eep.dll", "pyeep.pyd"]
elif platform.system() == "Darwin":
    lib_files = ["libEep.dylib", "pyeep.abi3.so"]
else:
    lib_files = []


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class build_ext(_build_ext):
    def run(self):
        """Build libeep with cmake as part of the extension build process."""
        src_dir = Path(__file__).parent / "src" / "libeep"
        with TemporaryDirectory() as build_dir:  # str
            args = [
                    "cmake",
                    "-S",
                    str(src_dir),
                    "-B",
                    build_dir,
                    "-DCMAKE_BUILD_TYPE=Release",
                    f"-DPython3_EXECUTABLE={sys.executable}",
            ]
            if platform.system() == "Windows":
                args.append("-DCMAKE_GENERATOR=Visual Studio 17 2022")
            for key in ("CMAKE_GENERATOR_PLATFORM",):
                if key in os.environ:
                    args.append(f"-D{key}={os.environ[key]}")
            subprocess.run(args, check=True)
            subprocess.run(
                ["cmake", "--build", build_dir, "--config", "Release"], check=True
            )
            # locate the built files and move then to antio.libeep
            build_dir = Path(build_dir)
            if platform.system() == "Windows":
                lib = build_dir / "Release" / lib_files[0]
                pyeep = build_dir / "python" / "Release" / lib_files[1]
            else:
                lib = build_dir / lib_files[0]
                pyeep = build_dir / "python" / lib_files[1]
            for elt in (lib, pyeep):
                move(elt, Path(self.build_lib) / "antio" / "libeep" / elt.name)
        super().run()


# Adapted from
# https://github.com/joerick/python-abi3-package-sample/blob/main/setup.py
class bdist_wheel_abi3(bdist_wheel):
    def get_tag(self):
        python, abi, plat = super().get_tag()

        if python.startswith("cp"):
            # on CPython, our wheels are abi3 and compatible back to 3.2,
            # but let's set it to our min version anyway
            return "cp39", "abi3", plat

        return python, abi, plat


setup(
    cmdclass={
        "build_ext": build_ext,
        "bdist_wheel": bdist_wheel_abi3,
    },
    distclass=BinaryDistribution,  # to handle binary files
    include_package_data=False,
)
