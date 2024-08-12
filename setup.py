import platform
import subprocess
from pathlib import Path
from shutil import move
from tempfile import TemporaryDirectory

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.dist import Distribution

# set the platform-specific files, libeep first, pyeep second.
if platform.system() == "Linux":
    lib_files = ["libEep.so", "pyeep.so"]
elif platform.system() == "Windows":
    lib_files = ["Eep.dll", "pyeep.pyd"]
elif platform.system() == "Darwin":
    lib_files = ["libEep.dylib", "pyeep.so"]
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
            subprocess.run(
                [
                    "cmake",
                    "-S",
                    str(src_dir),
                    "-B",
                    build_dir,
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
                check=True,
            )
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


setup(
    cmdclass={
        "build_ext": build_ext,
    },
    distclass=BinaryDistribution,  # to handle binary files
    include_package_data=False,
    package_data={"antio.libeep": lib_files},
)
