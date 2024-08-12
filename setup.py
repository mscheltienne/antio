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
    author="Mathieu Scheltienne",
    author_email="mathieu.scheltienne@fcbg.ch",
    classifiers=[
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    cmdclass={
        "build_ext": build_ext,
    },
    description="Python package to handle I/O with the CNT format from ANT Neuro.",
    distclass=BinaryDistribution,  # to handle binary files
    entry_points={
        "console_scripts": [
            "antio = antio.commands.main:run",
        ],
    },
    extras_require={
        "all": [
            "antio[build]",
            "antio[mne]",
            "antio[style]",
            "antio[test]",
        ],
        "build": [
            "build",
            "cibuildwheel",
            "setuptools",
            "twine",
        ],
        "full": [
            "antio[all]",
        ],
        "mne": [
            "mne",
        ],
        "style": [
            "codespell[toml]>=2.2.4",
            "isort",
            "pydocstyle[toml]",
            "ruff>=0.1.8",
            "toml-sort",
            "yamllint",
        ],
        "test": [
            "pytest-cov",
            "pytest>=8.0",
        ],
    },
    include_package_data=False,
    install_requires=[
        "click",
        "numpy>=1.21,<3",
        "packaging",
        "psutil",
    ],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    maintainer="Mathieu Scheltienne",
    maintainer_email="mathieu.scheltienne@fcbg.ch",
    name="antio",
    package_data={"antio.libeep": lib_files},
    project_urls={
        "Documentation": "https://github.com/mscheltienne/antio",
        "Homepage": "https://github.com/mscheltienne/antio",
        "Source": "https://github.com/mscheltienne/antio",
        "Tracker": "https://github.com/mscheltienne/antio/issues",
    },
    python_requires=">=3.9",
    version="0.1.0",
)
