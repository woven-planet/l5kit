#!/usr/bin/env python
from setuptools import find_packages, setup

from l5kit import __version__

setup(
    name="l5kit",
    version=__version__,
    description="Lyft Autonomous Vehicle Research library",
    author="Lyft Level 5",
    author_email="l5kit@lyft.com",
    url="https://github.com/lyft/l5kit",
    license="apache2",
    install_requires=[
        "imageio",
        "matplotlib",
        "numpy",
        "opencv-contrib-python-headless",
        "protobuf>=3.12.2",
        "pymap3d",
        "scipy",
        "setuptools",
        "torch>=1.5.0,<2.0.0",
        "torchvision>=0.6.0,<1.0.0",
        "tqdm",
        "transforms3d",
        "zarr",
        "pyyaml",
        "notebook",
        "ptable",
        "ipywidgets",
        "shapely",
    ],
    extras_require={
        "dev": ["pytest", "mypy", "setuptools", "twine", "wheel", "pytest-cov", "flake8",
                "isort", "Sphinx", "sphinx-rtd-theme", "recommonmark",
                "pre-commit"]
    },
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
