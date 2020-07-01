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
        "protobuf",
        "pymap3d",
        "scipy",
        "setuptools",
        "torch",
        "torchvision",
        "tqdm",
        "transforms3d",
        "zarr",
        "strictyaml",
        "notebook",
        "ptable",
        "ipywidgets"
    ],
    extras_require={
        "dev": ["pytest", "mypy", "setuptools", "twine", "wheel", "pytest-cov", "flake8", "black", "isort",
                "Sphinx", "sphinx-rtd-theme", "recommonmark", "pre-commit"]
    },
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
