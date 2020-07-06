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
        "protobuf==3.12.2",
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
        "dev": ["pytest==5.4.3", "mypy==0.782", "setuptools", "twine", "wheel", "pytest-cov==2.10.0", "flake8==3.8.3",
                "black==19.10b0", "isort==5.0.3", "Sphinx==3.1.1", "sphinx-rtd-theme==0.5.0", "recommonmark==0.6.0",
                "pre-commit==2.5.1"]
    },
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
