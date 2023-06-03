#!/usr/bin/env python
from setuptools import find_packages, setup

from l5kit import __version__


setup(
    name="l5kit",
    version=__version__,
    description="Woven by Toyota Autonomous Vehicle Research library",
    author="Woven by Toyota",
    author_email="l5kit@woven-planet.global",
    url="https://github.com/woven-planet/l5kit",
    license="apache2",
    install_requires=[
        "imageio",
        "matplotlib",
        "numpy>=1.19.0",
        "opencv-contrib-python-headless",
        "protobuf>=3.12.2,<=3.20",
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
        "shapely<2.0.0",
        "typing_extensions",
        "bokeh<3.0.0",
        "importlib-metadata>=4.10.0,<5.0.0",
        "gym==0.22.0",
        "typed-ast==1.5.4"
    ],
    extras_require={
        "dev": ["pytest", "mypy", "types-PyYAML", "setuptools", "twine", "wheel", "pytest-cov",
                "flake8", "isort", "Sphinx", "recommonmark",
                "pre-commit", "sphinx-press-theme"]
    },
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
