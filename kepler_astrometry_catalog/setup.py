# -*- coding: utf-8 -*-

"""
setup.py - Kris Pardo (kmpardo@usc.edu)
"""
__version__ = "0.0.0"

import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import shlex
        import pytest

        if not self.pytest_args:
            targs = []
        else:
            targs = shlex.split(self.pytest_args)

        errno = pytest.main(targs)
        sys.exit(errno)


def readme():
    with open("README.md") as f:
        return f.read()


INSTALL_REQUIRES = [
    "numpy>=1.4.0",
    "scipy",
    "astropy>=1.3",
    "matplotlib",
]

EXTRAS_REQUIRE = {"all": ["pandas", "urllib"]}

###############
## RUN SETUP ##
###############

# run setup.
setup(
    name="kepler_astrometry_catalog",
    version=__version__,
    description=("code that creates a relative astrometry catalog for Kepler data"),
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 1 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    keywords=["astronomy", "astrometry", "Kepler"],
    url="https://github.com/kepler-gws/kepler_astrometry_catalog",
    author=["Kris Pardo", "Ali Wang"],
    license="MIT",
    packages=[
        "kepler_astrometry_catalog",
    ],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    tests_require=[
        "pytest==3.8.2",
    ],
    cmdclass={"test": PyTest},
    include_package_data=True,
    zip_safe=False,
)
