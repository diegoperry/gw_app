# -*- coding: utf-8 -*-

"""
setup.py - Kris Pardo (kpardo@caltech.edu)
"""
__version__ = '0.0.0'

import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

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
    with open('README.md') as f:
        return f.read()

INSTALL_REQUIRES = [
    'numpy>=1.4.0',
    'scipy',
    'astropy>=1.3',
    'matplotlib',
]

EXTRAS_REQUIRE = {
    'all':[
        'pymc3',
        'corner'
    ]
}

###############
## RUN SETUP ##
###############

# run setup.
setup(
    name='estoiles',
    version=__version__,
    description=('routines for modeling GW signatures in photometric surveys.'),
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 1 - Alpha',
        'License :: OSI Approved :: MIT License',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    keywords=['astronomy', 'gravitational waves'],
    url='https://github.com/kpardo/estoiles/',
    author=['Kris Pardo', 'Ali Wang'],
    license='MIT',
    packages=[
        'estoiles',
    ],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    tests_require=['pytest==3.8.2',],
    cmdclass={'test':PyTest},
    include_package_data=True,
    zip_safe=False,
)
