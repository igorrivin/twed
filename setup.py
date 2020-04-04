#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import subprocess
import pip
import sys

REQUIREMENTS = [
    'pandas',
    'numpy',
    'numba',
]


setup(
    name='twed',
    version='0.1.0',
    description="Time warp edit distance (TWED)",
    author="Igor Rivin",
    author_email='rivinh@temple.edu',
    url='https://github.com/igorrivin/twed.git',
    packages=find_packages(),
    package_dir={'twed':'twed'},
    include_package_data=True,
    install_requires=REQUIREMENTS,
    zip_safe=False,
    keywords='Time Warp Edit Distance',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
#© 2020 GitHub, Inc.