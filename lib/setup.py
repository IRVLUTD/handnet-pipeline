#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="100doh",
    version="0.1",
    description="100doh dataset utils",
    packages=find_packages(exclude=("configs", "tests",)),
)