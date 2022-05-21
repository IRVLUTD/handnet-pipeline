#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="pose2mesh",
    version="0.1",
    description="pose2mesh lib",
    packages=find_packages(exclude=("configs", "tests",)),
)