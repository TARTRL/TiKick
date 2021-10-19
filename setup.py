#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 The TARTRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""

import os
from setuptools import setup, find_packages
import setuptools

def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("tmarl", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]

setup(
    name="tmarl",  # Replace with your own username
    version=get_version(),
    description="marl algorithms",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    author="tmarl",
    author_email="tmarl_contact@tartrl.cn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    keywords="multi-agent reinforcement learning algorithms pytorch",
    python_requires='>=3.6',
)
