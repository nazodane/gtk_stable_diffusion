# Copyright 2022 Toshimitsu Kimura <lovesyao@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


install_requires = [
# https://pygobject.readthedocs.io/en/latest/getting_started.html
    "pycairo",
    "PyGObject",

    "torch",
    "diffusers",
    "numpy",

    "xformers", # for memory efficient attention

## https://pytorch.org/TensorRT/getting_started/installation.html
#    'nvidia-pyindex',
#    'nvidia-tensorrt',
#    'torch-tensorrt',

    "libtorrent", # for model weights downloader
    "omegaconf", # for convert_original_stable_diffusion_to_diffusers.py

    "toml",
    "nltk", # for showing candidates of synonyms and antonyms
]

setup(
    name = "gtk_stable_diffusion",
    version = "0.0.8.1",
    author = "Toshimitsu Kimura",
    author_email = "lovesyao@gmail.com",
    description = ("A simple GTK UI for Stable Diffusion."),
    license = "Apache",
    keywords = "stable_diffusion diffusers",
    url = "https://github.com/nazodane/gtk_stable_diffusion",
    packages=['gtk_stable_diffusion'],
    include_package_data = True,
    long_description=read('README.md'),
    python_requires=">=3.10.0",
    install_requires=install_requires,
    scripts=["gtk-stable-diffusion"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Artistic Software",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: POSIX :: Linux", # XXX: for now
        "Environment :: GPU :: NVIDIA CUDA :: 11.7",  # XXX: for now
        "Intended Audience :: End Users/Desktop",
    ],
)
