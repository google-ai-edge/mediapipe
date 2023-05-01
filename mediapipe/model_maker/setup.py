"""Copyright 2020-2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Setup for Mediapipe-Model-Maker package with setuptools.
"""

import glob
import os
import shutil
import setuptools


__version__ = 'dev'
MM_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
# Build dir to copy all necessary files and build package
SRC_NAME = 'pip_src'
BUILD_DIR = os.path.join(MM_ROOT_PATH, SRC_NAME)
BUILD_MM_DIR = os.path.join(BUILD_DIR, 'mediapipe_model_maker')


def _parse_requirements(path):
  with open(os.path.join(MM_ROOT_PATH, path)) as f:
    return [
        line.rstrip()
        for line in f
        if not (line.isspace() or line.startswith('#'))
    ]


def _setup_build_dir():
  """Setup the BUILD_DIR directory to build the mediapipe_model_maker package.

  We need to create a new BUILD_DIR directory because any references to the path
  `mediapipe/model_maker` needs to be renamed to `mediapipe_model_maker` to
  avoid conflicting with the mediapipe package name.
  This setup function performs the following actions:
  1. Copy python source code into BUILD_DIR and rename imports to
    mediapipe_model_maker
  2. Download models from GCS into BUILD_DIR
  """
  # Copy python source code into BUILD_DIR
  if os.path.exists(BUILD_DIR):
    shutil.rmtree(BUILD_DIR)
  python_files = glob.glob('python/**/*.py', recursive=True)
  python_files.append('__init__.py')
  for python_file in python_files:
    # Exclude test files from pip package
    if '_test.py' in python_file:
      continue
    build_target_file = os.path.join(BUILD_MM_DIR, python_file)
    with open(python_file, 'r') as file:
      filedata = file.read()
    # Rename all mediapipe.model_maker imports to mediapipe_model_maker
    filedata = filedata.replace('from mediapipe.model_maker',
                                'from mediapipe_model_maker')
    os.makedirs(os.path.dirname(build_target_file), exist_ok=True)
    with open(build_target_file, 'w') as file:
      file.write(filedata)

_setup_build_dir()

setuptools.setup(
    name='mediapipe-model-maker',
    version=__version__,
    url='https://github.com/google/mediapipe/tree/master/mediapipe/model_maker',
    description='MediaPipe Model Maker is a simple, low-code solution for customizing on-device ML models',
    author='The MediaPipe Authors',
    author_email='mediapipe@google.com',
    long_description='',
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(where=SRC_NAME),
    package_dir={'': SRC_NAME},
    install_requires=_parse_requirements('requirements.txt'),
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords=['mediapipe', 'model', 'maker'],
)
