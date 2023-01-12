"""Copyright 2020-2022 The MediaPipe Authors. All Rights Reserved.

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
import subprocess
import sys
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


def _copy_to_pip_src_dir(file):
  """Copy a file from bazel-bin to the pip_src dir."""
  dst = file
  dst_dir = os.path.dirname(dst)
  if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
  src_file = os.path.join('../../bazel-bin/mediapipe/model_maker', file)
  shutil.copyfile(src_file, file)


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

  # Use bazel to download GCS model files
  model_build_files = [
      'models/gesture_recognizer/BUILD',
      'models/text_classifier/BUILD',
  ]
  for model_build_file in model_build_files:
    build_target_file = os.path.join(BUILD_MM_DIR, model_build_file)
    os.makedirs(os.path.dirname(build_target_file), exist_ok=True)
    shutil.copy(model_build_file, build_target_file)
  external_files = [
      'models/gesture_recognizer/canned_gesture_classifier.tflite',
      'models/gesture_recognizer/gesture_embedder.tflite',
      'models/gesture_recognizer/hand_landmark_full.tflite',
      'models/gesture_recognizer/palm_detection_full.tflite',
      'models/gesture_recognizer/gesture_embedder/keras_metadata.pb',
      'models/gesture_recognizer/gesture_embedder/saved_model.pb',
      'models/gesture_recognizer/gesture_embedder/variables/variables.data-00000-of-00001',
      'models/gesture_recognizer/gesture_embedder/variables/variables.index',
      'models/text_classifier/mobilebert_tiny/keras_metadata.pb',
      'models/text_classifier/mobilebert_tiny/saved_model.pb',
      'models/text_classifier/mobilebert_tiny/assets/vocab.txt',
      'models/text_classifier/mobilebert_tiny/variables/variables.data-00000-of-00001',
      'models/text_classifier/mobilebert_tiny/variables/variables.index',
  ]
  for elem in external_files:
    external_file = os.path.join(f'{SRC_NAME}/mediapipe_model_maker', elem)
    sys.stderr.write('downloading file: %s\n' % external_file)
    fetch_model_command = [
        'bazel',
        'build',
        external_file,
    ]
    if subprocess.call(fetch_model_command) != 0:
      sys.exit(-1)
    _copy_to_pip_src_dir(external_file)

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
