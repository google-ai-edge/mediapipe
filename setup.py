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

Setup for MediaPipe package with setuptools.
"""

import glob
import os
import platform
import posixpath
import re
import shutil
import subprocess
import sys

import setuptools
from setuptools.command import build_ext
from setuptools.command import build_py
from setuptools.command import install

__version__ = 'dev'
IS_WINDOWS = (platform.system() == 'Windows')
MP_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
MP_DIR_INIT_PY = os.path.join(MP_ROOT_PATH, 'mediapipe/__init__.py')
MP_THIRD_PARTY_BUILD = os.path.join(MP_ROOT_PATH, 'third_party/BUILD')
MP_ROOT_INIT_PY = os.path.join(MP_ROOT_PATH, '__init__.py')


def _normalize_path(path):
  return path.replace('\\', '/') if IS_WINDOWS else path


def _get_backup_file(path):
  return path + '.backup'


def _parse_requirements(path):
  with open(os.path.join(MP_ROOT_PATH, path)) as f:
    return [
        line.rstrip()
        for line in f
        if not (line.isspace() or line.startswith('#'))
    ]


def _get_long_description():
  # Fix the image urls.
  return re.sub(
      r'(docs/images/|docs/images/mobile/)([A-Za-z0-9_]*\.(png|gif))',
      r'https://github.com/google/mediapipe/blob/master/\g<1>\g<2>?raw=true',
      open(os.path.join(MP_ROOT_PATH, 'README.md'),
           'rb').read().decode('utf-8'))


def _check_bazel():
  """Check Bazel binary as well as its version."""

  if not shutil.which('bazel'):
    sys.stderr.write('could not find bazel executable. Please install bazel to'
                     'build the MediaPipe Python package.')
    sys.exit(-1)
  try:
    bazel_version_info = subprocess.check_output(['bazel', '--version'])
  except subprocess.CalledProcessError as e:
    sys.stderr.write('fail to get bazel version by $ bazel --version: ' +
                     str(e.output))
    sys.exit(-1)
  bazel_version_info = bazel_version_info.decode('UTF-8').strip()
  version = bazel_version_info.split('bazel ')[1].split('-')[0]
  version_segments = version.split('.')
  # Treat "0.24" as "0.24.0"
  if len(version_segments) == 2:
    version_segments.append('0')
  for seg in version_segments:
    if not seg.isdigit():
      sys.stderr.write('invalid bazel version number: %s\n' % version_segments)
      sys.exit(-1)
  bazel_version = int(''.join(['%03d' % int(seg) for seg in version_segments]))
  if bazel_version < 3004000:
    sys.stderr.write(
        'the current bazel version is older than the minimum version that MediaPipe can support. Please upgrade bazel.'
    )
    sys.exit(-1)


def _modify_opencv_cmake_rule(link_opencv):
  """Modify opencv_cmake rule to build the static opencv libraries."""

  # Ask the opencv_cmake rule to build the static opencv libraries for the
  # mediapipe python package. By doing this, we can avoid copying the opencv
  # .so file into the package.
  # On Windows, the opencv_cmake rule may need Visual Studio to compile OpenCV
  # from source. For simplicity, we continue to link the prebuilt version of
  # the OpenCV library through "@windows_opencv//:opencv".
  if not link_opencv and not IS_WINDOWS:
    content = open(MP_THIRD_PARTY_BUILD,
                   'r').read().replace('OPENCV_SHARED_LIBS = True',
                                       'OPENCV_SHARED_LIBS = False')
    shutil.move(MP_THIRD_PARTY_BUILD, _get_backup_file(MP_THIRD_PARTY_BUILD))
    build_file = open(MP_THIRD_PARTY_BUILD, 'w')
    build_file.write(content)
    build_file.close()


def _add_mp_init_files():
  """Add __init__.py to mediapipe root directories to make the subdirectories indexable."""
  open(MP_ROOT_INIT_PY, 'w').close()
  # Save the original mediapipe/__init__.py file.
  shutil.copyfile(MP_DIR_INIT_PY, _get_backup_file(MP_DIR_INIT_PY))
  mp_dir_init_file = open(MP_DIR_INIT_PY, 'a')
  mp_dir_init_file.writelines([
      '\n', 'from mediapipe.python import *\n',
      'import mediapipe.python.solutions as solutions', '\n\n',
      '__version__ = \'{}\''.format(__version__), '\n'
  ])
  mp_dir_init_file.close()


class GeneratePyProtos(build_ext.build_ext):
  """Generate MediaPipe Python protobuf files by Protocol Compiler."""

  def run(self):
    if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
      self._protoc = os.environ['PROTOC']
    else:
      self._protoc = shutil.which('protoc')
    if self._protoc is None:
      sys.stderr.write(
          'protoc is not found. Please run \'apt install -y protobuf'
          '-compiler\' (linux) or \'brew install protobuf\'(macos) to install '
          'protobuf compiler binary.')
      sys.exit(-1)

    # Add __init__.py to mediapipe proto directories to make the py protos
    # indexable.
    proto_dirs = ['mediapipe/calculators'] + [
        x[0] for x in os.walk('mediapipe/modules')
    ] + [x[0] for x in os.walk('mediapipe/tasks/cc')]
    for proto_dir in proto_dirs:
      self._add_empty_init_file(
          os.path.abspath(
              os.path.join(MP_ROOT_PATH, self.build_lib, proto_dir,
                           '__init__.py')))

    # Build framework and calculator py protos.
    for pattern in [
        'mediapipe/framework/**/*.proto', 'mediapipe/calculators/**/*.proto',
        'mediapipe/gpu/**/*.proto', 'mediapipe/modules/**/*.proto',
        'mediapipe/tasks/cc/**/*.proto', 'mediapipe/util/**/*.proto'
    ]:
      for proto_file in glob.glob(pattern, recursive=True):
        # Ignore test protos.
        if proto_file.endswith('test.proto'):
          continue
        # Ignore tensorflow protos in mediapipe/calculators/tensorflow.
        if 'tensorflow' in proto_file:
          continue
        # Ignore testdata dir.
        if 'testdata' in proto_file:
          continue
        self._add_empty_init_file(
            os.path.abspath(
                os.path.join(MP_ROOT_PATH, self.build_lib,
                             os.path.dirname(proto_file), '__init__.py')))
        self._generate_proto(proto_file)

  def _add_empty_init_file(self, init_file):
    init_py_dir = os.path.dirname(init_file)
    if not os.path.exists(init_py_dir):
      os.makedirs(init_py_dir)
    if not os.path.exists(init_file):
      open(init_file, 'w').close()

  def _generate_proto(self, source):
    """Invokes the Protocol Compiler to generate a _pb2.py."""
    output = os.path.join(self.build_lib, source.replace('.proto', '_pb2.py'))
    if not os.path.exists(output):
      sys.stderr.write('generating proto file: %s\n' % output)
      protoc_command = [
          self._protoc, '-I.',
          '--python_out=' + os.path.abspath(self.build_lib), source
      ]
      if subprocess.call(protoc_command) != 0:
        sys.exit(-1)


class BuildModules(build_ext.build_ext):
  """Build binary graphs and download external files of various MediaPipe modules."""

  user_options = build_ext.build_ext.user_options + [
      ('link-opencv', None, 'if true, build opencv from source.'),
  ]
  boolean_options = build_ext.build_ext.boolean_options + ['link-opencv']

  def initialize_options(self):
    self.link_opencv = False
    build_ext.build_ext.initialize_options(self)

  def finalize_options(self):
    build_ext.build_ext.finalize_options(self)

  def run(self):
    _check_bazel()
    external_files = [
        'face_detection/face_detection_full_range_sparse.tflite',
        'face_detection/face_detection_short_range.tflite',
        'face_landmark/face_landmark.tflite',
        'face_landmark/face_landmark_with_attention.tflite',
        'hand_landmark/hand_landmark_full.tflite',
        'hand_landmark/hand_landmark_lite.tflite',
        'holistic_landmark/hand_recrop.tflite',
        'iris_landmark/iris_landmark.tflite',
        'palm_detection/palm_detection_full.tflite',
        'palm_detection/palm_detection_lite.tflite',
        'pose_detection/pose_detection.tflite',
        'pose_landmark/pose_landmark_full.tflite',
        'selfie_segmentation/selfie_segmentation.tflite',
        'selfie_segmentation/selfie_segmentation_landscape.tflite',
    ]
    for elem in external_files:
      external_file = os.path.join('mediapipe/modules/', elem)
      sys.stderr.write('downloading file: %s\n' % external_file)
      self._download_external_file(external_file)

    binary_graphs = [
        'face_detection/face_detection_short_range_cpu',
        'face_detection/face_detection_full_range_cpu',
        'face_landmark/face_landmark_front_cpu',
        'hand_landmark/hand_landmark_tracking_cpu',
        'holistic_landmark/holistic_landmark_cpu', 'objectron/objectron_cpu',
        'pose_landmark/pose_landmark_cpu',
        'selfie_segmentation/selfie_segmentation_cpu'
    ]
    for elem in binary_graphs:
      binary_graph = os.path.join('mediapipe/modules/', elem)
      sys.stderr.write('generating binarypb: %s\n' % binary_graph)
      self._generate_binary_graph(binary_graph)

  def _download_external_file(self, external_file):
    """Download an external file from GCS via Bazel."""

    fetch_model_command = [
        'bazel',
        'build',
        external_file,
    ]
    if subprocess.call(fetch_model_command) != 0:
      sys.exit(-1)
    self._copy_to_build_lib_dir(external_file)

  def _generate_binary_graph(self, binary_graph_target):
    """Generate binary graph for a particular MediaPipe binary graph target."""

    bazel_command = [
        'bazel',
        'build',
        '--compilation_mode=opt',
        '--copt=-DNDEBUG',
        '--define=MEDIAPIPE_DISABLE_GPU=1',
        '--action_env=PYTHON_BIN_PATH=' + _normalize_path(sys.executable),
        binary_graph_target,
    ]
    if not self.link_opencv and not IS_WINDOWS:
      bazel_command.append('--define=OPENCV=source')
    if subprocess.call(bazel_command) != 0:
      sys.exit(-1)
    self._copy_to_build_lib_dir(binary_graph_target + '.binarypb')

  def _copy_to_build_lib_dir(self, file):
    """Copy a file from bazel-bin to the build lib dir."""
    dst = os.path.join(self.build_lib + '/', file)
    dst_dir = os.path.dirname(dst)
    if not os.path.exists(dst_dir):
      os.makedirs(dst_dir)
    shutil.copyfile(os.path.join('bazel-bin/', file), dst)


class BazelExtension(setuptools.Extension):
  """A C/C++ extension that is defined as a Bazel BUILD target."""

  def __init__(self, bazel_target, target_name=''):
    self.bazel_target = bazel_target
    self.relpath, self.target_name = (
        posixpath.relpath(bazel_target, '//').split(':'))
    if target_name:
      self.target_name = target_name
    ext_name = os.path.join(
        self.relpath.replace(posixpath.sep, os.path.sep), self.target_name)
    setuptools.Extension.__init__(self, ext_name, sources=[])


class BuildExtension(build_ext.build_ext):
  """A command that runs Bazel to build a C/C++ extension."""

  user_options = build_ext.build_ext.user_options + [
      ('link-opencv', None, 'if true, build opencv from source.'),
  ]
  boolean_options = build_ext.build_ext.boolean_options + ['link-opencv']

  def initialize_options(self):
    self.link_opencv = False
    build_ext.build_ext.initialize_options(self)

  def finalize_options(self):
    build_ext.build_ext.finalize_options(self)

  def run(self):
    _check_bazel()
    for ext in self.extensions:
      self._build_binary(ext)
    build_ext.build_ext.run(self)

  def _build_binary(self, ext):
    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)
    bazel_command = [
        'bazel',
        'build',
        '--compilation_mode=opt',
        '--copt=-DNDEBUG',
        '--define=MEDIAPIPE_DISABLE_GPU=1',
        '--action_env=PYTHON_BIN_PATH=' + _normalize_path(sys.executable),
        str(ext.bazel_target + '.so'),
    ]
    if not self.link_opencv and not IS_WINDOWS:
      bazel_command.append('--define=OPENCV=source')
    if subprocess.call(bazel_command) != 0:
      sys.exit(-1)
    ext_bazel_bin_path = os.path.join('bazel-bin', ext.relpath,
                                      ext.target_name + '.so')
    ext_dest_path = self.get_ext_fullpath(ext.name)
    ext_dest_dir = os.path.dirname(ext_dest_path)
    if not os.path.exists(ext_dest_dir):
      os.makedirs(ext_dest_dir)
    shutil.copyfile(ext_bazel_bin_path, ext_dest_path)
    if IS_WINDOWS:
      for opencv_dll in glob.glob(
          os.path.join('bazel-bin', ext.relpath, '*opencv*.dll')):
        shutil.copy(opencv_dll, ext_dest_dir)


class BuildPy(build_py.build_py):
  """Build command that generates protos, builds binary graphs and extension, builds python source, and performs a cleanup afterwards."""

  user_options = build_py.build_py.user_options + [
      ('link-opencv', None, 'if true, use the installed opencv library.'),
  ]
  boolean_options = build_py.build_py.boolean_options + ['link-opencv']

  def initialize_options(self):
    self.link_opencv = False
    build_py.build_py.initialize_options(self)

  def finalize_options(self):
    build_py.build_py.finalize_options(self)

  def run(self):
    _modify_opencv_cmake_rule(self.link_opencv)
    _add_mp_init_files()
    build_modules_obj = self.distribution.get_command_obj('build_modules')
    build_modules_obj.link_opencv = self.link_opencv
    build_ext_obj = self.distribution.get_command_obj('build_ext')
    build_ext_obj.link_opencv = self.link_opencv
    self.run_command('gen_protos')
    self.run_command('build_modules')
    self.run_command('build_ext')
    build_py.build_py.run(self)
    self.run_command('restore')


class Install(install.install):
  """Install command that generates protos, builds binary graphs and extension, builds python source, and performs a cleanup afterwards."""

  user_options = install.install.user_options + [
      ('link-opencv', None, 'if true, use the installed opencv library.'),
  ]
  boolean_options = install.install.boolean_options + ['link-opencv']

  def initialize_options(self):
    self.link_opencv = False
    install.install.initialize_options(self)

  def finalize_options(self):
    install.install.finalize_options(self)

  def run(self):
    build_py_obj = self.distribution.get_command_obj('build_py')
    build_py_obj.link_opencv = self.link_opencv
    install.install.run(self)


class Restore(setuptools.Command):
  """Restore the modified mediapipe source files."""

  user_options = []

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    # Restore the original init file from the backup.
    if os.path.exists(_get_backup_file(MP_DIR_INIT_PY)):
      os.remove(MP_DIR_INIT_PY)
      shutil.move(_get_backup_file(MP_DIR_INIT_PY), MP_DIR_INIT_PY)
    # Restore the original BUILD file from the backup.
    if os.path.exists(_get_backup_file(MP_THIRD_PARTY_BUILD)):
      os.remove(MP_THIRD_PARTY_BUILD)
      shutil.move(_get_backup_file(MP_THIRD_PARTY_BUILD), MP_THIRD_PARTY_BUILD)
    os.remove(MP_ROOT_INIT_PY)


setuptools.setup(
    name='mediapipe',
    version=__version__,
    url='https://github.com/google/mediapipe',
    description='MediaPipe is the simplest way for researchers and developers to build world-class ML solutions and applications for mobile, edge, cloud and the web.',
    author='The MediaPipe Authors',
    author_email='mediapipe@google.com',
    long_description=_get_long_description(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(exclude=['mediapipe.examples.desktop.*']),
    install_requires=_parse_requirements('requirements.txt'),
    cmdclass={
        'build_py': BuildPy,
        'gen_protos': GeneratePyProtos,
        'build_modules': BuildModules,
        'build_ext': BuildExtension,
        'install': Install,
        'restore': Restore,
    },
    ext_modules=[
        BazelExtension('//mediapipe/python:_framework_bindings'),
    ],
    zip_safe=False,
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
        'Programming Language :: Python :: 3.7',
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
    keywords='mediapipe',
)
