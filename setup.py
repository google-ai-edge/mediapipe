"""Copyright 2020-2021 The MediaPipe Authors.

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
import setuptools.command.build_ext as build_ext
import setuptools.command.build_py as build_py
import setuptools.command.install as install

__version__ = 'dev'
IS_WINDOWS = (platform.system() == 'Windows')
MP_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
MP_DIR_INIT_PY = os.path.join(MP_ROOT_PATH, 'mediapipe/__init__.py')
MP_THIRD_PARTY_BUILD = os.path.join(MP_ROOT_PATH, 'third_party/BUILD')
DIR_INIT_PY_FILES = [
    os.path.join(MP_ROOT_PATH, '__init__.py'),
    os.path.join(MP_ROOT_PATH, 'mediapipe/calculators/__init__.py'),
    os.path.join(MP_ROOT_PATH, 'mediapipe/modules/__init__.py'),
    os.path.join(MP_ROOT_PATH,
                 'mediapipe/modules/holistic_landmark/__init__.py'),
    os.path.join(MP_ROOT_PATH, 'mediapipe/modules/objectron/__init__.py')
]


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


class GeneratePyProtos(setuptools.Command):
  """Generate MediaPipe Python protobuf files by Protocol Compiler."""

  user_options = []

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

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
    self._modify_inits()
    # Build framework and calculator protos.
    for pattern in [
        'mediapipe/framework/**/*.proto', 'mediapipe/calculators/**/*.proto',
        'mediapipe/gpu/**/*.proto', 'mediapipe/modules/**/*.proto',
        'mediapipe/util/**/*.proto'
    ]:
      for proto_file in glob.glob(pattern, recursive=True):
        proto_dir = os.path.dirname(os.path.abspath(proto_file))
        # Ignore test protos.
        if proto_file.endswith('test.proto'):
          continue
        # Ignore tensorflow protos in mediapipe/calculators/tensorflow.
        if 'tensorflow' in proto_dir:
          continue
        # Ignore testdata dir.
        if proto_dir.endswith('testdata'):
          continue
        init_py = os.path.join(proto_dir, '__init__.py')
        if not os.path.exists(init_py):
          sys.stderr.write('adding __init__ file: %s\n' % init_py)
          open(init_py, 'w').close()
        self._generate_proto(proto_file)

  def _modify_inits(self):
    # Add __init__.py to make the dirs indexable.
    for init_py in DIR_INIT_PY_FILES:
      if not os.path.exists(init_py):
        sys.stderr.write('adding __init__ file: %s\n' % init_py)
        open(init_py, 'w').close()
    # Save the original init file.
    shutil.copyfile(MP_DIR_INIT_PY, _get_backup_file(MP_DIR_INIT_PY))
    mp_dir_init_file = open(MP_DIR_INIT_PY, 'a')
    mp_dir_init_file.writelines(
        ['\n', 'from mediapipe.python import *\n',
         'import mediapipe.python.solutions as solutions',
         '\n'])
    mp_dir_init_file.close()

  def _generate_proto(self, source):
    """Invokes the Protocol Compiler to generate a _pb2.py."""

    output = source.replace('.proto', '_pb2.py')
    sys.stderr.write('generating proto file: %s\n' % output)
    if (not os.path.exists(output) or
        (os.path.exists(source) and
         os.path.getmtime(source) > os.path.getmtime(output))):

      if not os.path.exists(source):
        sys.stderr.write('cannot find required file: %s\n' % source)
        sys.exit(-1)

      protoc_command = [self._protoc, '-I.', '--python_out=.', source]
      if subprocess.call(protoc_command) != 0:
        sys.exit(-1)


class BuildBinaryGraphs(build_ext.build_ext):
  """Build MediaPipe solution binary graphs."""

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
    binary_graphs = [
        'face_detection/face_detection_short_range_cpu',
        'face_detection/face_detection_full_range_cpu',
        'face_landmark/face_landmark_front_cpu',
        'hand_landmark/hand_landmark_tracking_cpu',
        'holistic_landmark/holistic_landmark_cpu', 'objectron/objectron_cpu',
        'pose_landmark/pose_landmark_cpu',
        'selfie_segmentation/selfie_segmentation_cpu'
    ]
    for binary_graph in binary_graphs:
      sys.stderr.write('generating binarypb: %s\n' %
                       os.path.join('mediapipe/modules/', binary_graph))
      self._generate_binary_graph(binary_graph)

  def _generate_binary_graph(self, graph_path):
    """Generate binary graph for a particular MediaPipe binary graph target."""

    bazel_command = [
        'bazel',
        'build',
        '--compilation_mode=opt',
        '--copt=-DNDEBUG',
        '--define=MEDIAPIPE_DISABLE_GPU=1',
        '--action_env=PYTHON_BIN_PATH=' + _normalize_path(sys.executable),
        os.path.join('mediapipe/modules/', graph_path),
    ]
    if not self.link_opencv and not IS_WINDOWS:
      bazel_command.append('--define=OPENCV=source')
    if subprocess.call(bazel_command) != 0:
      sys.exit(-1)
    output_name = graph_path + '.binarypb'
    output_file = os.path.join('mediapipe/modules', output_name)
    shutil.copyfile(
        os.path.join('bazel-bin/mediapipe/modules/', output_name), output_file)


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
    build_binary_graphs_obj = self.distribution.get_command_obj(
        'build_binary_graphs')
    build_binary_graphs_obj.link_opencv = self.link_opencv
    build_ext_obj = self.distribution.get_command_obj('build_ext')
    build_ext_obj.link_opencv = self.link_opencv
    self.run_command('build_binary_graphs')
    self.run_command('build_ext')
    build_py.build_py.run(self)
    self.run_command('remove_generated')


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
    _modify_opencv_cmake_rule(self.link_opencv)
    build_binary_graphs_obj = self.distribution.get_command_obj(
        'build_binary_graphs')
    build_binary_graphs_obj.link_opencv = self.link_opencv
    build_ext_obj = self.distribution.get_command_obj('build_ext')
    build_ext_obj.link_opencv = self.link_opencv
    self.run_command('build_binary_graphs')
    self.run_command('build_ext')
    install.install.run(self)
    self.run_command('remove_generated')


class RemoveGenerated(setuptools.Command):
  """Remove the generated files."""

  user_options = []

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    for pattern in [
        'mediapipe/calculators/**/*pb2.py',
        'mediapipe/framework/**/*pb2.py',
        'mediapipe/gpu/**/*pb2.py',
        'mediapipe/modules/**/*pb2.py',
        'mediapipe/util/**/*pb2.py',
    ]:
      for py_file in glob.glob(pattern, recursive=True):
        sys.stderr.write('removing generated files: %s\n' % py_file)
        os.remove(py_file)
        init_py = os.path.join(
            os.path.dirname(os.path.abspath(py_file)), '__init__.py')
        if os.path.exists(init_py):
          sys.stderr.write('removing __init__ file: %s\n' % init_py)
          os.remove(init_py)
    for binarypb_file in glob.glob(
        'mediapipe/modules/**/*.binarypb', recursive=True):
      sys.stderr.write('removing generated binary graphs: %s\n' % binarypb_file)
      os.remove(binarypb_file)
    # Restore the original init file from the backup.
    if os.path.exists(_get_backup_file(MP_DIR_INIT_PY)):
      os.remove(MP_DIR_INIT_PY)
      shutil.move(_get_backup_file(MP_DIR_INIT_PY), MP_DIR_INIT_PY)
    # Restore the original BUILD file from the backup.
    if os.path.exists(_get_backup_file(MP_THIRD_PARTY_BUILD)):
      os.remove(MP_THIRD_PARTY_BUILD)
      shutil.move(_get_backup_file(MP_THIRD_PARTY_BUILD), MP_THIRD_PARTY_BUILD)
    for init_py in DIR_INIT_PY_FILES:
      if os.path.exists(init_py):
        os.remove(init_py)


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
        'build_binary_graphs': BuildBinaryGraphs,
        'build_ext': BuildExtension,
        'install': Install,
        'remove_generated': RemoveGenerated,
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
