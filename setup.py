"""Copyright 2020 The MediaPipe Authors.

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

from distutils import spawn
import distutils.command.build as build
import distutils.command.clean as clean
import glob
import os
import posixpath
import re
import shutil
import subprocess
import sys

import setuptools
import setuptools.command.build_ext as build_ext
import setuptools.command.install as install

__version__ = '0.7'
MP_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_INIT_PY = os.path.join(MP_ROOT_PATH, '__init__.py')
MP_DIR_INIT_PY = os.path.join(MP_ROOT_PATH, 'mediapipe/__init__.py')
MP_DIR_INIT_PY_BACKUP = os.path.join(MP_ROOT_PATH,
                                     'mediapipe/__init__.py.backup')
MP_THIRD_PARTY_BUILD = os.path.join(MP_ROOT_PATH, 'third_party/BUILD')
MP_THIRD_PARTY_BUILD_BACKUP = os.path.join(MP_ROOT_PATH,
                                           'third_party/BUILD.backup')
if not os.path.exists(ROOT_INIT_PY):
  open(ROOT_INIT_PY, 'w').close()


def _parse_requirements(path):
  with open(os.path.join(MP_ROOT_PATH, path)) as f:
    return [
        line.rstrip()
        for line in f
        if not (line.isspace() or line.startswith('#'))
    ]


def _get_long_description():
  # fix the image urls.
  return re.sub(
      r'(docs/images/|docs/images/mobile/)([A-Za-z0-9_]*\.(png|gif))',
      r'https://github.com/google/mediapipe/blob/master/\g<1>\g<2>?raw=true',
      open(os.path.join(MP_ROOT_PATH, 'README.md')).read())


def _check_bazel():
  """Check Bazel binary as well as its version."""

  if not spawn.find_executable('bazel'):
    sys.stderr.write('could not find bazel executable. Please install bazel to'
                     'build the MediaPipe Python package.')
    sys.exit(-1)
  try:
    bazel_version_info = subprocess.check_output(['bazel', '--version'])
  except subprocess.CalledProcessError:
    sys.stderr.write('fail to get bazel version by $ bazel --version.')
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
  if bazel_version < 2000000:
    sys.stderr.write(
        'the current bazel version is older than the minimum version that MediaPipe can support. Please upgrade bazel.'
    )


class ModifyInitFiles(setuptools.Command):
  """Modify the init files for building MediaPipe Python package."""

  user_options = []

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    # Save the original init file.
    shutil.copyfile(MP_DIR_INIT_PY, MP_DIR_INIT_PY_BACKUP)
    mp_dir_init_file = open(MP_DIR_INIT_PY, 'a')
    mp_dir_init_file.writelines([
        '\n', 'import mediapipe.examples.python as examples\n',
        'from mediapipe.python import *\n', '\n'
    ])
    mp_dir_init_file.close()


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
      self._protoc = spawn.find_executable('protoc')
    if self._protoc is None:
      sys.stderr.write(
          'protoc is not found. Please run \'apt install -y protobuf'
          '-compiler\' (linux) or \'brew install protobuf\'(macos) to install '
          'protobuf compiler binary.')
      sys.exit(-1)
    # Build framework protos.
    for proto_file in glob.glob(
        'mediapipe/framework/**/*.proto', recursive=True):
      if proto_file.endswith('test.proto'):
        continue
      proto_dir = os.path.dirname(os.path.abspath(proto_file))
      if proto_dir.endswith('testdata'):
        continue
      init_py = os.path.join(proto_dir, '__init__.py')
      if not os.path.exists(init_py):
        sys.stderr.write('adding necessary __init__ file: %s\n' % init_py)
        open(init_py, 'w').close()
      self._generate_proto(proto_file)

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


class BuildBinaryGraphs(build.build):
  """Build binary graphs for Python examples."""

  def run(self):
    _check_bazel()
    binary_graphs = ['pose_tracking/upper_body_pose_tracking_cpu_binary_graph']
    for binary_graph in binary_graphs:
      sys.stderr.write('generating binarypb: %s\n' %
                       os.path.join('mediapipe/graphs/', binary_graph))
      self._generate_binary_graph(binary_graph)

  def _generate_binary_graph(self, graph_path):
    """Generate binary graph for a particular MediaPipe binary graph target."""

    bazel_command = [
        'bazel',
        'build',
        '--compilation_mode=opt',
        '--define=MEDIAPIPE_DISABLE_GPU=1',
        '--action_env=PYTHON_BIN_PATH=' + sys.executable,
        os.path.join('mediapipe/graphs/', graph_path),
    ]
    if subprocess.call(bazel_command) != 0:
      sys.exit(-1)
    output_name = graph_path.replace('_binary_graph', '.binarypb')
    output_file = os.path.join('mediapipe/graphs', output_name)
    shutil.copyfile(
        os.path.join('bazel-bin/mediapipe/graphs/', output_name), output_file)


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


class BuildBazelExtension(build_ext.build_ext):
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
      self.bazel_build(ext)
    build_ext.build_ext.run(self)

  def bazel_build(self, ext):
    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)
    bazel_argv = [
        'bazel',
        'build',
        '--compilation_mode=opt',
        '--define=MEDIAPIPE_DISABLE_GPU=1',
        '--action_env=PYTHON_BIN_PATH=' + sys.executable,
        str(ext.bazel_target + '.so'),
    ]
    if not self.link_opencv:
      # Ask the opencv_cmake rule to build the static opencv libraries for
      # mediapipe python package. By doing this, we can avoid copying the opencv
      # .so file into the package.
      content = open(MP_THIRD_PARTY_BUILD,
                     'r').read().replace('OPENCV_SHARED_LIBS = True',
                                         'OPENCV_SHARED_LIBS = False')
      shutil.move(MP_THIRD_PARTY_BUILD, MP_THIRD_PARTY_BUILD_BACKUP)
      build_file = open(MP_THIRD_PARTY_BUILD, 'w')
      build_file.write(content)
      build_file.close()
      bazel_argv.append('--define=OPENCV=source')
    self.spawn(bazel_argv)
    ext_bazel_bin_path = os.path.join('bazel-bin', ext.relpath,
                                      ext.target_name + '.so')
    ext_dest_path = self.get_ext_fullpath(ext.name)
    ext_dest_dir = os.path.dirname(ext_dest_path)
    if not os.path.exists(ext_dest_dir):
      os.makedirs(ext_dest_dir)
    shutil.copyfile(ext_bazel_bin_path, ext_dest_path)


class Build(build.build):
  """Build command that builds binary graphs and extension and does a cleanup afterwards."""

  user_options = build.build.user_options + [
      ('link-opencv', None, 'if true, use the installed opencv library.'),
  ]
  boolean_options = build.build.boolean_options + ['link-opencv']

  def initialize_options(self):
    self.link_opencv = False
    build.build.initialize_options(self)

  def finalize_options(self):
    build.build.finalize_options(self)

  def run(self):
    build_ext_obj = self.distribution.get_command_obj('build_ext')
    build_ext_obj.link_opencv = self.link_opencv
    self.run_command('build_binary_graphs')
    self.run_command('build_ext')
    self.run_command('modify_inits')
    build.build.run(self)
    self.run_command('remove_generated')


class Install(install.install):
  """Install command that builds binary graphs and extension and does a cleanup afterwards."""

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
    build_ext_obj = self.distribution.get_command_obj('build_ext')
    build_ext_obj.link_opencv = self.link_opencv
    self.run_command('build_binary_graphs')
    self.run_command('build_ext')
    self.run_command('modify_inits')
    install.install.run(self)
    self.run_command('remove_generated')


class RemoveGenerated(clean.clean):
  """Remove the generated files."""

  def run(self):
    for py_file in glob.glob('mediapipe/framework/**/*.py', recursive=True):
      sys.stderr.write('removing generated files: %s\n' % py_file)
      os.remove(py_file)
    for binarypb_file in glob.glob(
        'mediapipe/graphs/**/*.binarypb', recursive=True):
      sys.stderr.write('removing generated binary graphs: %s\n' % binarypb_file)
      os.remove(binarypb_file)
    # Restore the original init file from the backup.
    if os.path.exists(MP_DIR_INIT_PY_BACKUP):
      os.remove(MP_DIR_INIT_PY)
      shutil.move(MP_DIR_INIT_PY_BACKUP, MP_DIR_INIT_PY)
    # Restore the original BUILD file from the backup.
    if os.path.exists(MP_THIRD_PARTY_BUILD_BACKUP):
      os.remove(MP_THIRD_PARTY_BUILD)
      shutil.move(MP_THIRD_PARTY_BUILD_BACKUP, MP_THIRD_PARTY_BUILD)
    clean.clean.run(self)


setuptools.setup(
    name='mediapipe',
    version=__version__,
    url='https://github.com/google/mediapipe',
    description='MediaPipe is the simplest way for researchers and developers to build world-class ML solutions and applications for mobile, edge, cloud and the web.',
    author='MediaPipe Authors',
    author_email='mediapipe@google.com',
    long_description=_get_long_description(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(exclude=['mediapipe.examples.desktop.*']),
    install_requires=_parse_requirements('requirements.txt'),
    cmdclass={
        'build': Build,
        'gen_protos': GeneratePyProtos,
        'modify_inits': ModifyInitFiles,
        'build_binary_graphs': BuildBinaryGraphs,
        'build_ext': BuildBazelExtension,
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
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
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

os.remove(ROOT_INIT_PY)
