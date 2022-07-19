#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2016 The Tulsi Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bridge between Xcode and Bazel for the "build" action."""

import atexit
import errno
import fcntl
import hashlib
import inspect
import io
import json
import os
import pipes
import plistlib
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
import zipfile

from apfs_clone_copy import CopyOnWrite
import bazel_build_events
import bazel_build_settings
import bazel_options
from bootstrap_lldbinit import BootstrapLLDBInit
from bootstrap_lldbinit import TULSI_LLDBINIT_FILE
import tulsi_logging
from update_symbol_cache import UpdateSymbolCache


# List of frameworks that Xcode injects into test host targets that should be
# re-signed when running the tests on devices.
XCODE_INJECTED_FRAMEWORKS = [
    'libXCTestBundleInject.dylib',
    'libXCTestSwiftSupport.dylib',
    'IDEBundleInjection.framework',
    'XCTAutomationSupport.framework',
    'XCTest.framework',
    'XCTestCore.framework',
    'XCUnit.framework',
    'XCUIAutomation.framework',
]

_logger = None


def _PrintUnbuffered(msg):
  sys.stdout.write('%s\n' % msg)
  sys.stdout.flush()


def _PrintXcodeWarning(msg):
  sys.stdout.write(':: warning: %s\n' % msg)
  sys.stdout.flush()


def _PrintXcodeError(msg):
  sys.stderr.write(':: error: %s\n' % msg)
  sys.stderr.flush()


def _Fatal(msg, fatal_frame=None):
  """Print a fatal error pointing to the failure line inside the script."""
  if not fatal_frame:
    fatal_frame = inspect.currentframe().f_back
  filename, line_number, _, _, _ = inspect.getframeinfo(fatal_frame)
  _PrintUnbuffered('%s:%d: error: %s' % (os.path.abspath(filename),
                                         line_number, msg))


CLEANUP_BEP_FILE_AT_EXIT = False


# Function to be called atexit to clean up the BEP file if one is present.
# This is especially useful in cases of abnormal termination (such as what
# happens when Xcode is killed).
def _BEPFileExitCleanup(bep_file_path):
  if not CLEANUP_BEP_FILE_AT_EXIT:
    return
  try:
    os.remove(bep_file_path)
  except OSError as e:
    _PrintXcodeWarning('Failed to remove BEP file from %s. Error: %s' %
                       (bep_file_path, e.strerror))


def _InterruptHandler(signum, frame):
  """Gracefully exit on SIGINT."""
  del signum, frame  # Unused.
  _PrintUnbuffered('Caught interrupt signal. Exiting...')
  sys.exit(0)


def _FindDefaultLldbInit():
  """Returns the path to the primary lldbinit file that Xcode would load or None when no file exists."""
  for lldbinit_shortpath in ['~/.lldbinit-Xcode', '~/.lldbinit']:
    lldbinit_path = os.path.expanduser(lldbinit_shortpath)
    if os.path.isfile(lldbinit_path):
      return lldbinit_path

  return None


class Timer(object):
  """Simple profiler."""

  def __init__(self, action_name, action_id):
    """Creates a new Timer object.

    Args:
      action_name: A human-readable action name, shown in the build log.
      action_id: A machine-readable action identifier, can be used for metrics.

    Returns:
      A Timer instance.

    Raises:
      RuntimeError: if Timer is created without initializing _logger.
    """
    if _logger is None:
      raise RuntimeError('Attempted to create Timer without a logger.')
    self.action_name = action_name
    self.action_id = action_id
    self._start = None

  def Start(self):
    self._start = time.time()
    return self

  def End(self, log_absolute_times=False):
    end = time.time()
    seconds = end - self._start
    if log_absolute_times:
      _logger.log_action(self.action_name, self.action_id, seconds,
                         self._start, end)
    else:
      _logger.log_action(self.action_name, self.action_id, seconds)


def _LockFileCreate():
  # This relies on this script running at the root of the bazel workspace.
  cwd = os.environ['PWD']
  cwd_hash = hashlib.sha256(cwd.encode()).hexdigest()
  return '/tmp/tulsi_bazel_build_{}.lock'.format(cwd_hash)


# Function to be called atexit to release the file lock on script termination.
def _LockFileExitCleanup(lock_file_handle):
  lock_file_handle.close()


def _LockFileAcquire(lock_path):
  """Force script to wait on file lock to serialize build target actions.

  Args:
    lock_path: Path to the lock file.
  """
  _PrintUnbuffered('Queuing Tulsi build...')
  lockfile = open(lock_path, 'w')
  # Register "fclose(...)" as early as possible, before acquiring lock.
  atexit.register(_LockFileExitCleanup, lockfile)
  while True:
    try:
      fcntl.lockf(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
      break
    except IOError as err:
      if err.errno != errno.EAGAIN:
        raise
      else:
        time.sleep(0.1)


class CodesignBundleAttributes(object):
  """Wrapper class for codesigning attributes of a signed bundle."""

  # List of codesigning attributes that this script requires.
  _ATTRIBUTES = ['Authority', 'Identifier', 'TeamIdentifier']

  def __init__(self, codesign_output):
    self.attributes = {}

    pending_attributes = list(self._ATTRIBUTES)
    for line in codesign_output.split('\n'):
      if not pending_attributes:
        break

      for attribute in pending_attributes:
        if line.startswith(attribute):
          value = line[len(attribute) + 1:]
          self.attributes[attribute] = value
          pending_attributes.remove(attribute)
          break

    for attribute in self._ATTRIBUTES:
      if attribute not in self.attributes:
        _PrintXcodeError(
            'Failed to extract %s from %s.\n' % (attribute, codesign_output))

  def Get(self, attribute):
    """Returns the value for the given attribute, or None if it wasn't found."""
    value = self.attributes.get(attribute)
    if attribute not in self._ATTRIBUTES:
      _PrintXcodeError(
          'Attribute %s not declared to be parsed. ' % attribute +
          'Available attributes are %s.\n' % self._ATTRIBUTES)
    return value


class _OptionsParser(object):
  """Handles parsing script options."""

  # List of all supported Xcode configurations.
  KNOWN_CONFIGS = ['Debug', 'Release']

  def __init__(self, build_settings, sdk_version, platform_name, arch):
    self.targets = []
    self.build_settings = build_settings
    self.common_build_options = [
        '--verbose_failures',
        '--bes_outerr_buffer_size=0',  # Don't buffer Bazel output.
    ]

    self.sdk_version = sdk_version
    self.platform_name = platform_name

    if self.platform_name.startswith('watch'):
      config_platform = 'watchos'
    elif self.platform_name.startswith('iphone'):
      config_platform = 'ios'
    elif self.platform_name.startswith('macos'):
      config_platform = 'macos'
    elif self.platform_name.startswith('appletv'):
      config_platform = 'tvos'
    else:
      self._WarnUnknownPlatform()
      config_platform = 'ios'
    self.bazel_build_config = '{}_{}'.format(config_platform, arch)
    if self.bazel_build_config not in build_settings.platformConfigFlags:
      _PrintXcodeError('Unknown active compilation target of "{}". '
                       'Please report a Tulsi bug.'
                       .format(self.bazel_build_config))
      sys.exit(1)

    self.verbose = 0
    self.bazel_bin_path = 'bazel-bin'
    self.bazel_executable = None

  @staticmethod
  def _UsageMessage():
    """Returns a usage message string."""
    usage = textwrap.dedent("""\
      Usage: %s <target> [<target2> ...] --bazel <bazel_binary_path> [options]

      Where options are:
        --verbose [-v]
            Increments the verbosity of the script by one level. This argument
            may be provided multiple times to enable additional output levels.

        --bazel_bin_path <path>
            Path at which Bazel-generated artifacts may be retrieved.
      """ % sys.argv[0])

    return usage

  def ParseOptions(self, args):
    """Parses arguments, returning (message, exit_code)."""

    bazel_executable_index = args.index('--bazel')

    self.targets = args[:bazel_executable_index]
    if not self.targets or len(args) < bazel_executable_index + 2:
      return (self._UsageMessage(), 10)
    self.bazel_executable = args[bazel_executable_index + 1]

    return self._ParseVariableOptions(args[bazel_executable_index + 2:])

  def GetBaseFlagsForTargets(self, config):
    is_debug = config == 'Debug'
    return self.build_settings.flags_for_target(
        self.targets[0],
        is_debug,
        self.bazel_build_config)

  def GetEnabledFeatures(self):
    """Returns a list of enabled Bazel features for the active target."""
    return self.build_settings.features_for_target(self.targets[0])

  def GetBazelOptions(self, config):
    """Returns the full set of build options for the given config."""
    bazel, start_up, build = self.GetBaseFlagsForTargets(config)
    all_build = []
    all_build.extend(self.common_build_options)
    all_build.extend(build)

    xcode_version_flag = self._ComputeXcodeVersionFlag()
    if xcode_version_flag:
      all_build.append('--xcode_version=%s' % xcode_version_flag)

    return bazel, start_up, all_build

  def _WarnUnknownPlatform(self):
    _PrintUnbuffered('Warning: unknown platform "%s" will be treated as '
                     'iOS' % self.platform_name)

  def _ParseVariableOptions(self, args):
    """Parses flag-based args, returning (message, exit_code)."""

    verbose_re = re.compile('-(v+)$')

    while args:
      arg = args[0]
      args = args[1:]

      if arg == '--bazel_bin_path':
        if not args:
          return ('Missing required parameter for %s' % arg, 2)
        self.bazel_bin_path = args[0]
        args = args[1:]

      elif arg == '--verbose':
        self.verbose += 1

      else:
        match = verbose_re.match(arg)
        if match:
          self.verbose += len(match.group(1))
        else:
          return ('Unknown option "%s"\n%s' % (arg, self._UsageMessage()), 1)

    return (None, 0)

  @staticmethod
  def _GetXcodeBuildVersionString():
    """Returns Xcode build version from the environment as a string."""
    return os.environ['XCODE_PRODUCT_BUILD_VERSION']

  @staticmethod
  def _GetXcodeVersionString():
    """Returns Xcode version info from the Xcode's version.plist.

    Just reading XCODE_VERSION_ACTUAL from the environment seems like
    a more reasonable implementation, but has shown to be unreliable,
    at least when using Xcode 11.3.1 and opening the project within an
    Xcode workspace.
    """
    developer_dir = os.environ['DEVELOPER_DIR']
    app_dir = developer_dir.split('.app')[0] + '.app'
    version_plist_path = os.path.join(app_dir, 'Contents', 'version.plist')
    try:
      with open(version_plist_path, 'rb') as f:
        plist = plistlib.load(f)
    except IOError:
      _PrintXcodeWarning('Tulsi cannot determine Xcode version, error '
                         'reading from {}'.format(version_plist_path))
      return None
    try:
      # Example: "11.3.1", "11.3", "11.0"
      key = 'CFBundleShortVersionString'
      version_string = plist[key]
    except KeyError:
      _PrintXcodeWarning('Tulsi cannot determine Xcode version from {}, no '
                         '"{}" key'.format(version_plist_path, key))
      return None

    # But we need to normalize to major.minor.patch, e.g. 11.3.0 or
    # 11.0.0, so add one or two ".0" if needed (two just in case
    # there is ever just a single version number like "12")
    dots_count = version_string.count('.')
    dot_zeroes_to_add = 2 - dots_count
    version_string += '.0' * dot_zeroes_to_add
    return version_string

  @staticmethod
  def _ComputeXcodeVersionFlag():
    """Returns a string for the --xcode_version build flag, if any.

    The flag should be used if the active Xcode version was not the same one
    used during project generation.

    Note this a best-attempt only; this may not be accurate as Bazel itself
    caches the active DEVELOPER_DIR path and the user may have changed their
    installed Xcode version.
    """
    xcode_version = _OptionsParser._GetXcodeVersionString()
    build_version = _OptionsParser._GetXcodeBuildVersionString()

    if not xcode_version or not build_version:
      return None

    # Of the form Major.Minor.Fix.Build (new Bazel form) or Major.Min.Fix (old).
    full_bazel_version = os.environ.get('TULSI_XCODE_VERSION')
    if not full_bazel_version:  # Unexpected: Tulsi gen didn't set the flag.
      return xcode_version

    # Newer Bazel versions specify the version as Major.Minor.Fix.Build.
    if full_bazel_version.count('.') == 3:
      components = full_bazel_version.rsplit('.', 1)
      bazel_xcode_version = components[0]
      bazel_build_version = components[1]

      if (xcode_version != bazel_xcode_version
          or build_version != bazel_build_version):
        return '{}.{}'.format(xcode_version, build_version)
      else:
        return None
    else:  # Old version of Bazel. We need to use form Major.Minor.Fix.
      return xcode_version if xcode_version != full_bazel_version else None


class BazelBuildBridge(object):
  """Handles invoking Bazel and unpacking generated binaries."""

  BUILD_EVENTS_FILE = 'build_events.json'

  XCODE_MODULE_CACHE_DIRECTORY = os.path.expanduser(
      '~/Library/Developer/Xcode/DerivedData/ModuleCache.noindex')
  MODULE_CACHE_PRUNER_EXECUTABLE = os.path.expanduser(
      '~/Library/Application Support/Tulsi/Scripts/module_cache_pruner')

  def __init__(self, build_settings):
    self.build_settings = build_settings
    self.verbose = 0
    self.bazel_bin_path = None
    self.codesign_attributes = {}

    self.codesigning_folder_path = os.environ['CODESIGNING_FOLDER_PATH']

    self.xcode_action = os.environ['ACTION']  # The Xcode build action.
    # When invoked as an external build system script, Xcode will set ACTION to
    # an empty string.
    if not self.xcode_action:
      self.xcode_action = 'build'

    if int(os.environ['XCODE_VERSION_MAJOR']) < 900:
      xcode_build_version = os.environ['XCODE_PRODUCT_BUILD_VERSION']
      _PrintXcodeWarning('Tulsi officially supports Xcode 9+. You are using an '
                         'earlier Xcode, build %s.' % xcode_build_version)

    self.tulsi_version = os.environ.get('TULSI_VERSION', 'UNKNOWN')

    self.custom_lldbinit = os.environ.get('TULSI_LLDBINIT_FILE')

    # TODO(b/69857078): Remove this when wrapped_clang is updated.
    self.direct_debug_prefix_map = False
    self.normalized_prefix_map = False

    self.update_symbol_cache = None
    if os.environ.get('TULSI_USE_BAZEL_CACHE_READER') is not None:
      self.update_symbol_cache = UpdateSymbolCache()

    # Path into which generated artifacts should be copied.
    self.built_products_dir = os.environ['BUILT_PRODUCTS_DIR']
    # Path where Xcode expects generated sources to be placed.
    self.derived_sources_folder_path = os.environ.get('DERIVED_SOURCES_DIR')
    # Full name of the target artifact (e.g., "MyApp.app" or "Test.xctest").
    self.full_product_name = os.environ['FULL_PRODUCT_NAME']
    # Whether to generate runfiles for this target.
    self.gen_runfiles = os.environ.get('GENERATE_RUNFILES')
    # Target SDK version.
    self.sdk_version = os.environ.get('SDK_VERSION')
    # TEST_HOST for unit tests.
    self.test_host_binary = os.environ.get('TEST_HOST')
    # Whether this target is a test or not.
    self.is_test = os.environ.get('WRAPPER_EXTENSION') == 'xctest'
    # Target platform.
    self.platform_name = os.environ['PLATFORM_NAME']
    # Type of the target artifact.
    self.product_type = os.environ['PRODUCT_TYPE']
    # Path to the parent of the xcodeproj bundle.
    self.project_dir = os.environ['PROJECT_DIR']
    # Path to the xcodeproj bundle.
    self.project_file_path = os.environ['PROJECT_FILE_PATH']
    # Path to the directory containing the WORKSPACE file.
    self.workspace_root = os.path.abspath(os.environ['TULSI_WR'])
    # Set to the name of the generated bundle for bundle-type targets, None for
    # single file targets (like static libraries).
    self.wrapper_name = os.environ.get('WRAPPER_NAME')
    self.wrapper_suffix = os.environ.get('WRAPPER_SUFFIX', '')

    # Path where Xcode expects the artifacts to be written to. This is not the
    # codesigning_path as device vs simulator builds have different signing
    # requirements, so Xcode expects different paths to be signed. This is
    # mostly apparent on XCUITests where simulator builds set the codesigning
    # path to be the .xctest bundle, but for device builds it is actually the
    # UI runner app (since it needs to be codesigned to run on the device.) The
    # FULL_PRODUCT_NAME variable is a stable path on where to put the expected
    # artifacts. For static libraries (objc_library, swift_library),
    # FULL_PRODUCT_NAME corresponds to the .a file name, which coincides with
    # the expected location for a single artifact output.
    # TODO(b/35811023): Check these paths are still valid.
    self.artifact_output_path = os.path.join(
        os.environ['TARGET_BUILD_DIR'],
        os.environ['FULL_PRODUCT_NAME'])

    # Path to where Xcode expects the binary to be placed.
    self.binary_path = os.path.join(
        os.environ['TARGET_BUILD_DIR'], os.environ['EXECUTABLE_PATH'])

    self.is_simulator = self.platform_name.endswith('simulator')
    self.codesigning_allowed = not self.is_simulator

    # Target architecture.  Must be defined for correct setting of
    # the --cpu flag. Note that Xcode will set multiple values in
    # ARCHS when building for a Generic Device.
    archs = os.environ.get('ARCHS')
    if not archs:
      _PrintXcodeError('Tulsi requires env variable ARCHS to be '
                       'set.  Please file a bug against Tulsi.')
      sys.exit(1)
    arch = archs.split()[-1]
    if self.is_simulator and arch == "arm64":
      self.arch = "sim_" + arch
    else:
      self.arch = arch

    if self.codesigning_allowed:
      platform_prefix = 'iOS'
      if self.platform_name.startswith('macos'):
        platform_prefix = 'macOS'
      entitlements_filename = '%sXCTRunner.entitlements' % platform_prefix
      self.runner_entitlements_template = os.path.join(self.project_file_path,
                                                       '.tulsi',
                                                       'Resources',
                                                       entitlements_filename)

    self.bazel_executable = None

  def Run(self, args):
    """Executes a Bazel build based on the environment and given arguments."""
    if self.xcode_action != 'build':
      sys.stderr.write('Xcode action is %s, ignoring.' % self.xcode_action)
      return 0

    parser = _OptionsParser(self.build_settings,
                            self.sdk_version,
                            self.platform_name,
                            self.arch)
    timer = Timer('Parsing options', 'parsing_options').Start()
    message, exit_code = parser.ParseOptions(args[1:])
    timer.End()
    if exit_code:
      _PrintXcodeError('Option parsing failed: %s' % message)
      return exit_code

    self.verbose = parser.verbose
    self.bazel_bin_path = os.path.abspath(parser.bazel_bin_path)
    self.bazel_executable = parser.bazel_executable
    self.bazel_exec_root = self.build_settings.bazelExecRoot
    self.bazel_output_base = self.build_settings.bazelOutputBase

    # Update feature flags.
    features = parser.GetEnabledFeatures()
    self.direct_debug_prefix_map = 'DirectDebugPrefixMap' in features
    self.normalized_prefix_map = 'DebugPathNormalization' in features

    # Path to the Build Events JSON file uses pid and is removed if the
    # build is successful.
    filename = '%d_%s' % (os.getpid(), BazelBuildBridge.BUILD_EVENTS_FILE)
    self.build_events_file_path = os.path.join(
        self.project_file_path,
        '.tulsi',
        filename)

    (command, retval) = self._BuildBazelCommand(parser)
    if retval:
      return retval

    timer = Timer('Running Bazel', 'running_bazel').Start()
    exit_code, outputs = self._RunBazelAndPatchOutput(command)
    timer.End()
    if exit_code:
      _Fatal('Bazel build failed with exit code %d. Please check the build '
             'log in Report Navigator (⌘9) for more information.'
             % exit_code)
      return exit_code

    post_bazel_timer = Timer('Total Tulsi Post-Bazel time', 'total_post_bazel')
    post_bazel_timer.Start()


    # This needs to run after `bazel build`, since it depends on the Bazel
    # output directories

    if not os.path.exists(self.bazel_exec_root):
      _Fatal('No Bazel execution root was found at %r. Debugging experience '
             'will be compromised. Please report a Tulsi bug.'
             % self.bazel_exec_root)
      return 404
    if not os.path.exists(self.bazel_output_base):
      _Fatal('No Bazel output base was found at %r. Editing experience '
             'will be compromised for external workspaces. Please report a'
             ' Tulsi bug.'
             % self.bazel_output_base)
      return 404

    exit_code = self._LinkTulsiToBazel('tulsi-execution-root', self.bazel_exec_root)
    if exit_code:
      return exit_code
    # Old versions of Tulsi mis-referred to the execution root as the workspace.
    # We preserve the old symlink name for backwards compatibility.
    exit_code = self._LinkTulsiToBazel('tulsi-workspace', self.bazel_exec_root)
    if exit_code:
      return exit_code
    exit_code = self._LinkTulsiToBazel(
        'tulsi-output-base', self.bazel_output_base)
    if exit_code:
      return exit_code


    exit_code, outputs_data = self._ExtractAspectOutputsData(outputs)
    if exit_code:
      return exit_code

    # Generated headers are installed on a thread since we are launching
    # a separate process to do so. This gives us clean timings.
    install_thread = threading.Thread(
        target=self._InstallGeneratedHeaders, args=(outputs,))
    install_thread.start()
    timer = Timer('Installing artifacts', 'installing_artifacts').Start()
    exit_code = self._InstallArtifact(outputs_data)
    timer.End()
    install_thread.join()
    if exit_code:
      return exit_code

    exit_code, dsym_paths = self._InstallDSYMBundles(
        self.built_products_dir, outputs_data)
    if exit_code:
      return exit_code

    if not dsym_paths:
      # Clean any bundles from a previous build that can interfere with
      # debugging in LLDB.
      self._CleanExistingDSYMs()
    else:
      for path in dsym_paths:
        # Starting with Xcode 9.x, a plist based remapping exists for dSYM
        # bundles that works with Swift as well as (Obj-)C(++).
        #
        # This solution also works for Xcode 8.x for (Obj-)C(++) but not
        # for Swift.
        timer = Timer('Adding remappings as plists to dSYM',
                      'plist_dsym').Start()
        exit_code = self._PlistdSYMPaths(path)
        timer.End()
        if exit_code:
          _PrintXcodeError('Remapping dSYMs process returned %i, please '
                           'report a Tulsi bug and attach a full Xcode '
                           'build log.' % exit_code)
          return exit_code

    # Starting with Xcode 7.3, XCTests inject several supporting frameworks
    # into the test host that need to be signed with the same identity as
    # the host itself.
    if (self.is_test and not self.platform_name.startswith('macos') and
        self.codesigning_allowed):
      exit_code = self._ResignTestArtifacts()
      if exit_code:
        return exit_code

    self._PruneLLDBModuleCache(outputs)

    # Starting with Xcode 8, .lldbinit files are honored during Xcode debugging
    # sessions. This allows use of the target.source-map field to remap the
    # debug symbol paths encoded in the binary to the paths expected by Xcode.
    #
    # This will not work with dSYM bundles, or a direct -fdebug-prefix-map from
    # the Bazel-built locations to Xcode-visible sources.
    timer = Timer('Updating .lldbinit', 'updating_lldbinit').Start()
    clear_source_map = dsym_paths or self.direct_debug_prefix_map
    exit_code = self._UpdateLLDBInit(clear_source_map)
    timer.End()
    if exit_code:
      _PrintXcodeWarning('Updating .lldbinit action failed with code %d' %
                         exit_code)

    post_bazel_timer.End(log_absolute_times=True)

    return 0

  def _BuildBazelCommand(self, options):
    """Builds up a commandline string suitable for running Bazel."""
    configuration = os.environ['CONFIGURATION']
    # Treat the special testrunner build config as a Debug compile.
    test_runner_config_prefix = '__TulsiTestRunner_'
    if configuration.startswith(test_runner_config_prefix):
      configuration = configuration[len(test_runner_config_prefix):]
    elif os.environ.get('TULSI_TEST_RUNNER_ONLY') == 'YES':
      _PrintXcodeError('Building test targets with configuration "%s" is not '
                       'allowed. Please use the "Test" action or "Build for" > '
                       '"Testing" instead.' % configuration)
      return (None, 1)

    if configuration not in _OptionsParser.KNOWN_CONFIGS:
      _PrintXcodeError('Unknown build configuration "%s"' % configuration)
      return (None, 1)

    bazel, start_up, build = options.GetBazelOptions(configuration)
    bazel_command = [bazel]
    bazel_command.extend(start_up)
    bazel_command.append('build')
    bazel_command.extend(build)

    bazel_command.extend([
        # The following flags are used by Tulsi to identify itself and read
        # build information from Bazel. They shold not affect Bazel anaylsis
        # caching.
        '--tool_tag=tulsi:bazel_build',
        '--build_event_json_file=%s' % self.build_events_file_path,
        '--noexperimental_build_event_json_file_path_conversion',
        '--aspects', '@tulsi//:tulsi/tulsi_aspects.bzl%tulsi_outputs_aspect'])

    bazel_command.append('--output_groups=+tulsi_outputs')
    bazel_command.extend(options.targets)

    extra_options = bazel_options.BazelOptions(os.environ)
    bazel_command.extend(extra_options.bazel_feature_flags())

    return (bazel_command, 0)

  def _RunBazelAndPatchOutput(self, command):
    """Runs subprocess command, patching output as it's received."""
    self._PrintVerbose('Running "%s", patching output for workspace root at '
                       '"%s" with project path at "%s".' %
                       (' '.join([pipes.quote(x) for x in command]),
                        self.workspace_root,
                        self.project_dir))
    # Clean up bazel output to make it look better in Xcode.
    bazel_line_regex = re.compile(
        r'(INFO|DEBUG|WARNING|ERROR|FAILED): ([^:]+:\d+:(?:\d+:)?)\s+(.+)')

    bazel_generic_regex = re.compile(r'(INFO|DEBUG|WARNING|ERROR|FAILED): (.*)')

    def PatchBazelDiagnosticStatements(output_line):
      """Make Bazel output more Xcode friendly."""

      def BazelLabelToXcodeLabel(bazel_label):
        """Map Bazel labels to xcode labels for build output."""
        xcode_labels = {
            'INFO': 'note',
            'DEBUG': 'note',
            'WARNING': 'warning',
            'ERROR': 'error',
            'FAILED': 'error'
        }
        return xcode_labels.get(bazel_label, bazel_label)

      match = bazel_line_regex.match(output_line)
      if match:
        xcode_label = BazelLabelToXcodeLabel(match.group(1))
        output_line = '%s %s: %s' % (match.group(2), xcode_label,
                                     match.group(3))
      else:
        match = bazel_generic_regex.match(output_line)
        if match:
          xcode_label = BazelLabelToXcodeLabel(match.group(1))
          output_line = '%s: %s' % (xcode_label, match.group(2))
      return output_line

    if self.workspace_root != self.project_dir:
      # Match (likely) filename:line_number: lines.
      xcode_parsable_line_regex = re.compile(r'([^/][^:]+):\d+:')

      def PatchOutputLine(output_line):
        output_line = PatchBazelDiagnosticStatements(output_line)
        if xcode_parsable_line_regex.match(output_line):
          output_line = '%s/%s' % (self.workspace_root, output_line)
        return output_line
      patch_xcode_parsable_line = PatchOutputLine
    else:
      patch_xcode_parsable_line = PatchBazelDiagnosticStatements

    def HandleOutput(output):
      for line in output.splitlines():
        _logger.log_bazel_message(patch_xcode_parsable_line(line))

    def WatcherUpdate(watcher):
      """Processes any new events in the given watcher.

      Args:
        watcher: a BazelBuildEventsWatcher object.

      Returns:
        A list of new tulsiout file names seen.
      """
      new_events = watcher.check_for_new_events()
      new_outputs = []
      for build_event in new_events:
        if build_event.stderr:
          HandleOutput(build_event.stderr)
        if build_event.stdout:
          HandleOutput(build_event.stdout)
        if build_event.files:
          outputs = [x for x in build_event.files if x.endswith('.tulsiouts')]
          new_outputs.extend(outputs)
      return new_outputs

    def ReaderThread(file_handle, out_buffer):
      out_buffer.append(file_handle.read())
      file_handle.close()

    # Make sure the BEP JSON file exists and is empty. We do this to prevent
    # any sort of race between the watcher, bazel, and the old file contents.
    open(self.build_events_file_path, 'w').close()

    # Capture the stderr and stdout from Bazel. We only display it if it we're
    # unable to read any BEP events.
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1)

    # Register atexit function to clean up BEP file.
    atexit.register(_BEPFileExitCleanup, self.build_events_file_path)
    global CLEANUP_BEP_FILE_AT_EXIT
    CLEANUP_BEP_FILE_AT_EXIT = True

    # Start capturing output from Bazel.
    reader_buffer = []
    reader_thread = threading.Thread(target=ReaderThread,
                                     args=(process.stdout, reader_buffer))
    reader_thread.daemon = True
    reader_thread.start()

    with io.open(self.build_events_file_path, 'r', -1, 'utf-8', 'ignore'
                ) as bep_file:
      watcher = bazel_build_events.BazelBuildEventsWatcher(bep_file,
                                                           _PrintXcodeWarning)
      output_locations = []
      while process.returncode is None:
        output_locations.extend(WatcherUpdate(watcher))
        time.sleep(0.1)
        process.poll()

      output_locations.extend(WatcherUpdate(watcher))

      # If BEP JSON parsing failed, we should display the raw stdout and
      # stderr from Bazel.
      reader_thread.join()
      if not watcher.has_read_events():
        HandleOutput(reader_buffer[0])

      if process.returncode == 0 and not output_locations:
        CLEANUP_BEP_FILE_AT_EXIT = False
        _PrintXcodeError('Unable to find location of the .tulsiouts file.'
                         'Please report this as a Tulsi bug, including the'
                         'contents of %s.' % self.build_events_file_path)
        return 1, output_locations
      return process.returncode, output_locations

  def _ExtractAspectOutputsData(self, output_files):
    """Converts aspect output from paths to json to a list of dictionaries.

    Args:
      output_files: A list of strings to files representing Bazel aspect output
                    in UTF-8 JSON format.

    Returns:
      return_code, [dict]: A tuple with a return code as its first argument and
                           for its second argument, a list of dictionaries for
                           each output_file that could be interpreted as valid
                           JSON, representing the returned Bazel aspect
                           information.
      return_code, None: If an error occurred while converting the list of
                         files into JSON.
    """
    outputs_data = []
    for output_file in output_files:
      try:
        with io.open(output_file, 'rb') as f:
          output_data = json.load(f)
      except (ValueError, IOError) as e:
        _PrintXcodeError('Failed to load output map ""%s". '
                         '%s' % (output_file, e))
        return 600, None
      outputs_data.append(output_data)
    return 0, outputs_data

  def _InstallArtifact(self, outputs_data):
    """Installs Bazel-generated artifacts into the Xcode output directory."""
    xcode_artifact_path = self.artifact_output_path

    if not outputs_data:
      _PrintXcodeError('Failed to load top level output file.')
      return 600

    primary_output_data = outputs_data[0]

    if 'artifact' not in primary_output_data:
      _PrintXcodeError(
          'Failed to find an output artifact for target %s in output map %r' %
          (xcode_artifact_path, primary_output_data))
      return 601

    primary_artifact = primary_output_data['artifact']
    artifact_archive_root = primary_output_data.get('archive_root')
    bundle_name = primary_output_data.get('bundle_name')

    # The PRODUCT_NAME used by the Xcode project is not trustable as it may be
    # modified by the user and, more importantly, may have been modified by
    # Tulsi to disambiguate multiple targets with the same name.
    self.bazel_product_name = bundle_name

    # We need to handle IPAs (from {ios, tvos}_application) differently from
    # ZIPs (from the other bundled rules) because they output slightly different
    # directory structures.
    is_ipa = primary_artifact.endswith('.ipa')
    is_zip = primary_artifact.endswith('.zip')

    if is_ipa or is_zip:
      expected_bundle_name = bundle_name + self.wrapper_suffix

      # The directory structure within the IPA is then determined based on
      # Bazel's package and/or product type.
      if is_ipa:
        bundle_subpath = os.path.join('Payload', expected_bundle_name)
      else:
        # If the artifact is a ZIP, assume that the bundle is the top-level
        # directory (this is the way in which Skylark rules package artifacts
        # that are not standalone IPAs).
        bundle_subpath = expected_bundle_name

      # Prefer to copy over files from the archive root instead of unzipping the
      # ipa/zip in order to help preserve timestamps. Note that the archive root
      # is only present for local builds; for remote builds we must extract from
      # the zip file.
      if self._IsValidArtifactArchiveRoot(artifact_archive_root, bundle_name):
        source_location = os.path.join(artifact_archive_root, bundle_subpath)
        exit_code = self._RsyncBundle(os.path.basename(primary_artifact),
                                      source_location,
                                      xcode_artifact_path)
      else:
        exit_code = self._UnpackTarget(primary_artifact,
                                       xcode_artifact_path,
                                       bundle_subpath)
      if exit_code:
        return exit_code

    elif os.path.isfile(primary_artifact):
      # Remove the old artifact before copying.
      if os.path.isfile(xcode_artifact_path):
        try:
          os.remove(xcode_artifact_path)
        except OSError as e:
          _PrintXcodeError('Failed to remove stale output file ""%s". '
                           '%s' % (xcode_artifact_path, e))
          return 600
      exit_code = self._CopyFile(os.path.basename(primary_artifact),
                                 primary_artifact,
                                 xcode_artifact_path)
      if exit_code:
        return exit_code
    else:
      self._RsyncBundle(os.path.basename(primary_artifact),
                        primary_artifact,
                        xcode_artifact_path)

      # When the rules output a tree artifact, Tulsi will copy the bundle as is
      # into the expected Xcode output location. But because they're copied as
      # is from the bazel output, they come with bazel's permissions, which are
      # read only. Here we set them to write as well, so Xcode can modify the
      # bundle too (for example, for codesigning).
      chmod_timer = Timer('Modifying permissions of output bundle',
                          'bundle_chmod').Start()

      self._PrintVerbose('Spawning subprocess to add write permissions to '
                         'copied bundle...')
      process = subprocess.Popen(['chmod', '-R', 'uga+w', xcode_artifact_path])
      process.wait()
      chmod_timer.End()

    # No return code check as this is not an essential operation.
    self._InstallEmbeddedBundlesIfNecessary(primary_output_data)

    return 0

  def _IsValidArtifactArchiveRoot(self, archive_root, bundle_name):
    """Returns true if the archive root is valid for use."""
    if not archive_root or not os.path.isdir(archive_root):
      return False

    # The archive root will not be updated for any remote builds, but will be
    # valid for local builds. We detect this by using an implementation detail
    # of the rules_apple bundler: archives will always be transformed from
    # <name>.unprocessed.zip (locally or remotely) to <name>.archive-root.
    #
    # Thus if the mod time on the archive root is not greater than the mod
    # time on the on the zip, the archive root is not valid. Remote builds
    # will end up copying the <name>.unprocessed.zip but not the
    # <name>.archive-root, making this a valid temporary solution.
    #
    # In the future, it would be better to have this handled by the rules;
    # until then this should suffice as a work around to improve build times.
    unprocessed_zip = os.path.join(os.path.dirname(archive_root),
                                   '%s.unprocessed.zip' % bundle_name)
    if not os.path.isfile(unprocessed_zip):
      return False
    return os.path.getmtime(archive_root) > os.path.getmtime(unprocessed_zip)

  def _InstallEmbeddedBundlesIfNecessary(self, output_data):
    """Install embedded bundles next to the current target's output."""

    # In order to find and load symbols for the binary installed on device,
    # Instruments needs to "see" it in Spotlight index somewhere on the local
    # filesystem. This is only needed for on-device instrumentation.
    #
    # Unfortunatelly, it does not seem to be possible to detect when a build is
    # being made for profiling, thus we can't exclude this step for on-device
    # non-profiling builds.

    if self.is_simulator or ('embedded_bundles' not in output_data):
      return

    timer = Timer('Installing embedded bundles',
                  'installing_embedded_bundles').Start()

    for bundle_info in output_data['embedded_bundles']:
      bundle_name = bundle_info['bundle_name']
      bundle_extension = bundle_info['bundle_extension']
      full_name = bundle_name + bundle_extension
      output_path = os.path.join(self.built_products_dir, full_name)
      # TODO(b/68936732): See if copying just the binary (not the whole bundle)
      # is enough to make Instruments work.
      if self._IsValidArtifactArchiveRoot(bundle_info['archive_root'],
                                          bundle_name):
        source_path = os.path.join(bundle_info['archive_root'], full_name)
        self._RsyncBundle(full_name, source_path, output_path)
      else:
        # Try to find the embedded bundle within the installed main bundle.
        bundle_path = self._FindEmbeddedBundleInMain(bundle_name,
                                                     bundle_extension)
        if bundle_path:
          self._RsyncBundle(full_name, bundle_path, output_path)
        else:
          _PrintXcodeWarning('Could not find bundle %s in main bundle. ' %
                             (full_name) +
                             'Device-level Instruments debugging will be '
                             'disabled for this bundle. Please report a '
                             'Tulsi bug and attach a full Xcode build log.')

    timer.End()

  # Maps extensions to anticipated subfolders.
  _EMBEDDED_BUNDLE_PATHS = {
      '.appex': 'PlugIns',
      '.framework': 'Frameworks'
  }

  def _FindEmbeddedBundleInMain(self, bundle_name, bundle_extension):
    """Retrieves the first embedded bundle found within our main bundle."""
    main_bundle = os.environ.get('EXECUTABLE_FOLDER_PATH')

    if not main_bundle:
      return None

    main_bundle_path = os.path.join(self.built_products_dir,
                                    main_bundle)

    return self._FindEmbeddedBundle(bundle_name,
                                    bundle_extension,
                                    main_bundle_path)

  def _FindEmbeddedBundle(self, bundle_name, bundle_extension, bundle_path):
    """Retrieves the first embedded bundle found within this bundle path."""
    embedded_subfolder = self._EMBEDDED_BUNDLE_PATHS.get(bundle_extension)

    if not embedded_subfolder:
      return None

    projected_bundle_path = os.path.join(bundle_path,
                                         embedded_subfolder,
                                         bundle_name + bundle_extension)

    if os.path.isdir(projected_bundle_path):
      return projected_bundle_path

    # For frameworks not in the main app bundle, and possibly other executable
    # bundle content in the future, we recurse through every .appex in PlugIns
    # to find those frameworks.
    #
    # This won't support frameworks that could potentially have the same name
    # but are different between the app and extensions, but we intentionally
    # choose not to handle that case. Xcode build system only supports
    # uniquely named frameworks, and we shouldn't confuse the dynamic loader
    # with frameworks that have the same image names but different content.
    appex_root_path = os.path.join(bundle_path, 'PlugIns')
    if not os.path.isdir(appex_root_path):
      return None

    # Find each directory within appex_root_path and attempt to find a bundle.
    # If one can't be found, return None.
    appex_dirs = os.listdir(appex_root_path)
    for appex_dir in appex_dirs:
      appex_path = os.path.join(appex_root_path, appex_dir)
      path = self._FindEmbeddedBundle(bundle_name,
                                      bundle_extension,
                                      appex_path)
      if path:
        return path
    return None

  def _InstallGeneratedHeaders(self, outputs):
    """Invokes install_genfiles.py to install generated Bazel files."""
    genfiles_timer = Timer('Installing generated headers',
                           'installing_generated_headers').Start()
    # Resolve the path to the install_genfiles.py script.
    # It should be in the same directory as this script.
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'install_genfiles.py')

    args = [path, self.bazel_exec_root]
    args.extend(outputs)

    self._PrintVerbose('Spawning subprocess install_genfiles.py to copy '
                       'generated files in the background...')
    process = subprocess.Popen(args)
    process.wait()
    genfiles_timer.End()

  def _InstallBundle(self, source_path, output_path):
    """Copies the bundle at source_path to output_path."""
    if not os.path.isdir(source_path):
      return 0, None

    if os.path.isdir(output_path):
      try:
        shutil.rmtree(output_path)
      except OSError as e:
        _PrintXcodeError('Failed to remove stale bundle ""%s". '
                         '%s' % (output_path, e))
        return 700, None

    exit_code = self._CopyBundle(os.path.basename(source_path),
                                 source_path,
                                 output_path)
    return exit_code, output_path

  def _RsyncBundle(self, source_path, full_source_path, output_path):
    """Rsyncs the given bundle to the given expected output path."""
    self._PrintVerbose('Rsyncing %s to %s' % (source_path, output_path))

    # rsync behavior changes based on presence of a trailing slash.
    if not full_source_path.endswith('/'):
      full_source_path += '/'

    try:
      # Use -c to check differences by checksum, -v for verbose,
      # and --delete to delete stale files.
      # The rest of the flags are the same as -a but without preserving
      # timestamps, which is done intentionally so the timestamp will
      # only change when the file is changed.
      subprocess.check_output(['rsync',
                               '-vcrlpgoD',
                               '--delete',
                               full_source_path,
                               output_path],
                              stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
      _PrintXcodeError('Rsync failed. %s' % e)
      return 650
    return 0

  def _CopyBundle(self, source_path, full_source_path, output_path):
    """Copies the given bundle to the given expected output path."""
    self._PrintVerbose('Copying %s to %s' % (source_path, output_path))
    try:
      CopyOnWrite(full_source_path, output_path, tree=True)
    except OSError as e:
      _PrintXcodeError('Copy failed. %s' % e)
      return 650
    return 0

  def _CopyFile(self, source_path, full_source_path, output_path):
    """Copies the given file to the given expected output path."""
    self._PrintVerbose('Copying %s to %s' % (source_path, output_path))
    output_path_dir = os.path.dirname(output_path)
    if not os.path.exists(output_path_dir):
      try:
        os.makedirs(output_path_dir)
      except OSError as e:
        _PrintXcodeError('Failed to create output directory "%s". '
                         '%s' % (output_path_dir, e))
        return 650
    try:
      CopyOnWrite(full_source_path, output_path)
    except OSError as e:
      _PrintXcodeError('Copy failed. %s' % e)
      return 650
    return 0

  def _UnpackTarget(self, bundle_path, output_path, bundle_subpath):
    """Unpacks generated bundle into the given expected output path."""
    self._PrintVerbose('Unpacking %s to %s' % (bundle_path, output_path))

    if not os.path.isfile(bundle_path):
      _PrintXcodeError('Generated bundle not found at "%s"' % bundle_path)
      return 670

    if os.path.isdir(output_path):
      try:
        shutil.rmtree(output_path)
      except OSError as e:
        _PrintXcodeError('Failed to remove stale output directory ""%s". '
                         '%s' % (output_path, e))
        return 600

    # We need to handle IPAs (from {ios, tvos}_application) differently from
    # ZIPs (from the other bundled rules) because they output slightly different
    # directory structures.
    is_ipa = bundle_path.endswith('.ipa')

    with zipfile.ZipFile(bundle_path, 'r') as zf:
      for item in zf.infolist():
        filename = item.filename

        # Support directories do not seem to be needed by the debugger and are
        # skipped.
        basedir = filename.split(os.sep)[0]
        if basedir.endswith('Support') or basedir.endswith('Support2'):
          continue

        if len(filename) < len(bundle_subpath):
          continue

        attributes = (item.external_attr >> 16) & 0o777
        self._PrintVerbose('Extracting %s (%o)' % (filename, attributes),
                           level=1)

        if not filename.startswith(bundle_subpath):
          _PrintXcodeWarning('Mismatched extraction path. Bundle content '
                             'at "%s" expected to have subpath of "%s"' %
                             (filename, bundle_subpath))

        dir_components = self._SplitPathComponents(filename)

        # Get the file's path, ignoring the payload components if the archive
        # is an IPA.
        if is_ipa:
          subpath = os.path.join(*dir_components[2:])
        else:
          subpath = os.path.join(*dir_components[1:])
        target_path = os.path.join(output_path, subpath)

        # Ensure the target directory exists.
        try:
          target_dir = os.path.dirname(target_path)
          if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        except OSError as e:
          _PrintXcodeError(
              'Failed to create target path "%s" during extraction. %s' % (
                  target_path, e))
          return 671

        # If the archive item looks like a file, extract it.
        if not filename.endswith(os.sep):
          with zf.open(item) as src, open(target_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)

        # Patch up the extracted file's attributes to match the zip content.
        if attributes:
          os.chmod(target_path, attributes)

    return 0

  def _InstallDSYMBundles(self, output_dir, outputs_data):
    """Copies any generated dSYM bundles to the given directory."""

    dsym_to_process = set()
    primary_output_data = outputs_data[0]
    if primary_output_data['has_dsym']:
      # Declares the Xcode-generated name of our main target's dSYM.
      # This environment variable is always set, for any possible Xcode output
      # that could generate a dSYM bundle.
      #
      # Note that this may differ from the Bazel name as Tulsi may modify the
      # Xcode `BUNDLE_NAME`, so we need to make sure we use Bazel as the source
      # of truth for Bazel's dSYM name, but copy it over to where Xcode expects.
      xcode_target_dsym = os.environ.get('DWARF_DSYM_FILE_NAME')

      if xcode_target_dsym:
        dsym_path = primary_output_data.get('dsym_path')
        if dsym_path:
          dsym_to_process.add((dsym_path, xcode_target_dsym))
        else:
          _PrintXcodeWarning('Unable to resolve dSYM paths for main bundle %s' %
                             primary_output_data)

    # Collect additional dSYM bundles generated by the dependencies of this
    # build such as extensions or frameworks. Note that a main target may not
    # have dSYMs while subtargets (like an xctest) still can have them.
    child_dsyms = set()
    for data in outputs_data:
      for bundle_info in data.get('embedded_bundles', []):
        if not bundle_info['has_dsym']:
          continue
        dsym_path = bundle_info.get('dsym_path')
        if dsym_path:
          child_dsyms.add((dsym_path, os.path.basename(dsym_path)))
        else:
          _PrintXcodeWarning(
              'Unable to resolve dSYM paths for embedded bundle %s'
              % bundle_info)
    dsym_to_process.update(child_dsyms)

    if not dsym_to_process:
      return 0, None

    # Start the timer now that we know we have dSYM bundles to install.
    timer = Timer('Installing dSYM bundles', 'installing_dsym').Start()

    dsyms_found = []
    for input_dsym_full_path, xcode_dsym_name in dsym_to_process:
      output_full_path = os.path.join(output_dir, xcode_dsym_name)
      exit_code, path = self._InstallBundle(input_dsym_full_path,
                                            output_full_path)
      if exit_code:
        _PrintXcodeWarning('Failed to install dSYM to "%s" (%s)'
                           % (input_dsym_full_path, exit_code))
      elif path is None:
        _PrintXcodeWarning('Did not find a dSYM bundle at %s'
                           % input_dsym_full_path)
      else:
        dsyms_found.append(path)

    timer.End()
    return 0, dsyms_found

  def _ResignBundle(self, bundle_path, signing_identity, entitlements=None):
    """Re-signs the bundle with the given signing identity and entitlements."""
    if not self.codesigning_allowed:
      return 0

    timer = Timer('\tSigning ' + bundle_path, 'signing_bundle').Start()
    command = [
        'xcrun',
        'codesign',
        '-f',
        '--timestamp=none',
        '-s',
        signing_identity,
    ]

    if entitlements:
      command.extend(['--entitlements', entitlements])
    else:
      command.append('--preserve-metadata=entitlements')

    command.append(bundle_path)

    returncode, output = self._RunSubprocess(command)
    timer.End()
    if returncode:
      _PrintXcodeError('Re-sign command %r failed. %s' % (command, output))
      return 800 + returncode
    return 0

  def _ResignTestArtifacts(self):
    """Resign test related artifacts that Xcode injected into the outputs."""
    if not self.is_test:
      return 0
    # Extract the signing identity from the bundle at the expected output path
    # since that's where the signed bundle from bazel was placed.
    signing_identity = self._ExtractSigningIdentity(self.artifact_output_path)
    if not signing_identity:
      return 800

    exit_code = 0
    timer = Timer('Re-signing injected test host artifacts',
                  'resigning_test_host').Start()

    if self.test_host_binary:
      # For Unit tests, we need to resign the frameworks that Xcode injected
      # into the test host bundle.
      test_host_bundle = os.path.dirname(self.test_host_binary)
      exit_code = self._ResignXcodeTestFrameworks(
          test_host_bundle, signing_identity)
    else:
      # For UI tests, we need to resign the UI test runner app and the
      # frameworks that Xcode injected into the runner app. The UI Runner app
      # also needs to be signed with entitlements.
      exit_code = self._ResignXcodeTestFrameworks(
          self.codesigning_folder_path, signing_identity)
      if exit_code == 0:
        entitlements_path = self._InstantiateUIRunnerEntitlements()
        if entitlements_path:
          exit_code = self._ResignBundle(
              self.codesigning_folder_path,
              signing_identity,
              entitlements_path)
        else:
          _PrintXcodeError('Could not instantiate UI runner entitlements.')
          exit_code = 800

    timer.End()
    return exit_code

  def _ResignXcodeTestFrameworks(self, bundle, signing_identity):
    """Re-signs the support frameworks injected by Xcode in the given bundle."""
    if not self.codesigning_allowed:
      return 0

    for framework in XCODE_INJECTED_FRAMEWORKS:
      framework_path = os.path.join(
          bundle, 'Frameworks', framework)
      if os.path.isdir(framework_path) or os.path.isfile(framework_path):
        exit_code = self._ResignBundle(framework_path, signing_identity)
        if exit_code != 0:
          return exit_code
    return 0

  def _InstantiateUIRunnerEntitlements(self):
    """Substitute team and bundle identifiers into UI runner entitlements.

    This method throws an IOError exception if the template wasn't found in
    its expected location, or an OSError if the expected output folder could
    not be created.

    Returns:
      The path to where the entitlements file was generated.
    """
    if not self.codesigning_allowed:
      return None
    if not os.path.exists(self.derived_sources_folder_path):
      os.makedirs(self.derived_sources_folder_path)

    output_file = os.path.join(
        self.derived_sources_folder_path,
        self.bazel_product_name + '_UIRunner.entitlements')
    if os.path.exists(output_file):
      os.remove(output_file)

    with io.open(
        self.runner_entitlements_template, 'r', encoding='utf-8') as template:
      contents = template.read()
      contents = contents.replace(
          '$(TeamIdentifier)',
          self._ExtractSigningTeamIdentifier(self.artifact_output_path))
      contents = contents.replace(
          '$(BundleIdentifier)',
          self._ExtractSigningBundleIdentifier(self.artifact_output_path))
      with open(output_file, 'w') as output:
        output.write(contents)
    return output_file

  def _ExtractSigningIdentity(self, signed_bundle):
    """Returns the identity used to sign the given bundle path."""
    return self._ExtractSigningAttribute(signed_bundle, 'Authority')

  def _ExtractSigningTeamIdentifier(self, signed_bundle):
    """Returns the team identifier used to sign the given bundle path."""
    return self._ExtractSigningAttribute(signed_bundle, 'TeamIdentifier')

  def _ExtractSigningBundleIdentifier(self, signed_bundle):
    """Returns the bundle identifier used to sign the given bundle path."""
    return self._ExtractSigningAttribute(signed_bundle, 'Identifier')

  def _ExtractSigningAttribute(self, signed_bundle, attribute):
    """Returns the attribute used to sign the given bundle path."""
    if not self.codesigning_allowed:
      return '<CODE_SIGNING_ALLOWED=NO>'

    cached = self.codesign_attributes.get(signed_bundle)
    if cached:
      return cached.Get(attribute)

    timer = Timer('\tExtracting signature for ' + signed_bundle,
                  'extracting_signature').Start()
    output = subprocess.check_output(
        ['xcrun', 'codesign', '-dvv', signed_bundle],
        stderr=subprocess.STDOUT,
        encoding='utf-8')
    timer.End()

    bundle_attributes = CodesignBundleAttributes(output)
    self.codesign_attributes[signed_bundle] = bundle_attributes
    return bundle_attributes.Get(attribute)

  def _PruneLLDBModuleCache(self, output_files):
    """Run the module cache pruner tool as a subprocess."""
    if not os.path.exists(BazelBuildBridge.MODULE_CACHE_PRUNER_EXECUTABLE):
      _PrintXcodeWarning(
          'Could find module cache pruner executable at %s. '
          'You may need to manually remove %s if lldb-rpc-server crashes.' %
          (BazelBuildBridge.MODULE_CACHE_PRUNER_EXECUTABLE,
           BazelBuildBridge.XCODE_MODULE_CACHE_DIRECTORY))
      return

    timer = Timer('Pruning module cache', 'prune_module_cache').Start()
    for output_file in output_files:
      self._RunSubprocess([
          BazelBuildBridge.MODULE_CACHE_PRUNER_EXECUTABLE,
          BazelBuildBridge.XCODE_MODULE_CACHE_DIRECTORY, output_file
      ])
    timer.End()

  def _UpdateLLDBInit(self, clear_source_map=False):
    """Updates lldbinit to enable debugging of Bazel binaries."""

    # An additional lldbinit file that we should load in the lldbinit file
    # we are about to write.
    additional_lldbinit = None

    if self.custom_lldbinit is None:
      # Write our settings to the global ~/.lldbinit-tulsiproj file when no
      # custom lldbinit is provided.
      lldbinit_file = TULSI_LLDBINIT_FILE
      # Make sure a reference to ~/.lldbinit-tulsiproj exists in ~/.lldbinit or
      # ~/.lldbinit-Xcode. Priority is given to ~/.lldbinit-Xcode if it exists,
      # otherwise the bootstrapping will be written to ~/.lldbinit.
      BootstrapLLDBInit(True)
    else:
      # Remove any reference to ~/.lldbinit-tulsiproj if the global lldbinit was
      # previously bootstrapped. This prevents the global lldbinit from having
      # side effects on the custom lldbinit file.
      BootstrapLLDBInit(False)
      # When using a custom lldbinit, Xcode will directly load our custom file
      # so write our settings to this custom file. Retain standard Xcode
      # behavior by loading the default file in our custom file.
      lldbinit_file = self.custom_lldbinit
      additional_lldbinit = _FindDefaultLldbInit()

    project_basename = os.path.basename(self.project_file_path)
    workspace_root = self._NormalizePath(self.workspace_root)

    with open(lldbinit_file, 'w') as out:
      out.write('# This file is autogenerated by Tulsi and should not be '
                'edited.\n')

      if additional_lldbinit is not None:
        out.write('# This loads the default lldbinit file to retain standard '
                  'Xcode behavior.\n')
        out.write('command source "%s"\n' % additional_lldbinit)

      out.write('# This sets lldb\'s working directory to the Bazel workspace '
                'root used by %r.\n' % project_basename)
      out.write('platform settings -w "%s"\n' % workspace_root)

      out.write('# This enables implicitly loading Clang modules which can be '
                'disabled when a Swift module was built with explicit modules '
                'enabled.\n')
      out.write(
          'settings set -- target.swift-extra-clang-flags "-fimplicit-module-maps"\n'
      )

      if clear_source_map:
        out.write('settings clear target.source-map\n')
        return 0

      if self.normalized_prefix_map:
        source_map = ('./', workspace_root)
        out.write('# This maps the normalized root to that used by '
                  '%r.\n' % project_basename)
      else:
        # NOTE: settings target.source-map is different from
        # DBGSourcePathRemapping; the former is an LLDB target-level
        # remapping API that rewrites breakpoints, the latter is an LLDB
        # module-level remapping API that changes DWARF debug info in memory.
        #
        # If we had multiple remappings, it would not make sense for the
        # two APIs to share the same mappings. They have very different
        # side-effects in how they individually handle debug information.
        source_map = self._ExtractTargetSourceMap()
        out.write('# This maps Bazel\'s execution root to that used by '
                  '%r.\n' % project_basename)

      out.write('settings set target.source-map "%s" "%s"\n' % source_map)

    return 0

  def _DWARFdSYMBinaries(self, dsym_bundle_path):
    """Returns an array of abs paths to DWARF binaries in the dSYM bundle.

    Args:
      dsym_bundle_path: absolute path to the dSYM bundle.

    Returns:
      str[]: a list of strings representing the absolute paths to each binary
             found within the dSYM bundle.
    """
    dwarf_dir = os.path.join(dsym_bundle_path,
                             'Contents',
                             'Resources',
                             'DWARF')

    dsym_binaries = []

    for f in os.listdir(dwarf_dir):
      # Ignore hidden files, such as .DS_Store files.
      if not f.startswith('.'):
        # Append full path info.
        dsym_binary = os.path.join(dwarf_dir, f)
        dsym_binaries.append(dsym_binary)

    return dsym_binaries

  def _UUIDInfoForBinary(self, source_binary_path):
    """Returns exit code of dwarfdump along with every UUID + arch found.

    Args:
      source_binary_path: absolute path to the binary file.

    Returns:
      (Int, str[(str, str)]): a tuple containing the return code of dwarfdump
                              as its first element, and a list of strings
                              representing each UUID found for each given
                              binary slice found within the binary with its
                              given architecture, if no error has occcured.
    """

    returncode, output = self._RunSubprocess(
        ['xcrun', 'dwarfdump', '--uuid', source_binary_path])
    if returncode:
      _PrintXcodeWarning('dwarfdump returned %d while finding the UUID for %s'
                         % (returncode, source_binary_path))
      return (returncode, [])

    # All UUIDs for binary slices will be returned as the second from left,
    # from output; "UUID: D4DE5AA2-79EE-36FE-980C-755AED318308 (x86_64)
    # /Applications/Calendar.app/Contents/MacOS/Calendar"

    uuids_found = []
    for dwarfdump_output in output.split('\n'):
      if not dwarfdump_output:
        continue
      found_output = re.match(r'^(?:UUID: )([^ ]+) \(([^)]+)', dwarfdump_output)
      if not found_output:
        continue
      found_uuid = found_output.group(1)
      if not found_uuid:
        continue
      found_arch = found_output.group(2)
      if not found_arch:
        continue
      uuids_found.append((found_uuid, found_arch))

    return (0, uuids_found)

  def _CreateUUIDPlist(self, dsym_bundle_path, uuid, arch, source_maps):
    """Creates a UUID.plist in a dSYM bundle to redirect sources.

    Args:
      dsym_bundle_path: absolute path to the dSYM bundle.
      uuid: string representing the UUID of the binary slice with paths to
            remap in the dSYM bundle.
      arch: the architecture of the binary slice.
      source_maps:  list of tuples representing all absolute paths to source
                    files compiled by Bazel as strings ($0) associated with the
                    paths to Xcode-visible sources used for the purposes of
                    Tulsi debugging as strings ($1).

    Returns:
      Bool: True if no error was found, or False, representing a failure to
            write when creating the plist.
    """

    # Create a UUID plist at (dsym_bundle_path)/Contents/Resources/.
    remap_plist = os.path.join(dsym_bundle_path,
                               'Contents',
                               'Resources',
                               '%s.plist' % uuid)

    # Via an XML plist, add the mappings from  _ExtractTargetSourceMap().
    try:
      with open(remap_plist, 'w') as out:
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                  '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                  '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
                  '<plist version="1.0">\n'
                  '<dict>\n'
                  '<key>DBGSourcePathRemapping</key>\n'
                  '<dict>\n')
        for source_map in source_maps:
          # Add the mapping as a DBGSourcePathRemapping to the UUID plist here.
          out.write('<key>%s</key>\n<string>%s</string>\n' % source_map)

        # Make sure that we also set DBGVersion to 3.
        out.write('</dict>\n'
                  '<key>DBGVersion</key>\n'
                  '<string>3</string>\n'
                  '</dict>\n'
                  '</plist>\n')
    except OSError as e:
      _PrintXcodeError('Failed to write %s, received error %s' %
                       (remap_plist, e))
      return False

    # Update the dSYM symbol cache with a reference to this dSYM bundle.
    if self.update_symbol_cache is not None:
      err_msg = self.update_symbol_cache.UpdateUUID(uuid,
                                                    dsym_bundle_path,
                                                    arch)
      if err_msg:
        _PrintXcodeWarning('Attempted to save (uuid, dsym_bundle_path, arch) '
                           'to DBGShellCommands\' dSYM cache, but got error '
                           '\"%s\".' % err_msg)

    return True

  def _CleanExistingDSYMs(self):
    """Clean dSYM bundles that were left over from a previous build."""

    output_dir = self.built_products_dir
    output_dir_list = os.listdir(output_dir)
    for item in output_dir_list:
      if item.endswith('.dSYM'):
        shutil.rmtree(os.path.join(output_dir, item))

  def _PlistdSYMPaths(self, dsym_bundle_path):
    """Adds Plists to a given dSYM bundle to redirect DWARF data."""

    # Retrieve the paths that we are expected to remap.

    # Always include a direct path from the execroot to Xcode-visible sources.
    source_maps = [self._ExtractTargetSourceMap()]

    # Remap relative paths from the workspace root.
    if self.normalized_prefix_map:
      # Take the normalized path and map that to Xcode-visible sources.
      source_maps.append(('./', self._NormalizePath(self.workspace_root)))

    # Find the binaries within the dSYM bundle. UUIDs will match that of the
    # binary it was based on.
    dsym_binaries = self._DWARFdSYMBinaries(dsym_bundle_path)

    if not dsym_binaries:
      _PrintXcodeWarning('Could not find the binaries that the dSYM %s was '
                         'based on to determine DWARF binary slices to patch. '
                         'Debugging will probably fail.' % (dsym_bundle_path))
      return 404

    # Find the binary slice UUIDs with dwarfdump from each binary.
    for source_binary_path in dsym_binaries:

      returncode, uuid_info_found = self._UUIDInfoForBinary(source_binary_path)
      if returncode:
        return returncode

      # Create a plist per UUID, each indicating a binary slice to remap paths.
      for uuid, arch in uuid_info_found:
        plist_created = self._CreateUUIDPlist(dsym_bundle_path,
                                              uuid,
                                              arch,
                                              source_maps)
        if not plist_created:
          return 405

    return 0

  def _NormalizePath(self, path):
    """Returns paths with a common form, normalized with a trailing slash.

    Args:
      path: a file system path given in the form of a string.

    Returns:
      str: a normalized string with a trailing slash, based on |path|.
    """
    return os.path.normpath(path) + os.sep

  def _ExtractTargetSourceMap(self, normalize=True):
    """Extracts the source path as a tuple associated with the WORKSPACE path.

    Args:
      normalize: Defines if all paths should be normalized. Preferred for APIs
                 like DBGSourcePathRemapping and target.source-map but won't
                 work for the purposes of -fdebug-prefix-map.

    Returns:
      None: if an error occurred.
      (str, str): a single tuple representing all absolute paths to source
                  files compiled by Bazel as strings ($0) associated with
                  the paths to Xcode-visible sources used for the purposes
                  of Tulsi debugging as strings ($1).
    """
    # All paths route to the "workspace root" for sources visible from Xcode.
    sm_destpath = self.workspace_root
    if normalize:
      sm_destpath = self._NormalizePath(sm_destpath)

    # Add a redirection for the Bazel execution root, the path where sources
    # are referenced by Bazel.
    sm_execroot = self.bazel_exec_root
    if normalize:
      sm_execroot = self._NormalizePath(sm_execroot)
    return (sm_execroot, sm_destpath)

  def _LinkTulsiToBazel(self, symlink_name, destination):
    """Links symlink_name (in project/.tulsi) to the specified destination."""
    symlink_path = os.path.join(self.project_file_path,
                                   '.tulsi',
                                   symlink_name)
    if os.path.islink(symlink_path):
      os.unlink(symlink_path)
    os.symlink(destination, symlink_path)
    if not os.path.exists(symlink_path):
      _PrintXcodeError(
          'Linking %s to %s failed.' % (symlink_path, destination))
      return -1

  @staticmethod
  def _SplitPathComponents(path):
    """Splits the given path into an array of all of its components."""
    components = path.split(os.sep)
    # Patch up the first component if path started with an os.sep
    if not components[0]:
      components[0] = os.sep
    return components

  def _RunSubprocess(self, cmd):
    """Runs the given command as a subprocess, returning (exit_code, output)."""
    self._PrintVerbose('%r' % cmd, 1)
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
    output, _ = process.communicate()
    return (process.returncode, output)

  def _PrintVerbose(self, msg, level=0):
    if self.verbose > level:
      _PrintUnbuffered(msg)


def main(argv):
  build_settings = bazel_build_settings.BUILD_SETTINGS
  if build_settings is None:
    _Fatal('Unable to resolve build settings. Please report a Tulsi bug.')
    return 1
  return BazelBuildBridge(build_settings).Run(argv)


if __name__ == '__main__':
  # Register the interrupt handler immediately in case we receive SIGINT while
  # trying to acquire the lock.
  signal.signal(signal.SIGINT, _InterruptHandler)
  _LockFileAcquire(_LockFileCreate())
  _logger = tulsi_logging.Logger()
  logger_warning = tulsi_logging.validity_check()
  if logger_warning:
    _PrintXcodeWarning(logger_warning)
  _timer = Timer('Everything', 'complete_build').Start()
  _exit_code = main(sys.argv)
  _timer.End()
  sys.exit(_exit_code)
