#!/usr/bin/python3
# Copyright 2018 The Tulsi Authors. All rights reserved.
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

"""Invokes Bazel builds for the given target using Tulsi specific flags."""


import argparse
import pipes
import subprocess
import sys
from bazel_build_settings import BUILD_SETTINGS


def _FatalError(msg, exit_code=1):
  """Prints a fatal error message to stderr and exits."""
  sys.stderr.write(msg)
  sys.exit(exit_code)


def _BuildSettingsTargetForTargets(targets):
  """Returns the singular target to use when fetching build settings."""
  return targets[0] if len(targets) == 1 else None


def _CreateCommand(targets, build_settings, test, release,
                   config, xcode_version, force_swift):
  """Creates a Bazel command for targets with the specified settings."""
  target = _BuildSettingsTargetForTargets(targets)
  bazel, startup, flags = build_settings.flags_for_target(
      target, not release, config, is_swift_override=force_swift)
  bazel_action = 'test' if test else 'build'

  command = [bazel]
  command.extend(startup)
  command.append(bazel_action)
  command.extend(flags)
  if xcode_version:
    command.append('--xcode_version=%s' % xcode_version)
  command.append('--tool_tag=tulsi:user_build')
  command.extend(targets)

  return command


def _QuoteCommandForShell(cmd):
  cmd = [pipes.quote(x) for x in cmd]
  return ' '.join(cmd)


def _InterruptSafeCall(cmd):
  p = subprocess.Popen(cmd)
  try:
    return p.wait()
  except KeyboardInterrupt:
    return p.wait()


def main():
  if not BUILD_SETTINGS:
    _FatalError('Unable to fetch build settings. Please report a Tulsi bug.')

  default_config = BUILD_SETTINGS.defaultPlatformConfigId
  config_options = BUILD_SETTINGS.platformConfigFlags
  config_help = (
      'Bazel apple config (used for flags). Default: {}').format(default_config)

  parser = argparse.ArgumentParser(description='Invoke a Bazel build or test '
                                               'with the same flags as Tulsi.')
  parser.add_argument('--test', dest='test', action='store_true', default=False)
  parser.add_argument('--release', dest='release', action='store_true',
                      default=False)
  parser.add_argument('--noprint_cmd', dest='print_cmd', action='store_false',
                      default=True)
  parser.add_argument('--norun', dest='run', action='store_false', default=True)
  parser.add_argument('--config', help=config_help, default=default_config,
                      choices=config_options)
  parser.add_argument('--xcode_version', help='Bazel --xcode_version flag.')
  parser.add_argument('--force_swift', dest='swift', action='store_true',
                      default=None, help='Forcibly treat the given targets '
                                         'as containing Swift.')
  parser.add_argument('--force_noswift', dest='swift', action='store_false',
                      default=None, help='Forcibly treat the given targets '
                                         'as not containing Swift.')
  parser.add_argument('targets', nargs='+')

  args = parser.parse_args()
  command = _CreateCommand(args.targets, BUILD_SETTINGS, args.test,
                           args.release, args.config, args.xcode_version,
                           args.swift)
  if args.print_cmd:
    print(_QuoteCommandForShell(command))

  if args.run:
    return _InterruptSafeCall(command)
  return 0


if __name__ == '__main__':
  sys.exit(main())
