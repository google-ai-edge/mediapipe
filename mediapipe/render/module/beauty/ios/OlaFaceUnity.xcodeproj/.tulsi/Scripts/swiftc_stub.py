#!/usr/bin/python3
# Copyright 2022 The Tulsi Authors. All rights reserved.
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

"""Stub to avoid swiftc but create the expected swiftc outputs."""

import json
import os
import pathlib
import subprocess
import sys


def _TouchFile(filepath):
  """Touch the given file: create if necessary and update its mtime."""
  pathlib.Path(filepath).touch()


def _HandleOutputMapFile(filepath):
  # Touch all output files referenced in the map. See the documentation here:
  # https://github.com/apple/swift/blob/main/docs/Driver.md#output-file-maps
  with open(filepath, 'rb') as file:
    output_map = json.load(file)
    for single_file_outputs in output_map.values():
      for output in single_file_outputs.values():
        _TouchFile(output)


def _CreateModuleFiles(module_path):
  _TouchFile(module_path)
  filename_no_ext = os.path.splitext(module_path)[0]
  _TouchFile(filename_no_ext + '.swiftdoc')
  _TouchFile(filename_no_ext + '.swiftsourceinfo')


def main(args):
  # Xcode may call `swiftc -v` which we need to pass through.
  if args == ['-v'] or args == ['--version']:
    return subprocess.call(['swiftc', '-v'])

  index = 0
  num_args = len(args)
  # Compare against length - 1 since we only care about arguments which come in
  # pairs.
  while index < num_args - 1:
    cur_arg = args[index]

    if cur_arg == '-output-file-map':
      index += 1
      output_file_map = args[index]
      _HandleOutputMapFile(output_file_map)
    elif cur_arg == '-emit-module-path':
      index += 1
      module_path = args[index]
      _CreateModuleFiles(module_path)
    elif cur_arg == '-emit-objc-header-path':
      index += 1
      header_path = args[index]
      _TouchFile(header_path)
    index += 1
  return 0


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
