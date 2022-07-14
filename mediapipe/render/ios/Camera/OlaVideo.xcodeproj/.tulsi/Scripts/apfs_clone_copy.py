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

"""Copy on write with similar behavior to shutil.copy2, when available."""

import errno
import os
import re
import shutil
import subprocess


def _APFSCheck(volume_path):
  """Reports if the given path belongs to an APFS volume.

  Args:
    volume_path: Absolute path to the volume we want to test.

  Returns:
    True if the volume has been formatted as APFS.
    False if not.
  """
  output = subprocess.check_output(['diskutil', 'info', volume_path],
                                   encoding='utf-8')
  # Match the output's "Type (Bundle): ..." entry to determine if apfs.
  target_fs = re.search(r'(?:Type \(Bundle\):) +([^ ]+)', output)
  if not target_fs:
    return False
  filesystem = target_fs.group(1)
  if 'apfs' not in filesystem:
    return False
  return True


def _IsOnDevice(path, st_dev):
  """Checks if a given path belongs to a FS on a given device.

  Args:
    path: a filesystem path, possibly to a non-existent file or directory.
    st_dev: the ID of a device with a filesystem, as in os.stat(...).st_dev.

  Returns:
    True if the path or (if the path does not exist) its closest existing
    ancestor exists on the device.
    False if not.
  """
  if not os.path.isabs(path):
    path = os.path.abspath(path)
  try:
    return os.stat(path).st_dev == st_dev
  except OSError as err:
    if err.errno == errno.ENOENT:
      dirname = os.path.dirname(path)
      if len(dirname) < len(path):
        return _IsOnDevice(dirname, st_dev)
  return False

# At launch, determine if the root filesystem is APFS.
IS_ROOT_APFS = _APFSCheck('/')

# At launch, determine the root filesystem device ID.
ROOT_ST_DEV = os.stat('/').st_dev


def CopyOnWrite(source, dest, tree=False):
  """Invokes cp -c to perform a CoW copy2 of all files, like clonefile(2).

  Args:
    source: Source path to copy.
    dest: Destination for copying.
    tree: "True" to copy all child files and folders, like shutil.copytree().
  """
  # Note that this is based on cp, so permissions are copied, unlike shutil's
  # copyfile method.
  #
  # Identical to shutil's copy2 method, used by shutil's move and copytree.
  cmd = ['cp']
  if IS_ROOT_APFS and _IsOnDevice(source, ROOT_ST_DEV) and _IsOnDevice(
      dest, ROOT_ST_DEV):
    # Copy on write (clone) is possible if both source and destination reside in
    # the same APFS volume. For simplicity, and since checking FS type can be
    # expensive, allow CoW only for the root volume.
    cmd.append('-c')
  if tree:
    # Copy recursively if indicated.
    cmd.append('-R')
    # Follow symlinks, emulating shutil.copytree defaults.
    cmd.append('-L')
  # Preserve all possible file attributes and permissions (copystat/copy2).
  cmd.extend(['-p', source, dest])
  try:
    # Attempt the copy action with cp.
    subprocess.check_output(cmd)
  except subprocess.CalledProcessError:
    # If -c is not supported, use shutil's copy2-based methods directly.
    if tree:
      # A partial tree might be left over composed of dirs but no files.
      # Remove them with rmtree so that they don't interfere with copytree.
      if os.path.exists(dest):
        shutil.rmtree(dest)
      shutil.copytree(source, dest)
    else:
      shutil.copy2(source, dest)
