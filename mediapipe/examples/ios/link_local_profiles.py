#!/usr/bin/env python3

# Copyright 2020 The MediaPipe Authors.
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

"""This script is used to set up automatic provisioning for iOS examples.

It scans the provisioning profiles used by Xcode, looking for one matching the
application identifier for each example app. If found, it symlinks the profile
in the appropriate location for Bazel to find it.

It also checks whether the bundle_id.bzl file has a placeholder bundle ID, and
replaces it with a unique ID if so.
"""

import os
import plistlib
import re
import subprocess
from typing import Optional
import uuid

# This script is meant to be located in the MediaPipe iOS examples directory
# root. The logic below will have to be changed if the directory structure is
# reorganized.
examples_ios = os.path.dirname(os.path.realpath(__file__))
example_names = {
    f for f in os.listdir(examples_ios)
    if os.path.isdir(os.path.join(examples_ios, f))
}


def configure_bundle_id_prefix(
    bundle_id_bzl=os.path.join(examples_ios, "bundle_id.bzl")) -> str:
  """Configures the bundle id prefix to use.

  Gets the bundle id prefix in use from bundle_id.bzl; sets up a unique
  prefix if not already set.

  Args:
    bundle_id_bzl: Path to the bzl file where the bundle id is stored.

  Returns:
    The bundle id prefix to use.

  Raises:
    Exception: If parsing of bundle_id.bzl fails.
  """
  bundle_id_re = re.compile(
      r'^BUNDLE_ID_PREFIX\s*=\s*"(.*)"', flags=re.MULTILINE)

  with open(bundle_id_bzl, "r") as f:
    contents = f.read()
  match = bundle_id_re.search(contents)
  if not match:
    raise Exception("could not find BUNDLE_ID_PREFIX")

  bundle_id_prefix = match.group(1)
  # The default value contains a *, which is an invalid character in bundle IDs.
  if "*" in bundle_id_prefix:
    bundle_id_prefix = str(uuid.uuid4()) + ".mediapipe.examples"
    contents = contents[:match.start(1)] + bundle_id_prefix + contents[match
                                                                       .end(1):]
    with open(bundle_id_bzl, "w") as f:
      f.write(contents)
    print("Set up a unique bundle ID prefix: " + bundle_id_prefix)

  return bundle_id_prefix


def get_app_id(profile_path) -> Optional[str]:
  try:
    plist = subprocess.check_output(
        ["security", "cms", "-D", "-i", profile_path])
    profile = plistlib.loads(plist)
    return profile["Entitlements"]["application-identifier"]
  except Exception:  # pylint: disable=broad-except
    return None


def update_symlink(target_path, link_path):
  if os.path.islink(link_path):
    print(f"  Removing existing symlink at {link_path}")
    os.remove(link_path)
  elif os.path.exists(link_path):
    print(f"  Unexpected existing file at {link_path}; skipping")
    return
  os.symlink(target_path, link_path)
  print(f"  Created symlink to {target_path} at {link_path}")


def process_profile(profile_path, our_app_id_re):
  """Processes one mobileprovision file.

  Checks if its app ID matches one of our example apps, and symlinks it in the
  appropriate location if so.

  Args:
    profile_path: Path to the mobileprovision file.
    our_app_id_re: Regular expression to extract the example name from one of
      out app ids.
  """
  app_id = get_app_id(profile_path)
  if not app_id:
    print(f"Could not parse '{profile_path}', skipping")
    return
  match = our_app_id_re.match(app_id)
  if not match:
    return
  app_name = match.group(1)
  app_dir_name = app_name.lower()
  if app_dir_name not in example_names:
    print(f"The app id '{app_id}' has our prefix, but does not seem to match" +
          "any of our examples. Skipping.")
    return

  print(f"Found profile for {app_name}")

  link_path = os.path.join(examples_ios, app_dir_name,
                           "provisioning_profile.mobileprovision")
  update_symlink(profile_path, link_path)


def main():
  bundle_id_prefix = configure_bundle_id_prefix()
  our_app_id_re = re.compile(r"[0-9A-Z]+\." + re.escape(bundle_id_prefix) +
                             r"\.(.*)")

  profile_dir = os.path.expanduser(
      "~/Library/MobileDevice/Provisioning Profiles")
  if not os.path.isdir(profile_dir):
    print(f"Could not find provisioning profiles directory at {profile_dir}")
    return 2

  print(
      f"Looking for profiles for app ids with prefix '{bundle_id_prefix}' in '{profile_dir}'"
  )

  profiles_found = False
  for name in os.listdir(profile_dir):
    if not name.endswith(".mobileprovision"):
      continue
    profiles_found = True
    profile_path = os.path.join(profile_dir, name)
    process_profile(profile_path, our_app_id_re)

  if not profiles_found:
    print("Error: Unable to find any provisioning profiles " +
          f"(*.mobileprovision files) in '{profile_dir}'")


if __name__ == "__main__":
  main()
