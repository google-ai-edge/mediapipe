# Copyright 2017 The Tulsi Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logic to translate Xcode options to Bazel options."""


class BazelOptions(object):
  """Converts Xcode features into Bazel command line flags."""

  def __init__(self, xcode_env):
    """Creates a new BazelOptions object.

    Args:
      xcode_env: A dictionary of Xcode environment variables.

    Returns:
      A BazelOptions instance.
    """
    self.xcode_env = xcode_env

  def bazel_feature_flags(self):
    """Returns a list of bazel flags for the current Xcode env configuration."""
    flags = []
    if self.xcode_env.get('ENABLE_ADDRESS_SANITIZER') == 'YES':
      flags.append('--features=asan')
    if self.xcode_env.get('ENABLE_THREAD_SANITIZER') == 'YES':
      flags.append('--features=tsan')
    if self.xcode_env.get('ENABLE_UNDEFINED_BEHAVIOR_SANITIZER') == 'YES':
      flags.append('--features=ubsan')

    return flags
