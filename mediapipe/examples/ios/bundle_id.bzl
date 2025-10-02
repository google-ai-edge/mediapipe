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

"""Configuration helper for iOS app bundle ids and provisioning profiles.
"""

# Set a valid bundle identifier prefix for local builds.
# For simulator builds, the provisioning profile is ignored, but a valid
# bundle id format is still required by rules_apple.
BUNDLE_ID_PREFIX = "com.codexmp.mediapipe.examples"

# Look for a provisioning profile in the example's directory first,
# otherwise look for a common one.
def example_provisioning():
    local_profile = native.glob(["provisioning_profile.mobileprovision"])
    if local_profile:
        return local_profile[0]
    return "//mediapipe/examples/ios:provisioning_profile"
