# Copyright 2019 The MediaPipe Authors.
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

load(
    "@build_bazel_rules_apple//apple:ios.bzl",
    "ios_application",
)
load(
    "//mediapipe/examples/ios:bundle_id.bzl",
    "BUNDLE_ID_PREFIX",
    "example_provisioning",
)

licenses(["notice"])

MIN_IOS_VERSION = "10.0"

alias(
    name = "helloworld",
    actual = "HelloWorldApp",
)

ios_application(
    name = "HelloWorldApp",
    app_icons = ["//mediapipe/examples/ios/common:AppIcon"],
    bundle_id = BUNDLE_ID_PREFIX + ".HelloWorld",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//mediapipe/examples/ios/common:Info.plist",
        "Info.plist",
    ],
    minimum_os_version = MIN_IOS_VERSION,
    provisioning_profile = example_provisioning(),
    deps = [":HelloWorldAppLibrary"],
)

objc_library(
    name = "HelloWorldAppLibrary",
    data = [
        "//mediapipe/graphs/edge_detection:mobile_gpu_binary_graph",
    ],
    deps = [
        "//mediapipe/examples/ios/common:CommonMediaPipeAppLibrary",
        "//mediapipe/graphs/edge_detection:mobile_calculators",
    ],
)
