# Copyright 2023 The MediaPipe Authors.
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

load("@rules_cc//cc:objc_library.bzl", "objc_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPInteractiveSegmenterOptionsHelpers",
    srcs = ["sources/MPPInteractiveSegmenterOptions+Helpers.mm"],
    hdrs = ["sources/MPPInteractiveSegmenterOptions+Helpers.h"],
    deps = [
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:image_segmenter_graph_options_cc_proto",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/core:MPPTaskOptionsProtocol",
        "//mediapipe/tasks/ios/core/utils:MPPBaseOptionsHelpers",
        "//mediapipe/tasks/ios/vision/interactive_segmenter:MPPInteractiveSegmenterOptions",
    ],
)

objc_library(
    name = "MPPInteractiveSegmenterResultHelpers",
    srcs = ["sources/MPPInteractiveSegmenterResult+Helpers.mm"],
    hdrs = ["sources/MPPInteractiveSegmenterResult+Helpers.h"],
    deps = [
        "//mediapipe/framework:packet",
        "//mediapipe/framework/formats:image",
        "//mediapipe/tasks/ios/vision/interactive_segmenter:MPPInteractiveSegmenterResult",
    ],
)
