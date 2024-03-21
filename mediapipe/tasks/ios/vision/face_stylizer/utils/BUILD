# Copyright 2024 The MediaPipe Authors.
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

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPFaceStylizerOptionsHelpers",
    srcs = ["sources/MPPFaceStylizerOptions+Helpers.mm"],
    hdrs = ["sources/MPPFaceStylizerOptions+Helpers.h"],
    deps = [
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/tasks/cc/vision/face_stylizer/proto:face_stylizer_graph_options_cc_proto",
        "//mediapipe/tasks/ios/core:MPPTaskOptionsProtocol",
        "//mediapipe/tasks/ios/core/utils:MPPBaseOptionsHelpers",
        "//mediapipe/tasks/ios/vision/face_stylizer:MPPFaceStylizerOptions",
    ],
)

objc_library(
    name = "MPPFaceStylizerResultHelpers",
    srcs = ["sources/MPPFaceStylizerResult+Helpers.mm"],
    hdrs = ["sources/MPPFaceStylizerResult+Helpers.h"],
    deps = [
        "//mediapipe/framework:packet",
        "//mediapipe/framework/formats:image",
        "//mediapipe/tasks/ios/vision/core/utils:MPPImageUtils",
        "//mediapipe/tasks/ios/vision/face_stylizer:MPPFaceStylizerResult",
    ],
)
