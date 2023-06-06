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

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPHandLandmarkerOptionsHelpers",
    srcs = ["sources/MPPHandLandmarkerOptions+Helpers.mm"],
    hdrs = ["sources/MPPHandLandmarkerOptions+Helpers.h"],
    deps = [
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/tasks/cc/vision/hand_detector/proto:hand_detector_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarker_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarks_detector_graph_options_cc_proto",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/core:MPPTaskOptionsProtocol",
        "//mediapipe/tasks/ios/core/utils:MPPBaseOptionsHelpers",
        "//mediapipe/tasks/ios/vision/hand_landmarker:MPPHandLandmarkerOptions",
    ],
)

objc_library(
    name = "MPPHandLandmarkerResultHelpers",
    srcs = ["sources/MPPHandLandmarkerResult+Helpers.mm"],
    hdrs = ["sources/MPPHandLandmarkerResult+Helpers.h"],
    deps = [
        "//mediapipe/framework:packet",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/tasks/ios/components/containers/utils:MPPCategoryHelpers",
        "//mediapipe/tasks/ios/components/containers/utils:MPPLandmarkHelpers",
        "//mediapipe/tasks/ios/vision/hand_landmarker:MPPHandLandmarkerResult",
    ],
)
