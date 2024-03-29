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
    name = "MPPHolisticLandmarkerResultProtobufHelpers",
    srcs = ["sources/MPPHolisticLandmarkerResult+ProtobufHelpers.mm"],
    hdrs = ["sources/MPPHolisticLandmarkerResult+ProtobufHelpers.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
        "-x objective-c++",
    ],
    deps = [
        "//mediapipe/tasks/cc/vision/holistic_landmarker/proto:holistic_result_cc_proto",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/test/vision/utils:parse_proto_utils",
        "//mediapipe/tasks/ios/vision/holistic_landmarker:MPPHolisticLandmarkerResult",
        "//mediapipe/tasks/ios/vision/holistic_landmarker/utils:MPPHolisticLandmarkerResultHelpers",
    ],
)
