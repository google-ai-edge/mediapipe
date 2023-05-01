# Copyright 2022 The MediaPipe Authors.
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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "keypoint",
    hdrs = ["keypoint.h"],
)

cc_library(
    name = "rect",
    srcs = ["rect.cc"],
    hdrs = ["rect.h"],
)

cc_library(
    name = "category",
    srcs = ["category.cc"],
    hdrs = ["category.h"],
    deps = [
        "//mediapipe/framework/formats:classification_cc_proto",
    ],
)

cc_library(
    name = "classification_result",
    srcs = ["classification_result.cc"],
    hdrs = ["classification_result.h"],
    deps = [
        ":category",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_cc_proto",
    ],
)

cc_library(
    name = "detection_result",
    srcs = ["detection_result.cc"],
    hdrs = ["detection_result.h"],
    deps = [
        ":category",
        ":keypoint",
        ":rect",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
    ],
)

cc_library(
    name = "embedding_result",
    srcs = ["embedding_result.cc"],
    hdrs = ["embedding_result.h"],
    deps = [
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_cc_proto",
    ],
)

cc_library(
    name = "landmark",
    srcs = ["landmark.cc"],
    hdrs = ["landmark.h"],
    deps = [
        "//mediapipe/framework/formats:landmark_cc_proto",
    ],
)
