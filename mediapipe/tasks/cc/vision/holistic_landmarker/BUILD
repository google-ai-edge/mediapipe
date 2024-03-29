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

package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "holistic_face_tracking",
    srcs = ["holistic_face_tracking.cc"],
    hdrs = ["holistic_face_tracking.h"],
    deps = [
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2/stream:detections_to_rects",
        "//mediapipe/framework/api2/stream:image_size",
        "//mediapipe/framework/api2/stream:landmarks_to_detection",
        "//mediapipe/framework/api2/stream:loopback",
        "//mediapipe/framework/api2/stream:rect_transformation",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/modules/holistic_landmark/calculators:roi_tracking_calculator",
        "//mediapipe/modules/holistic_landmark/calculators:roi_tracking_calculator_cc_proto",
        "//mediapipe/tasks/cc/vision/face_detector:face_detector_graph",
        "//mediapipe/tasks/cc/vision/face_detector/proto:face_detector_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_blendshapes_graph",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker_graph",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarks_detector_graph",
        "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_blendshapes_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_landmarks_detector_graph_options_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "holistic_hand_tracking",
    srcs = ["holistic_hand_tracking.cc"],
    hdrs = ["holistic_hand_tracking.h"],
    deps = [
        "//mediapipe/calculators/util:align_hand_to_pose_in_world_calculator",
        "//mediapipe/calculators/util:align_hand_to_pose_in_world_calculator_cc_proto",
        "//mediapipe/calculators/util:landmark_visibility_calculator",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2/stream:image_size",
        "//mediapipe/framework/api2/stream:landmarks_to_detection",
        "//mediapipe/framework/api2/stream:loopback",
        "//mediapipe/framework/api2/stream:rect_transformation",
        "//mediapipe/framework/api2/stream:split",
        "//mediapipe/framework/api2/stream:threshold",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/modules/holistic_landmark/calculators:hand_detections_from_pose_to_rects_calculator",
        "//mediapipe/modules/holistic_landmark/calculators:roi_tracking_calculator",
        "//mediapipe/modules/holistic_landmark/calculators:roi_tracking_calculator_cc_proto",
        "//mediapipe/tasks/cc/components/utils:gate",
        "//mediapipe/tasks/cc/vision/hand_landmarker:hand_landmarker_graph",
        "//mediapipe/tasks/cc/vision/hand_landmarker:hand_landmarks_detector_graph",
        "//mediapipe/tasks/cc/vision/hand_landmarker:hand_roi_refinement_graph",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarker_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarks_detector_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_roi_refinement_graph_options_cc_proto",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "holistic_pose_tracking",
    srcs = ["holistic_pose_tracking.cc"],
    hdrs = ["holistic_pose_tracking.h"],
    deps = [
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2/stream:detections_to_rects",
        "//mediapipe/framework/api2/stream:image_size",
        "//mediapipe/framework/api2/stream:landmarks_to_detection",
        "//mediapipe/framework/api2/stream:loopback",
        "//mediapipe/framework/api2/stream:merge",
        "//mediapipe/framework/api2/stream:presence",
        "//mediapipe/framework/api2/stream:rect_transformation",
        "//mediapipe/framework/api2/stream:segmentation_smoothing",
        "//mediapipe/framework/api2/stream:smoothing",
        "//mediapipe/framework/api2/stream:split",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc/components/utils:gate",
        "//mediapipe/tasks/cc/vision/pose_detector:pose_detector_graph",
        "//mediapipe/tasks/cc/vision/pose_detector/proto:pose_detector_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/pose_landmarker:pose_landmarks_detector_graph",
        "//mediapipe/tasks/cc/vision/pose_landmarker/proto:pose_landmarks_detector_graph_options_cc_proto",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "holistic_landmarker_graph",
    srcs = ["holistic_landmarker_graph.cc"],
    visibility = ["//visibility:public"],  # Visibility is public until we have a C++ API.
    deps = [
        ":holistic_face_tracking",
        ":holistic_hand_tracking",
        ":holistic_pose_tracking",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2/stream:split",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc/core:model_asset_bundle_resources",
        "//mediapipe/tasks/cc/core:model_resources_cache",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/core:utils",
        "//mediapipe/tasks/cc/metadata/utils:zip_utils",
        "//mediapipe/tasks/cc/vision/face_detector/proto:face_detector_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_blendshapes_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_landmarks_detector_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarks_detector_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_roi_refinement_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/holistic_landmarker/proto:holistic_landmarker_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/pose_detector/proto:pose_detector_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/pose_landmarker:pose_topology",
        "//mediapipe/tasks/cc/vision/pose_landmarker/proto:pose_landmarks_detector_graph_options_cc_proto",
        "//mediapipe/util:graph_builder_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)
