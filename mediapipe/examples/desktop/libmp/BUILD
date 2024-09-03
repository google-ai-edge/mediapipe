# Copyright 2021 The MediaPipe Authors.
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

## Thank you to willstot101 on github for this build
# https://github.com/willstott101/meta-mediapipe/blob/master/recipes-framework/mediapipe/files/0002-Build-library.patch

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

BIN_DATA = [
    # Face detection
    "//mediapipe/modules/face_detection:face_detection_full_range.tflite",
    "//mediapipe/modules/face_detection:face_detection_full_range_sparse.tflite",
    "//mediapipe/modules/face_detection:face_detection_short_range.tflite",

    # Face mesh
    "//mediapipe/modules/face_landmark:face_landmark.tflite",
    "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",

    # Hand landmark
    "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
    "//mediapipe/modules/hand_landmark:hand_landmark_lite.tflite",

    # Holistic
#     "//mediapipe/modules/holistic_landmark:hand_recrop.tflite",

    # Iris
#     "//mediapipe/modules/iris_landmark:iris_landmark.tflite",

    # Object detection
    "//mediapipe/models:ssdlite_object_detection.tflite",
    "//mediapipe/modules/objectron:object_detection_3d_camera.tflite",
    "//mediapipe/modules/objectron:object_detection_3d_chair.tflite",
    "//mediapipe/modules/objectron:object_detection_3d_chair_1stage.tflite",
    "//mediapipe/modules/objectron:object_detection_3d_cup.tflite",
    "//mediapipe/modules/objectron:object_detection_3d_sneakers.tflite",
    "//mediapipe/modules/objectron:object_detection_3d_sneakers_1stage.tflite",
    "//mediapipe/modules/objectron:object_detection_ssd_mobilenetv2_oidv4_fp16.tflite",

    # Palm detection
    "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
    "//mediapipe/modules/palm_detection:palm_detection_lite.tflite",
    
    # Pose detection
    "//mediapipe/modules/pose_detection:pose_detection.tflite",

    # Pose landmark
    "//mediapipe/modules/pose_landmark:pose_landmark_full.tflite",
    "//mediapipe/modules/pose_landmark:pose_landmark_heavy.tflite",
    "//mediapipe/modules/pose_landmark:pose_landmark_lite.tflite",

#     # Selfie segmentation
#     "//mediapipe/modules/selfie_segmentation:selfie_segmentation.tflite",
#     "//mediapipe/modules/selfie_segmentation:selfie_segmentation_landscape.tflite",

#     # Hair segmentation
#     "//mediapipe/models:hair_segmentation.tflite",

#     "//mediapipe/models:ssdlite_object_detection.tflite",
    "//mediapipe/models:ssdlite_object_detection_labelmap.txt"
]

LIB_DEPS = [
    # Deps for libmp_impl.cpp
    "//mediapipe/framework/port:parse_text_proto",
    "@com_google_absl//absl/flags:parse",

#     # Remove any of these if you need to build only one example
#     # Selfie segmentation
#     "//mediapipe/graphs/selfie_segmentation:selfie_segmentation_cpu_deps",

    # Hand tracking
    "//mediapipe/modules/hand_landmark:hand_landmark_tracking_cpu",
    "//mediapipe/calculators/util:annotation_overlay_calculator",
    "//mediapipe/calculators/util:detections_to_render_data_calculator",

    # Face detection
#     "//mediapipe/graphs/face_detection:face_detection_full_range_desktop_live_deps",

    # Face mesh
#     "//mediapipe/graphs/face_mesh:desktop_live_calculators",

    # Face geometry (thanks, @oUp2Uo!)
    # - Fixes from https://github.com/google/mediapipe/issues/2867 have been applied,
    #   allowing for use of face mesh with attention (discards the 10 iris landmarks)
    # - See https://github.com/google/mediapipe/issues/1162#issuecomment-1434268379 for details
    "//mediapipe/modules/face_geometry:face_geometry_from_detection",
    "//mediapipe/modules/face_geometry:env_generator_calculator",
    "//mediapipe/modules/face_geometry:geometry_pipeline_calculator",

#     # Hair segmentation
#     "//mediapipe/graphs/hair_segmentation:desktop_calculators",

    # Holistic tracking
#     "//mediapipe/graphs/holistic_tracking:holistic_tracking_cpu_graph_deps",

    # Iris tracking
#     "//mediapipe/graphs/iris_tracking:iris_tracking_cpu_deps",

    # Object detection
    "//mediapipe/calculators/core:concatenate_vector_calculator",
    "//mediapipe/calculators/core:flow_limiter_calculator",
    "//mediapipe/calculators/core:previous_loopback_calculator",
    "//mediapipe/calculators/core:split_vector_calculator",
    "//mediapipe/calculators/image:image_transformation_calculator",
    "//mediapipe/calculators/tflite:ssd_anchors_calculator",
    "//mediapipe/calculators/tflite:tflite_converter_calculator",
    "//mediapipe/calculators/tflite:tflite_inference_calculator",
    "//mediapipe/calculators/tflite:tflite_tensors_to_detections_calculator",
    "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
    "//mediapipe/calculators/util:non_max_suppression_calculator",

    # Pose tracking
    "//mediapipe/graphs/pose_tracking:pose_tracking_cpu_deps",


    # tasks
    "//mediapipe/tasks/cc/audio/audio_classifier",
    "//mediapipe/tasks/cc/vision/image_classifier", 
    "//mediapipe/tasks/cc/vision/object_detector", 
    "//mediapipe/tasks/cc/vision/hand_landmarker",
    "//mediapipe/tasks/cc/vision/pose_landmarker",
    "//mediapipe/graphs/tracking:desktop_calculators", 
    "//mediapipe/calculators/tensor:bert_preprocessor_calculator",    
    "//mediapipe/calculators/core:pass_through_calculator",
    "//mediapipe/calculators/tensor:tensors_to_floats_calculator",
    "//mediapipe/tasks/cc/metadata:metadata_extractor",
    "//mediapipe/framework:packet",
    "//mediapipe/tasks/cc/text/tokenizers:bert_tokenizer",
    "//mediapipe/calculators/tensorflow:tensorflow_session",

]

cc_binary(
    target_compatible_with = [
        "@platforms//os:linux",
    ],
    name = "libmp.so",
    deps = [":libmp_linux"],
    data = BIN_DATA,
    linkopts = ["-Wl,-soname,libmp.so"],    
    linkshared = 1, ## Very important
)

# cc_library(
#     name = "glog_static",
#     deps = ["@com_github_glog_glog//:glog"],
#     linkstatic = 1,
# )

cc_library(
    target_compatible_with = [
        "@platforms//os:linux",
    ],
    name = "libmp_linux",
    srcs = ["libmp.cpp", "libmp_impl.cpp"],
    hdrs = ["libmp.h", "libmp_impl.h"],
    visibility = ["//visibility:public"],
    deps = LIB_DEPS,
#     deps = LIB_DEPS + [":glog_static"],
    alwayslink = 1,
)

