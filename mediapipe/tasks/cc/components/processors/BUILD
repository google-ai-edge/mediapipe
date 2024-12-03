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

load("@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

cc_library(
    name = "classifier_options",
    srcs = ["classifier_options.cc"],
    hdrs = ["classifier_options.h"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/tasks/cc/components/processors/proto:classifier_options_cc_proto"],
)

cc_library(
    name = "classification_postprocessing_graph",
    srcs = ["classification_postprocessing_graph.cc"],
    hdrs = ["classification_postprocessing_graph.h"],
    deps = [
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/core:split_vector_calculator_cc_proto",
        "//mediapipe/calculators/tensor:tensors_dequantization_calculator",
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator",
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/calculators:classification_aggregation_calculator",
        "//mediapipe/tasks/cc/components/calculators:classification_aggregation_calculator_cc_proto",
        "//mediapipe/tasks/cc/components/calculators:score_calibration_calculator",
        "//mediapipe/tasks/cc/components/calculators:score_calibration_calculator_cc_proto",
        "//mediapipe/tasks/cc/components/calculators:score_calibration_utils",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_cc_proto",
        "//mediapipe/tasks/cc/components/processors/proto:classification_postprocessing_graph_options_cc_proto",
        "//mediapipe/tasks/cc/components/processors/proto:classifier_options_cc_proto",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "//mediapipe/tasks/metadata:metadata_schema_cc",
        "//mediapipe/util:label_map_cc_proto",
        "//mediapipe/util:label_map_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
    alwayslink = 1,
)

cc_library(
    name = "embedder_options",
    srcs = ["embedder_options.cc"],
    hdrs = ["embedder_options.h"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/tasks/cc/components/processors/proto:embedder_options_cc_proto"],
)

cc_library(
    name = "embedding_postprocessing_graph",
    srcs = ["embedding_postprocessing_graph.cc"],
    hdrs = ["embedding_postprocessing_graph.h"],
    deps = [
        "//mediapipe/calculators/tensor:tensors_dequantization_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/tool:options_map",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/calculators:embedding_aggregation_calculator",
        "//mediapipe/tasks/cc/components/calculators:tensors_to_embeddings_calculator",
        "//mediapipe/tasks/cc/components/calculators:tensors_to_embeddings_calculator_cc_proto",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_cc_proto",
        "//mediapipe/tasks/cc/components/processors/proto:embedder_options_cc_proto",
        "//mediapipe/tasks/cc/components/processors/proto:embedding_postprocessing_graph_options_cc_proto",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
    alwayslink = 1,
)

cc_library_with_tflite(
    name = "image_preprocessing_graph",
    srcs = ["image_preprocessing_graph.cc"],
    hdrs = ["image_preprocessing_graph.h"],
    tflite_deps = [
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/vision/utils:image_tensor_specs",
    ],
    deps = [
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/calculators/image:image_clone_calculator",
        "//mediapipe/calculators/image:image_clone_calculator_cc_proto",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator_cc_proto",
        "//mediapipe/calculators/tensor:inference_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/gpu:gpu_origin_cc_proto",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/processors/proto:image_preprocessing_graph_options_cc_proto",
        "//mediapipe/tasks/cc/core/proto:acceleration_cc_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
    alwayslink = 1,
)

# TODO: Enable this test

cc_library(
    name = "text_preprocessing_graph",
    srcs = ["text_preprocessing_graph.cc"],
    hdrs = ["text_preprocessing_graph.h"],
    deps = [
        "//mediapipe/calculators/tensor:bert_preprocessor_calculator",
        "//mediapipe/calculators/tensor:bert_preprocessor_calculator_cc_proto",
        "//mediapipe/calculators/tensor:regex_preprocessor_calculator",
        "//mediapipe/calculators/tensor:regex_preprocessor_calculator_cc_proto",
        "//mediapipe/calculators/tensor:text_to_tensor_calculator",
        "//mediapipe/calculators/tensor:universal_sentence_encoder_preprocessor_calculator",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/processors/proto:text_model_type_cc_proto",
        "//mediapipe/tasks/cc/components/processors/proto:text_preprocessing_graph_options_cc_proto",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "//mediapipe/tasks/cc/text/utils:text_model_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "detection_postprocessing_graph",
    srcs = ["detection_postprocessing_graph.cc"],
    hdrs = ["detection_postprocessing_graph.h"],
    deps = [
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/core:split_vector_calculator_cc_proto",
        "//mediapipe/calculators/tensor:tensors_dequantization_calculator",
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator",
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator_cc_proto",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator_cc_proto",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator_cc_proto",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/formats/object_detection:anchor_cc_proto",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/calculators:score_calibration_calculator",
        "//mediapipe/tasks/cc/components/calculators:score_calibration_calculator_cc_proto",
        "//mediapipe/tasks/cc/components/calculators:score_calibration_utils",
        "//mediapipe/tasks/cc/components/processors/proto:detection_postprocessing_graph_options_cc_proto",
        "//mediapipe/tasks/cc/components/processors/proto:detector_options_cc_proto",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/core:utils",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "//mediapipe/tasks/cc/vision/object_detector/proto:object_detector_options_cc_proto",
        "//mediapipe/tasks/metadata:metadata_schema_cc",
        "//mediapipe/tasks/metadata:object_detector_metadata_schema_cc",
        "//mediapipe/util:label_map_cc_proto",
        "//mediapipe/util:label_map_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers//:runtime_cc",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
    alwayslink = 1,
)
