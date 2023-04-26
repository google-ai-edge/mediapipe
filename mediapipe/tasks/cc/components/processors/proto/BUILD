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

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_proto_library(
    name = "classifier_options_proto",
    srcs = ["classifier_options.proto"],
)

mediapipe_proto_library(
    name = "detector_options_proto",
    srcs = ["detector_options.proto"],
)

mediapipe_proto_library(
    name = "classification_postprocessing_graph_options_proto",
    srcs = ["classification_postprocessing_graph_options.proto"],
    deps = [
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator_proto",
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/tasks/cc/components/calculators:classification_aggregation_calculator_proto",
        "//mediapipe/tasks/cc/components/calculators:score_calibration_calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "detection_postprocessing_graph_options_proto",
    srcs = ["detection_postprocessing_graph_options.proto"],
    deps = [
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator_proto",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator_proto",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator_proto",
        "//mediapipe/calculators/util:non_max_suppression_calculator_proto",
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/tasks/cc/components/calculators:score_calibration_calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "embedder_options_proto",
    srcs = ["embedder_options.proto"],
)

mediapipe_proto_library(
    name = "embedding_postprocessing_graph_options_proto",
    srcs = ["embedding_postprocessing_graph_options.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/tasks/cc/components/calculators:tensors_to_embeddings_calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "image_preprocessing_graph_options_proto",
    srcs = ["image_preprocessing_graph_options.proto"],
    deps = [
        "//mediapipe/calculators/tensor:image_to_tensor_calculator_proto",
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "text_model_type_proto",
    srcs = ["text_model_type.proto"],
)

mediapipe_proto_library(
    name = "text_preprocessing_graph_options_proto",
    srcs = ["text_preprocessing_graph_options.proto"],
    deps = [
        ":text_model_type_proto",
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)
