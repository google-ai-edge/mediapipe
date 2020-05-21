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
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

mediapipe_simple_subgraph(
    name = "hand_detection_cpu",
    graph = "hand_detection_cpu.pbtxt",
    register_as = "HandDetectionSubgraph",
    deps = [
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_detections_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:detection_letterbox_removal_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "multi_hand_detection_cpu",
    graph = "multi_hand_detection_cpu.pbtxt",
    register_as = "MultiHandDetectionSubgraph",
    deps = [
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:clip_vector_size_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_detections_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:detection_letterbox_removal_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "hand_landmark_cpu",
    graph = "hand_landmark_cpu.pbtxt",
    register_as = "HandLandmarkSubgraph",
    deps = [
        "//mediapipe/calculators/core:split_normalized_landmark_list_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/image:image_cropping_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_classification_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_floats_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_landmarks_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:landmark_letterbox_removal_calculator",
        "//mediapipe/calculators/util:landmark_projection_calculator",
        "//mediapipe/calculators/util:landmarks_to_detection_calculator",
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator",
        "//mediapipe/calculators/util:thresholding_calculator",
        "//mediapipe/graphs/hand_tracking/calculators:hand_detections_to_rects_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "multi_hand_landmark_cpu",
    graph = "multi_hand_landmark.pbtxt",
    register_as = "MultiHandLandmarkSubgraph",
    deps = [
        ":hand_landmark_cpu",
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/util:filter_collection_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "renderer_cpu",
    graph = "renderer_cpu.pbtxt",
    register_as = "RendererSubgraph",
    deps = [
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:labels_to_render_data_calculator",
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "//mediapipe/calculators/util:rect_to_render_data_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "multi_hand_renderer_cpu",
    graph = "multi_hand_renderer_cpu.pbtxt",
    register_as = "MultiHandRendererSubgraph",
    deps = [
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "//mediapipe/calculators/util:rect_to_render_data_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "hand_detection_gpu",
    graph = "hand_detection_gpu.pbtxt",
    register_as = "HandDetectionSubgraph",
    deps = [
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_detections_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:detection_letterbox_removal_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "multi_hand_detection_gpu",
    graph = "multi_hand_detection_gpu.pbtxt",
    register_as = "MultiHandDetectionSubgraph",
    deps = [
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:clip_vector_size_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_detections_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:detection_letterbox_removal_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "hand_landmark_gpu",
    graph = "hand_landmark_gpu.pbtxt",
    register_as = "HandLandmarkSubgraph",
    deps = [
        "//mediapipe/calculators/core:split_normalized_landmark_list_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/image:image_cropping_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_classification_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_floats_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_landmarks_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:landmark_letterbox_removal_calculator",
        "//mediapipe/calculators/util:landmark_projection_calculator",
        "//mediapipe/calculators/util:landmarks_to_detection_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator",
        "//mediapipe/calculators/util:thresholding_calculator",
        "//mediapipe/graphs/hand_tracking/calculators:hand_detections_to_rects_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "multi_hand_landmark_gpu",
    graph = "multi_hand_landmark.pbtxt",
    register_as = "MultiHandLandmarkSubgraph",
    deps = [
        ":hand_landmark_gpu",
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/util:filter_collection_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "renderer_gpu",
    graph = "renderer_gpu.pbtxt",
    register_as = "RendererSubgraph",
    deps = [
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:labels_to_render_data_calculator",
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "//mediapipe/calculators/util:rect_to_render_data_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "multi_hand_renderer_gpu",
    graph = "multi_hand_renderer_gpu.pbtxt",
    register_as = "MultiHandRendererSubgraph",
    deps = [
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "//mediapipe/calculators/util:rect_to_render_data_calculator",
    ],
)
