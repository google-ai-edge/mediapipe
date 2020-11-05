# Copyright 2020 The MediaPipe Authors.
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

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_simple_subgraph(
    name = "objectron_detection_gpu",
    graph = "objectron_detection_gpu.pbtxt",
    register_as = "ObjectronDetectionSubgraphGpu",
    deps = [
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/graphs/object_detection_3d/calculators:tflite_tensors_to_objects_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "objectron_tracking_gpu",
    graph = "objectron_tracking_gpu.pbtxt",
    register_as = "ObjectronTrackingSubgraphGpu",
    deps = [
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/video:box_tracker_calculator",
        "//mediapipe/calculators/video:flow_packager_calculator",
        "//mediapipe/calculators/video:motion_analysis_calculator",
        "//mediapipe/framework/stream_handler:sync_set_input_stream_handler",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/graphs/object_detection_3d/calculators:frame_annotation_to_timed_box_list_calculator",
        "//mediapipe/graphs/object_detection_3d/calculators:frame_annotation_tracker_calculator",
        "//mediapipe/graphs/object_detection_3d/calculators:lift_2d_frame_annotation_to_3d_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "box_landmark_gpu",
    graph = "box_landmark_gpu.pbtxt",
    register_as = "BoxLandmarkSubgraph",
    deps = [
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/image:image_cropping_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_floats_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_landmarks_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:landmark_letterbox_removal_calculator",
        "//mediapipe/calculators/util:landmark_projection_calculator",
        "//mediapipe/calculators/util:landmarks_smoothing_calculator",
        "//mediapipe/calculators/util:landmarks_to_detection_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator",
        "//mediapipe/calculators/util:thresholding_calculator",
        "//mediapipe/graphs/object_detection_3d/calculators:frame_annotation_to_rect_calculator",
        "//mediapipe/graphs/object_detection_3d/calculators:landmarks_to_frame_annotation_calculator",
        "//mediapipe/graphs/object_detection_3d/calculators:lift_2d_frame_annotation_to_3d_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "object_detection_oid_v4_gpu",
    graph = "object_detection_oid_v4_gpu.pbtxt",
    register_as = "ObjectDetectionOidV4Subgraph",
    deps = [
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_detections_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator",
        "//mediapipe/graphs/object_detection_3d/calculators:filter_detection_calculator",
    ],
)
