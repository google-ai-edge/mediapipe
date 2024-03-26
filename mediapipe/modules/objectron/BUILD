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
    "//mediapipe/framework/tool:mediapipe_files.bzl",
    "mediapipe_files",
)
load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_files(srcs = [
    "object_detection_3d_camera.tflite",
    "object_detection_3d_chair.tflite",
    "object_detection_3d_chair_1stage.tflite",
    "object_detection_3d_cup.tflite",
    "object_detection_3d_sneakers.tflite",
    "object_detection_3d_sneakers_1stage.tflite",
    "object_detection_ssd_mobilenetv2_oidv4_fp16.tflite",
])

exports_files(srcs = [
    "object_detection_oidv4_labelmap.txt",
])

mediapipe_simple_subgraph(
    name = "objectron_detection_1stage_gpu",
    graph = "objectron_detection_1stage_gpu.pbtxt",
    register_as = "ObjectronDetection1StageSubgraphGpu",
    deps = [
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/modules/objectron/calculators:tflite_tensors_to_objects_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "objectron_tracking_1stage_gpu",
    graph = "objectron_tracking_1stage_gpu.pbtxt",
    register_as = "ObjectronTracking1StageSubgraphGpu",
    deps = [
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/video:box_tracker_calculator",
        "//mediapipe/calculators/video:flow_packager_calculator",
        "//mediapipe/calculators/video:motion_analysis_calculator",
        "//mediapipe/framework/stream_handler:sync_set_input_stream_handler",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/modules/objectron/calculators:frame_annotation_to_timed_box_list_calculator",
        "//mediapipe/modules/objectron/calculators:frame_annotation_tracker_calculator",
        "//mediapipe/modules/objectron/calculators:lift_2d_frame_annotation_to_3d_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "box_landmark_gpu",
    graph = "box_landmark_gpu.pbtxt",
    register_as = "BoxLandmarkSubgraph",
    deps = [
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensors_to_floats_calculator",
        "//mediapipe/calculators/tensor:tensors_to_landmarks_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:landmark_letterbox_removal_calculator",
        "//mediapipe/calculators/util:landmark_projection_calculator",
        "//mediapipe/calculators/util:landmarks_to_detection_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator",
        "//mediapipe/calculators/util:thresholding_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "box_landmark_cpu",
    graph = "box_landmark_cpu.pbtxt",
    register_as = "BoxLandmarkSubgraph",
    deps = [
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensors_to_floats_calculator",
        "//mediapipe/calculators/tensor:tensors_to_landmarks_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:landmark_letterbox_removal_calculator",
        "//mediapipe/calculators/util:landmark_projection_calculator",
        "//mediapipe/calculators/util:landmarks_to_detection_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator",
        "//mediapipe/calculators/util:thresholding_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "object_detection_oid_v4_gpu",
    graph = "object_detection_oid_v4_gpu.pbtxt",
    register_as = "ObjectDetectionOidV4Subgraph",
    deps = [
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/modules/objectron/calculators:filter_detection_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "object_detection_oid_v4_cpu",
    graph = "object_detection_oid_v4_cpu.pbtxt",
    register_as = "ObjectDetectionOidV4Subgraph",
    deps = [
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/modules/objectron/calculators:filter_detection_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "objectron_cpu",
    graph = "objectron_cpu.pbtxt",
    register_as = "ObjectronCpuSubgraph",
    deps = [
        ":box_landmark_cpu",
        ":object_detection_oid_v4_cpu",
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:clip_vector_size_calculator",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/tflite:tflite_model_calculator",
        "//mediapipe/calculators/util:association_norm_rect_calculator",
        "//mediapipe/calculators/util:collection_has_min_size_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:local_file_contents_calculator",
        "//mediapipe/modules/objectron/calculators:frame_annotation_to_rect_calculator",
        "//mediapipe/modules/objectron/calculators:landmarks_to_frame_annotation_calculator",
        "//mediapipe/modules/objectron/calculators:lift_2d_frame_annotation_to_3d_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "objectron_gpu",
    graph = "objectron_gpu.pbtxt",
    register_as = "ObjectronGpuSubgraph",
    deps = [
        ":box_landmark_gpu",
        ":object_detection_oid_v4_gpu",
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:clip_vector_size_calculator",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/util:association_norm_rect_calculator",
        "//mediapipe/calculators/util:collection_has_min_size_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/modules/objectron/calculators:frame_annotation_to_rect_calculator",
        "//mediapipe/modules/objectron/calculators:landmarks_to_frame_annotation_calculator",
        "//mediapipe/modules/objectron/calculators:lift_2d_frame_annotation_to_3d_calculator",
    ],
)
