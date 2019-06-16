# Copyright 2019 The MediaPipeOSS Authors.
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
#

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "android_calculators",
    deps = [
        "//mediapipe/calculators/core:real_time_flow_limiter_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_detections_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:detection_letterbox_removal_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",
    ],
)

cc_library(
    name = "desktop_tensorflow_calculators",
    deps = [
        "//mediapipe/calculators/tensorflow:image_frame_to_tensor_calculator",
        "//mediapipe/calculators/tensorflow:lapped_tensor_buffer_calculator",
        "//mediapipe/calculators/tensorflow:object_detection_tensors_to_detections_calculator",
        "//mediapipe/calculators/tensorflow:tensor_squeeze_dimensions_calculator",
        "//mediapipe/calculators/tensorflow:tensorflow_inference_calculator",
        "//mediapipe/calculators/tensorflow:tensorflow_session_from_saved_model_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "//mediapipe/calculators/video:opencv_video_encoder_calculator",
    ],
)

cc_library(
    name = "desktop_tflite_calculators",
    deps = [
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_detections_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "//mediapipe/calculators/video:opencv_video_encoder_calculator",
    ],
)

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)

mediapipe_binary_graph(
    name = "android_cpu_binary_graph",
    graph = "object_detection_android_cpu.pbtxt",
    output_name = "android_cpu.binarypb",
    deps = [
        "//mediapipe/calculators/image:image_transformation_calculator_proto",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator_proto",
        "//mediapipe/calculators/tflite:tflite_converter_calculator_proto",
        "//mediapipe/calculators/tflite:tflite_inference_calculator_proto",
        "//mediapipe/calculators/tflite:tflite_tensors_to_detections_calculator_proto",
        "//mediapipe/calculators/util:annotation_overlay_calculator_proto",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator_proto",
        "//mediapipe/calculators/util:detections_to_render_data_calculator_proto",
        "//mediapipe/calculators/util:non_max_suppression_calculator_proto",
    ],
)

mediapipe_binary_graph(
    name = "android_gpu_binary_graph",
    graph = "object_detection_android_gpu.pbtxt",
    output_name = "android_gpu.binarypb",
    deps = [
        "//mediapipe/calculators/image:image_transformation_calculator_proto",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator_proto",
        "//mediapipe/calculators/tflite:tflite_converter_calculator_proto",
        "//mediapipe/calculators/tflite:tflite_inference_calculator_proto",
        "//mediapipe/calculators/tflite:tflite_tensors_to_detections_calculator_proto",
        "//mediapipe/calculators/util:annotation_overlay_calculator_proto",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator_proto",
        "//mediapipe/calculators/util:detections_to_render_data_calculator_proto",
        "//mediapipe/calculators/util:non_max_suppression_calculator_proto",
    ],
)
