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

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_library(
    name = "object_detection_tensorflow_deps",
    deps = [
        "@org_tensorflow//tensorflow/c/kernels:bitcast_op",
        "@org_tensorflow//tensorflow/core:direct_session",
        "@org_tensorflow//tensorflow/core/kernels:argmax_op",
        "@org_tensorflow//tensorflow/core/kernels:bias_op",
        "@org_tensorflow//tensorflow/core/kernels:cast_op",
        "@org_tensorflow//tensorflow/core/kernels:concat_op",
        "@org_tensorflow//tensorflow/core/kernels:constant_op",
        "@org_tensorflow//tensorflow/core/kernels:control_flow_ops",
        "@org_tensorflow//tensorflow/core/kernels:conv_ops",
        "@org_tensorflow//tensorflow/core/kernels:cwise_op",
        "@org_tensorflow//tensorflow/core/kernels:depthwise_conv_op",
        "@org_tensorflow//tensorflow/core/kernels:fused_batch_norm_op",
        "@org_tensorflow//tensorflow/core/kernels:gather_op",
        "@org_tensorflow//tensorflow/core/kernels:identity_op",
        "@org_tensorflow//tensorflow/core/kernels:logging_ops",
        "@org_tensorflow//tensorflow/core/kernels:matmul_op",
        "@org_tensorflow//tensorflow/core/kernels:non_max_suppression_op",
        "@org_tensorflow//tensorflow/core/kernels:pack_op",
        "@org_tensorflow//tensorflow/core/kernels:reduction_ops",
        "@org_tensorflow//tensorflow/core/kernels:relu_op",
        "@org_tensorflow//tensorflow/core/kernels:reshape_op",
        "@org_tensorflow//tensorflow/core/kernels:resize_bilinear_op",
        "@org_tensorflow//tensorflow/core/kernels:sequence_ops",
        "@org_tensorflow//tensorflow/core/kernels:shape_ops",
        "@org_tensorflow//tensorflow/core/kernels:slice_op",
        "@org_tensorflow//tensorflow/core/kernels:split_op",
        "@org_tensorflow//tensorflow/core/kernels:tensor_array_ops",
        "@org_tensorflow//tensorflow/core/kernels:tile_ops",
        "@org_tensorflow//tensorflow/core/kernels:topk_op",
        "@org_tensorflow//tensorflow/core/kernels:transpose_op",
        "@org_tensorflow//tensorflow/core/kernels:unpack_op",
        "@org_tensorflow//tensorflow/core/kernels/data:tensor_dataset_op",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "object_detection_tensorflow",
    deps = [
        ":object_detection_tensorflow_deps",
        "//mediapipe/examples/desktop:simple_run_graph_main",
        "//mediapipe/graphs/object_detection:desktop_tensorflow_calculators",
        "@org_tensorflow//tensorflow/core:direct_session",
    ],
)

cc_binary(
    name = "object_detection_tflite",
    deps = [
        "//mediapipe/examples/desktop:simple_run_graph_main",
        "//mediapipe/graphs/object_detection:desktop_tflite_calculators",
    ],
)
