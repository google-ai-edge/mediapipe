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
#

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:private"])

proto_library(
    name = "graph_tensors_packet_generator_proto",
    srcs = ["graph_tensors_packet_generator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/framework:packet_generator_proto",
    ],
)

proto_library(
    name = "matrix_to_tensor_calculator_options_proto",
    srcs = ["matrix_to_tensor_calculator_options.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

proto_library(
    name = "lapped_tensor_buffer_calculator_proto",
    srcs = ["lapped_tensor_buffer_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

proto_library(
    name = "object_detection_tensors_to_detections_calculator_proto",
    srcs = ["object_detection_tensors_to_detections_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

proto_library(
    name = "tensorflow_inference_calculator_proto",
    srcs = ["tensorflow_inference_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

proto_library(
    name = "tensor_squeeze_dimensions_calculator_proto",
    srcs = ["tensor_squeeze_dimensions_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

proto_library(
    name = "tensor_to_image_frame_calculator_proto",
    srcs = ["tensor_to_image_frame_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

proto_library(
    name = "tensor_to_matrix_calculator_proto",
    srcs = ["tensor_to_matrix_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/framework/formats:time_series_header_proto",
    ],
)

proto_library(
    name = "tensor_to_vector_float_calculator_options_proto",
    srcs = ["tensor_to_vector_float_calculator_options.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

proto_library(
    name = "tensor_to_vector_string_calculator_options_proto",
    srcs = ["tensor_to_vector_string_calculator_options.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

proto_library(
    name = "unpack_media_sequence_calculator_proto",
    srcs = ["unpack_media_sequence_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/calculators/core:packet_resampler_calculator_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:audio_decoder_proto",
    ],
)

proto_library(
    name = "vector_float_to_tensor_calculator_options_proto",
    srcs = ["vector_float_to_tensor_calculator_options.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

proto_library(
    name = "vector_string_to_tensor_calculator_options_proto",
    srcs = ["vector_string_to_tensor_calculator_options.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "graph_tensors_packet_generator_cc_proto",
    srcs = ["graph_tensors_packet_generator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:packet_generator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":graph_tensors_packet_generator_proto"],
)

mediapipe_cc_proto_library(
    name = "image_frame_to_tensor_calculator_cc_proto",
    srcs = ["image_frame_to_tensor_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
    visibility = ["//visibility:public"],
    deps = [":image_frame_to_tensor_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "matrix_to_tensor_calculator_options_cc_proto",
    srcs = ["matrix_to_tensor_calculator_options.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":matrix_to_tensor_calculator_options_proto"],
)

mediapipe_cc_proto_library(
    name = "lapped_tensor_buffer_calculator_cc_proto",
    srcs = ["lapped_tensor_buffer_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":lapped_tensor_buffer_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "object_detection_tensors_to_detections_calculator_cc_proto",
    srcs = ["object_detection_tensors_to_detections_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":object_detection_tensors_to_detections_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "pack_media_sequence_calculator_cc_proto",
    srcs = ["pack_media_sequence_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
    visibility = ["//visibility:public"],
    deps = [":pack_media_sequence_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "tensorflow_inference_calculator_cc_proto",
    srcs = ["tensorflow_inference_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":tensorflow_inference_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "tensorflow_session_from_frozen_graph_generator_cc_proto",
    srcs = ["tensorflow_session_from_frozen_graph_generator.proto"],
    cc_deps = [
        "//mediapipe/framework:packet_generator_cc_proto",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
    visibility = ["//visibility:public"],
    deps = [":tensorflow_session_from_frozen_graph_generator_proto"],
)

mediapipe_cc_proto_library(
    name = "tensorflow_session_from_frozen_graph_calculator_cc_proto",
    srcs = ["tensorflow_session_from_frozen_graph_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
    visibility = ["//visibility:public"],
    deps = [":tensorflow_session_from_frozen_graph_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "tensorflow_session_from_saved_model_generator_cc_proto",
    srcs = ["tensorflow_session_from_saved_model_generator.proto"],
    cc_deps = [
        "//mediapipe/framework:packet_generator_cc_proto",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
    visibility = ["//visibility:public"],
    deps = [":tensorflow_session_from_saved_model_generator_proto"],
)

mediapipe_cc_proto_library(
    name = "tensorflow_session_from_saved_model_calculator_cc_proto",
    srcs = ["tensorflow_session_from_saved_model_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
    visibility = ["//visibility:public"],
    deps = [":tensorflow_session_from_saved_model_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "tensor_squeeze_dimensions_calculator_cc_proto",
    srcs = ["tensor_squeeze_dimensions_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":tensor_squeeze_dimensions_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "tensor_to_image_frame_calculator_cc_proto",
    srcs = ["tensor_to_image_frame_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":tensor_to_image_frame_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "tensor_to_matrix_calculator_cc_proto",
    srcs = ["tensor_to_matrix_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/formats:time_series_header_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":tensor_to_matrix_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "tensor_to_vector_float_calculator_options_cc_proto",
    srcs = ["tensor_to_vector_float_calculator_options.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":tensor_to_vector_float_calculator_options_proto"],
)

mediapipe_cc_proto_library(
    name = "tensor_to_vector_string_calculator_options_cc_proto",
    srcs = ["tensor_to_vector_string_calculator_options.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":tensor_to_vector_string_calculator_options_proto"],
)

mediapipe_cc_proto_library(
    name = "unpack_media_sequence_calculator_cc_proto",
    srcs = ["unpack_media_sequence_calculator.proto"],
    cc_deps = [
        "//mediapipe/calculators/core:packet_resampler_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/util:audio_decoder_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":unpack_media_sequence_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "vector_int_to_tensor_calculator_options_cc_proto",
    srcs = ["vector_int_to_tensor_calculator_options.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
    visibility = ["//visibility:public"],
    deps = [":vector_int_to_tensor_calculator_options_proto"],
)

mediapipe_cc_proto_library(
    name = "vector_float_to_tensor_calculator_options_cc_proto",
    srcs = ["vector_float_to_tensor_calculator_options.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":vector_float_to_tensor_calculator_options_proto"],
)

mediapipe_cc_proto_library(
    name = "vector_string_to_tensor_calculator_options_cc_proto",
    srcs = ["vector_string_to_tensor_calculator_options.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":vector_string_to_tensor_calculator_options_proto"],
)

cc_library(
    name = "graph_tensors_packet_generator",
    srcs = ["graph_tensors_packet_generator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_tensors_packet_generator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:status_util",
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "image_frame_to_tensor_calculator",
    srcs = ["image_frame_to_tensor_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":image_frame_to_tensor_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ] + select({
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:framework",
        ],
        "//mediapipe:android": [
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "matrix_to_tensor_calculator",
    srcs = ["matrix_to_tensor_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/formats:time_series_header_cc_proto",
        ":matrix_to_tensor_calculator_options_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:ret_check",
    ] + select({
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:framework",
        ],
        "//mediapipe:android": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "lapped_tensor_buffer_calculator",
    srcs = ["lapped_tensor_buffer_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":lapped_tensor_buffer_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/profiler:circular_buffer",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "object_detection_tensors_to_detections_calculator",
    srcs = ["object_detection_tensors_to_detections_calculator.cc"],
    features = [
        # Layering check doesn't play nicely with portable proto wrappers.
        "no_layering_check",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":object_detection_tensors_to_detections_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:tensor_to_detection",
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "pack_media_sequence_calculator",
    srcs = ["pack_media_sequence_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/calculators/image:opencv_image_encoder_calculator_cc_proto",
        "//mediapipe/calculators/tensorflow:pack_media_sequence_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util/sequence:media_sequence",
        "//mediapipe/util/sequence:media_sequence_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

cc_library(
    name = "string_to_sequence_example_calculator",
    srcs = ["string_to_sequence_example_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

# On android, this calculator is configured to run with lite protos. Therefore,
# compile your binary with the flag TENSORFLOW_PROTOS=lite.
cc_library(
    name = "tensorflow_inference_calculator",
    srcs = ["tensorflow_inference_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":tensorflow_session",
        ":tensorflow_inference_calculator_cc_proto",
        "//mediapipe/framework:timestamp",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/tool:status_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "//mediapipe/framework/deps:clock",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:map_util",
        "//mediapipe/framework:packet",
    ] + select({
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:framework",
        ],
        "//mediapipe:android": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//mediapipe:ios": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "tensorflow_session",
    hdrs = [
        "tensorflow_session.h",
    ],
    features = ["no_layering_check"],
    visibility = ["//visibility:public"],
    deps = select({
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:core",
        ],
        "//mediapipe:android": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//mediapipe:ios": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib",
        ],
    }),
)

cc_library(
    name = "tensorflow_session_from_frozen_graph_calculator",
    srcs = ["tensorflow_session_from_frozen_graph_calculator.cc"],
    features = ["no_layering_check"],
    visibility = ["//visibility:public"],
    deps = [
        ":tensorflow_session",
        "//mediapipe/calculators/tensorflow:tensorflow_session_from_frozen_graph_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/tool:status_util",
        "//mediapipe/framework/deps:clock",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:ret_check",
    ] + select({
        "//conditions:default": [
            "//mediapipe/framework/port:file_helpers",
            "@org_tensorflow//tensorflow/core:core",
        ],
        "//mediapipe:android": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
            "//mediapipe/android/file/base",
        ],
        "//mediapipe:ios": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib",
            "//mediapipe/android/file/base",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "tensorflow_session_from_frozen_graph_generator",
    srcs = ["tensorflow_session_from_frozen_graph_generator.cc"],
    features = ["no_layering_check"],
    visibility = ["//visibility:public"],
    deps = [
        ":tensorflow_session",
        ":tensorflow_session_from_frozen_graph_generator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/tool:status_util",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/deps:clock",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
    ] + select({
        "//conditions:default": [
            "//mediapipe/framework/port:file_helpers",
            "@org_tensorflow//tensorflow/core:core",
        ],
        "//mediapipe:android": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
            "//mediapipe/android/file/base",
        ],
        "//mediapipe:ios": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib",
            "//mediapipe/android/file/base",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "tensorflow_session_from_saved_model_calculator",
    srcs = ["tensorflow_session_from_saved_model_calculator.cc"],
    defines = select({
        "//mediapipe:android": ["__ANDROID__"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":tensorflow_session",
        ":tensorflow_session_from_saved_model_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/cc/saved_model:constants",
        "@org_tensorflow//tensorflow/cc/saved_model:loader_lite",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:ret_check",
    ] + select({
        "//conditions:default": [
        ],
        "//mediapipe:android": [],
    }) + select(
        {
            "//conditions:default": [
            ],
        },
    ) + select(
        {
            "//conditions:default": [
            ],
            "//mediapipe:android": [
            ],
        },
    ),
    alwayslink = 1,
)

cc_library(
    name = "tensorflow_session_from_saved_model_generator",
    srcs = ["tensorflow_session_from_saved_model_generator.cc"],
    defines = select({
        "//mediapipe:android": ["__ANDROID__"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":tensorflow_session",
        ":tensorflow_session_from_saved_model_generator_cc_proto",
        "//mediapipe/framework:packet_generator",
        "//mediapipe/framework:packet_type",
        "//mediapipe/framework/tool:status_util",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/cc/saved_model:constants",
        "@org_tensorflow//tensorflow/cc/saved_model:loader_lite",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:ret_check",
    ] + select({
        "//conditions:default": [
            "//mediapipe/framework/port:file_helpers",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "tensor_squeeze_dimensions_calculator",
    srcs = ["tensor_squeeze_dimensions_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":tensor_squeeze_dimensions_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tensor_to_image_frame_calculator",
    srcs = ["tensor_to_image_frame_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":tensor_to_image_frame_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tensor_to_matrix_calculator",
    srcs = ["tensor_to_matrix_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/formats:time_series_header_cc_proto",
        ":tensor_to_matrix_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:ret_check",
    ] + select({
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:framework",
        ],
        "//mediapipe:android": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "tfrecord_reader_calculator",
    srcs = ["tfrecord_reader_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tensor_to_vector_float_calculator",
    srcs = ["tensor_to_vector_float_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:ret_check",
        ":tensor_to_vector_float_calculator_options_cc_proto",
    ] + select({
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:framework",
        ],
        "//mediapipe:android": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "tensor_to_vector_string_calculator",
    srcs = ["tensor_to_vector_string_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:ret_check",
        ":tensor_to_vector_string_calculator_options_cc_proto",
    ] + select({
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:framework",
        ],
        "//mediapipe:android": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "unpack_media_sequence_calculator",
    srcs = ["unpack_media_sequence_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/calculators/core:packet_resampler_calculator_cc_proto",
        "//mediapipe/calculators/tensorflow:unpack_media_sequence_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:audio_decoder_cc_proto",
        "//mediapipe/util/sequence:media_sequence",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

cc_library(
    name = "vector_int_to_tensor_calculator",
    srcs = ["vector_int_to_tensor_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":vector_int_to_tensor_calculator_options_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "vector_float_to_tensor_calculator",
    srcs = ["vector_float_to_tensor_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":vector_float_to_tensor_calculator_options_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "vector_string_to_tensor_calculator",
    srcs = ["vector_string_to_tensor_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":vector_string_to_tensor_calculator_options_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "unpack_yt8m_sequence_example_calculator",
    srcs = ["unpack_yt8m_sequence_example_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":lapped_tensor_buffer_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/port:status",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

cc_test(
    name = "graph_tensors_packet_generator_test",
    srcs = ["graph_tensors_packet_generator_test.cc"],
    linkstatic = 1,
    deps = [
        ":graph_tensors_packet_generator",
        ":graph_tensors_packet_generator_cc_proto",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:packet_generator_cc_proto",
        "//mediapipe/framework:packet_set",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/tool:validate_type",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_test(
    name = "image_frame_to_tensor_calculator_test",
    size = "small",
    srcs = ["image_frame_to_tensor_calculator_test.cc"],
    linkstatic = 1,
    deps = [
        ":image_frame_to_tensor_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_test(
    name = "matrix_to_tensor_calculator_test",
    size = "small",
    srcs = ["matrix_to_tensor_calculator_test.cc"],
    linkstatic = 1,
    deps = [
        ":matrix_to_tensor_calculator",
        ":matrix_to_tensor_calculator_options_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_test(
    name = "lapped_tensor_buffer_calculator_test",
    size = "small",
    srcs = ["lapped_tensor_buffer_calculator_test.cc"],
    linkstatic = 1,
    deps = [
        ":lapped_tensor_buffer_calculator",
        ":lapped_tensor_buffer_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/memory",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "object_detection_tensors_to_detections_calculator_test",
    srcs = ["object_detection_tensors_to_detections_calculator_test.cc"],
    linkstatic = 1,
    deps = [
        ":object_detection_tensors_to_detections_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:testlib",
    ],
)

cc_test(
    name = "pack_media_sequence_calculator_test",
    srcs = ["pack_media_sequence_calculator_test.cc"],
    deps = [
        ":pack_media_sequence_calculator",
        "//mediapipe/calculators/image:opencv_image_encoder_calculator_cc_proto",
        "//mediapipe/calculators/tensorflow:pack_media_sequence_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/util/sequence:media_sequence",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "tensorflow_session_from_frozen_graph_calculator_test",
    srcs = ["tensorflow_session_from_frozen_graph_calculator_test.cc"],
    data = [":test_frozen_graph"],
    linkstatic = 1,
    deps = [
        ":tensorflow_inference_calculator",
        ":tensorflow_session",
        ":tensorflow_session_from_frozen_graph_calculator",
        ":tensorflow_session_from_frozen_graph_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:tag_map_helper",
        "//mediapipe/framework/tool:validate_type",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:direct_session",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:testlib",
        "@org_tensorflow//tensorflow/core/kernels:conv_ops",
        "@org_tensorflow//tensorflow/core/kernels:math",
    ],
)

cc_test(
    name = "tensorflow_session_from_frozen_graph_generator_test",
    srcs = ["tensorflow_session_from_frozen_graph_generator_test.cc"],
    data = [":test_frozen_graph"],
    linkstatic = 1,
    deps = [
        ":tensorflow_inference_calculator",
        ":tensorflow_session",
        ":tensorflow_session_from_frozen_graph_generator",
        ":tensorflow_session_from_frozen_graph_generator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:packet_generator_cc_proto",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:tag_map_helper",
        "//mediapipe/framework/tool:validate_type",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:direct_session",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:testlib",
        "@org_tensorflow//tensorflow/core/kernels:conv_ops",
        "@org_tensorflow//tensorflow/core/kernels:math",
    ],
)

cc_test(
    name = "tensorflow_session_from_saved_model_generator_test",
    srcs = ["tensorflow_session_from_saved_model_generator_test.cc"],
    data = [":test_saved_model"],
    linkstatic = 1,
    deps = [
        ":tensorflow_inference_calculator",
        ":tensorflow_session",
        ":tensorflow_session_from_saved_model_generator",
        ":tensorflow_session_from_saved_model_generator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:packet_generator_cc_proto",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:tag_map_helper",
        "//mediapipe/framework/tool:validate_type",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:all_kernels",
        "@org_tensorflow//tensorflow/core:direct_session",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "tensorflow_session_from_saved_model_calculator_test",
    srcs = ["tensorflow_session_from_saved_model_calculator_test.cc"],
    data = [":test_saved_model"],
    linkstatic = 1,
    deps = [
        ":tensorflow_inference_calculator",
        ":tensorflow_session",
        ":tensorflow_session_from_saved_model_calculator",
        ":tensorflow_session_from_saved_model_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:tag_map_helper",
        "//mediapipe/framework/tool:validate_type",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:all_kernels",
        "@org_tensorflow//tensorflow/core:direct_session",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "tensor_squeeze_dimensions_calculator_test",
    srcs = ["tensor_squeeze_dimensions_calculator_test.cc"],
    linkstatic = 1,
    deps = [
        ":tensor_squeeze_dimensions_calculator",
        ":tensor_squeeze_dimensions_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "tensor_to_image_frame_calculator_test",
    size = "small",
    srcs = ["tensor_to_image_frame_calculator_test.cc"],
    linkstatic = 1,
    deps = [
        ":tensor_to_image_frame_calculator",
        ":tensor_to_image_frame_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "tensor_to_matrix_calculator_test",
    size = "small",
    srcs = ["tensor_to_matrix_calculator_test.cc"],
    linkstatic = 1,
    deps = [
        ":tensor_to_matrix_calculator",
        ":tensor_to_matrix_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/formats:time_series_header_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "tensor_to_vector_float_calculator_test",
    srcs = ["tensor_to_vector_float_calculator_test.cc"],
    linkstatic = 1,
    deps = [
        ":tensor_to_vector_float_calculator",
        ":tensor_to_vector_float_calculator_options_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "tensor_to_vector_string_calculator_test",
    srcs = ["tensor_to_vector_string_calculator_test.cc"],
    deps = [
        ":tensor_to_vector_string_calculator",
        ":tensor_to_vector_string_calculator_options_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "unpack_media_sequence_calculator_test",
    srcs = ["unpack_media_sequence_calculator_test.cc"],
    deps = [
        ":unpack_media_sequence_calculator",
        "//mediapipe/calculators/core:packet_resampler_calculator_cc_proto",
        "//mediapipe/calculators/tensorflow:unpack_media_sequence_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:rectangle",
        "//mediapipe/util:audio_decoder_cc_proto",
        "//mediapipe/util/sequence:media_sequence",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "vector_int_to_tensor_calculator_test",
    srcs = ["vector_int_to_tensor_calculator_test.cc"],
    linkstatic = 1,
    deps = [
        ":vector_int_to_tensor_calculator",
        ":vector_int_to_tensor_calculator_options_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "vector_float_to_tensor_calculator_test",
    srcs = ["vector_float_to_tensor_calculator_test.cc"],
    linkstatic = 1,
    deps = [
        ":vector_float_to_tensor_calculator",
        ":vector_float_to_tensor_calculator_options_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "vector_string_to_tensor_calculator_test",
    srcs = ["vector_string_to_tensor_calculator_test.cc"],
    deps = [
        ":vector_string_to_tensor_calculator",
        ":vector_string_to_tensor_calculator_options_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

test_suite(
    name = "ios",
    tags = ["ios"],
)

test_suite(
    name = "android",
    tags = ["android"],
)

cc_test(
    name = "tensorflow_inference_calculator_test",
    size = "medium",
    srcs = ["tensorflow_inference_calculator_test.cc"],
    data = [":test_frozen_graph"],
    linkstatic = 1,
    deps = [
        ":tensorflow_session",
        ":tensorflow_inference_calculator",
        ":tensorflow_session_from_frozen_graph_generator",
        ":tensorflow_session_from_frozen_graph_generator_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/tool:sink",
        "//mediapipe/framework/tool:validate_type",
        "//mediapipe/framework/port:gtest_main",
    ] + select({
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:testlib",
            "@org_tensorflow//tensorflow/core/kernels:math",
            "@org_tensorflow//tensorflow/core/kernels:conv_ops",
            "@org_tensorflow//tensorflow/core:direct_session",
        ],
        "//mediapipe:android": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib",
            "@org_tensorflow//tensorflow/core:portable_tensorflow_test_lib",
        ],
        "//mediapipe:ios": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_test_lib",
        ],
    }),
)

filegroup(
    name = "test_frozen_graph",
    srcs = [
        "testdata/bundle/00000000/checkpoint",
        "testdata/bundle/00000000/export.meta",
        "testdata/bundle/00000000/export-00000-of-00001",
        "testdata/frozen_graph_def.pb",
        "testdata/model.chkpt-0",
        "testdata/model.chkpt-0.meta",
        "testdata/tf_graph_def.pb",
    ],
)

filegroup(
    name = "test_saved_model",
    srcs = [
        "testdata/tensorflow_saved_model/00000000/saved_model.pb",
        "testdata/tensorflow_saved_model/00000000/variables/variables.data-00000-of-00001",
        "testdata/tensorflow_saved_model/00000000/variables/variables.index",
    ],
)
