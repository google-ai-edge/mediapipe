package(
    default_visibility = [
        "//mediapipe/tasks:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "zip_writable_mem_file",
    srcs = ["zip_writable_mem_file.cc"],
    hdrs = ["zip_writable_mem_file.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@zlib//:zlib_minizip",
    ],
)

cc_library(
    name = "zip_readonly_mem_file",
    srcs = ["zip_readonly_mem_file.cc"],
    hdrs = ["zip_readonly_mem_file.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@zlib//:zlib_minizip",
    ],
)

cc_library(
    name = "zip_utils",
    srcs = ["zip_utils.cc"],
    hdrs = ["zip_utils.h"],
    deps = [
        ":zip_readonly_mem_file",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@zlib//:zlib_minizip",
    ],
)
