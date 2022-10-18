# Placeholder for internal Python strict library compatibility macro.

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

py_library(
    name = "metadata_info",
    srcs = [
        "metadata_info.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/metadata:metadata_schema_py",
        "//mediapipe/tasks/metadata:schema_py",
    ],
)

py_library(
    name = "metadata_writer",
    srcs = ["metadata_writer.py"],
    deps = [
        ":metadata_info",
        ":writer_utils",
        "//mediapipe/tasks/metadata:metadata_schema_py",
        "//mediapipe/tasks/python/metadata",
        "@flatbuffers//:runtime_py",
    ],
)

py_library(
    name = "writer_utils",
    srcs = ["writer_utils.py"],
    deps = ["//mediapipe/tasks/metadata:schema_py"],
)
