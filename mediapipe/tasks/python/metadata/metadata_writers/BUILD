# Placeholder for internal Python strict library and test compatibility macro.

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
    visibility = ["//visibility:public"],
    deps = [
        ":writer_utils",
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

py_library(
    name = "image_classifier",
    srcs = ["image_classifier.py"],
    deps = [":metadata_writer"],
)

py_library(
    name = "text_classifier",
    srcs = ["text_classifier.py"],
    deps = [":metadata_writer"],
)
