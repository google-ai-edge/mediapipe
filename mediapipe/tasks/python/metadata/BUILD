load("//mediapipe/tasks/metadata:build_defs.bzl", "stamp_metadata_parser_version")
# Placeholder: load py_binary
# Placeholder: load py_library

package(
    licenses = ["notice"],  # Apache 2.0
)

stamp_metadata_parser_version(
    name = "metadata_parser_py",
    srcs = ["metadata_parser.py.template"],
    outs = ["metadata_parser.py"],
)

py_library(
    name = "metadata",
    srcs = [
        "metadata.py",
        ":metadata_parser_py",
    ],
    data = ["//mediapipe/tasks/metadata:metadata_schema.fbs"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/cc/metadata/python:_pywrap_metadata_version",
        "//mediapipe/tasks/metadata:metadata_schema_py",
        "//mediapipe/tasks/metadata:schema_py",
        "//mediapipe/tasks/python/metadata/flatbuffers_lib:_pywrap_flatbuffers",
        "@flatbuffers//:runtime_py",
    ],
)

py_binary(
    name = "metadata_displayer_cli",
    srcs = ["metadata_displayer_cli.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [":metadata"],
)
