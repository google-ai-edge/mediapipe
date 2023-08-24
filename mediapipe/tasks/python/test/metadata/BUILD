# Placeholder: load py_test

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

py_test(
    name = "metadata_test",
    srcs = ["metadata_test.py"],
    data = ["//mediapipe/tasks/testdata/metadata:data_files"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//mediapipe/tasks/metadata:metadata_schema_py",
        "//mediapipe/tasks/metadata:schema_py",
        "//mediapipe/tasks/python/metadata",
        "//mediapipe/tasks/python/test:test_utils",
        "@flatbuffers//:runtime_py",
    ],
)

py_test(
    name = "metadata_parser_test",
    srcs = ["metadata_parser_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = ["//mediapipe/tasks/python/metadata"],
)
