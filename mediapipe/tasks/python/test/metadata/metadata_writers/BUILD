# Placeholder for internal Python strict test compatibility macro.

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

py_test(
    name = "metadata_info_test",
    srcs = ["metadata_info_test.py"],
    data = [
        "//mediapipe/tasks/testdata/metadata:data_files",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//mediapipe/tasks/metadata:metadata_schema_py",
        "//mediapipe/tasks/metadata:schema_py",
        "//mediapipe/tasks/python/metadata",
        "//mediapipe/tasks/python/metadata/metadata_writers:metadata_info",
        "//mediapipe/tasks/python/test:test_utils",
        "@flatbuffers//:runtime_py",
    ],
)

py_test(
    name = "metadata_writer_test",
    srcs = ["metadata_writer_test.py"],
    data = [
        "//mediapipe/tasks/testdata/metadata:data_files",
        "//mediapipe/tasks/testdata/metadata:model_files",
    ],
    deps = [
        "//mediapipe/tasks/metadata:metadata_schema_py",
        "//mediapipe/tasks/python/metadata/metadata_writers:metadata_info",
        "//mediapipe/tasks/python/metadata/metadata_writers:metadata_writer",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)

py_test(
    name = "image_classifier_test",
    srcs = ["image_classifier_test.py"],
    data = [
        "//mediapipe/tasks/testdata/metadata:data_files",
        "//mediapipe/tasks/testdata/metadata:model_files",
    ],
    deps = [
        "//mediapipe/tasks/metadata:metadata_schema_py",
        "//mediapipe/tasks/python/metadata",
        "//mediapipe/tasks/python/metadata/metadata_writers:image_classifier",
        "//mediapipe/tasks/python/metadata/metadata_writers:metadata_writer",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)

py_test(
    name = "text_classifier_test",
    srcs = ["text_classifier_test.py"],
    data = [
        "//mediapipe/tasks/testdata/metadata:data_files",
        "//mediapipe/tasks/testdata/metadata:model_files",
    ],
    deps = [
        "//mediapipe/tasks/python/metadata/metadata_writers:metadata_writer",
        "//mediapipe/tasks/python/metadata/metadata_writers:text_classifier",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)

py_test(
    name = "model_asset_bundle_utils_test",
    srcs = ["model_asset_bundle_utils_test.py"],
    deps = ["//mediapipe/tasks/python/metadata/metadata_writers:model_asset_bundle_utils"],
)

py_test(
    name = "object_detector_test",
    srcs = ["object_detector_test.py"],
    data = [
        "//mediapipe/tasks/testdata/metadata:data_files",
        "//mediapipe/tasks/testdata/metadata:model_files",
    ],
    deps = [
        "//mediapipe/tasks/metadata:metadata_schema_py",
        "//mediapipe/tasks/python/metadata",
        "//mediapipe/tasks/python/metadata/metadata_writers:metadata_info",
        "//mediapipe/tasks/python/metadata/metadata_writers:metadata_writer",
        "//mediapipe/tasks/python/metadata/metadata_writers:object_detector",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)

py_test(
    name = "image_segmenter_test",
    srcs = ["image_segmenter_test.py"],
    data = [
        "//mediapipe/tasks/testdata/metadata:data_files",
        "//mediapipe/tasks/testdata/metadata:model_files",
    ],
    deps = [
        "//mediapipe/tasks/python/metadata",
        "//mediapipe/tasks/python/metadata/metadata_writers:image_segmenter",
        "//mediapipe/tasks/python/metadata/metadata_writers:metadata_writer",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)

py_test(
    name = "face_stylizer_test",
    srcs = ["face_stylizer_test.py"],
    data = [
        "//mediapipe/tasks/testdata/metadata:data_files",
        "//mediapipe/tasks/testdata/metadata:model_files",
    ],
    deps = [
        "//mediapipe/tasks/python/metadata/metadata_writers:face_stylizer",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)
