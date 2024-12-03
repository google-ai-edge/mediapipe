# Placeholder for internal Python strict library and test compatibility macro.
# Placeholder: load py_library

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

py_library(
    name = "image_segmenter",
    srcs = ["image_segmenter.py"],
    data = ["//mediapipe/tasks/metadata:image_segmenter_metadata_schema.fbs"],
    deps = [
        ":metadata_info",
        ":metadata_writer",
        ":writer_utils",
        "//mediapipe/tasks/metadata:image_segmenter_metadata_schema_py",
        "//mediapipe/tasks/metadata:metadata_schema_py",
        "//mediapipe/tasks/python/metadata",
        "@flatbuffers//:runtime_py",
    ],
)

py_library(
    name = "object_detector",
    srcs = ["object_detector.py"],
    data = ["//mediapipe/tasks/metadata:object_detector_metadata_schema.fbs"],
    deps = [
        ":metadata_info",
        ":metadata_writer",
        "//mediapipe/tasks/metadata:metadata_schema_py",
        "//mediapipe/tasks/metadata:object_detector_metadata_schema_py",
        "//mediapipe/tasks/python/metadata",
        "@flatbuffers//:runtime_py",
    ],
)

py_library(
    name = "face_stylizer",
    srcs = ["face_stylizer.py"],
    deps = [
        ":metadata_writer",
        ":model_asset_bundle_utils",
    ],
)

py_library(
    name = "model_asset_bundle_utils",
    srcs = ["model_asset_bundle_utils.py"],
)
