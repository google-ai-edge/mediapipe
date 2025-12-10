load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = ["//mediapipe/tasks:internal"],
    licenses = ["notice"],  # Apache 2.0
)

pybind_extension(
    name = "_pywrap_metadata_version",
    srcs = [
        "metadata_version.cc",
    ],
    features = ["-use_header_modules"],
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": ["-undefined", "dynamic_lookup"],
        "//conditions:default": [],
    }),
    module_name = "_pywrap_metadata_version",
    deps = [
        "//mediapipe/tasks/cc/metadata:metadata_version",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@pybind11",
    ],
)
