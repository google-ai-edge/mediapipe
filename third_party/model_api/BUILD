load("@rules_foreign_cc//foreign_cc:cmake.bzl", "cmake")

_ALL_CONTENT = """\
filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
"""



visibility = ["//visibility:public"]

filegroup(
    name = "all_srcs",
    srcs = glob(["model_api/cpp/**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "model_api_cmake",
    cache_entries = {
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "OpenVINO_DIR": "/opt/intel/openvino/runtime/cmake",
    },
    env = {
        "HTTP_PROXY": "http://proxy-dmz.intel.com:911",
        "HTTPS_PROXY": "http://proxy-dmz.intel.com:912",
    },
    #lib_source = "@model_api_geti//:all_srcs",
    lib_source = ":all_srcs",
    out_static_libs = ["libmodel_api.a"],
    tags = ["requires-network"]
)

cc_library(
    name = "model_api",
    deps = [
        "@mediapipe//mediapipe/framework/port:opencv_core",
        "@linux_openvino//:openvino",
        ":model_api_cmake",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
   name = "adapter_api",
   hdrs = ["model_api/cpp/adapters/include/adapters/inference_adapter.h",],
   includes = ["model_api/cpp/adapters/include"],
   deps = ["@linux_openvino//:openvino"],
   visibility = ["//visibility:public"],
)
