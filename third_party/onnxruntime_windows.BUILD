cc_library(
    name = "onnxruntime",
    srcs = [
        "lib/onnxruntime.dll",
        "lib/onnxruntime.lib",
        "lib/onnxruntime_providers_cuda.dll",
        "lib/onnxruntime_providers_cuda.lib",
        "lib/onnxruntime_providers_shared.dll",
        "lib/onnxruntime_providers_shared.lib",
        "lib/onnxruntime_providers_tensorrt.dll",
        "lib/onnxruntime_providers_tensorrt.lib",
    ],
    hdrs = glob(["include/*.h*"]),
    includes = ["include/"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
